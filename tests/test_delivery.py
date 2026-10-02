"""Entrega FIFO sem bloquear inferencia, sem GPU, clipboard ou teclado reais."""
import queue
import sys
import threading
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app


_END = object()


class DeliveryTests(unittest.TestCase):
    def transcriber(self, texts=('primeiro', 'segundo'), inject=True):
        with patch.object(threading.Thread, 'start'), patch.object(app.keyboard, 'Controller'):
            t = app.Transcriber(queue.Queue(), queue.Queue())
        t.library.apply = lambda text: (text, 0)
        t._session_id = 1
        t._session_inject = inject
        t._drained = False
        t._finalize_session = Mock()
        t._press_enter = Mock()
        outputs = iter(texts)
        t._transcribe_locked = Mock(side_effect=lambda *a, **k: (
            [SimpleNamespace(text=next(outputs), start=0.0, end=1.0)], None))
        for _ in texts:
            t._enqueue_segment(np.ones(app.SAMPLE_RATE, dtype=np.float32))
        t._segment_queue.put((None, None, None, 1, 0.0))
        return t

    @contextmanager
    def workers(self, t, release):
        errors = []
        threads = []
        queues = [t._segment_queue]
        targets = [t._transcribe_loop]
        if hasattr(t, '_delivery_loop'):
            queues.append(t._delivery_queue)
            targets.append(t._delivery_loop)
        for q, target in zip(queues, targets):
            get = q.get
            def finite_get(*args, get=get, **kwargs):
                item = get(*args, **kwargs)
                if item is _END:
                    raise StopIteration
                return item
            q.get = finite_get
            def run(target=target):
                try:
                    target()
                except StopIteration:
                    pass
                except BaseException as error:
                    errors.append(error)
            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            threads.append(thread)
        try:
            yield
        finally:
            release.set()
            for q, thread in zip(queues, threads):
                q.put(_END)
                thread.join(2)
                self.assertFalse(thread.is_alive(), 'consumidor nao terminou')
            self.assertEqual(errors, [])

    def test_second_inference_runs_while_first_paste_is_blocked(self):
        t = self.transcriber()
        t._session_auto_enter = True
        blocked, release, second = (threading.Event() for _ in range(3))
        delivered = []
        def paste(text):
            if not delivered:
                blocked.set()
                if not release.wait(2):
                    raise RuntimeError('paste nao liberado')
            delivered.append(text)
        infer = t._transcribe_locked.side_effect
        def transcribe(*args, **kwargs):
            result = infer(*args, **kwargs)
            if t._transcribe_locked.call_count == 2:
                second.set()
            return result
        t._transcribe_locked.side_effect = transcribe
        t._paste = paste
        t._press_enter.side_effect = lambda *a: delivered.append('ENTER')
        with patch.object(app, '_perf') as perf, self.workers(t, release):
            self.assertTrue(blocked.wait(1))
            self.assertTrue(second.wait(.5), '2o trecho espera o paste do 1o na thread de transcricao')
            self.assertTrue(t.busy())
            self.assertEqual(delivered, [])
            t._finalize_session.assert_not_called()
            # Tempo controlado da entrega deve aparecer so em delivery_ms.
            time.sleep(.06)
        self.assertEqual(delivered, ['primeiro', ' segundo', 'ENTER'])
        self.assertEqual(t._pending, 0)
        self.assertTrue(t._drained)
        t._finalize_session.assert_called_once()
        metrics = [call.kwargs for call in perf.call_args_list if call.args[0] == 'segment_done']
        self.assertEqual(len(metrics), 2)
        self.assertGreaterEqual(metrics[0]['delivery_ms'], 60)
        self.assertLess(metrics[0]['inference_ms'], metrics[0]['delivery_ms'])
        self.assertTrue(all(metric['injected'] for metric in metrics))

    def test_fifo_foot_keeps_ctrl_shift_v_and_paragraphs_before_enter(self):
        t = self.transcriber()
        first = t._segment_queue.get_nowait()
        second = t._segment_queue.get_nowait()
        marker = t._segment_queue.get_nowait()
        t._segment_queue.put(first)
        t._segment_queue.put((*second[:4], app.PARAGRAPH_SILENCE_S))
        t._segment_queue.put(marker)
        t._session_auto_enter = True
        release = threading.Event()
        actions = []
        t._press_enter.side_effect = lambda *a: actions.append('ENTER')
        with patch.object(app, '_perf'), \
             patch.object(app, 'IS_WIN', False), \
             patch.object(app, 'prepare_paste_target', return_value='foot'), \
             patch.object(app, 'set_clipboard_text', side_effect=lambda text: actions.append(text) or True), \
             patch.object(app, '_is_wayland', return_value=True), \
             patch.object(app, '_ydotool_keys', return_value=True) as keys, \
             self.workers(t, release):
            pass
        self.assertEqual(actions, ['primeiro', '\n\nsegundo', 'ENTER'])
        self.assertEqual(keys.call_count, 2)
        for call in keys.call_args_list:
            self.assertEqual(call.args, ('29:1', '42:1', '47:1', '47:0', '42:0', '29:0'))

    def test_cancel_drops_queued_paste_and_enter_without_decrementing_new_session(self):
        t = self.transcriber()
        t._session_auto_enter = True
        blocked, release, inferred = (threading.Event() for _ in range(3))
        pasted = []
        def paste(text):
            blocked.set()
            if not release.wait(2):
                raise RuntimeError('paste nao liberado')
            pasted.append(text)
        t._paste = paste
        # O marker de fim so pode entrar na entrega depois da 2a inferencia.
        put = t._delivery_queue.put
        def enqueue(item):
            put(item)
            if item is not _END and item[0] is None:
                inferred.set()
        t._delivery_queue.put = enqueue
        with patch.object(app, '_perf'), self.workers(t, release):
            self.assertTrue(blocked.wait(1))
            self.assertTrue(inferred.wait(1))
            t.cancel(from_processing=True)
            self.assertEqual(t._pending, 0)
            # O envio ja iniciado nao pode ser desfeito, mas sua baixa nao pertence a este sid.
            t._session_id += 1
            t._pending = 3
        self.assertEqual(pasted, ['primeiro'])
        self.assertEqual(t._pending, 3)
        t._press_enter.assert_not_called()
        t._finalize_session.assert_not_called()

    def test_delivery_failure_keeps_worker_alive_and_suppresses_enter(self):
        t = self.transcriber()
        t._session_auto_enter = True
        t._paste = Mock(side_effect=[RuntimeError('falha na colagem'), None])
        with patch.object(app, '_perf'), patch.object(app.traceback, 'print_exc'), \
             self.workers(t, threading.Event()):
            pass
        self.assertEqual([call.args[0] for call in t._paste.call_args_list], ['primeiro', ' segundo'])
        self.assertEqual(t._session_errors, ['RuntimeError: falha na colagem'])
        t._press_enter.assert_not_called()
        t._finalize_session.assert_called_once()
        self.assertEqual(t._pending, 0)
        self.assertTrue(t._drained)

    def test_typing_uses_delivery_worker_in_order(self):
        t = self.transcriber()
        t.inject_method = 'digitar'
        t._paste = Mock()
        t._type_fallback = Mock()
        with patch.object(app, '_perf'), self.workers(t, threading.Event()):
            pass
        t._paste.assert_not_called()
        self.assertEqual([call.args[0] for call in t._type_fallback.call_args_list],
                         ['primeiro', ' segundo'])
        self.assertEqual(t._pending, 0)

    def test_non_injected_and_empty_text_finish_without_keyboard(self):
        for texts, inject in [(('primeiro', 'segundo'), False), (('', ''), True)]:
            with self.subTest(texts=texts, inject=inject):
                t = self.transcriber(texts, inject)
                t._session_auto_enter = True
                t._paste = Mock()
                t._type_fallback = Mock()
                with patch.object(app, '_perf') as perf, self.workers(t, threading.Event()):
                    pass
                t._paste.assert_not_called()
                t._type_fallback.assert_not_called()
                t._press_enter.assert_not_called()
                t._finalize_session.assert_called_once()
                self.assertEqual(t._pending, 0)
                self.assertTrue(t._drained)
                self.assertTrue(all(not call.kwargs['injected'] for call in perf.call_args_list))

    def test_cancel_during_enter_delay_does_not_dispatch_key(self):
        t = self.transcriber()
        # Usa o metodo real de Enter, mas nenhuma tecla real.
        with patch.object(app.time, 'sleep', side_effect=lambda seconds: setattr(t, '_session_id', 2)), \
             patch.object(app, '_ydotool_keys') as keys:
            app.Transcriber._press_enter(t, 1)
        keys.assert_not_called()
        t._keyboard.press.assert_not_called()

    def test_stale_session_items_are_ignored_even_if_enqueued_after_cancel(self):
        t = self.transcriber(texts=())
        t.cancel(from_processing=True)
        t._pending = 2
        t._drained = False
        t._paste = Mock()
        t._delivery_queue.put((1.0, 'texto cancelado', 'colar', 1, time.perf_counter(), .01, True))
        t._delivery_queue.put((None, None, None, 1, None, None, False))
        with patch.object(app, '_perf') as perf, self.workers(t, threading.Event()):
            pass
        t._paste.assert_not_called()
        t._press_enter.assert_not_called()
        t._finalize_session.assert_not_called()
        perf.assert_not_called()
        self.assertEqual(t._pending, 2)
        self.assertFalse(t._drained)


if __name__ == '__main__':
    unittest.main()
