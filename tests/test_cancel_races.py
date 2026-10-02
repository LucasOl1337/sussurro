"""Corridas do cancel com threads reais, modelo fake e historico temporario."""
import json
import queue
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app


class CancelRaceTests(unittest.TestCase):
    def transcriber(self):
        with patch.object(threading.Thread, 'start'), patch.object(app.keyboard, 'Controller'):
            t = app.Transcriber(queue.Queue(), queue.Queue())
        t._session_id = 1
        t._drained = False
        t._session_inject = True
        t._session_started = app.datetime(2026, 10, 2, 12, 0, 0)
        t.library = SimpleNamespace(apply=lambda text: (text, 0))
        return t

    def consume(self, loop, source, items):
        values = iter(items)
        original = source.get
        def get(block=True, timeout=None):
            return next(values) if block else original(block=False)
        with patch.object(source, 'get', side_effect=get):
            with self.assertRaises(StopIteration):
                loop()

    def test_cancel_during_vad_cannot_relabel_old_audio_as_new_session(self):
        t = self.transcriber()
        audio = np.ones(app.SAMPLE_RATE, dtype=np.float32) * .05
        # Alimenta pelo mixer real, tal como a captura faz.
        t._mix_buffers = [[audio]]
        t._drain_mix_locked()
        item = t._audio_queue.get_nowait()
        def cancel_in_vad(_audio, _options):
            t.cancel(from_processing=True)
            return [{'start': 0, 'end': app.SAMPLE_RATE // 8}]
        with patch.object(app, 'get_speech_timestamps', side_effect=cancel_in_vad):
            self.consume(t._segmenter_loop, t._audio_queue, [item])
        self.assertTrue(t._segment_queue.empty(), 'audio cancelado foi reenfileirado com sid novo')
        self.assertEqual(t._session_audio, [])
        self.assertEqual(t._pending, 0)

    def test_cancel_during_stop_vad_cannot_finalize_replacement_session(self):
        t = self.transcriber()
        audio = np.ones(app.SAMPLE_RATE, dtype=np.float32) * .05
        t.recording.set()
        t._mix_buffers = [[audio]]
        t.stop()
        items = [t._audio_queue.get_nowait(), t._audio_queue.get_nowait()]
        calls = 0
        def vad(_audio, _options):
            nonlocal calls
            calls += 1
            if calls == 2:
                t.cancel(from_processing=True)
                t._session_id += 1  # nova gravacao depois do cancel
                t._drained = False
                t.recording.set()
            return [{'start': 0, 'end': app.SAMPLE_RATE}]
        with patch.object(app, 'get_speech_timestamps', side_effect=vad):
            self.consume(t._segmenter_loop, t._audio_queue, items)
        with patch.object(t, '_transcribe_locked', return_value=([], None)) as transcribe, \
             patch.object(t, '_finalize_session') as finalize:
            queued = []
            while not t._segment_queue.empty():
                queued.append(t._segment_queue.get_nowait())
            self.consume(t._transcribe_loop, t._segment_queue, queued)
            deliveries = []
            while not t._delivery_queue.empty():
                deliveries.append(t._delivery_queue.get_nowait())
            self.consume(t._delivery_loop, t._delivery_queue, deliveries)
            transcribe.assert_not_called()
            finalize.assert_not_called()
        self.assertFalse(t._drained, 'marcador antigo deu baixa na nova gravacao')

    def test_cancel_while_writing_wav_never_publishes_history(self):
        t = self.transcriber()
        audio = np.ones(app.SAMPLE_RATE, dtype=np.float32) * .05
        t._session_audio = [audio]
        t._session_parts = [(audio, 'texto cancelado', 0, False)]
        entered = threading.Event()
        release = threading.Event()
        original = app.wave.Wave_write.writeframes
        errors = []
        def delayed_write(writer, data):
            entered.set()
            if not release.wait(3):
                raise TimeoutError('escrita de WAV nao liberada')
            return original(writer, data)
        def finalize():
            try:
                t._finalize_session()
            except BaseException as exc:
                errors.append(exc)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(app, 'HISTORY_DIR', Path(directory)), \
             patch.object(app, 'HISTORY_INDEX', Path(directory) / 'history.jsonl'), \
             patch.object(app.wave.Wave_write, 'writeframes', delayed_write):
            worker = threading.Thread(target=finalize)
            worker.start()
            try:
                self.assertTrue(entered.wait(2))
                t.cancel(from_processing=True)
            finally:
                release.set()
                worker.join(3)
            self.assertFalse(worker.is_alive())
            self.assertEqual(errors, [])
            self.assertTrue(t.history_queue.empty(), 'sessao cancelada chegou ao historico da UI')
            self.assertEqual(list(Path(directory).iterdir()), [], 'cancel deixou WAV ou indice')

    def test_new_final_session_drops_old_buffer_even_without_cancel_marker(self):
        t = self.transcriber()
        t._session_mode = 'final'
        old = np.ones(app.SAMPLE_RATE, dtype=np.float32) * .05
        new = np.ones(app.SAMPLE_RATE, dtype=np.float32) * .15
        values = iter([(1, old), (2, new), (1, old), (2, None)])
        def get():
            item = next(values)
            t._session_id = 2 if item[0] == 2 else t._session_id
            return item
        with patch.object(t._audio_queue, 'get', side_effect=get):
            with self.assertRaises(StopIteration):
                t._segmenter_loop()
        segment = t._segment_queue.get_nowait()
        np.testing.assert_array_equal(segment[0], new)
        self.assertEqual(segment[3], 2)
        self.assertEqual(t._segment_queue.get_nowait()[3], 2)
        self.assertTrue(t._segment_queue.empty())

    def test_failed_wav_write_cleans_temporary_file_without_history(self):
        t = self.transcriber()
        t._session_audio = [np.ones(app.SAMPLE_RATE, dtype=np.float32)]
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(app, 'HISTORY_DIR', Path(directory)), \
             patch.object(app, 'HISTORY_INDEX', Path(directory) / 'history.jsonl'), \
             patch.object(app.wave.Wave_write, 'writeframes', side_effect=OSError('disco cheio')):
            with self.assertRaisesRegex(OSError, 'disco cheio'):
                t._finalize_session()
            self.assertEqual(list(Path(directory).iterdir()), [])
            self.assertTrue(t.history_queue.empty())

    def test_cancel_during_inference_discards_result_without_paste(self):
        t = self.transcriber()
        audio = np.ones(app.SAMPLE_RATE, dtype=np.float32) * .05
        t._enqueue_segment(audio)
        item = t._segment_queue.get_nowait()
        def infer(*args, **kwargs):
            t.cancel(from_processing=True)
            return ([SimpleNamespace(text='nao colar', start=0.0, end=1.0)], None)
        with patch.object(t, '_transcribe_locked', side_effect=infer), \
             patch.object(t, '_paste') as paste, patch.object(t, '_type_fallback') as typing:
            self.consume(t._transcribe_loop, t._segment_queue, [item])
        paste.assert_not_called()
        typing.assert_not_called()
        self.assertTrue(t.text_queue.empty())
        self.assertTrue(t.history_queue.empty())
        self.assertEqual(t._session_parts, [])
        self.assertEqual(t._pending, 0)

    def test_normal_final_dictation_still_archives_one_wav_and_entry(self):
        t = self.transcriber()
        t._session_mode = 'final'
        t._session_inject = False
        t.recording.set()
        audio = np.ones(app.SAMPLE_RATE, dtype=np.float32) * .05
        t._mix_buffers = [[audio]]
        t.stop()
        items = [t._audio_queue.get_nowait(), t._audio_queue.get_nowait()]
        self.consume(t._segmenter_loop, t._audio_queue, items)
        queued = [t._segment_queue.get_nowait(), t._segment_queue.get_nowait()]
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(app, 'HISTORY_DIR', Path(directory)), \
             patch.object(app, 'HISTORY_INDEX', Path(directory) / 'history.jsonl'), \
             patch.object(t, '_transcribe_locked', return_value=(
                 [SimpleNamespace(text='ditado confirmado', start=0.0, end=1.0)], None)), \
             patch.object(app, '_perf'):
            self.consume(t._transcribe_loop, t._segment_queue, queued)
            deliveries = [t._delivery_queue.get_nowait(), t._delivery_queue.get_nowait()]
            self.consume(t._delivery_loop, t._delivery_queue, deliveries)
            entry = json.loads(app.HISTORY_INDEX.read_text())
            self.assertEqual(entry['text'], 'ditado confirmado')
            self.assertTrue((app.HISTORY_DIR / entry['wav']).exists())
            self.assertEqual(t.history_queue.get_nowait(), entry)
            self.assertEqual(t.text_queue.get_nowait(), 'ditado confirmado')
            self.assertTrue(t._drained)
            self.assertEqual(t._pending, 0)


if __name__ == '__main__':
    unittest.main()
