"""Captura fake: sem microfone, GPU, desktop ou socket da sessao viva."""
import queue
import threading
import unittest
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app


class CaptureWatchdogTests(unittest.TestCase):
    def setUp(self):
        with patch.object(threading.Thread, 'start'), patch.object(app.keyboard, 'Controller'):
            self.t = app.Transcriber(queue.Queue(), queue.Queue())
        self.now = 10.0
        self.clock = patch.object(app.time, 'monotonic', side_effect=lambda: self.now)
        self.clock.start()
        self.addCleanup(self.clock.stop)
        self.callbacks = []

    def start(self, mode='os_dois'):
        def open_fake(_device, _loopback, label):
            self.t._resamplers.append(None)
            self.callbacks.append(self.t._make_callback(label, self.t._slot))
        with patch.object(self.t, '_open_stream', side_effect=open_fake):
            self.t.start(None, False, capture_mode=mode)
        self.t.status_queue.get_nowait()

    def callback(self, slot, values=(.1, .2)):
        data = np.asarray(values, dtype=np.float32).reshape(-1, 1)
        self.callbacks[slot](data, len(data), None, None)

    def audio_item(self, expected_sid=None):
        sid, audio = self.t._audio_queue.get_nowait()
        self.assertEqual(sid, self.t._session_id if expected_sid is None else expected_sid)
        return audio

    def mix_tick(self):
        with patch.object(app.time, 'sleep', side_effect=[None, StopIteration]):
            with self.assertRaises(StopIteration):
                self.t._mixer_loop()

    def test_stopped_mic_callback_reports_error_and_unblocks_live_slot(self):
        self.start()
        self.callback(0)
        self.callback(1, (.3, .4))
        self.mix_tick()
        np.testing.assert_allclose(self.audio_item(), [.4, .6])
        self.now += 2.01
        self.callback(1, (.5, .6))
        self.mix_tick()
        self.assertFalse(self.t._audio_queue.empty(), 'slot vivo ficou preso esperando o mic morto')
        np.testing.assert_allclose(self.audio_item(), [.5, .6])
        self.assertEqual(self.t.status_queue.get_nowait(), 'ERRO: microfone parou de enviar áudio')
        self.mix_tick()
        self.assertTrue(self.t.status_queue.empty(), 'erro repetido a cada tick')
        self.assertTrue(self.t.recording.is_set())

    def test_single_mic_without_any_callback_reports_error(self):
        self.start('microfone')
        self.now += 2.01
        self.mix_tick()
        self.assertFalse(self.t.status_queue.empty(), 'mic sem primeira callback ficou sem aviso')
        self.assertEqual(self.t.status_queue.get_nowait(), 'ERRO: microfone parou de enviar áudio')

    def test_missing_slot_still_waits_at_exactly_two_seconds(self):
        self.start()
        self.now += 2.0
        self.callback(1)
        self.mix_tick()
        self.assertTrue(self.t._audio_queue.empty())
        self.assertTrue(self.t.status_queue.empty())

    def test_continuous_silent_callbacks_are_not_dead_microphones(self):
        self.start('microfone')
        for _ in range(4):
            self.now += 1.5
            self.callback(0, (0., 0.))
            self.mix_tick()
            np.testing.assert_array_equal(self.audio_item(), [0., 0.])
        self.assertTrue(self.t.status_queue.empty())

    def test_no_watchdog_error_after_stop(self):
        self.start('microfone')
        self.t.stop()
        self.t.status_queue.get_nowait()
        self.now += 3.0
        self.mix_tick()
        self.assertTrue(self.t.status_queue.empty())
        self.assertIsNone(self.audio_item())

    def test_silent_loopback_unblocks_microphone_without_error(self):
        self.start()
        self.now += 2.01
        self.callback(0)
        self.mix_tick()
        np.testing.assert_allclose(self.audio_item(), [.1, .2])
        self.assertTrue(self.t.status_queue.empty())

    def test_resumed_callback_rejoins_mixing_and_can_timeout_again(self):
        self.start()
        self.now += 2.01
        self.callback(1)
        self.mix_tick()
        self.audio_item()
        self.t.status_queue.get_nowait()
        self.callback(0)
        self.mix_tick()
        self.assertTrue(self.t._audio_queue.empty(), 'slot recuperado deve esperar o vizinho vivo')
        self.callback(1, (.3, .4))
        self.mix_tick()
        np.testing.assert_allclose(self.audio_item(), [.4, .6])
        self.now += 2.01
        self.callback(1)
        self.mix_tick()
        self.assertEqual(self.t.status_queue.get_nowait(), 'ERRO: microfone parou de enviar áudio')

    def test_timeout_preserves_buffered_tail_and_stop_marker(self):
        self.start()
        self.callback(0, (.1, .2, .3))
        self.callback(1, (.4,))
        self.mix_tick()
        np.testing.assert_allclose(self.audio_item(), [.5])
        self.now += 2.01
        self.callback(1, (.4, .4, .4))
        self.mix_tick()
        np.testing.assert_allclose(self.audio_item(), [.6, .7])
        np.testing.assert_allclose(self.audio_item(), [.4])
        self.callback(1, (.8,))
        self.t.stop()
        np.testing.assert_allclose(self.audio_item(), [.8])
        self.assertIsNone(self.audio_item())
        self.assertTrue(self.t._audio_queue.empty())

    def test_new_session_resets_dead_slots_and_start_deadline(self):
        self.start('microfone')
        old_sid = self.t._session_id
        self.now += 2.01
        self.mix_tick()
        self.t.status_queue.get_nowait()
        self.t.stop()
        self.t.status_queue.get_nowait()
        self.t._drained = True  # fake nao tem segmentador consumindo o marcador
        self.callbacks.clear()
        self.now += 10.0
        self.start()
        self.callback(1)
        self.mix_tick()
        self.assertTrue(self.audio_item(expected_sid=old_sid) is None)
        self.assertTrue(self.t._audio_queue.empty())
        self.assertTrue(self.t.status_queue.empty())

    def test_loopback_blocks_refresh_watchdog_without_device_or_model(self):
        self.start('audio_pc')
        handle = SimpleNamespace(stop_flag=threading.Event())
        recorder = Mock()
        blocks = iter([np.ones((4800, 2), dtype=np.float32) * .25] * 3)
        def record(**_kwargs):
            self.now += 1.5
            try:
                return next(blocks)
            except StopIteration:
                handle.stop_flag.set()
                return None
        recorder.record.side_effect = record
        loop = Mock()
        loop.recorder.return_value.__enter__ = Mock(return_value=recorder)
        loop.recorder.return_value.__exit__ = Mock(return_value=False)
        with patch.object(app, '_open_loopback_mic', return_value=loop):
            self.t._loopback_loop(0, 'audio do PC', None, handle)
        self.mix_tick()
        self.assertFalse(self.t._audio_queue.empty())
        self.assertTrue(self.t.status_queue.empty())
        self.now += .51
        self.mix_tick()
        self.assertTrue(self.t.status_queue.empty())

    def test_loopback_only_without_blocks_does_not_report_microphone_error(self):
        self.start('audio_pc')
        self.now += 10.0
        self.mix_tick()
        self.assertTrue(self.t.status_queue.empty())

    def test_unexpected_loopback_exit_reports_error_and_unblocks_microphone(self):
        self.start()
        handle = SimpleNamespace(stop_flag=threading.Event())
        with patch.object(app, '_open_loopback_mic', side_effect=RuntimeError('processo morreu')):
            self.t._loopback_loop(1, 'audio do PC', None, handle)
        self.assertEqual(self.t.status_queue.get_nowait(), 'ERRO (audio do PC): processo morreu')
        self.assertEqual(self.t.status_queue.get_nowait(), 'ERRO: áudio do PC parou de enviar áudio')
        self.callback(0)
        self.mix_tick()
        np.testing.assert_allclose(self.audio_item(), [.1, .2])
        self.assertTrue(self.t.status_queue.empty())


if __name__ == '__main__':
    unittest.main()
