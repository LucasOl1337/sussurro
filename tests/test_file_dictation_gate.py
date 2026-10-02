"""Gate de arquivo sem GPU/mic, contrato Hermes pelo socket e CLI reais."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import queue
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import wave
from unittest.mock import Mock, patch

import app


class FileDictationGateTests(unittest.TestCase):
    def setUp(self):
        with patch.object(threading.Thread, 'start'), patch.object(app.keyboard, 'Controller'):
            self.t = app.Transcriber(queue.Queue(), queue.Queue())
        self.t.model = Mock()
        self.t.model.transcribe.return_value = (iter([]), None)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.folder = Path(self.directory.name)
        for name, value in [('HISTORY_DIR', self.folder), ('HISTORY_INDEX', self.folder / 'history.jsonl')]:
            patcher = patch.object(app, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.source = self.folder / 'quiet.wav'
        with wave.open(str(self.source), 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(app.SAMPLE_RATE)
            wav.writeframes(b'\x00\x00' * app.SAMPLE_RATE)

    def assert_unchanged(self):
        self.assertTrue(self.t.history_queue.empty())
        self.assertTrue(self.t.status_queue.empty())
        self.assertFalse(app.HISTORY_INDEX.exists())
        self.assertEqual(list(self.folder.glob('*.wav')), [self.source])
        self.assertEqual(self.t._file_jobs, 0)

    def test_active_dictation_rejects_before_decode_or_model(self):
        self.t.recording.set()
        with patch.object(app, 'load_audio_16k_mono', wraps=app.load_audio_16k_mono) as decode:
            with self.assertRaisesRegex(RuntimeError, 'ditado'):
                self.t.transcribe_file(str(self.source))
        decode.assert_not_called()
        self.t.model.transcribe.assert_not_called()
        self.assert_unchanged()

    def test_stopped_dictation_still_draining_is_busy_for_files(self):
        self.t._drained = False
        with patch.object(app, 'load_audio_16k_mono', wraps=app.load_audio_16k_mono) as decode:
            with self.assertRaisesRegex(RuntimeError, 'ditado'):
                self.t.transcribe_file(str(self.source))
        decode.assert_not_called()
        self.assert_unchanged()

    def test_idle_silent_file_keeps_diagnosis_without_history_or_status(self):
        with self.assertRaisesRegex(ValueError, 'Microfone não enviou sinal'):
            self.t.transcribe_file(str(self.source))
        self.t.model.transcribe.assert_called_once()
        self.assert_unchanged()

    def test_pending_dictation_rejects_even_if_recording_is_clear(self):
        self.t._pending = 1
        with self.assertRaisesRegex(RuntimeError, 'ditado'):
            self.t.transcribe_file(str(self.source))
        self.t.model.transcribe.assert_not_called()
        self.assert_unchanged()

    def cli(self):
        server = app.IpcServer(queue.Queue(), self.folder / 'sussurro.sock', self.t)
        worker = threading.Thread(target=server.run, daemon=True)
        worker.start()
        def stop():
            server._stop.set()
            worker.join(timeout=2)
            self.assertFalse(worker.is_alive())
        self.addCleanup(stop)
        deadline = time.monotonic() + 2
        while not server.path.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertTrue(server.path.exists())
        env = {**os.environ, 'XDG_RUNTIME_DIR': str(self.folder), 'DISPLAY': '', 'WAYLAND_DISPLAY': ''}
        return subprocess.run([sys.executable, '-S', app.__file__, 'transcribe', str(self.source)],
                              env=env, capture_output=True, text=True, timeout=5)

    def test_hermes_cli_busy_response_is_json_with_exit_one_on_private_ipc(self):
        self.t.recording.set()
        result = self.cli()
        self.assertEqual(result.returncode, 1, result.stderr)
        payload = json.loads(result.stdout)
        self.assertFalse(payload['ok'])
        self.assertIn('ditado', payload['error'])
        self.assertEqual(set(payload), {'ok', 'error'})
        self.t.model.transcribe.assert_not_called()
        self.assert_unchanged()

    def test_hermes_cli_idle_silent_file_returns_diagnosis_and_no_side_effects(self):
        result = self.cli()
        self.assertEqual(result.returncode, 1)
        payload = json.loads(result.stdout)
        self.assertFalse(payload['ok'])
        self.assertIn('Microfone não enviou sinal', payload['error'])
        self.assert_unchanged()

    def test_hermes_cli_idle_success_retains_json_history_contract(self):
        self.t.model.transcribe.return_value = (iter([SimpleNamespace(text='texto normal', start=0., end=1.)]), None)
        result = self.cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertTrue(payload.pop('ok'))
        self.assertEqual(payload['text'], 'texto normal')
        self.assertEqual(payload, self.t.history_queue.get_nowait())
        self.assertEqual(payload, json.loads(app.HISTORY_INDEX.read_text()))
        self.assertEqual(self.t._file_jobs, 0)


if __name__ == '__main__':
    unittest.main()
