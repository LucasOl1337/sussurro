"""Regressoes do caminho de ditado; nao abrem mic, colam no desktop nem usam a GPU."""
import collections
import json
import os
import queue
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app
import numpy as np


class LatencyTests(unittest.TestCase):
    def setUp(self):
        perf = patch.object(app, '_perf')
        perf.start()
        self.addCleanup(perf.stop)

    def transcriber(self):
        with patch.object(threading.Thread, 'start'), patch.object(app.keyboard, 'Controller'):
            return app.Transcriber(queue.Queue(), queue.Queue())

    def test_cli_without_site_packages_or_display(self):
        with tempfile.TemporaryDirectory() as directory:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
                server.bind(str(Path(directory) / 'sussurro.sock'))
                server.listen(1)
                seen = []
                def serve():
                    connection, _ = server.accept()
                    with connection:
                        seen.append(connection.recv(128))
                        connection.sendall(b'ok\n')
                thread = threading.Thread(target=serve, daemon=True)
                thread.start()
                env = {**os.environ, 'XDG_RUNTIME_DIR': directory, 'DISPLAY': '', 'WAYLAND_DISPLAY': ''}
                result = subprocess.run([sys.executable, '-S', app.__file__, 'toggle'],
                                        env=env, capture_output=True, text=True, timeout=3)
                thread.join(timeout=1)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(seen, [b'toggle\n'])
                self.assertEqual(result.stdout, 'ok\n')

    def test_warmup_exhausts_generator_before_publishing_model(self):
        t = self.transcriber()
        decoded = []
        def generate():
            self.assertIsNone(t.model)
            decoded.append(True)
            yield None
        model = SimpleNamespace(model=SimpleNamespace(device='cuda', compute_type='float16'),
                                transcribe=Mock(return_value=(generate(), None)))
        with patch.object(app, 'WhisperModel', return_value=model), patch.object(app, 'get_speech_timestamps') as vad:
            t.load_model()
        self.assertEqual(decoded, [True])
        self.assertIs(t.model, model)
        vad.assert_called_once()

    def test_stop_flushes_all_channels_before_marker(self):
        t = self.transcriber()
        t.recording.set()
        t._mix_buffers = [[np.array([.1, .2, .3], dtype=np.float32)],
                          [np.array([.4], dtype=np.float32)]]
        t.stop()
        chunks = []
        while True:
            chunk = t._audio_queue.get_nowait()
            if chunk is None:
                break
            chunks.append(chunk)
        np.testing.assert_allclose(np.concatenate(chunks), [.5, .2, .3])
        self.assertTrue(t._audio_queue.empty())
        self.assertFalse(t.recording.is_set())

    def test_new_session_cannot_overwrite_pending_history(self):
        t = self.transcriber()
        t._drained = False
        t._session_parts = ['previous']
        with self.assertRaises(RuntimeError):
            t.start(None, False)
        self.assertEqual(t._session_parts, ['previous'])
        self.assertEqual(t._session_id, 0)

    def test_paste_returns_before_restore_and_restores_original(self):
        t = self.transcriber()
        clipboard = [b'original']
        restored = threading.Event()
        def write(text):
            clipboard[0] = text.encode(); return True
        def restore(data):
            clipboard[0] = data; restored.set()
        with patch.object(app, 'backup_clipboard', side_effect=lambda: clipboard[0]), \
             patch.object(app, 'set_clipboard_text', side_effect=write), \
             patch.object(app, 'restore_clipboard', side_effect=restore), \
             patch.object(t, '_send_paste_key', return_value=True):
            start = time.perf_counter()
            t._paste_linux('ação rápida')
            self.assertLess(time.perf_counter() - start, .25)
            self.assertEqual(clipboard[0], 'ação rápida'.encode())
            self.assertFalse(restored.is_set())
            self.assertTrue(restored.wait(2))
            self.assertEqual(clipboard[0], b'original')

    @unittest.skipIf(app.IS_WIN, 'Clipboard owner process is a Unix behavior')
    def test_copy_and_restore_do_not_wait_for_background_clipboard_owner(self):
        # Simula wl-copy: pai termina; filho continua com os descritores herdados.
        # Nenhum clipboard real e acessado por este teste.
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'selection'
            code = ('import os,sys,time; from pathlib import Path; '
                    'Path(sys.argv[1]).write_bytes(sys.stdin.buffer.read()); '
                    'child=os.fork(); '
                    'time.sleep(1) if child == 0 else None; os._exit(0)')
            with patch.object(app, '_CLIP_WRITE', [sys.executable, '-c', code, str(output)]):
                started = time.perf_counter()
                self.assertTrue(app.set_clipboard_text('ação'))
                self.assertLess(time.perf_counter() - started, .6)
                self.assertEqual(output.read_bytes(), 'ação'.encode())
                started = time.perf_counter()
                app.restore_clipboard(b'original')
                self.assertLess(time.perf_counter() - started, .6)
                self.assertEqual(output.read_bytes(), b'original')

    def test_late_restore_does_not_overwrite_user_copy_or_new_paste(self):
        t = self.transcriber()
        pending = (b'original', b'dictation', 0)
        t._clipboard_restore = pending
        with patch.object(app, 'backup_clipboard', return_value=b'user copied this'), \
             patch.object(app, 'restore_clipboard') as restore:
            t._restore_linux_clipboard(pending)
            restore.assert_not_called()
            newer = (b'user copied this', b'new dictation', 0)
            t._clipboard_restore = newer
            t._restore_linux_clipboard(pending)
            self.assertIs(t._clipboard_restore, newer)
            restore.assert_not_called()

    def test_no_second_vad_scan_for_final_mode(self):
        t = self.transcriber()
        t._session_mode = 'final'
        audio = np.ones(app.SAMPLE_RATE, dtype=np.float32)
        events = iter([audio, None])
        t._audio_queue.get = lambda: next(events)
        with patch.object(app, 'get_speech_timestamps') as vad:
            with self.assertRaises(StopIteration):
                t._segmenter_loop()
            vad.assert_not_called()
        emitted = t._segment_queue.get_nowait()
        np.testing.assert_array_equal(emitted[0], audio)
        self.assertIsNone(t._segment_queue.get_nowait()[0])

    def test_status_reports_real_busy_state(self):
        t = self.transcriber()
        t._drained = False
        ipc = app.IpcServer(queue.Queue(), app.IPC_SOCK, t)
        status = json.loads(ipc._handle('status'))
        self.assertFalse(status['ready'])
        self.assertTrue(status['busy'])
        self.assertFalse(status['recording'])


if __name__ == '__main__':
    unittest.main()
