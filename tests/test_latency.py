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
from sussurro_models import ModelConfig
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
        with patch.object(app, 'WhisperModel', return_value=model), patch.object(app, 'get_speech_timestamps') as vad, \
             patch.object(app, 'download_model', return_value='/cached/model'), \
             patch.object(app, 'resolve_model_config', return_value=ModelConfig('base', 'cpu', 'int8')):
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

    def test_paste_strategy_routes_terminals_away_from_ctrl_v(self):
        self.assertEqual(app.paste_strategy('foot'), 'terminal')
        self.assertEqual(app.paste_strategy('kitty'), 'terminal')
        self.assertEqual(app.paste_strategy('ChatGPT'), 'ctrl_v')
        self.assertEqual(app.paste_strategy('chromium'), 'ctrl_v')
        self.assertEqual(app.paste_strategy(None), 'ctrl_v')

    def test_terminal_paste_sends_ctrl_shift_v(self):
        # Com ydotoold no ar as teclas saem por uinput (keycodes evdev), nunca
        # por wtype: o teclado virtual do wtype faz o Hyprland reenviar o keymap
        # a cada evento e trava a sessao (ver _ydotool_keys).
        t = self.transcriber()
        with patch.object(app, '_is_wayland', return_value=True), \
             patch.object(app.shutil, 'which', return_value='/usr/bin/ydotool'), \
             patch.object(app, '_ydotool_env', return_value={'YDOTOOL_SOCKET': '/tmp/x'}), \
             patch.object(app.subprocess, 'run') as run:
            run.return_value = SimpleNamespace(returncode=0)
            self.assertTrue(t._send_paste_key('terminal'))
            cmd = run.call_args[0][0]
            self.assertEqual(cmd[:2], ['ydotool', 'key'])
            self.assertIn('42:1', cmd)  # shift
            self.assertTrue(t._send_paste_key('ctrl_v'))
            cmd = run.call_args[0][0]
            self.assertEqual(cmd[:2], ['ydotool', 'key'])
            self.assertNotIn('42:1', cmd)
            self.assertEqual(cmd[-4:], ['29:1', '47:1', '47:0', '29:0'])

    def test_paste_falls_back_to_wtype_without_ydotoold(self):
        t = self.transcriber()
        with patch.object(app, '_is_wayland', return_value=True), \
             patch.object(app, '_hypr', return_value=SimpleNamespace(available=False)), \
             patch.object(app.shutil, 'which', return_value='/usr/bin/wtype'), \
             patch.object(app, '_ydotool_env', return_value={}), \
             patch.object(app.subprocess, 'run') as run:
            run.return_value = SimpleNamespace(returncode=0)
            self.assertTrue(t._send_paste_key('terminal'))
            cmd = run.call_args[0][0]
            self.assertEqual(cmd[0], 'wtype')
            self.assertIn('shift', cmd)
            self.assertTrue(t._send_paste_key('ctrl_v'))
            self.assertEqual(run.call_args[0][0], ['wtype', '-M', 'ctrl', '-k', 'v', '-m', 'ctrl'])

    def test_hyprland_does_not_fallback_to_virtual_keymap_for_paste_or_enter(self):
        t = self.transcriber()
        with patch.object(app, '_is_wayland', return_value=True), \
             patch.object(app, '_hypr', return_value=SimpleNamespace(available=True)), \
             patch.object(app, '_ydotool_keys', return_value=False), \
             patch.object(app.subprocess, 'run') as run, \
             patch.object(app.time, 'sleep'):
            self.assertFalse(t._send_paste_key('terminal'))
            with self.assertRaises(RuntimeError):
                t._press_enter()
            run.assert_not_called()

    def test_hyprland_clipboard_failure_preserves_text_without_virtual_typing(self):
        t = self.transcriber()
        with patch.object(app, 'prepare_paste_target', return_value='foot'), \
             patch.object(app, '_is_wayland', return_value=True), \
             patch.object(app, '_hypr', return_value=SimpleNamespace(available=True)), \
             patch.object(app, 'backup_clipboard', return_value=b'original'), \
             patch.object(app, 'set_clipboard_text', return_value=False), \
             patch.object(t, '_type_fallback') as fallback:
            with self.assertRaises(RuntimeError):
                t._paste_linux('ditado')
            fallback.assert_not_called()

    def test_foot_paste_uses_terminal_chord(self):
        t = self.transcriber()
        with patch.object(app, 'prepare_paste_target', return_value='foot'), \
             patch.object(app, 'backup_clipboard', return_value=b'orig'), \
             patch.object(app, 'set_clipboard_text', return_value=True), \
             patch.object(t, '_send_paste_key', return_value=True) as send:
            t._paste_linux('texto ditado')
            send.assert_called_once_with('terminal')

    def test_codex_desktop_paste_uses_ctrl_v(self):
        t = self.transcriber()
        with patch.object(app, 'prepare_paste_target', return_value='ChatGPT'), \
             patch.object(t, '_type_fallback') as type_fallback, \
             patch.object(app, 'backup_clipboard', return_value=b'orig'), \
             patch.object(threading.Timer, 'start'), \
             patch.object(app, 'set_clipboard_text', return_value=True) as write, \
             patch.object(t, '_send_paste_key', return_value=True) as send:
            t._paste_linux('cole isto no Codex')
            type_fallback.assert_not_called()
            write.assert_called_once_with('cole isto no Codex')
            send.assert_called_once_with('ctrl_v')

    def test_paste_leaves_dictation_without_background_restore(self):
        t = self.transcriber()
        clipboard = [b'original']
        restored = threading.Event()
        def write(text):
            clipboard[0] = text.encode(); return True
        def restore(data):
            clipboard[0] = data; restored.set()
        with patch.object(app, 'prepare_paste_target', return_value='chromium'), \
             patch.object(app, 'backup_clipboard', side_effect=lambda: clipboard[0]), \
             patch.object(app, 'set_clipboard_text', side_effect=write), \
             patch.object(app, 'restore_clipboard', side_effect=restore), \
             patch.object(t, '_send_paste_key', return_value=True), \
             patch.object(app.threading, 'Timer') as timer:
            start = time.perf_counter()
            t._paste_linux('ação rápida')
            self.assertLess(time.perf_counter() - start, .25)
            self.assertEqual(clipboard[0], 'ação rápida'.encode())
            self.assertFalse(restored.is_set())
            timer.assert_not_called()

    def test_delayed_web_field_reads_dictation_after_old_restore_deadline(self):
        t = self.transcriber()
        clipboard = [b'original']
        received = []
        target_read = threading.Event()
        def write(text):
            clipboard[0] = text.encode()
            return True
        def restore(data):
            clipboard[0] = data
        def read_later():
            received.append(clipboard[0])
            target_read.set()
        def send(_strategy):
            timer = threading.Timer(1.25, read_later)
            timer.daemon = True
            timer.start()
            return True
        with patch.object(app, 'prepare_paste_target', return_value='chromium'), \
             patch.object(app, 'backup_clipboard', side_effect=lambda: clipboard[0]), \
             patch.object(app, 'set_clipboard_text', side_effect=write), \
             patch.object(app, 'restore_clipboard', side_effect=restore), \
             patch.object(t, '_send_paste_key', side_effect=send):
            t._paste_linux('texto para o campo web')
            self.assertTrue(target_read.wait(2))
        self.assertEqual(received, [b'texto para o campo web'])

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

    def test_consecutive_pastes_do_not_restore_old_clipboard(self):
        t = self.transcriber()
        clipboard = [b'old image']
        def write(text):
            clipboard[0] = text.encode()
            return True
        with patch.object(app, 'prepare_paste_target', return_value='foot'), \
             patch.object(app, 'set_clipboard_text', side_effect=write), \
             patch.object(app, 'backup_clipboard') as backup, \
             patch.object(app, 'restore_clipboard') as restore, \
             patch.object(t, '_send_paste_key', return_value=True):
            t._paste_linux('primeiro')
            self.assertEqual(clipboard[0], b'primeiro')
            clipboard[0] = b'user copy'
            t._paste_linux('segundo')
            self.assertEqual(clipboard[0], b'segundo')
            backup.assert_not_called()
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

    def test_failed_delivery_does_not_press_enter_in_wrong_window(self):
        t = self.transcriber()
        t._session_auto_enter = True
        t._session_emitted = True
        t._session_inject = True
        t._session_errors = ['Nao foi possivel focar o destino']
        events = iter([(None, None, None, t._session_id, 0.0)])
        t._segment_queue.get = lambda: next(events)
        with patch.object(t, '_press_enter') as enter, \
             patch.object(t, '_finalize_session'):
            with self.assertRaises(StopIteration):
                t._transcribe_loop()
        enter.assert_not_called()

    def test_transcription_failure_still_archives_audio(self):
        t = self.transcriber()
        audio = np.linspace(-.2, .2, app.SAMPLE_RATE, dtype=np.float32)
        t._session_id = 1
        t._session_started = app.datetime(2026, 9, 10, 8, 34, 22)
        t._drained = False
        t._session_mode = 'final'
        t._enqueue_segment(audio)
        t._segment_queue.put((None, None, None, 1, 0.0))
        events = iter([t._segment_queue.get_nowait(), t._segment_queue.get_nowait()])
        t._segment_queue.get = lambda: next(events)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(app, 'HISTORY_DIR', Path(directory)), \
             patch.object(app, 'HISTORY_INDEX', Path(directory) / 'history.jsonl'), \
             patch.object(t, '_transcribe_locked', side_effect=RuntimeError('CUDA indisponivel')), \
             patch.object(app.traceback, 'print_exc'):
            with self.assertRaises(StopIteration):
                t._transcribe_loop()
            entries = [json.loads(line) for line in app.HISTORY_INDEX.read_text().splitlines()]
            self.assertEqual(len(entries), 1)
            self.assertTrue(entries[0]['failed'])
            self.assertIn('CUDA indisponivel', entries[0]['error'])
            self.assertEqual(entries[0]['text'], '')
            self.assertTrue((app.HISTORY_DIR / entries[0]['wav']).is_file())

    def test_retry_replaces_failed_history_entry(self):
        t = self.transcriber()
        audio = np.ones(app.SAMPLE_RATE, dtype=np.float32) * .05
        segment = SimpleNamespace(text=' instrucao recuperada', start=0.0, end=1.0)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(app, 'HISTORY_DIR', Path(directory)), \
             patch.object(app, 'HISTORY_INDEX', Path(directory) / 'history.jsonl'), \
             patch.object(app, 'load_audio_16k_mono', return_value=audio), \
             patch.object(t, '_transcribe_locked', return_value=([segment], None)):
            entry = t._archive_audio(audio, '', 0, app.datetime(2026, 9, 10, 8, 34, 22),
                                     failed=True, error='falha inicial')
            t.history_queue.get_nowait()
            t._retrying.set()
            t._retry_history_worker(entry, app.HISTORY_DIR / entry['wav'])
            entries = [json.loads(line) for line in app.HISTORY_INDEX.read_text().splitlines()]
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]['text'], 'instrucao recuperada')
            self.assertNotIn('failed', entries[0])
            self.assertNotIn('error', entries[0])
            self.assertEqual(t.history_queue.get_nowait(), entries[0])
            self.assertFalse(t._retrying.is_set())


if __name__ == '__main__':
    unittest.main()
