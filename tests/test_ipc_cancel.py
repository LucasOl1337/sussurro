"""Cancelamento pelo CLI/socket, sem Tk, microfone, entrada no desktop ou GPU."""
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


class IpcCancelTests(unittest.TestCase):
    def make_app(self, state):
        ui = app.App.__new__(app.App)
        ui.settings = {'feedback_sounds': False}
        ui.hotkey_queue = queue.Queue()
        ui.text_queue = queue.Queue()
        ui.status_queue = queue.Queue()
        ui._ui_queue = queue.Queue()
        with patch.object(threading.Thread, 'start'), patch.object(app.keyboard, 'Controller'):
            ui.transcriber = app.Transcriber(ui.text_queue, ui.status_queue)
        ui.hotkey = SimpleNamespace(active=state == 'recording')
        ui.record_btn = Mock()
        ui.bar = Mock()
        ui.bar.visivel.return_value = False
        ui.root = Mock()
        ui.status = Mock()
        ui.text = Mock()
        ui.transcriber._session_id = 42
        if state in ('recording', 'processing'):
            ui.transcriber._drained = False
            ui.transcriber._session_parts = ['trecho pendente']
            ui.transcriber._session_audio = ['audio pendente']
        if state == 'recording':
            ui.transcriber.recording.set()
            ui.transcriber._streams = [Mock()]
        return ui

    def run_cli(self, ui, command):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sussurro.sock'
            server = app.IpcServer(ui.hotkey_queue, path, ui.transcriber)
            # Thread externa: nao depende do atributo _stop do IpcServer para join.
            thread = threading.Thread(target=server.run, daemon=True)
            thread.start()
            try:
                deadline = time.monotonic() + 2
                while True:
                    try:
                        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
                            probe.settimeout(1)
                            probe.connect(str(path))
                            probe.sendall(b'status\n')
                            probe.recv(4096)
                        break
                    except (FileNotFoundError, ConnectionRefusedError):
                        if time.monotonic() >= deadline:
                            self.fail('socket temporario nao ficou pronto')
                        time.sleep(.01)
                env = {**os.environ, 'XDG_RUNTIME_DIR': directory,
                       'DISPLAY': '', 'WAYLAND_DISPLAY': ''}
                return subprocess.run([sys.executable, '-S', app.__file__, command],
                                      env=env, capture_output=True, text=True, timeout=3)
            finally:
                server._stop.set()
                thread.join(timeout=2)
                self.assertFalse(thread.is_alive())

    def test_cli_cancel_reaches_ui_and_discards_recording_or_processing(self):
        for state in ('recording', 'processing'):
            with self.subTest(state=state):
                ui = self.make_app(state)
                streams = list(ui.transcriber._streams)
                with patch.object(app, '_perf'), patch.object(ui.transcriber, 'stop') as stop:
                    result = self.run_cli(ui, 'cancel')
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout, 'ok\n')
                    # O socket so agenda. Cancelamento e widgets rodam na thread da UI.
                    self.assertEqual(ui.transcriber._session_id, 42)
                    app.App._poll(ui)
                    stop.assert_not_called()
                self.assertFalse(ui.transcriber.recording.is_set())
                self.assertFalse(ui.hotkey.active)
                self.assertEqual(ui.transcriber._session_id, 43)
                self.assertEqual(ui.transcriber._session_parts, [])
                self.assertEqual(ui.transcriber._session_audio, [])
                self.assertTrue(ui.transcriber.history_queue.empty())
                self.assertTrue(ui.text_queue.empty())
                ui.record_btn.configure.assert_called_once_with(text='GRAVAR')
                ui.bar.hide.assert_called_once()
                for stream in streams:
                    stream.stop.assert_called_once()
                    stream.close.assert_called_once()
                ui.hotkey_queue.put(('cancel', None))
                app.App._poll(ui)
                self.assertEqual(ui.transcriber._session_id, 43)
                ui.bar.hide.assert_called_once()

    def test_stop_and_toggle_still_confirm_instead_of_cancelling(self):
        for command in ('stop', 'toggle', 'stop-enter', 'toggle-enter'):
            with self.subTest(command=command):
                ui = self.make_app('recording')
                server = app.IpcServer(ui.hotkey_queue, Path('/unused.sock'))
                with patch.object(app, '_perf'), patch.object(ui.transcriber, 'stop') as stop, \
                     patch.object(ui.transcriber, 'cancel') as cancel, \
                     patch.object(ui.transcriber, 'arm_auto_enter') as enter:
                    self.assertEqual(server._handle(command), 'ok\n')
                    app.App._poll(ui)
                    stop.assert_called_once()
                    cancel.assert_not_called()
                    self.assertEqual(enter.call_count, int(command.endswith('-enter')))
                ui.bar.show.assert_called_once_with('proc')

    def test_cli_cancel_is_noop_without_dictation_even_when_file_is_busy(self):
        for file_jobs in (0, 1):
            with self.subTest(file_jobs=file_jobs):
                ui = self.make_app('idle')
                ui.transcriber._file_jobs = file_jobs
                result = self.run_cli(ui, 'cancel')
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, 'ok\n')
                with patch.object(ui.transcriber, 'cancel', wraps=ui.transcriber.cancel) as cancel:
                    app.App._poll(ui)
                    cancel.assert_not_called()
                self.assertEqual(ui.transcriber._session_id, 42)
                self.assertTrue(ui.transcriber._audio_queue.empty())
                ui.record_btn.configure.assert_not_called()
                ui.bar.hide.assert_not_called()
                ui.status.configure.assert_not_called()

    def test_cancel_does_not_change_other_ipc_commands(self):
        commands = {
            'toggle': ('toggle', None), 'start': ('start', None), 'stop': ('stop', None),
            'toggle-enter': ('toggle', {'enter': True}),
            'start-enter': ('start', {'enter': True}), 'stop-enter': ('stop', {'enter': True}),
            'meeting-start': ('meeting', 'start'), 'meeting-stop': ('meeting', 'stop'),
            'meeting-pause': ('meeting', 'pause'), 'cancel': ('cancel', None),
        }
        events = queue.Queue()
        server = app.IpcServer(events, Path('/unused.sock'))
        with patch.object(app, '_perf'):
            for command, event in commands.items():
                with self.subTest(command=command):
                    self.assertEqual(server._handle(command), 'ok\n')
                    self.assertEqual(events.get_nowait(), event)
            for command in ('cancel-enter', 'cancel-now', 'unknown'):
                self.assertEqual(server._handle(command), 'err unknown\n')
            self.assertEqual(server._handle('status'), 'ok\n')
        self.assertTrue(events.empty())


if __name__ == '__main__':
    unittest.main()
