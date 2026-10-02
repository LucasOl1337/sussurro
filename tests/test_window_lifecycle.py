"""Janela Linux oculta sem encerrar o motor; Tk opt-in e socket temporario."""
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import app


def ui_fixture(root):
    ui = app.App.__new__(app.App)
    ui.root = root
    ui._close_notice_shown = False
    ui.compare_panel = Mock()
    ui.meeting_panel = Mock()
    ui.status = Mock()
    ui.bar = Mock()
    ui.bar.visivel.return_value = False
    ui.hotkey = SimpleNamespace(_listener=Mock(), active=False)
    ui.gestures = Mock()
    ui._ipc = Mock()
    ui._ui_queue = queue.Queue()
    ui.hotkey_queue = queue.Queue()
    ui.text_queue = queue.Queue()
    ui.status_queue = queue.Queue()
    ui.transcriber = Mock()
    ui.transcriber.recording = threading.Event()
    ui.transcriber.history_queue = queue.Queue()
    ui.text = Mock()
    ui._start = Mock(return_value=True)
    return ui


class WindowLifecycleTests(unittest.TestCase):
    def test_linux_close_withdraws_without_touching_workers_and_warns_once(self):
        ui = ui_fixture(Mock())
        with patch.object(app, 'IS_WIN', False):
            app.App._close(ui)
            app.App._close(ui)
        self.assertEqual(ui.root.withdraw.call_count, 2)
        ui.root.destroy.assert_not_called()
        ui.compare_panel.close.assert_not_called()
        ui.meeting_panel.close.assert_not_called()
        ui.transcriber.cancel.assert_not_called()
        ui.hotkey._listener.stop.assert_not_called()
        ui._ipc._stop.set.assert_not_called()
        self.assertEqual(ui.bar.flash.call_count, 1)
        self.assertIn('sussurro show', ui.bar.flash.call_args[0][1])
        self.assertIn('sussurro quit', ui.bar.flash.call_args[0][1])

    def test_windows_close_keeps_destroying_as_before(self):
        ui = ui_fixture(Mock())
        with patch.object(app, 'IS_WIN', True):
            app.App._close(ui)
        ui.root.withdraw.assert_not_called()
        ui.root.destroy.assert_called_once()
        ui.compare_panel.close.assert_called_once()
        ui.meeting_panel.close.assert_called_once()

    def test_show_runs_on_ui_queue_and_quit_does_not_schedule_after_destroy(self):
        ui = ui_fixture(Mock())
        server = app.IpcServer(ui.hotkey_queue, Path('/unused.sock'))
        self.assertEqual(server._handle('show'), 'ok\n')
        ui.root.deiconify.assert_not_called()
        app.App._poll(ui)
        ui.root.deiconify.assert_called_once()
        ui.root.lift.assert_called_once()
        ui.root.reset_mock()
        self.assertEqual(server._handle('quit'), 'ok\n')
        with patch.object(app, 'IS_WIN', False):
            app.App._poll(ui)
        ui.root.destroy.assert_called_once()
        ui.root.after.assert_not_called()
        ui._ipc._stop.set.assert_called_once()
        ui.gestures.stop.assert_called_once()
        ui.hotkey._listener.stop.assert_called_once()

    def test_quit_without_listener_gestures_or_meeting_still_destroys_root(self):
        ui = ui_fixture(Mock())
        ui.hotkey._listener = None
        ui.gestures = None
        ui.meeting_panel = None
        with patch.object(app, 'IS_WIN', False):
            app.App._quit(ui)
        ui.transcriber.cancel.assert_called_once_with(from_processing=True)
        ui.bar.hide.assert_called_once()
        ui.root.destroy.assert_called_once()


@unittest.skipUnless(os.environ.get('SUSSURRO_TEST_TK') == '1', 'Tk isolado opt-in')
class HiddenWindowTkTests(unittest.TestCase):
    def test_wm_close_keeps_socket_and_toggle_and_show_or_duplicate_restore_window(self):
        root = tk.Tk()
        ui = ui_fixture(root)
        root.protocol('WM_DELETE_WINDOW', ui._close)
        errors = []
        root.report_callback_exception = lambda *error: errors.append(error)
        def cleanup():
            try:
                for timer in root.tk.call('after', 'info'):
                    root.after_cancel(timer)
                root.destroy()
            except tk.TclError:
                pass
        self.addCleanup(cleanup)
        root.update()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sussurro.sock'
            ui._ipc = app.IpcServer(ui.hotkey_queue, path)
            worker = threading.Thread(target=ui._ipc.run, daemon=True)
            worker.start()
            self.addCleanup(lambda: (ui._ipc._stop.set(), worker.join(2)))
            deadline = time.monotonic() + 2
            while not path.exists():
                self.assertLess(time.monotonic(), deadline)
                time.sleep(.01)
            env = {**os.environ, 'XDG_RUNTIME_DIR': directory}
            def cli(*args):
                result = subprocess.run([sys.executable, '-S', app.__file__, *args],
                                        env=env, capture_output=True, text=True, timeout=3)
                self.assertEqual(result.returncode, 0, result.stderr)
                return result.stdout
            with patch.object(app, 'IS_WIN', False), patch.object(app, '_perf'):
                # Invoca o callback real registrado pelo protocolo do gerenciador.
                root.tk.call(root.protocol('WM_DELETE_WINDOW'))
                root.update()
                self.assertEqual(root.state(), 'withdrawn')
                self.assertEqual(cli('status'), 'ok\n')
                self.assertEqual(cli('toggle'), 'ok\n')
                app.App._poll(ui)
                root.update()
                ui._start.assert_called_once_with(inject=True, auto_enter=False)
                self.assertEqual(cli('show'), 'ok\n')
                app.App._poll(ui)
                root.update()
                self.assertEqual(root.state(), 'normal')
                ui._close()
                root.update()
                self.assertEqual(root.state(), 'withdrawn')
                self.assertIn('ja esta rodando', cli())
                app.App._poll(ui)
                root.update()
                self.assertEqual(root.state(), 'normal')
                self.assertEqual(ui.bar.flash.call_count, 1)
                self.assertEqual(cli('quit'), 'ok\n')
                app.App._poll(ui)
                worker.join(2)
                self.assertFalse(worker.is_alive())
                self.assertFalse(path.exists())
            self.assertEqual(errors, [])


if __name__ == '__main__':
    unittest.main()
