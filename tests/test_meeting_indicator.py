"""Indicador e resposta de reuniao sem captura ou sessao viva."""
import queue
import threading
import tempfile
import subprocess
import sys
import os
import time
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import app
import sussurro_meeting_ui as meeting_ui
import test_bar_feedback
import sussurro_ipc


class MeetingIndicatorTests(unittest.TestCase):
    def test_meeting_recording_is_visible_when_dictation_is_idle(self):
        bar = test_bar_feedback.BarFeedbackTests().bar()
        bar._meeting = None
        bar.set_meeting(True, False, 65)
        bar.finish()
        self.assertEqual(bar._state, 'meeting')
        bar.win.withdraw.assert_not_called()
        bar.set_meeting(True, True, 65)
        self.assertEqual(bar._meeting, (True, 65))
        bar.set_meeting(False, False, 65)
        self.assertFalse(bar.visivel())

    def test_dictation_has_priority_and_meeting_returns_after_finish(self):
        bar = test_bar_feedback.BarFeedbackTests().bar('rec')
        bar._meeting = None
        bar.set_meeting(True, False, 12)
        self.assertEqual(bar._state, 'rec')
        bar.show('proc')
        bar.set_meeting(True, False, 13)
        self.assertEqual(bar._state, 'proc')
        bar.finish()
        self.assertEqual(bar._state, 'meeting')

    def test_poll_tracks_meeting_even_outside_its_tab(self):
        ui = test_bar_feedback.AppFeedbackTests().ui()
        ui.meeting_panel = SimpleNamespace(recording=True, paused=False, elapsed=5, resumed_at=10.)
        with patch.object(app.time, 'monotonic', return_value=12.):
            ui._poll()
        ui.bar.set_meeting.assert_called_once_with(True, False, 7.)

    def test_error_notice_returns_to_meeting_after_expiry(self):
        bar = test_bar_feedback.BarFeedbackTests().bar()
        bar.set_meeting(True, False, 25)
        with patch.object(app.time, 'monotonic', return_value=10.):
            bar.flash('error', 'Falha no ditado.', 2000)
        self.assertEqual(bar._state, 'error')
        with patch.object(app.time, 'monotonic', return_value=12.1):
            bar._tick()
        self.assertEqual(bar._state, 'meeting')
        bar.win.withdraw.assert_not_called()


class MeetingCommandTests(unittest.TestCase):
    def panel(self):
        panel = meeting_ui.MeetingPanel.__new__(meeting_ui.MeetingPanel)
        panel.recording = False
        panel.abort = None
        panel.closed = False
        panel.sources = [Mock(), Mock()]
        panel.ready_status = Mock()
        panel._write_state = Mock()
        panel._set_ready_text = Mock()
        for name in ('start_btn', 'pause_btn', 'stop_btn', 'import_btn', 'previous'):
            setattr(panel, name, Mock())
        return panel

    def test_recovery_blocks_start_and_returns_error_not_success(self):
        panel = self.panel()
        with patch.object(meeting_ui.rec, 'pending_recording', return_value={'duration_secs': 12}):
            self.assertTrue(panel.command('start').startswith('err '))
        self.assertFalse(panel.recording)
        panel.sources[0].start_recording.assert_not_called()

    def test_start_failure_closes_partial_capture_and_returns_error(self):
        panel = self.panel()
        panel.sources[1].start_recording.side_effect = OSError('disco cheio')
        sources = list(panel.sources)
        with patch.object(meeting_ui.rec, 'pending_recording', return_value=None), \
             patch.object(meeting_ui.rec, 'raw_paths', return_value=('mic', 'pc', 'state')):
            self.assertIn('disco cheio', panel.command('start'))
        self.assertFalse(panel.recording)
        for source in sources:
            source.close.assert_called_once()

    def test_start_pause_and_repeated_start_report_state_correctly(self):
        panel = self.panel()
        with patch.object(meeting_ui.rec, 'pending_recording', return_value=None), \
             patch.object(meeting_ui.rec, 'raw_paths', return_value=('mic', 'pc', 'state')):
            self.assertEqual(panel.command('start'), 'ok\n')
            self.assertEqual(panel.command('start'), 'ok\n')
        self.assertTrue(panel.recording)
        panel.sources[0].start_recording.assert_called_once()
        self.assertEqual(panel.command('pause'), 'ok\n')
        self.assertTrue(panel.paused)
        self.assertEqual(panel.command('pause'), 'ok\n')
        self.assertFalse(panel.paused)

    def test_pause_stop_without_recording_and_start_during_job_are_errors(self):
        panel = self.panel()
        self.assertTrue(panel.command('pause').startswith('err '))
        self.assertTrue(panel.command('stop').startswith('err '))
        panel.abort = threading.Event()
        self.assertTrue(panel.command('start').startswith('err '))
        panel.sources[0].start_recording.assert_not_called()


class MeetingIpcTests(unittest.TestCase):
    def test_real_cli_waits_for_ui_result_and_does_not_block_other_commands(self):
        for response in ('ok\n', 'err gravacao interrompida\n'):
            with self.subTest(response=response), tempfile.TemporaryDirectory() as directory:
                ui = test_bar_feedback.AppFeedbackTests().ui()
                ui.meeting_panel = SimpleNamespace(recording=False, paused=False,
                    elapsed=0., resumed_at=None, command=Mock(return_value=response))
                path = Path(directory) / 'sussurro.sock'
                server = app.IpcServer(ui.hotkey_queue, path)
                server_thread = threading.Thread(target=server.run, daemon=True)
                server_thread.start()
                self.addCleanup(lambda s=server, t=server_thread: (s._stop.set(), t.join(2)))
                deadline = time.monotonic() + 2
                while not path.exists():
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(.01)
                result = []
                env = {**os.environ, 'XDG_RUNTIME_DIR': directory, 'DISPLAY': '', 'WAYLAND_DISPLAY': ''}
                client = threading.Thread(target=lambda: result.append(subprocess.run(
                    [sys.executable, '-S', app.__file__, 'meeting-start'], env=env,
                    capture_output=True, text=True, timeout=3)))
                client.start()
                event = ui.hotkey_queue.get(timeout=2)
                self.assertEqual(event[0], 'meeting')
                self.assertEqual(sussurro_ipc.ipc_send('status', path=path), 'ok\n')
                self.assertEqual(sussurro_ipc.ipc_send('toggle', path=path), 'ok\n')
                ui.hotkey_queue.get_nowait()  # toggle ja foi confirmado, nao iniciar mic de teste
                ui.hotkey_queue.put(event)
                ui._poll()
                client.join(3)
                self.assertFalse(client.is_alive())
                self.assertEqual(result[0].stdout, response)
                self.assertEqual(result[0].returncode, int(response.startswith('err')))
                ui.meeting_panel.command.assert_called_once_with('start')
                server._stop.set()
                server_thread.join(2)

    def test_expired_request_does_not_start_late(self):
        ui = test_bar_feedback.AppFeedbackTests().ui()
        ui.meeting_panel = SimpleNamespace(recording=False, paused=False,
            elapsed=0., resumed_at=None, command=Mock())
        reply = queue.Queue()
        ui.hotkey_queue.put(('meeting', ('start', reply, 1.)))
        with patch.object(app.time, 'monotonic', return_value=2.):
            ui._poll()
        ui.meeting_panel.command.assert_not_called()
        self.assertTrue(reply.empty())

    def test_missing_panel_reports_error_instead_of_ok(self):
        ui = test_bar_feedback.AppFeedbackTests().ui()
        reply = queue.Queue()
        ui.hotkey_queue.put(('meeting', ('start', reply, time.monotonic() + 1)))
        ui._poll()
        self.assertTrue(reply.get_nowait().startswith('err'))
        ui.bar.flash.assert_called_once()

    def test_timeout_never_claims_success_and_late_poll_does_not_execute(self):
        ui = test_bar_feedback.AppFeedbackTests().ui()
        ui.meeting_panel = SimpleNamespace(recording=False, paused=False,
            elapsed=0., resumed_at=None, command=Mock())
        server = app.IpcServer(ui.hotkey_queue, Path('/unused.sock'))
        self.assertTrue(server._handle('meeting-start').startswith('err '))
        ui._poll()
        ui.meeting_panel.command.assert_not_called()


if __name__ == '__main__':
    unittest.main()
