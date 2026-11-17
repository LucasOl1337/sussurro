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
    def pill(self):
        pill = app.MeetingPill.__new__(app.MeetingPill)
        pill.win = Mock()
        pill.hypr = False
        pill._shown = False
        pill._dismissed = False
        pill._meeting = None
        pill._place = Mock()
        pill._draw = Mock()
        pill.on_dismiss = Mock()
        pill.on_pause = Mock()
        pill._native_drag = None
        return pill

    def test_meeting_pill_is_its_own_window_and_shows_while_recording(self):
        pill = self.pill()
        pill.update(True, False, 65)
        self.assertTrue(pill.visible())
        self.assertEqual(pill._meeting, (False, 65))
        pill.win.deiconify.assert_called_once()
        pill.update(True, True, 66)
        self.assertEqual(pill._meeting, (True, 66))
        pill.win.deiconify.assert_called_once()  # nao reabre a cada poll
        pill.update(False, False, 66)
        self.assertFalse(pill.visible())
        pill.win.withdraw.assert_called()

    def test_dictation_bar_no_longer_carries_meeting_state(self):
        self.assertFalse(hasattr(app.RecorderBar, 'set_meeting'))
        bar = test_bar_feedback.BarFeedbackTests().bar('proc')
        bar.finish()
        self.assertIsNone(bar._state)

    def test_dismiss_hides_until_meeting_ends(self):
        pill = self.pill()
        pill.update(True, False, 5)
        pill.dismiss()
        self.assertFalse(pill.visible())
        pill.on_dismiss.assert_called_once()
        pill.update(True, False, 6)
        self.assertFalse(pill.visible())
        pill.reveal()
        self.assertTrue(pill.visible())
        pill.dismiss()
        pill.update(False, False, 7)
        pill.update(True, False, 0)  # reuniao nova volta a mostrar
        self.assertTrue(pill.visible())

    def test_click_on_buttons_pauses_or_hides(self):
        pill = self.pill()
        pill._k = 1.0
        pill.W, pill.H = app.MeetingPill.W, app.MeetingPill.H
        pill.CY, pill.PX, pill.HX = pill.H / 2, pill.W - 36, pill.W - 14
        pill.canvas = Mock()
        pill._hovered = None
        pill.update(True, False, 5)
        for x, called in ((pill.PX, pill.on_pause), (pill.HX, pill.on_dismiss)):
            pill._press(SimpleNamespace(x=x, y=pill.CY))
            pill._release(SimpleNamespace(x=x, y=pill.CY))
            called.assert_called_once()

    def test_poll_tracks_meeting_even_outside_its_tab(self):
        ui = test_bar_feedback.AppFeedbackTests().ui()
        ui.meeting_panel = SimpleNamespace(recording=True, paused=False, elapsed=5, resumed_at=10.)
        with patch.object(app.time, 'monotonic', return_value=12.):
            ui._poll()
        ui.meeting_pill.update.assert_called_once_with(True, False, 7.)
        ui.bar.set_meeting.assert_not_called()

    def test_pill_saves_monitor_and_relative_position_after_native_drag(self):
        hypr = SimpleNamespace(
            native_cursorpos=Mock(side_effect=[(5500, 2000)]),
            native_monitor_at=Mock(return_value=dict(name='DP-2', x=5360, y=0, width=2560,
                                                     height=1440, scale=1, transform=3,
                                                     reserved=[0, 30, 0, 0])),
            native_work_area=app._hypr().native_work_area)
        drag = ('0xabc', (5400, 1000), (5380, 990), (112, 26))
        with patch.object(app, '_hypr', return_value=hypr):
            monitor, rel = app._native_drag_end(drag)
        self.assertEqual(monitor, 'DP-2')
        # janela parou em (5480, 1990); area util 5360..6800 x 30..2560
        self.assertEqual(rel, [round(120 / (1440 - 112), 4), round(1960 / (2530 - 26), 4)])


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
        panel._refresh_list = Mock()
        for name in ('start_btn', 'pause_btn', 'stop_btn', 'import_btn', 'pill_btn'):
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

    def test_start_failure_does_not_leave_fake_interrupted_recording(self):
        with tempfile.TemporaryDirectory() as folder:
            mic, pc, state = (Path(folder) / name for name in ('mic.raw', 'pc.raw', 'estado.json'))
            panel = self.panel()
            panel._write_state = lambda: state.write_text('{}', encoding='utf-8')
            panel.sources[0].start_recording.side_effect = lambda path: Path(path).touch()
            panel.sources[1].start_recording.side_effect = OSError('parec saiu')
            with patch.object(meeting_ui.rec, 'raw_paths', return_value=(mic, pc, state)):
                self.assertTrue(panel.command('start').startswith('err '))
                self.assertIsNone(meeting_ui.rec.pending_recording())
                panel.sources = [Mock(), Mock()]
                self.assertEqual(panel.command('start'), 'ok\n')

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
