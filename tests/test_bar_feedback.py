"""Feedback fora da janela, sem iniciar Tk, microfone ou modelo."""
import queue
import threading
import unittest
from unittest.mock import Mock, patch

import app


class BarFeedbackTests(unittest.TestCase):
    def bar(self, state=None):
        bar = app.RecorderBar.__new__(app.RecorderBar)
        bar._state = state
        bar._feedback = None
        bar.hypr = False
        bar.win = Mock()
        bar.root = Mock()
        bar._draw = Mock()
        bar._follow = Mock()
        bar._ticking = True
        bar._phase = 0.
        bar._follow_n = 0
        return bar

    def test_idle_feedback_survives_idle_cleanup_until_expiry(self):
        bar = self.bar()
        with patch.object(app.time, 'monotonic', return_value=10.):
            bar.flash('busy', 'Modelo ainda carregando.', 1200)
            bar.finish()
        self.assertEqual(bar._state, 'busy')
        bar.win.withdraw.assert_not_called()
        with patch.object(app.time, 'monotonic', return_value=11.3):
            bar._tick()
        self.assertIsNone(bar._state)
        bar.win.withdraw.assert_called_once()

    def test_error_keeps_recording_and_processing_controls(self):
        for state in ('rec', 'proc'):
            with self.subTest(state=state):
                bar = self.bar(state)
                with patch.object(app.time, 'monotonic', return_value=10.):
                    bar.flash('error', 'Microfone parou.', 2000)
                self.assertEqual(bar._state, state)
                with patch.object(app.time, 'monotonic', return_value=12.1):
                    bar._tick()
                self.assertEqual(bar._state, state)
                self.assertIsNone(bar._feedback)
                bar.win.withdraw.assert_not_called()

    def test_finished_processing_preserves_remaining_error_notice(self):
        bar = self.bar('proc')
        bar.flash('error', 'Falha de colagem.', 2000)
        bar.finish()
        self.assertEqual(bar._state, 'error')
        bar.win.withdraw.assert_not_called()

    def test_new_recording_clears_notice_without_stale_expiry(self):
        bar = self.bar()
        with patch.object(app.time, 'monotonic', return_value=10.):
            bar.flash('busy', 'Aguarde.', 1200)
        bar.show('rec')
        with patch.object(app.time, 'monotonic', return_value=20.):
            bar._tick()
        self.assertEqual(bar._state, 'rec')
        bar.win.withdraw.assert_not_called()

    def test_cancel_hide_clears_pending_notice(self):
        bar = self.bar('rec')
        bar.flash('error', 'Falha na captura.', 2000)
        bar.hide()
        self.assertIsNone(bar._feedback)
        self.assertFalse(bar.visivel())


class AppFeedbackTests(unittest.TestCase):
    def ui(self):
        ui = app.App.__new__(app.App)
        ui.transcriber = Mock()
        ui.transcriber.comparing = threading.Event()
        ui.transcriber.model_loading = threading.Event()
        ui.transcriber.recording = threading.Event()
        ui.transcriber.busy.return_value = False
        ui.transcriber.history_queue = queue.Queue()
        ui.settings = {'capture_mode': 'microfone'}
        ui._device_index = Mock(return_value=None)
        ui._pc_device_index = Mock(return_value=None)
        ui.bar = Mock()
        ui.bar.visivel.return_value = False
        ui.record_btn = Mock()
        ui.status = Mock()
        ui.root = Mock()
        ui.hotkey_queue = queue.Queue()
        ui.text_queue = queue.Queue()
        ui.status_queue = queue.Queue()
        ui._ui_queue = queue.Queue()
        return ui

    def test_rejected_start_reports_busy_for_loading_comparison_and_draining(self):
        for reason in ('loading', 'comparison', 'draining'):
            with self.subTest(reason=reason):
                ui = self.ui()
                if reason == 'loading':
                    ui.transcriber.model_loading.set()
                elif reason == 'comparison':
                    ui.transcriber.comparing.set()
                else:
                    ui.transcriber.busy.return_value = True
                self.assertFalse(ui._start(inject=True))
                self.assertEqual(ui.bar.flash.call_args.args[0], 'busy')
                ui.transcriber.start.assert_not_called()

    def test_failed_capture_start_reports_error_and_does_not_show_recording(self):
        ui = self.ui()
        ui.transcriber.start.side_effect = RuntimeError('dispositivo ausente')
        self.assertFalse(ui._start(inject=True))
        self.assertEqual(ui.bar.flash.call_args.args[0], 'error')
        self.assertIn('dispositivo ausente', ui.bar.flash.call_args.args[1])
        ui.bar.show.assert_not_called()

    def test_error_status_never_hides_active_recording(self):
        ui = self.ui()
        ui.transcriber.recording.set()
        ui.transcriber.busy.return_value = True
        ui.status_queue.put('ERRO: microfone parou de enviar áudio')
        ui._poll()
        ui.bar.hide.assert_not_called()
        ui.bar.flash.assert_called_once_with('error', 'ERRO: microfone parou de enviar áudio', 2000)

    def test_successful_start_still_shows_recording(self):
        ui = self.ui()
        self.assertTrue(ui._start(inject=True))
        ui.bar.show.assert_called_once_with('rec')
        ui.bar.flash.assert_not_called()

    def test_poll_idle_does_not_erase_recent_error_on_next_poll(self):
        ui = self.ui()
        ui.bar = BarFeedbackTests().bar()
        ui.status_queue.put('ERRO ao colar: destino indisponivel')
        with patch.object(app.time, 'monotonic', return_value=10.):
            ui._poll()
            ui._poll()
        self.assertEqual(ui.bar._state, 'error')
        ui.bar.win.withdraw.assert_not_called()
        with patch.object(app.time, 'monotonic', return_value=12.1):
            ui.bar._tick()
        self.assertFalse(ui.bar.visivel())


if __name__ == '__main__':
    unittest.main()
