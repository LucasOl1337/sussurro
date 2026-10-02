"""Modelo descartado apos falha CUDA volta sozinho, sem depender de Aplicar modelo."""
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app
import test_bar_feedback


class ModelRecoveryTests(unittest.TestCase):
    def ui(self):
        ui = test_bar_feedback.AppFeedbackTests().ui()
        ui.transcriber.model = None
        ui.settings.update({'whisper_model': 'large-v3', 'whisper_device': 'cuda'})
        ui._begin_model_load = Mock()
        return ui

    def test_hotkey_after_cuda_discard_reloads_model_instead_of_waiting_forever(self):
        ui = self.ui()
        self.assertFalse(ui._start(inject=True))
        ui._begin_model_load.assert_called_once()
        selection = ui._begin_model_load.call_args.args[0]
        self.assertEqual((selection['whisper_model'], selection['whisper_device']), ('large-v3', 'cuda'))
        self.assertEqual(ui._begin_model_load.call_args.kwargs, {'persist': False})
        ui.transcriber.start.assert_not_called()

    def test_hotkey_while_loading_does_not_start_second_load(self):
        ui = self.ui()
        ui.transcriber.model_loading.set()
        self.assertFalse(ui._start(inject=True))
        ui._begin_model_load.assert_not_called()

    def test_busy_transcriber_does_not_reload_under_running_work(self):
        ui = self.ui()
        ui.transcriber.busy.return_value = True
        ui._recover_model()
        ui._begin_model_load.assert_not_called()

    def test_cuda_discard_status_schedules_reload_without_user_click(self):
        ui = self.ui()
        ui.status_queue.put(app.CUDA_DISCARD_STATUS)
        ui._poll()
        ui.root.after.assert_any_call(app.MODEL_RECOVERY_DELAY_MS, ui._recover_model)


if __name__ == '__main__':
    unittest.main()
