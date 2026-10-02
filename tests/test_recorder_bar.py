"""Cliques reais do Tk so rodam na bancada explicitamente habilitada."""
import os
import tkinter as tk
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import app


@unittest.skipUnless(os.environ.get('SUSSURRO_TEST_TK') == '1',
                     'habilite SUSSURRO_TEST_TK=1 apenas na bancada agent-bench')
class RecorderBarTkTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.addCleanup(self.root.destroy)
        self.root.withdraw()
        self.callback_errors = []
        self.root.report_callback_exception = lambda *error: self.callback_errors.append(error)
        self.cancel = Mock()
        self.confirm = Mock()
        self.save = Mock()
        self.hypr = SimpleNamespace(available=True, place_bar=Mock(return_value=False),
                                    reset_bar_placement=Mock(),
                                    work_area_at=Mock(return_value=(0, 0, 1600, 1000)))
        patcher = patch.object(app, '_hypr', return_value=self.hypr)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.bar = app.RecorderBar(self.root, lambda: (.5, .5), self.save,
                                   lambda: [0.], self.cancel, self.confirm)
        self.bar._target_xy = lambda: (300, 300)
        self.bar.show('rec')
        self.root.update()

    def tearDown(self):
        self.bar.hide()
        for timer in self.root.tk.call('after', 'info'):
            self.root.after_cancel(timer)
        self.assertEqual(self.callback_errors, [])

    def click(self, x, y):
        for event in ('<Button-1>', '<ButtonRelease-1>'):
            self.bar.canvas.event_generate(event, x=int(x), y=int(y))
            self.root.update()

    def test_cancel_receives_canvas_events_while_recording(self):
        self.click(self.bar.LX, self.bar.CY)
        self.cancel.assert_called_once_with()
        self.confirm.assert_not_called()
        self.save.assert_not_called()

    def test_cancel_receives_events_while_processing(self):
        self.bar.show('proc')
        self.root.update()
        self.click(self.bar.LX, self.bar.CY)
        self.cancel.assert_called_once_with()
        self.confirm.assert_not_called()

    def test_confirm_receives_canvas_events_while_recording(self):
        self.click(self.bar.RX, self.bar.CY)
        self.confirm.assert_called_once_with()
        self.cancel.assert_not_called()

    def test_processing_does_not_confirm(self):
        self.bar.show('proc')
        self.root.update()
        self.click(self.bar.RX, self.bar.CY)
        self.confirm.assert_not_called()
        self.cancel.assert_not_called()

    def test_releasing_outside_cancel_does_not_cancel(self):
        self.bar.canvas.event_generate('<Button-1>', x=int(self.bar.LX), y=int(self.bar.CY))
        self.bar.canvas.event_generate('<ButtonRelease-1>', x=self.bar.W // 2, y=int(self.bar.CY))
        self.root.update()
        self.cancel.assert_not_called()
        self.confirm.assert_not_called()

    def test_wave_click_does_not_save_or_cancel(self):
        self.click(self.bar.W // 2, self.bar.CY)
        self.save.assert_not_called()
        self.cancel.assert_not_called()
        self.confirm.assert_not_called()

    def test_drag_from_wave_saves_position_without_cancel(self):
        for event, x in (('<Button-1>', 76), ('<B1-Motion>', 86), ('<ButtonRelease-1>', 86)):
            self.bar.canvas.event_generate(event, x=x, y=int(self.bar.CY))
            self.root.update()
        self.save.assert_called_once()
        self.cancel.assert_not_called()
        self.confirm.assert_not_called()


class RecorderBarCancelTests(unittest.TestCase):
    def test_cancel_calls_processing_cancel_and_hides_bar(self):
        ui = SimpleNamespace(transcriber=Mock(), hotkey=SimpleNamespace(active=True),
                             record_btn=Mock(), bar=Mock())
        app.App._cancel(ui)
        ui.transcriber.cancel.assert_called_once_with(from_processing=True)
        self.assertFalse(ui.hotkey.active)
        ui.record_btn.configure.assert_called_once_with(text='GRAVAR')
        ui.bar.hide.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
