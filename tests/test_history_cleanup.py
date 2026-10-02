"""Limpeza opt-in testada apenas em arquivos temporarios, nunca no historico vivo."""
import json
import os
from pathlib import Path
import queue
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import app


class HistoryCleanupTests(unittest.TestCase):
    def setUp(self):
        with patch.object(threading.Thread, 'start'), patch.object(app.keyboard, 'Controller'):
            self.t = app.Transcriber(queue.Queue(), queue.Queue())
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.folder = Path(directory.name)
        self.now = app.datetime(2026, 10, 2, 15, 0)
        for name, value in [('HISTORY_DIR', self.folder), ('HISTORY_INDEX', self.folder / 'history.jsonl')]:
            patcher = patch.object(app, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.entries = [
            {'ts': '2026-08-01T10:00:00', 'wav': 'old.wav', 'text': 'texto antigo', 'dur': 2., 'fix': 1},
            {'ts': '2026-10-01T10:00:00', 'wav': 'recent.wav', 'text': 'texto recente', 'dur': 2., 'fix': 0},
            {'ts': '2026-08-01T10:00:01', 'wav': 'failed.wav', 'text': '', 'failed': True, 'dur': 2., 'fix': 0},
        ]
        for entry in self.entries:
            (self.folder / entry['wav']).write_bytes(b'WAV fixture')
        self.index = '\n'.join(json.dumps(entry) for entry in self.entries) + '\n'
        app.HISTORY_INDEX.write_text(self.index)

    def test_preview_then_confirm_deletes_only_old_transcribed_audio_keeps_index_bytes(self):
        preview = self.t.prune_history_audio(30, now=self.now)
        self.assertEqual(preview['files'], ['old.wav'])
        self.assertEqual(preview['bytes'], len(b'WAV fixture'))
        self.assertEqual(preview['deleted'], 0)
        self.assertTrue((self.folder / 'old.wav').exists())
        result = self.t.prune_history_audio(30, confirmed=True, now=self.now)
        self.assertEqual(result['deleted'], 1)
        self.assertFalse((self.folder / 'old.wav').exists())
        self.assertTrue((self.folder / 'recent.wav').exists())
        self.assertTrue((self.folder / 'failed.wav').exists())
        self.assertEqual(app.HISTORY_INDEX.read_text(), self.index)
        self.assertTrue(self.t.status_queue.empty())
        self.assertTrue(self.t.history_queue.empty())
        again = self.t.prune_history_audio(30, confirmed=True, now=self.now)
        self.assertEqual(again['deleted'], 0)

    def write_entries(self, entries):
        app.HISTORY_INDEX.write_text('\n'.join(json.dumps(entry) for entry in entries) + '\n')

    def test_exact_age_boundary_and_invalid_date_are_preserved(self):
        old = self.entries[0]
        self.write_entries([{**old, 'ts': (self.now - app.timedelta(days=30)).isoformat()},
                            {**self.entries[1], 'ts': 'data-invalida'}])
        self.assertEqual(self.t.prune_history_audio(30, confirmed=True, now=self.now)['deleted'], 0)
        self.assertTrue((self.folder / 'old.wav').exists())

    def test_unreferenced_non_wav_traversal_and_symlinks_are_never_deleted(self):
        outside = self.folder.parent / (self.folder.name + '-external.wav')
        outside.write_bytes(b'external')
        self.addCleanup(outside.unlink)
        (self.folder / 'link.wav').symlink_to(outside)
        (self.folder / 'other.txt').write_text('not audio')
        (self.folder / 'orphan.wav').write_bytes(b'orphan')
        old = self.entries[0]
        self.write_entries([{**old, 'wav': '../' + outside.name}, {**old, 'wav': str(outside)},
                            {**old, 'wav': 'link.wav'}, {**old, 'wav': 'other.txt'}])
        self.assertEqual(self.t.prune_history_audio(30, confirmed=True, now=self.now)['deleted'], 0)
        self.assertEqual(outside.read_bytes(), b'external')
        self.assertTrue((self.folder / 'link.wav').is_symlink())
        self.assertTrue((self.folder / 'orphan.wav').exists())

    def test_busy_jobs_and_invalid_days_never_delete(self):
        for days in (0, -1, True, 1.5, 36501):
            with self.subTest(days=days), self.assertRaises(ValueError):
                self.t.prune_history_audio(days, confirmed=True, now=self.now)
        for attribute, value in [('_drained', False), ('_pending', 1), ('_file_jobs', 1), ('_meeting_jobs', 1)]:
            before = getattr(self.t, attribute)
            setattr(self.t, attribute, value)
            with self.subTest(attribute=attribute), self.assertRaisesRegex(RuntimeError, 'Aguarde'):
                self.t.prune_history_audio(30, confirmed=True, now=self.now)
            setattr(self.t, attribute, before)
        for event in (self.t.recording, self.t._retrying):
            event.set()
            with self.assertRaises(RuntimeError):
                self.t.prune_history_audio(30, confirmed=True, now=self.now)
            event.clear()
        self.assertTrue((self.folder / 'old.wav').exists())

    def test_permission_error_is_reported_without_removing_text_or_claiming_bytes(self):
        with patch.object(Path, 'unlink', side_effect=PermissionError('negado')):
            result = self.t.prune_history_audio(30, confirmed=True, now=self.now)
        self.assertEqual(result['deleted'], 0)
        self.assertEqual(result['bytes'], 0)
        self.assertIn('old.wav: negado', result['errors'])
        self.assertEqual(app.HISTORY_INDEX.read_text(), self.index)

    def test_duplicate_with_missing_text_protects_original_recoverable_audio(self):
        self.write_entries([self.entries[0], {**self.entries[0], 'text': ''}])
        self.assertEqual(self.t.prune_history_audio(30, confirmed=True, now=self.now)['deleted'], 0)

    def test_new_candidate_after_preview_is_not_part_of_confirmed_removal(self):
        preview = self.t.prune_history_audio(30, now=self.now)
        other = {**self.entries[0], 'wav': 'newly-added.wav'}
        (self.folder / other['wav']).write_bytes(b'preserve')
        self.write_entries(self.entries + [other])
        result = self.t.prune_history_audio(30, now=self.now, confirmed=True, only_files=preview['files'])
        self.assertEqual(result['deleted'], 1)
        self.assertTrue((self.folder / other['wav']).exists())

    def test_duplicate_recent_entry_protects_shared_wav(self):
        self.write_entries([self.entries[0], {**self.entries[0], 'ts': self.now.isoformat()}])
        self.assertEqual(self.t.prune_history_audio(30, confirmed=True, now=self.now)['deleted'], 0)

    def ui(self):
        return SimpleNamespace(root=None, _playing=None, status=Mock(), transcriber=self.t, _render_history=Mock())

    def test_ui_cancel_age_or_confirmation_has_no_effect_on_disk(self):
        for days, approve in [(None, True), (30, False)]:
            with self.subTest(days=days), patch.object(app.simpledialog, 'askinteger', return_value=days), \
                 patch.object(app.messagebox, 'askyesno', return_value=approve) as confirm:
                app.App._prune_history_audio(self.ui())
            self.assertTrue((self.folder / 'old.wav').exists())
            if days is not None:
                self.assertEqual(confirm.call_args.kwargs['default'], app.messagebox.NO)
                self.assertIn('não pode ser desfeita', confirm.call_args.args[1])

    def test_ui_confirm_deletes_audio_and_keeps_copy_text_and_missing_audio_actions_clear(self):
        ui = self.ui()
        with patch.object(app.simpledialog, 'askinteger', return_value=30), \
             patch.object(app.messagebox, 'askyesno', return_value=True):
            app.App._prune_history_audio(ui)
        self.assertFalse((self.folder / 'old.wav').exists())
        self.assertEqual(app.HISTORY_INDEX.read_text(), self.index)
        app.App._play(ui, str(self.folder / 'old.wav'))
        self.assertIn('texto do histórico foi mantido', ui.status.configure.call_args.kwargs['text'])
        with patch.object(app, 'RetranscribeDialog') as dialog:
            app.App._retry_entry(ui, self.entries[0])
        dialog.assert_not_called()
        self.assertEqual(app.App._load_history()[-1]['text'], 'texto antigo')

    @unittest.skipUnless(os.environ.get('SUSSURRO_TEST_TK') == '1', 'Tk apenas na bancada isolada')
    def test_real_app_button_age_dialog_no_then_yes_preserves_text_and_index(self):
        from contextlib import ExitStack
        from sussurro_hardware import HardwareInfo
        root = app.ctk.CTk()
        self.addCleanup(root.destroy)
        with ExitStack() as stack:
            stack.enter_context(patch.object(app, 'load_settings', return_value=dict(app.DEFAULT_SETTINGS)))
            stack.enter_context(patch.object(app, 'list_input_devices', return_value={}))
            stack.enter_context(patch.object(app, 'list_loopback_devices', return_value={}))
            stack.enter_context(patch.object(app, 'detect_hardware', return_value=HardwareInfo()))
            stack.enter_context(patch.object(app.App, '_begin_model_load'))
            stack.enter_context(patch.object(app.App, '_build_devices_tab', side_effect=lambda: app.ctk.CTkFrame(root)))
            stack.enter_context(patch.object(app, 'MeetingPanel', side_effect=lambda parent, host: app.ctk.CTkFrame(parent)))
            stack.enter_context(patch.object(app, 'RecorderBar'))
            stack.enter_context(patch.object(threading.Thread, 'start'))
            stack.enter_context(patch.object(app.keyboard, 'Controller'))
            ui = app.App(root)
        ui.transcriber = self.t
        root.update()
        # Xvfb nao mapeia a janela de teste: simula o canvas ja dimensionado, como no app aberto.
        ui.hist_frame._on_cfg(SimpleNamespace(width=600))
        button = next(child for child in ui.hist_frame.winfo_children()
                      if isinstance(child, app.ctk.CTkButton) and child.cget('text') == 'Liberar espaço')
        decisions = []
        def answer_age_then_confirmation(approve):
            def age():
                dialogs = [w for w in root.winfo_children() if isinstance(w, app.simpledialog.Dialog)]
                if not dialogs:
                    root.after(10, age)
                    return
                dialog = dialogs[0]
                dialog.entry.delete(0, 'end')
                dialog.entry.insert(0, '30')
                root.after(30, confirmation)
                dialog.ok()
            def confirmation():
                name = '.__tk__messagebox'
                if not int(root.tk.call('winfo', 'exists', name)):
                    root.after(10, confirmation)
                    return
                decisions.append(approve)
                root.tk.call(name + ('.yes' if approve else '.no'), 'invoke')
            root.after(10, age)
        answer_age_then_confirmation(False)
        button.invoke()
        self.assertTrue((self.folder / 'old.wav').exists())
        answer_age_then_confirmation(True)
        button.invoke()
        root.update()
        self.assertEqual(decisions, [False, True])
        self.assertFalse((self.folder / 'old.wav').exists())
        self.assertEqual(app.HISTORY_INDEX.read_text(), self.index)
        rendered = [ui.hist_frame.canvas.itemcget(item, 'text') for item in ui.hist_frame.canvas.find_all()
                    if ui.hist_frame.canvas.type(item) == 'text']
        self.assertTrue(any('texto antigo' in text and 'Áudio removido' in text for text in rendered))
        self.assertTrue(any('Sem áudio' == text for text in rendered))
        self.assertIn('Textos mantidos', ui.status.cget('text'))
        ui._copy_entry('texto antigo')
        self.assertEqual(root.clipboard_get(), 'texto antigo')


if __name__ == '__main__':
    unittest.main()
