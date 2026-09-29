"""Refazer ditados prontos ou falhos sem perder texto, audio ou o modelo habitual."""
import json
import queue
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import app
from sussurro_models import ModelConfig, PARAKEET


class HistoryRetryTests(unittest.TestCase):
    def setUp(self):
        with patch.object(threading.Thread, 'start'), patch.object(app.keyboard, 'Controller'):
            self.t = app.Transcriber(queue.Queue(), queue.Queue())
        self.t.model = Mock()
        self.original = ModelConfig('base', 'cpu', 'int8')
        self.t.model_config = self.original
        self.audio = np.ones(app.SAMPLE_RATE, dtype=np.float32) * .05
        self.segment = SimpleNamespace(text=' texto corrigido', start=0., end=1.)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.folder = Path(self.directory.name)
        for target, value in [('HISTORY_DIR', self.folder),
                              ('HISTORY_INDEX', self.folder / 'history.jsonl')]:
            patcher = patch.object(app, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        perf = patch.object(app, '_perf')
        perf.start()
        self.addCleanup(perf.stop)
        trace = patch.object(app.traceback, 'print_exc')
        trace.start()
        self.addCleanup(trace.stop)
        self.entry = self.t._archive_audio(self.audio, 'texto anterior', 2,
                                           app.datetime(2026, 9, 29, 9, 0, 0))
        self.t.history_queue.get_nowait()
        self.path = self.folder / self.entry['wav']
        self.wav = self.path.read_bytes()

    def rows(self):
        return [json.loads(line) for line in app.HISTORY_INDEX.read_text().splitlines()]

    def test_success_replaces_only_selected_entry_and_preserves_wav(self):
        other = self.t._archive_audio(self.audio, 'outro ditado', 0,
                                      app.datetime(2026, 9, 29, 9, 1, 0))
        self.t.history_queue.get_nowait()
        self.entry['retrying'] = True
        self.t._retrying.set()
        with patch.object(self.t, '_transcribe_locked', return_value=([self.segment], None)) as transcribe, \
             patch.object(self.t.library, 'apply', return_value=('texto corrigido', 1)), \
             patch.object(self.t, '_paste') as paste:
            self.t._retry_history_worker(self.entry, self.path, language='en')
        updated, unchanged = self.rows()
        self.assertEqual(updated['text'], 'texto corrigido')
        self.assertEqual(updated['wav'], self.entry['wav'])
        self.assertEqual(updated['ts'], self.entry['ts'])
        self.assertEqual(updated['fix'], 1)
        self.assertEqual(updated['model'], 'base')
        self.assertEqual(updated['language'], 'en')
        self.assertNotIn('retrying', updated)
        self.assertNotIn('failed', updated)
        self.assertEqual(unchanged, other)
        self.assertEqual(self.path.read_bytes(), self.wav)
        self.assertEqual(transcribe.call_args.kwargs['language'], 'en')
        self.assertEqual(self.t.history_queue.get_nowait(), updated)
        self.assertFalse(self.t.busy())
        self.assertTrue(self.t.text_queue.empty())
        paste.assert_not_called()

    def test_failure_or_empty_result_preserves_successful_text_and_copyability(self):
        for result in (RuntimeError('modelo indisponivel'), ([], None)):
            with self.subTest(result=result):
                self.t._retrying.set()
                self.entry['retrying'] = True
                kwargs = {'side_effect': result} if isinstance(result, Exception) else {'return_value': result}
                with patch.object(self.t, '_transcribe_locked', **kwargs):
                    self.t._retry_history_worker(self.entry, self.path)
                updated = self.t.history_queue.get_nowait()
                self.assertEqual(updated['text'], 'texto anterior')
                self.assertEqual(updated['fix'], 2)
                self.assertNotIn('failed', updated)
                self.assertNotIn('retrying', updated)
                self.assertIn('retry_error', updated)
                self.assertEqual(self.rows(), [updated])
                self.assertEqual(self.path.read_bytes(), self.wav)
                self.assertFalse(self.t.busy())

    def test_failed_save_preserves_original_and_releases_ui(self):
        with patch.object(self.t, '_transcribe_locked', return_value=([self.segment], None)), \
             patch.object(app.os, 'replace', side_effect=OSError('disco cheio')):
            self.t._retrying.set()
            self.t._retry_history_worker(self.entry, self.path)
        self.assertEqual(self.rows(), [self.entry])
        updated = self.t.history_queue.get_nowait()
        self.assertEqual(updated['text'], self.entry['text'])
        self.assertIn('disco cheio', updated['retry_error'])
        self.assertFalse(self.t.busy())

    def test_async_accepts_completed_entry_and_snapshots_choices(self):
        selection = {'whisper_model': 'large-v3', 'whisper_device': 'cuda'}
        with patch.object(app.threading, 'Thread') as thread:
            self.t.retry_history_async(self.entry, selection, 'auto')
            args = thread.call_args.kwargs['args']
            selection['whisper_model'] = 'tiny'
            self.t.language = 'en'
            self.assertEqual(args[2]['whisper_model'], 'large-v3')
            self.assertEqual(args[3], 'auto')
            self.assertIsNot(args[0], self.entry)
            self.assertTrue(self.t.busy())
            with self.assertRaisesRegex(RuntimeError, 'Aguarde'):
                self.t.retry_history_async(self.entry)
            thread.return_value.start.assert_called_once()

    def test_rejects_missing_audio_and_traversal_without_leaving_busy(self):
        for name, error in [('missing.wav', FileNotFoundError), ('../outside.wav', ValueError)]:
            with self.subTest(name=name), self.assertRaises(error):
                self.t.retry_history_async({**self.entry, 'wav': name})
            self.assertFalse(self.t.busy())

    def test_meeting_blocks_retry_even_between_inferences(self):
        self.t.begin_meeting_job()
        with self.assertRaisesRegex(RuntimeError, 'Aguarde'):
            self.t.retry_history_async(self.entry)
        self.t.end_meeting_job()
        self.assertFalse(self.t.busy())

    def test_retry_blocks_capture_files_meetings_and_comparison(self):
        self.t._retrying.set()
        calls = [lambda: self.t.start(None, False), lambda: self.t.transcribe_file(str(self.path)),
                 self.t.begin_meeting_job, self.t.acquire_comparison,
                 lambda: self.t.load_model({})]
        for call in calls:
            with self.subTest(call=call), self.assertRaises(RuntimeError):
                call()
        self.assertEqual(self.t._file_jobs, 0)
        self.assertEqual(self.t._meeting_jobs, 0)

    def test_temporary_engine_restores_previous_after_success_or_failure(self):
        config = ModelConfig(PARAKEET, 'cuda', 'float16')
        for fail in (False, True):
            with self.subTest(fail=fail):
                models, loaded = [], []
                self.t.model_config = self.original
                self.t.model = Mock()
                def load(choice, path):
                    self.assertTrue(self.t._model_lock.locked())
                    self.assertIsNone(self.t.model)
                    loaded.append(choice)
                    model = Mock()
                    if fail and choice == config:
                        model.transcribe.side_effect = RuntimeError('falha na inferencia')
                    else:
                        model.transcribe.return_value = (iter([self.segment]), None)
                    models.append(model)
                    self.t.model = model
                    self.t.model_config = choice
                with patch.object(self.t, '_weights', return_value='/cache') as weights, \
                     patch.object(self.t, '_load_model_config', side_effect=load), \
                     patch.object(app, 'drop_hallucinations', side_effect=lambda segments, audio: segments):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, 'inferencia'):
                            self.t._transcribe_locked(self.audio, language='pt', vad_filter=True, config=config)
                    else:
                        segments, _ = self.t._transcribe_locked(self.audio, language='pt', vad_filter=True, config=config)
                        self.assertEqual(segments, [self.segment])
                self.assertEqual(loaded, [config, self.original])
                self.assertEqual(self.t.model_config, self.original)
                weights.assert_called_with(self.original, local_only=True)
                models[0].close.assert_called_once()
                self.assertFalse(self.t.busy())

    def test_same_model_does_not_reload_weights(self):
        self.t.model.transcribe.return_value = (iter([self.segment]), None)
        with patch.object(self.t, '_weights') as weights, \
             patch.object(app, 'drop_hallucinations', side_effect=lambda segments, audio: segments):
            self.t._transcribe_locked(self.audio, language=None, vad_filter=True, config=self.original)
        weights.assert_not_called()

    def test_restoration_failure_keeps_original_text_and_marks_model_unavailable(self):
        config = ModelConfig('small', 'cpu', 'int8')
        def load(choice, path):
            if choice == self.original:
                raise RuntimeError('carga falhou')
            self.t.model = Mock()
            self.t.model.transcribe.return_value = (iter([self.segment]), None)
            self.t.model_config = choice
        with patch.object(app, 'resolve_model_config', return_value=config), \
             patch.object(self.t, '_weights', return_value='/cache'), \
             patch.object(self.t, '_load_model_config', side_effect=load), \
             patch.object(app, 'drop_hallucinations', side_effect=lambda segments, audio: segments):
            self.t._retry_history_worker(self.entry, self.path, {'whisper_model': 'small'})
        self.assertEqual(self.rows()[0]['text'], self.entry['text'])
        self.assertIn('Aplicar modelo', self.rows()[0]['retry_error'])
        self.assertIsNone(self.t.model)
        self.assertIsNone(self.t.model_config)
        self.assertFalse(self.t.busy())


class HistoryActionsTests(unittest.TestCase):
    def test_completed_entry_has_independent_retry_copy_and_play_actions(self):
        entry = {'wav': 'saved.wav', 'text': 'texto pronto'}
        hit = {'i': 0, 'y0': 0, 'y1': 40, 'play': (300, 0, 328, 40),
               'copy': (330, 0, 358, 40), 'retry': (360, 0, 428, 40), 'text': (60, 0, 298, 40)}
        ui = SimpleNamespace(_entries=[entry], _hits=[hit], on_retry=Mock(), on_play=Mock(),
                             on_copy=Mock(), canvas=Mock(canvasx=lambda x: x, canvasy=lambda y: y))
        ui._hit = lambda x, y: app.HistoryList._hit(ui, x, y)
        for x, zone in [(310, 'play'), (340, 'copy'), (390, 'retry'), (100, 'text')]:
            self.assertEqual(ui._hit(x, 20)[1], zone)
            app.HistoryList._on_click(ui, SimpleNamespace(x=x, y=20))
        ui.on_retry.assert_called_once_with(entry)
        ui.on_play.assert_called_once_with(str(app.HISTORY_DIR / 'saved.wav'))
        self.assertEqual(ui.on_copy.call_count, 2)
        ui.on_copy.assert_called_with('texto pronto')


if __name__ == '__main__':
    unittest.main()
