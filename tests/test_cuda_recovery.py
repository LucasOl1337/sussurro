"""Recuperacao CUDA sem carregar pesos, abrir captura ou tocar GPU."""
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
from sussurro_models import ModelConfig


class CudaRecoveryTests(unittest.TestCase):
    def setUp(self):
        with patch.object(threading.Thread, 'start'), patch.object(app.keyboard, 'Controller'):
            self.t = app.Transcriber(queue.Queue(), queue.Queue())
        self.config = ModelConfig('base', 'cuda', 'float16')
        self.t.model_config = self.config
        self.t.model = Mock()
        self.audio = np.ones(app.SAMPLE_RATE, dtype=np.float32) * .05
        self.segment = SimpleNamespace(text=' recuperado', start=0., end=1.)
        hallucinations = patch.object(app, 'drop_hallucinations', side_effect=lambda segments, audio: segments)
        hallucinations.start()
        self.addCleanup(hallucinations.stop)

    def test_cuda_generator_failure_reloads_once_and_retries_same_audio(self):
        def fail_late():
            yield self.segment
            raise RuntimeError('CUDA failed with error out of memory')
        old = self.t.model
        old.transcribe.return_value = (fail_late(), None)
        recovered = Mock()
        recovered.transcribe.return_value = (iter([self.segment]), 'info')
        def load(config, path):
            self.assertTrue(self.t._model_lock.locked())
            self.assertIsNone(self.t.model)
            self.t.model, self.t.model_config = recovered, config
        with patch.object(self.t, '_weights', return_value='/cached') as weights, \
             patch.object(self.t, '_load_model_config', side_effect=load) as loading:
            segments, info = self.t._transcribe_locked(self.audio, language='pt', vad_filter=True)
        self.assertEqual(segments, [self.segment])
        self.assertEqual(info, 'info')
        loading.assert_called_once_with(self.config, '/cached')
        weights.assert_called_once_with(self.config, local_only=True)
        self.assertIs(old.transcribe.call_args.args[0], self.audio)
        self.assertIs(recovered.transcribe.call_args.args[0], self.audio)
        self.assertEqual(old.transcribe.call_args.kwargs, recovered.transcribe.call_args.kwargs)

    def test_second_cuda_failure_clears_model_and_ipc_ready(self):
        self.t.model.transcribe.side_effect = RuntimeError('cudaError invalid device')
        recovered = Mock()
        recovered.transcribe.side_effect = RuntimeError('CUDA out of memory')
        def load(config, path):
            self.t.model, self.t.model_config = recovered, config
        with patch.object(self.t, '_weights', return_value='/cached'), \
             patch.object(self.t, '_load_model_config', side_effect=load) as loading:
            with self.assertRaisesRegex(RuntimeError, 'Recarregando sozinho'):
                self.t._transcribe_locked(self.audio, language='pt', vad_filter=False)
        loading.assert_called_once()
        recovered.transcribe.assert_called_once()
        self.assertIsNone(self.t.model)
        self.assertIsNone(self.t.model_config)
        messages = []
        while not self.t.status_queue.empty():
            messages.append(self.t.status_queue.get_nowait())
        self.assertTrue(any('Recarregando sozinho' in message for message in messages))
        ipc = app.IpcServer(queue.Queue(), app.IPC_SOCK, self.t)
        self.assertFalse(json.loads(ipc._handle('status'))['ready'])

    def test_public_history_retry_recovers_or_preserves_text_after_double_failure(self):
        for fail_again in (False, True):
            with self.subTest(fail_again=fail_again), tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                with patch.object(app, 'HISTORY_DIR', folder), \
                     patch.object(app, 'HISTORY_INDEX', folder / 'history.jsonl'), \
                     patch.object(app, '_perf'), patch.object(app.traceback, 'print_exc'):
                    self.t.model_config = self.config
                    self.t.model = Mock()
                    self.t.model.transcribe.side_effect = RuntimeError('CUDA out of memory')
                    entry = self.t._archive_audio(self.audio, 'texto anterior', 0, app.datetime.now())
                    self.t.history_queue.get_nowait()
                    wav_before = (folder / entry['wav']).read_bytes()
                    recovered = Mock()
                    if fail_again:
                        recovered.transcribe.side_effect = RuntimeError('cudaError invalid device')
                    else:
                        recovered.transcribe.return_value = (iter([self.segment]), None)
                    def load(config, path):
                        self.t.model, self.t.model_config = recovered, config
                    with patch.object(self.t, '_weights', return_value='/cache'), \
                         patch.object(self.t, '_load_model_config', side_effect=load) as loading:
                        self.t.retry_history_async(entry, language='pt')
                        updated = self.t.history_queue.get(timeout=3)
                    loading.assert_called_once()
                    self.assertFalse(self.t._retrying.is_set())
                    self.assertEqual((folder / entry['wav']).read_bytes(), wav_before)
                    saved = json.loads(app.HISTORY_INDEX.read_text())
                    self.assertEqual(saved, updated)
                    if fail_again:
                        self.assertEqual(updated['text'], 'texto anterior')
                        self.assertIn('Recarregando sozinho', updated['retry_error'])
                        self.assertIsNone(self.t.model)
                    else:
                        self.assertEqual(updated['text'], 'recuperado')
                        self.assertNotIn('retry_error', updated)
                        self.assertIs(self.t.model, recovered)

    def test_non_cuda_runtime_error_preserves_model_without_reload(self):
        old = self.t.model
        old.transcribe.side_effect = RuntimeError('audio invalido')
        with patch.object(self.t, '_load_model_config') as loading:
            with self.assertRaisesRegex(RuntimeError, 'audio invalido'):
                self.t._transcribe_locked(self.audio, language=None, vad_filter=False)
        loading.assert_not_called()
        self.assertIs(self.t.model, old)
        self.assertTrue(self.t.status_queue.empty())

    def test_dictation_cuda_retry_emits_one_complete_text_and_releases_pending(self):
        def fail_late():
            yield SimpleNamespace(text=' parcial descartado', start=0., end=1.)
            raise RuntimeError('cudaError invalid device')
        self.t.model.transcribe.return_value = (fail_late(), None)
        recovered = Mock()
        recovered.transcribe.return_value = (iter([self.segment]), None)
        def load(config, path):
            self.t.model, self.t.model_config = recovered, config
        self.t._enqueue_segment(self.audio)
        events = iter([self.t._segment_queue.get_nowait()])
        self.t._segment_queue.get = lambda: next(events)
        with patch.object(self.t, '_weights', return_value='/cache'), \
             patch.object(self.t, '_load_model_config', side_effect=load) as loading:
            with self.assertRaises(StopIteration):
                self.t._transcribe_loop()
        loading.assert_called_once()
        self.assertEqual(self.t.text_queue.get_nowait(), 'recuperado')
        self.assertTrue(self.t.text_queue.empty())
        self.assertEqual(self.t._session_errors, [])
        delivery = iter([self.t._delivery_queue.get_nowait()])
        self.t._delivery_queue.get = lambda: next(delivery)
        with patch.object(app, '_perf'), patch.object(self.t, '_paste') as paste:
            with self.assertRaises(StopIteration):
                self.t._delivery_loop()
        paste.assert_not_called()
        self.assertEqual(self.t._pending, 0)

    def test_cuda_oom_or_invalid_device_without_cuda_prefix_retries(self):
        for error in ('out of memory', 'invalid device ordinal', 'cudaErrorUnknown'):
            with self.subTest(error=error):
                self.t.model_config = self.config
                self.t.model = Mock()
                self.t.model.transcribe.side_effect = RuntimeError(error)
                def load(config, path):
                    self.t.model = Mock()
                    self.t.model.transcribe.return_value = (iter([self.segment]), None)
                    self.t.model_config = config
                with patch.object(self.t, '_weights', return_value='/cache'), \
                     patch.object(self.t, '_load_model_config', side_effect=load) as loading:
                    segments, _ = self.t._transcribe_locked(self.audio, language='pt', vad_filter=True)
                self.assertEqual(segments, [self.segment])
                loading.assert_called_once()

    def test_cpu_oom_does_not_trigger_cuda_recovery(self):
        self.t.model_config = ModelConfig('base', 'cpu', 'int8')
        self.t.model.transcribe.side_effect = RuntimeError('out of memory')
        with patch.object(self.t, '_load_model_config') as loading:
            with self.assertRaisesRegex(RuntimeError, 'out of memory'):
                self.t._transcribe_locked(self.audio, language=None, vad_filter=False)
        loading.assert_not_called()
        self.assertIsNotNone(self.t.model)

    def test_reload_failure_clears_partial_model_and_requests_apply(self):
        self.t.model.transcribe.side_effect = RuntimeError('CUDA out of memory')
        def load(config, path):
            self.t.model, self.t.model_config = Mock(), config
            raise RuntimeError('carga falhou')
        with patch.object(self.t, '_weights', return_value='/cache'), \
             patch.object(self.t, '_load_model_config', side_effect=load) as loading:
            with self.assertRaisesRegex(RuntimeError, 'Recarregando sozinho'):
                self.t._transcribe_locked(self.audio, language='pt', vad_filter=True)
        loading.assert_called_once()
        self.assertIsNone(self.t.model)
        self.assertIsNone(self.t.model_config)

    def test_temporary_cuda_double_failure_does_not_restore_and_claim_ready(self):
        temporary = ModelConfig('small', 'cuda', 'float16')
        loaded = []
        def load(config, path):
            loaded.append(config)
            self.t.model, self.t.model_config = Mock(), config
            self.t.model.transcribe.side_effect = RuntimeError('CUDA invalid device')
        with patch.object(self.t, '_weights', return_value='/cache'), \
             patch.object(self.t, '_load_model_config', side_effect=load):
            with self.assertRaisesRegex(RuntimeError, 'Recarregando sozinho'):
                self.t._transcribe_locked(self.audio, language='pt', vad_filter=True, config=temporary)
        self.assertEqual(loaded, [temporary, temporary])
        self.assertIsNone(self.t.model)
        self.assertIsNone(self.t.model_config)


if __name__ == '__main__':
    unittest.main()
