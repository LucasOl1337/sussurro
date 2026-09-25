"""Model migration, hardware choices and replacement: no microphone or desktop input."""
import json
import queue
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import app
from sussurro_models import PARAKEET, ModelConfig, normalize_model_settings, resolve_model_config, model_path


class ModelSettingsTests(unittest.TestCase):
    def backend(self, cuda=1):
        return SimpleNamespace(get_cuda_device_count=Mock(return_value=cuda),
                               get_supported_compute_types=Mock(side_effect=lambda dev:
                                   {'int8_float16', 'float16'} if dev == 'cuda' else {'int8', 'float32'}))

    def test_old_install_uses_turbo_on_gpu_and_preserves_preferences(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'settings.json'
            path.write_text(json.dumps({'language': 'en', 'mouse_button': 'middle'}))
            with patch.object(app, 'SETTINGS_PATH', path):
                settings = app.load_settings()
        self.assertEqual(settings['language'], 'en')
        self.assertEqual(settings['mouse_button'], 'middle')
        self.assertEqual(resolve_model_config(settings, self.backend()),
                         ModelConfig('large-v3-turbo', 'cuda', 'int8_float16'))

    def test_no_gpu_selects_base_int8(self):
        self.assertEqual(resolve_model_config({}, self.backend(0)), ModelConfig('base', 'cpu', 'int8'))

    def test_explicit_cpu_does_not_probe_cuda(self):
        backend = self.backend()
        self.assertEqual(resolve_model_config({'whisper_device': 'cpu', 'whisper_model': 'small'}, backend),
                         ModelConfig('small', 'cpu', 'int8'))
        backend.get_cuda_device_count.assert_not_called()

    def test_explicit_gpu_reports_unavailable(self):
        with self.assertRaisesRegex(RuntimeError, 'CPU'):
            resolve_model_config({'whisper_device': 'cuda'}, self.backend(0))

    def test_broken_gpu_probe_falls_back_in_auto(self):
        backend = self.backend()
        backend.get_cuda_device_count.side_effect = RuntimeError('driver missing')
        self.assertEqual(resolve_model_config({}, backend).device, 'cpu')

    def test_unsupported_quantization_uses_supported_type(self):
        backend = self.backend()
        backend.get_supported_compute_types.side_effect = None
        backend.get_supported_compute_types.return_value = {'float32'}
        self.assertEqual(resolve_model_config({}, backend).compute_type, 'float32')

    def test_explicit_model_survives_update_and_roundtrip(self):
        settings = {'whisper_model': 'large-v3', 'whisper_device': 'cuda', 'language': 'pt'}
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(app, 'SETTINGS_PATH', Path(folder) / 'settings.json'):
                app.save_settings(settings)
                self.assertEqual(app.load_settings()['whisper_model'], 'large-v3')
        self.assertEqual(resolve_model_config(settings, self.backend()).model, 'large-v3')

    def test_invalid_preferences_recover_to_defaults(self):
        self.assertEqual(normalize_model_settings({'whisper_model': [], 'whisper_device': {'invalid': True}}),
                         {'whisper_model': 'auto', 'whisper_device': 'auto'})


class ModelReplacementTests(unittest.TestCase):
    def setUp(self):
        with patch.object(threading.Thread, 'start'), patch.object(app.keyboard, 'Controller'):
            self.t = app.Transcriber(queue.Queue(), queue.Queue())
        self.old = ModelConfig('base', 'cpu', 'int8')
        self.new = ModelConfig('large-v3-turbo', 'cuda', 'int8_float16')
        self.t.model = Mock()
        self.t.model_config = self.old

    def test_download_failure_keeps_ready_model(self):
        previous = self.t.model
        with patch.object(app, 'resolve_model_config', return_value=self.new), \
             patch.object(app, 'download_model', side_effect=RuntimeError('offline')):
            with self.assertRaisesRegex(RuntimeError, 'offline'):
                self.t.load_model({})
        self.assertIs(self.t.model, previous)
        self.assertEqual(self.t.model_config, self.old)
        self.assertFalse(self.t.model_loading.is_set())

    def test_replacement_frees_previous_model_and_recovers_if_new_fails(self):
        calls = []
        def load(config, path):
            self.assertTrue(self.t._model_lock.locked())
            self.assertTrue(self.t.model_loading.is_set())
            self.assertIsNone(self.t.model)
            calls.append(config)
            if config == self.new:
                raise RuntimeError('CUDA out of memory')
            self.t.model = Mock()
            self.t.model_config = config
        with patch.object(app, 'resolve_model_config', return_value=self.new), \
             patch.object(app, 'download_model', return_value='/cached/model') as download, \
             patch.object(self.t, '_load_model_config', side_effect=load):
            with self.assertRaisesRegex(RuntimeError, 'memory'):
                self.t.load_model({})
        self.assertEqual(calls, [self.new, self.old])
        download.assert_called_with('base', local_files_only=True)
        self.assertEqual(self.t.model_config, self.old)
        self.assertFalse(self.t.model_loading.is_set())

    def test_failed_recovery_leaves_explicit_unavailable_state(self):
        with patch.object(app, 'resolve_model_config', return_value=self.new), \
             patch.object(app, 'download_model', return_value='/cached/model'), \
             patch.object(self.t, '_load_model_config', side_effect=RuntimeError('failure')):
            with self.assertRaises(RuntimeError):
                self.t.load_model({})
        self.assertIsNone(self.t.model)
        self.assertIsNone(self.t.model_config)
        self.assertFalse(self.t.model_loading.is_set())

    def test_hotkey_cannot_start_capture_during_load(self):
        self.t.model_loading.set()
        with patch.object(self.t, '_open_stream') as stream:
            with self.assertRaisesRegex(RuntimeError, 'troca'):
                self.t.start(None, inject=False)
        stream.assert_not_called()

    def test_status_reports_actual_model_and_loading(self):
        ipc = app.IpcServer(queue.Queue(), Path('/unused.sock'), self.t)
        self.t.model_loading.set()
        status = json.loads(ipc._handle('status'))
        self.assertFalse(status['ready'])
        self.assertTrue(status['loading'])
        self.assertEqual(status['device'], 'cpu')
        self.assertEqual(status['model'], 'base')

    def test_ui_refuses_switch_during_recording(self):
        self.t.recording.set()
        ui = SimpleNamespace(transcriber=self.t, status=Mock(), _begin_model_load=Mock())
        app.App._apply_model(ui)
        ui._begin_model_load.assert_not_called()

    def test_ui_only_saves_successful_selection(self):
        ui = SimpleNamespace(transcriber=self.t, settings={'language': 'pt'}, _save=Mock(),
                             model_label=Mock(), record_btn=Mock(), apply_model_btn=Mock(),
                             model_choice=Mock(), device_choice=Mock(), status_queue=queue.Queue())
        selection = {'whisper_model': 'small', 'whisper_device': 'cpu'}
        app.App._model_load_done(ui, selection, True, 'failure')
        ui._save.assert_not_called()
        app.App._model_load_done(ui, selection, True, None)
        ui._save.assert_called_once()
        self.assertEqual(ui.settings['whisper_model'], 'small')
        self.assertEqual(ui.settings['language'], 'pt')

    def test_file_decoding_is_busy_and_clears_after_failure(self):
        with tempfile.NamedTemporaryFile(suffix='.wav') as f:
            def decode(path):
                self.assertTrue(self.t.busy())
                raise ValueError('bad audio')
            with patch.object(app, 'load_audio_16k_mono', side_effect=decode):
                with self.assertRaisesRegex(ValueError, 'bad audio'):
                    self.t.transcribe_file(f.name)
        self.assertFalse(self.t.busy())

    def test_loader_rechecks_file_activity_before_changing_weights(self):
        self.t._file_jobs = 1
        previous = self.t.model
        with patch.object(app, 'download_model') as download:
            with self.assertRaisesRegex(RuntimeError, 'trabalho'):
                self.t.load_model({})
        download.assert_not_called()
        self.assertIs(self.t.model, previous)
        self.assertFalse(self.t.model_loading.is_set())


class ModelCacheTests(unittest.TestCase):
    def test_complete_cache_never_contacts_network(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in ('model.bin', 'config.json', 'tokenizer.json'):
                (Path(folder) / name).write_bytes(b'test')
            download = Mock(return_value=folder)
            self.assertEqual(model_path('base', download), folder)
            download.assert_called_once_with('base', local_files_only=True)

    def test_partial_cache_completes_download(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / 'config.json').write_text('{}')
            download = Mock(return_value=folder)
            model_path('base', download)
            self.assertEqual(download.call_count, 2)
            download.assert_called_with('base')

    def test_first_use_downloads_model(self):
        from huggingface_hub.errors import LocalEntryNotFoundError
        download = Mock(side_effect=[LocalEntryNotFoundError('missing'), '/downloaded'])
        self.assertEqual(model_path('base', download), '/downloaded')


class ParakeetTests(unittest.TestCase):
    def backend(self, cuda=1):
        return SimpleNamespace(get_cuda_device_count=Mock(return_value=cuda),
                               get_supported_compute_types=Mock(return_value={'int8_float16'}))

    def test_parakeet_runs_on_gpu_in_fp16(self):
        config = resolve_model_config({'whisper_model': PARAKEET, 'whisper_device': 'auto'}, self.backend())
        self.assertEqual(config, ModelConfig(PARAKEET, 'cuda', 'float16'))
        self.assertEqual(config.engine, 'parakeet')
        self.assertEqual(ModelConfig('large-v3', 'cuda', 'int8_float16').engine, 'whisper')

    def test_parakeet_refuses_cpu(self):
        with self.assertRaisesRegex(RuntimeError, 'GPU'):
            resolve_model_config({'whisper_model': PARAKEET, 'whisper_device': 'cpu'}, self.backend())
        with self.assertRaisesRegex(RuntimeError, 'GPU'):
            resolve_model_config({'whisper_model': PARAKEET, 'whisper_device': 'auto'}, self.backend(0))

    def test_tokens_become_words_with_times(self):
        from sussurro_parakeet import words_from_tokens
        words = words_from_tokens([' No', ' Da', 'ily', ' W', 'ork', ','], [0.08, 0.32, 0.48, 0.72, 0.8, 0.96],
                                  [0.0] * 6, offset=10.0, end=11.2)
        self.assertEqual([w.word for w in words], [' No', ' Daily', ' Work,'])
        self.assertAlmostEqual(words[1].start, 10.32)
        self.assertAlmostEqual(words[1].end, 10.72)
        self.assertAlmostEqual(words[2].end, 11.04)

    def test_batches_bound_padded_audio(self):
        from sussurro_parakeet import batches, SAMPLE_RATE, BATCH_SECONDS
        chunks = [(0, s * SAMPLE_RATE) for s in (30, 2, 29, 5, 1, 30, 12)]
        groups = list(batches(chunks))
        self.assertEqual(sorted(i for g in groups for i in g), list(range(len(chunks))))
        for g in groups:
            longest = max(chunks[i][1] for i in g)
            self.assertTrue(len(g) == 1 or longest * len(g) <= BATCH_SECONDS * SAMPLE_RATE)

    def test_adapter_offsets_segments_and_skips_empty(self):
        import numpy as np
        import sussurro_parakeet as pk
        result = lambda text, tokens, times: SimpleNamespace(text=text, tokens=tokens, timestamps=times,
                                                             logprobs=[0.0] * len(tokens))
        model = pk.ParakeetModel.__new__(pk.ParakeetModel)
        model._asr = SimpleNamespace(recognize=lambda chunks: [
            result('Oi.', [' Oi', '.'], [0.1, 0.3]) if c.size > pk.SAMPLE_RATE else result('', [], [])
            for c in chunks])
        audio = np.zeros(pk.SAMPLE_RATE * 6, dtype=np.float32)
        spans = [(0, pk.SAMPLE_RATE // 2), (2 * pk.SAMPLE_RATE, 4 * pk.SAMPLE_RATE)]
        with patch.object(pk, 'speech_chunks', return_value=spans):
            segments, info = model.transcribe(audio, language='pt', vad_filter=True, word_timestamps=True)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].text, ' Oi.')
        self.assertAlmostEqual(segments[0].start, 2.1)
        self.assertEqual([w.word for w in segments[0].words], [' Oi.'])
        self.assertEqual(info.duration, 6)

    def test_neighbouring_speech_shares_one_window(self):
        from sussurro_parakeet import merge_spans
        self.assertEqual(merge_spans([(0, 5), (7, 12), (14, 30), (31, 33)], 20), [(0, 12), (14, 33)])
