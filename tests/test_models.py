"""Model migration, hardware choices and replacement: no microphone or desktop input."""
import json
import queue
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import app
from sussurro_cloud import ELEVENLABS, XAI, CloudError
from sussurro_models import (ModelConfig, fallback_config, normalize_model_settings, resolve_model_config,
                             model_path)


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

    def test_no_gpu_runs_turbo_int8_on_cpu(self):
        self.assertEqual(resolve_model_config({}, self.backend(0)), ModelConfig('large-v3-turbo', 'cpu', 'int8'))

    def test_explicit_cpu_does_not_probe_cuda(self):
        backend = self.backend()
        self.assertEqual(resolve_model_config({'whisper_device': 'cpu', 'whisper_model': 'large-v3'}, backend),
                         ModelConfig('large-v3', 'cpu', 'int8'))
        backend.get_cuda_device_count.assert_not_called()

    def test_removed_models_migrate_to_turbo_keeping_device(self):
        for old in ('auto', 'tiny', 'large-v3-turbo', 'small', 'medium', 'parakeet-tdt-0.6b-v3'):
            with self.subTest(old=old):
                self.assertEqual(normalize_model_settings({'whisper_model': old, 'whisper_device': 'cpu'}),
                                 {'whisper_model': 'large-v3-turbo', 'whisper_device': 'cpu'})

    def test_cloud_never_touches_cuda(self):
        backend = self.backend()
        for provider in (ELEVENLABS, XAI):
            with self.subTest(provider=provider):
                config = resolve_model_config({'whisper_model': provider, 'whisper_device': 'cuda'}, backend)
                self.assertEqual((config.engine, config.device), ('cloud', 'cloud'))
                self.assertIn('NUVEM', config.label)
        backend.get_cuda_device_count.assert_not_called()

    def test_cloud_fallback_is_local_turbo_on_chosen_device(self):
        self.assertEqual(fallback_config({'whisper_model': XAI, 'whisper_device': 'cpu'}, self.backend()),
                         ModelConfig('large-v3-turbo', 'cpu', 'int8'))

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
                         {'whisper_model': 'large-v3-turbo', 'whisper_device': 'auto'})


class ModelReplacementTests(unittest.TestCase):
    def setUp(self):
        with patch.object(threading.Thread, 'start'), patch.object(app.keyboard, 'Controller'):
            self.t = app.Transcriber(queue.Queue(), queue.Queue())
        self.old = ModelConfig('large-v3', 'cpu', 'int8')
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
        download.assert_called_with('large-v3', local_files_only=True)
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
        self.assertEqual(status['model'], 'large-v3')

    def test_ui_refuses_switch_during_recording(self):
        self.t.recording.set()
        ui = SimpleNamespace(transcriber=self.t, status=Mock(), _begin_model_load=Mock(),
                             _show_model_selection=Mock())
        app.App._apply_model(ui)
        ui._begin_model_load.assert_not_called()
        ui._show_model_selection.assert_called_once()  # combo volta pro modelo ativo

    def test_picking_a_model_loads_it_without_extra_click(self):
        ui = SimpleNamespace(transcriber=self.t, status=Mock(), _begin_model_load=Mock(),
                             settings={'whisper_model': 'large-v3', 'whisper_device': 'cuda'},
                             model_choice=Mock(get=Mock(return_value=app.MODEL_LABELS[XAI])),
                             device_choice=Mock(get=Mock(return_value='GPU NVIDIA')),
                             device_labels={'auto': 'Automatico', 'cuda': 'GPU NVIDIA', 'cpu': 'CPU'})
        app.App._apply_model(ui)
        ui._begin_model_load.assert_called_once_with(
            {'whisper_model': XAI, 'whisper_device': 'cuda'}, persist=True)
        ui._begin_model_load.reset_mock()
        ui.model_choice.get.return_value = app.MODEL_LABELS['large-v3']
        app.App._apply_model(ui)
        ui._begin_model_load.assert_not_called()  # mesma escolha ja carregada

    def test_engine_status_says_what_transcribes_next(self):
        self.assertEqual(self.t.engine_status(), ('Large-v3 · CPU', 'ok'))
        self.t.model_config = app.resolve_model_config({'whisper_model': XAI})
        self.t.model = SimpleNamespace(last_provider=None)
        self.assertEqual(self.t.engine_status(), ('Grok (xAI) · nuvem', 'ok'))
        self.t.model.last_provider = ELEVENLABS
        text, level = self.t.engine_status()
        self.assertEqual(level, 'aviso')
        self.assertTrue(text.startswith('ElevenLabs Scribe · nuvem'))
        self.t._cloud_down_until = app.time.monotonic() + 30
        text, level = self.t.engine_status()
        self.assertEqual(level, 'aviso')
        self.assertIn('Turbo', text)
        self.assertIn('Grok (xAI) fora', text)
        self.t.model_loading.set()
        self.assertEqual(self.t.engine_status()[1], 'aviso')

    def test_ui_only_saves_successful_selection(self):
        ui = SimpleNamespace(transcriber=self.t, settings={'language': 'pt'}, _save=Mock(),
                             model_label=Mock(), record_btn=Mock(), _set_model_picker=Mock(), _show_model_selection=Mock(),
                             model_choice=Mock(), device_choice=Mock(), status_queue=queue.Queue())
        selection = {'whisper_model': ELEVENLABS, 'whisper_device': 'cpu'}
        app.App._model_load_done(ui, selection, True, 'failure')
        ui._save.assert_not_called()
        app.App._model_load_done(ui, selection, True, None)
        ui._save.assert_called_once()
        self.assertEqual(ui.settings['whisper_model'], ELEVENLABS)
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
            self.assertEqual(model_path('large-v3-turbo', download), folder)
            download.assert_called_once_with('large-v3-turbo', local_files_only=True)

    def test_partial_cache_completes_download(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / 'config.json').write_text('{}')
            download = Mock(return_value=folder)
            model_path('large-v3-turbo', download)
            self.assertEqual(download.call_count, 2)
            download.assert_called_with('large-v3-turbo')

    def test_first_use_downloads_model(self):
        from huggingface_hub.errors import LocalEntryNotFoundError
        download = Mock(side_effect=[LocalEntryNotFoundError('missing'), '/downloaded'])
        self.assertEqual(model_path('large-v3-turbo', download), '/downloaded')


class CloudTranscriberTests(unittest.TestCase):
    """Nuvem no ditado: sem pesos na VRAM, Turbo local so quando a nuvem cai ou na reuniao."""

    def setUp(self):
        with patch.object(threading.Thread, 'start'), patch.object(app.keyboard, 'Controller'):
            self.t = app.Transcriber(queue.Queue(), queue.Queue())
        self.cloud = Mock()
        self.cloud.last_provider = ELEVENLABS
        self.t.model = self.cloud
        self.t.model_config = ModelConfig(ELEVENLABS, 'cloud', 'api')
        self.audio = np.ones(app.SAMPLE_RATE, dtype=np.float32) * .05
        self.segment = SimpleNamespace(text=' oi', start=0., end=1., no_speech_prob=0.)
        for target in ('_perf', 'drop_hallucinations'):
            patcher = patch.object(app, target, side_effect=(lambda segments, audio: segments)
                                   if target == 'drop_hallucinations' else None)
            patcher.start()
            self.addCleanup(patcher.stop)

    def local(self):
        model = Mock()
        model.transcribe.return_value = (iter([self.segment]), None)
        return model

    def test_cloud_answer_never_loads_local_weights(self):
        self.cloud.transcribe.return_value = ([self.segment], None)
        with patch.object(app, 'WhisperModel') as whisper:
            segments, _ = self.t._transcribe_locked(self.audio, language='pt', vad_filter=False)
        self.assertEqual(segments, [self.segment])
        whisper.assert_not_called()
        self.assertIsNone(self.t._local)

    def test_cloud_failure_falls_back_to_turbo_and_skips_cloud_for_a_while(self):
        self.cloud.transcribe.side_effect = CloudError('sem rede')
        local = self.local()
        with patch.object(app, 'WhisperModel', return_value=local) as whisper, \
             patch.object(app, 'model_path', return_value='/cache'), \
             patch.object(app, 'fallback_config', return_value=ModelConfig('large-v3-turbo', 'cuda', 'int8_float16')):
            segments, _ = self.t._transcribe_locked(self.audio, language='pt', vad_filter=True)
            self.assertEqual(segments, [self.segment])
            local.transcribe.return_value = (iter([self.segment]), None)
            self.t._transcribe_locked(self.audio, language='pt', vad_filter=True)
        self.assertEqual(self.cloud.transcribe.call_count, 1)  # segundo trecho nem tenta a nuvem
        whisper.assert_called_once()                            # Turbo carregado uma vez so
        self.assertEqual(local.transcribe.call_count, 2)
        self.assertIn('Turbo local', self.t.status_queue.get_nowait())
        self.assertEqual(self.t.last_engine, 'Turbo · GPU NVIDIA (nuvem caiu)')

    def test_cloud_answer_names_the_provider_that_actually_answered(self):
        self.cloud.transcribe.return_value = ([self.segment], None)
        self.t._transcribe_locked(self.audio, language='pt', vad_filter=False)
        self.assertEqual(self.t.last_engine, 'ElevenLabs Scribe · nuvem')
        self.t.model_config = ModelConfig(XAI, 'cloud', 'api')  # Grok falhou, ElevenLabs respondeu
        self.t._transcribe_locked(self.audio, language='pt', vad_filter=False)
        self.assertEqual(self.t.last_engine, 'ElevenLabs Scribe · nuvem (reserva: Grok (xAI) falhou)')

    def test_cloud_is_retried_after_cooldown_and_announces_recovery(self):
        self.t._cloud_down_until = time.monotonic() - 1
        self.cloud.transcribe.return_value = ([self.segment], None)
        self.t._transcribe_locked(self.audio, language='pt', vad_filter=False)
        self.assertEqual(self.t._cloud_down_until, 0.0)
        self.assertIn('voltou', self.t.status_queue.get_nowait())

    def test_idle_local_fallback_is_released(self):
        self.t._local = Mock()
        self.t._local_config = ModelConfig('large-v3-turbo', 'cuda', 'int8_float16')
        self.t._local_used = time.monotonic() - app.LOCAL_IDLE_S - 1
        with patch.object(app.time, 'sleep', side_effect=[None, StopIteration]):
            with self.assertRaises(StopIteration):
                self.t._local_reaper()
        self.assertIsNone(self.t._local)

    def test_meeting_keeps_local_fallback_alive(self):
        self.t._local = Mock()
        self.t._local_used = time.monotonic() - app.LOCAL_IDLE_S - 1
        self.t.begin_meeting_job()
        with patch.object(app.time, 'sleep', side_effect=[None, StopIteration]):
            with self.assertRaises(StopIteration):
                self.t._local_reaper()
        self.assertIsNotNone(self.t._local)

    def test_meeting_uses_local_turbo_when_dictation_is_cloud(self):
        local = self.local()
        config = ModelConfig('large-v3-turbo', 'cpu', 'int8')
        with patch.object(app, 'WhisperModel', return_value=local), \
             patch.object(app, 'model_path', return_value='/cache'), \
             patch.object(app, 'fallback_config', return_value=config):
            self.assertEqual(self.t.meeting_model(), (local, config))
        self.cloud.transcribe.assert_not_called()

    def test_meeting_uses_dictation_model_when_local(self):
        self.t.model_config = ModelConfig('large-v3', 'cuda', 'int8_float16')
        self.assertEqual(self.t.meeting_model(), (self.cloud, self.t.model_config))
        self.assertIsNone(self.t._local)

    def test_cloud_meeting_engine_gets_its_own_model_without_the_lock(self):
        own = Mock()
        self.t.model_config = ModelConfig('large-v3', 'cuda', 'int8_float16')
        with patch.object(app, 'CloudModel', return_value=own) as made, self.t._model_lock:
            model, config = self.t.meeting_model(XAI)  # lock tomado pelo ditado: nao espera
        made.assert_called_once_with(XAI)
        own.check.assert_called_once()
        self.assertIs(model, own)
        self.assertEqual(config, ModelConfig(XAI, 'cloud', 'api'))
        self.assertIsNone(self.t._local)  # nada de Turbo emprestado

    def test_cloud_meeting_does_not_hold_model_swaps(self):
        self.t.model = None  # modelo do ditado ainda carregando: reuniao na nuvem segue
        self.t.begin_meeting_job(cloud=True)
        self.assertEqual(self.t._meeting_jobs, 0)
        self.t.end_meeting_job(cloud=True)
        self.assertEqual(self.t._meeting_jobs, 0)
        with self.assertRaisesRegex(RuntimeError, 'carregando'):
            self.t.begin_meeting_job()

    def test_meeting_engine_setting_is_normalized(self):
        from sussurro_models import normalize_meeting_engine
        self.assertEqual([normalize_meeting_engine(v) for v in (XAI, ELEVENLABS, 'local', 'large-v3', None)],
                         [XAI, ELEVENLABS, 'local', 'local', 'local'])

    def test_switch_to_cloud_without_credentials_keeps_current_model(self):
        current, config = Mock(), ModelConfig('large-v3', 'cuda', 'int8_float16')
        self.t.model, self.t.model_config = current, config
        with patch.object(app, 'require_credentials', side_effect=CloudError('sem chave')):
            with self.assertRaisesRegex(CloudError, 'sem chave'):
                self.t.load_model({'whisper_model': XAI})
        self.assertIs(self.t.model, current)
        self.assertEqual(self.t.model_config, config)

    def test_switch_to_cloud_frees_local_weights(self):
        whisper = Mock()
        self.t.model, self.t.model_config = whisper, ModelConfig('large-v3', 'cuda', 'int8_float16')
        self.t._local = Mock()
        with patch.object(app, 'require_credentials'), \
             patch.object(app, 'CloudModel') as cloud:
            self.t.load_model({'whisper_model': XAI, 'whisper_device': 'cpu'})
        self.assertIs(self.t.model, cloud.return_value)
        self.assertEqual(self.t.model_config.engine, 'cloud')
        self.assertIsNone(self.t._local)
        self.assertEqual(self.t._fallback_device, 'cpu')
