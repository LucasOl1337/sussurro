"""Diagnostico de resultado vazio com niveis reais e historico temporario."""
import json
import queue
import sys
import tempfile
import threading
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app


class AudioDiagnosisTests(unittest.TestCase):
    def setUp(self):
        with patch.object(threading.Thread, 'start'), patch.object(app.keyboard, 'Controller'):
            self.t = app.Transcriber(queue.Queue(), queue.Queue())
        self.t.model = Mock()
        self.t.library.apply = lambda text: (text, 0)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.folder = Path(directory.name)
        for name, value in [('HISTORY_DIR', self.folder),
                            ('HISTORY_INDEX', self.folder / 'history.jsonl'), ('_perf', Mock())]:
            patcher = patch.object(app, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def finalize(self, audio, text='', errors=()):
        self.t._session_audio = [audio]
        self.t._session_parts = [(audio, text, 0, False)]
        self.t._session_errors = list(errors)
        self.t._finalize_session()
        return self.t.history_queue.get_nowait()

    def test_silent_session_reports_no_microphone_signal_with_levels(self):
        entry = self.finalize(np.zeros(app.SAMPLE_RATE, dtype=np.float32))
        self.assertEqual(entry['error'], 'Microfone não enviou sinal (confira a entrada).')
        self.assertEqual(entry['audio_level'], {'peak': 0.0, 'rms': 0.0})
        self.assertEqual(self.t.status_queue.get_nowait(), f"ERRO: {entry['error']}")

    def test_low_audio_reports_level_instead_of_generic_recognition_error(self):
        audio = np.ones(app.SAMPLE_RATE, dtype=np.float32) * .001
        entry = self.finalize(audio)
        self.assertEqual(entry['error'], 'Áudio muito baixo.')
        self.assertAlmostEqual(entry['audio_level']['peak'], .001, places=6)
        self.assertAlmostEqual(entry['audio_level']['rms'], .001, places=6)

    def test_normal_signal_with_empty_model_result_suggests_another_model(self):
        audio = (.1 * np.sin(np.arange(app.SAMPLE_RATE) * .1)).astype(np.float32)
        self.t._enqueue_segment(audio)
        self.t._segment_queue.put((None, None, None, self.t._session_id, 0.))
        events = iter(list(self.t._segment_queue.queue))
        self.t._segment_queue.get = lambda: next(events)
        with patch.object(self.t, '_transcribe_locked', return_value=([], None)):
            with self.assertRaises(StopIteration):
                self.t._transcribe_loop()
        deliveries = iter(list(self.t._delivery_queue.queue))
        self.t._delivery_queue.get = lambda: next(deliveries)
        with self.assertRaises(StopIteration):
            self.t._delivery_loop()
        entry = self.t.history_queue.get_nowait()
        self.assertEqual(entry['error'], 'Fala não reconhecida. Tente Refazer com outro modelo.')
        self.assertGreater(entry['audio_level']['rms'], .05)
        self.assertTrue(any(status.startswith('ERRO:') for status in self.t.status_queue.queue))

    def test_empty_file_result_is_archived_with_diagnosis(self):
        source = self.folder / 'input.wav'
        source.write_bytes(b'mocked decoder input')
        audio = np.zeros(app.SAMPLE_RATE, dtype=np.float32)
        with patch.object(app, 'load_audio_16k_mono', return_value=audio), \
             patch.object(self.t, '_transcribe_locked', return_value=([], None)):
            with self.assertRaisesRegex(ValueError, 'Microfone não enviou sinal'):
                self.t.transcribe_file(str(source))
        entry = self.t.history_queue.get_nowait()
        self.assertTrue(entry['failed'])
        self.assertEqual(entry['audio_level'], {'peak': 0., 'rms': 0.})
        self.assertTrue((self.folder / entry['wav']).is_file())
        self.assertEqual(self.t._file_jobs, 0)

    def test_normal_session_retains_old_entry_shape_and_status(self):
        audio = np.ones(app.SAMPLE_RATE, dtype=np.float32) * .1
        entry = self.finalize(audio, 'texto normal')
        self.assertEqual(set(entry), {'ts', 'wav', 'text', 'dur', 'fix'})
        self.assertEqual(entry['text'], 'texto normal')
        self.assertTrue(self.t.status_queue.empty())
        saved = json.loads(app.HISTORY_INDEX.read_text())
        self.assertEqual(saved, entry)

    def test_technical_error_is_not_replaced_by_signal_diagnosis(self):
        entry = self.finalize(np.zeros(app.SAMPLE_RATE, dtype=np.float32), errors=['CUDA indisponivel'])
        self.assertEqual(entry['error'], 'CUDA indisponivel')
        self.assertEqual(entry['audio_level'], {'peak': 0., 'rms': 0.})
        self.assertEqual(self.t.status_queue.get_nowait(), 'ERRO: CUDA indisponivel')

    def test_new_levels_and_old_entries_do_not_change_statistics(self):
        normal = {'ts': '2026-10-02T10:00:00', 'wav': 'old.wav',
                  'text': 'texto antigo normal', 'dur': 12., 'fix': 2}
        failed = self.finalize(np.zeros(app.SAMPLE_RATE, dtype=np.float32))
        self.assertEqual(app.compute_stats([normal, failed]), app.compute_stats([normal]))
        self.assertEqual(app.compute_stats([{**normal, 'audio_level': {'peak': .1, 'rms': .05}}]),
                         app.compute_stats([normal]))

    def test_public_file_decode_real_pcm_and_save_diagnostic_levels(self):
        source = self.folder / 'quiet.wav'
        pcm = np.full(app.SAMPLE_RATE, 32, dtype=np.int16)
        with wave.open(str(source), 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(app.SAMPLE_RATE)
            wav.writeframes(pcm.tobytes())
        with patch.object(self.t, '_transcribe_locked', return_value=([], None)):
            with self.assertRaisesRegex(ValueError, 'Áudio muito baixo'):
                self.t.transcribe_file(str(source))
        entry = self.t.history_queue.get_nowait()
        self.assertAlmostEqual(entry['audio_level']['rms'], 32 / 32768, places=7)
        self.assertAlmostEqual(entry['audio_level']['peak'], 32 / 32768, places=7)
        self.assertEqual(entry['error'], 'Áudio muito baixo.')
        self.assertEqual(json.loads(app.HISTORY_INDEX.read_text()), entry)
        self.assertTrue((self.folder / entry['wav']).is_file())

    def test_retry_empty_failed_legacy_entry_gains_levels_without_altering_wav(self):
        audio = np.zeros(app.SAMPLE_RATE, dtype=np.float32)
        entry = self.t._archive_audio(audio, '', 0, app.datetime(2026, 10, 2, 10, 1),
                                      failed=True, error='Nenhuma fala reconhecida.')
        self.t.history_queue.get_nowait()
        entry.pop('audio_level')  # formato antigo continua aceito
        self.t._replace_history_entry(entry)
        path = self.folder / entry['wav']
        before = path.read_bytes()
        with patch.object(self.t, '_transcribe_locked', return_value=([], None)), \
             patch.object(app.traceback, 'print_exc'):
            self.t._retry_history_worker(entry, path)
        updated = self.t.history_queue.get_nowait()
        self.assertEqual(updated['audio_level'], {'peak': 0., 'rms': 0.})
        self.assertIn('Microfone não enviou sinal', updated['error'])
        self.assertEqual(path.read_bytes(), before)

    def test_pcm_quantization_and_low_rms_boundaries(self):
        for amplitude, expected in [(1 / 32768, 'Microfone'),
                                    (2 / 32768, 'Áudio muito baixo'),
                                    (.003, 'Áudio muito baixo'), (.0033, 'Fala não reconhecida')]:
            with self.subTest(amplitude=amplitude):
                levels, diagnosis = self.t._audio_diagnosis(
                    np.full(app.SAMPLE_RATE, amplitude, dtype=np.float32))
                self.assertTrue(diagnosis.startswith(expected))
                self.assertAlmostEqual(levels['rms'], amplitude, places=7)

    def test_normal_file_still_returns_success_entry(self):
        source = self.folder / 'normal.wav'
        source.write_bytes(b'mocked input')
        audio = np.ones(app.SAMPLE_RATE, dtype=np.float32) * .1
        segment = SimpleNamespace(text='texto normal', start=0., end=1.)
        with patch.object(app, 'load_audio_16k_mono', return_value=audio), \
             patch.object(self.t, '_transcribe_locked', return_value=([segment], None)):
            entry = self.t.transcribe_file(str(source))
        self.assertEqual(set(entry), {'ts', 'wav', 'text', 'dur', 'fix'})
        self.assertEqual(entry['text'], 'texto normal')
        self.assertEqual(self.t._file_jobs, 0)
        self.assertFalse(any(status.startswith('ERRO') for status in self.t.status_queue.queue))


if __name__ == '__main__':
    unittest.main()
