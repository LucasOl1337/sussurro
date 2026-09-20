"""Same-clip scheduling, subprocess cleanup, capture limits and dictation isolation."""
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np
import app
from sussurro_compare import Comparison, ClipRecorder, MAX_SECONDS, SAMPLE_RATE, worker


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.script = Path(self.temp.name) / 'worker.py'
        self.script.write_text('''import hashlib,json,pathlib,sys,time
model,device,language,path=sys.argv[1:]
print(json.dumps({'event':'stage','stage':'Transcrevendo...'}),flush=True)
time.sleep(10 if language=='en' else .35)
if model=='medium':
    print('test failure',file=sys.stderr)
    raise SystemExit(3)
print(json.dumps({'event':'result','text':hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest(),
                 'infer_s':.1,'load_s':.2,'prep_s':.05,'device':device,'compute_type':'int8','audio_s':1}),flush=True)
''')
        self.events = []
        self.peak = 0
        self.paths = []
        self.runner = Comparison(self.emit)
        self.real_popen = subprocess.Popen

    def emit(self, event):
        self.peak = max(self.peak, len(self.runner.processes))
        self.events.append(event)

    def spawn(self, command, **kwargs):
        self.paths.append(command[-1])
        return self.real_popen([sys.executable, str(self.script), *command[3:]], **kwargs)

    def run_models(self, models, parallel=True, language='pt'):
        with patch('sussurro_compare.subprocess.Popen', side_effect=self.spawn):
            self.runner.run(np.ones(SAMPLE_RATE, dtype=np.float32) * .1, models, 'cpu', language, parallel)

    def test_parallel_models_receive_identical_wav_and_have_separate_timings(self):
        self.run_models(['tiny', 'base', 'small'])
        results = [e for e in self.events if e['event'] == 'result']
        self.assertEqual({e['model'] for e in results}, {'tiny','base','small'})
        self.assertEqual(len({e['text'] for e in results}), 1)
        self.assertEqual(self.peak, 2)
        self.assertTrue(all(e['elapsed_s'] >= e['infer_s'] for e in results))
        self.assertEqual(len(set(self.paths)), 1)
        self.assertTrue(all(not Path(p).exists() for p in self.paths))
        self.assertFalse(self.runner.processes)

    def test_individual_mode_runs_one_process_at_a_time(self):
        self.run_models(['tiny','base'], parallel=False)
        self.assertEqual(self.peak, 1)
        self.assertEqual(len([e for e in self.events if e['event']=='result']), 2)

    def test_one_model_failure_does_not_stop_other_models(self):
        self.run_models(['medium','tiny'])
        results = {e['model']:e for e in self.events if e['event']=='result'}
        self.assertIn('test failure', results['medium']['error'])
        self.assertTrue(results['tiny']['text'])
        self.assertFalse(self.runner.processes)

    def test_cancel_stops_running_processes_and_skips_queued_models(self):
        with patch('sussurro_compare.subprocess.Popen', side_effect=self.spawn):
            thread = threading.Thread(target=self.runner.run,
                                      args=(np.ones(SAMPLE_RATE), ['tiny','base','small'], 'cpu','en',True))
            thread.start()
            deadline = time.monotonic() + 5
            while len(self.runner.processes) < 2 and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertEqual(len(self.runner.processes), 2)
            self.runner.cancel()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertFalse(self.runner.processes)
        self.assertTrue(all(not Path(p).exists() for p in self.paths))
        self.assertEqual({e['model'] for e in self.events if e['event']=='cancelled'}, {'tiny','base','small'})
        self.assertTrue(self.events[-1]['cancelled'])

    def test_invalid_or_overlong_audio_never_starts_process(self):
        for audio in [np.zeros(0),np.zeros(SAMPLE_RATE*(MAX_SECONDS+1)),np.array([np.nan])]:
            with self.subTest(length=len(audio)), patch('sussurro_compare.subprocess.Popen') as spawn:
                self.runner.run(audio, ['base'], 'cpu', 'pt')
                spawn.assert_not_called()

    def test_unavailable_system_temp_does_not_block_comparison(self):
        cache = str(Path(self.temp.name) / 'cache')
        with patch.object(tempfile, 'tempdir', '/proc/sussurro-unavailable-temp'), \
             patch.dict(os.environ, {'SUSSURRO_CACHE_DIR': cache}):
            self.run_models(['tiny'])
        finished = [event for event in self.events if event['event'] == 'finished'][-1]
        self.assertIsNone(finished['error'])
        self.assertEqual(len([event for event in self.events if event['event'] == 'result']), 1)


class CaptureTests(unittest.TestCase):
    def test_capture_is_bounded_and_closes_stream(self):
        backend = Mock()
        backend.query_devices.return_value = {'default_samplerate':16000}
        recorder = ClipRecorder(None, Mock(), backend)
        recorder.start()
        samples = np.ones((SAMPLE_RATE * (MAX_SECONDS + 1),1),dtype=np.float32)
        recorder._capture(samples,len(samples),None,None)
        self.assertTrue(recorder.full.is_set())
        audio = recorder.stop()
        self.assertEqual(len(audio),SAMPLE_RATE*MAX_SECONDS)
        backend.InputStream.return_value.stop.assert_called_once()
        backend.InputStream.return_value.close.assert_called_once()

    def test_stream_failure_closes_partial_stream(self):
        backend = Mock()
        backend.query_devices.return_value = {'default_samplerate':48000}
        backend.InputStream.return_value.start.side_effect = RuntimeError('mic failure')
        recorder = ClipRecorder(None, Mock(), backend)
        with self.assertRaisesRegex(RuntimeError,'mic failure'):
            recorder.start()
        backend.InputStream.return_value.close.assert_called_once()

    def test_audio_overflow_is_reported_instead_of_compared_as_valid(self):
        backend = Mock()
        backend.query_devices.return_value = {'default_samplerate':16000}
        recorder = ClipRecorder(None, Mock(), backend)
        recorder.start()
        recorder._capture(np.zeros((100,1)),100,None,'input overflow')
        with self.assertRaisesRegex(RuntimeError,'overflow'):
            recorder.stop()


class ComparisonIsolationTests(unittest.TestCase):
    def setUp(self):
        with patch.object(threading.Thread,'start'),patch.object(app.keyboard,'Controller'):
            self.t = app.Transcriber(queue.Queue(),queue.Queue())

    def test_comparison_refuses_active_dictation(self):
        self.t.recording.set()
        with self.assertRaises(RuntimeError):
            self.t.acquire_comparison()
        self.assertFalse(self.t.comparing.is_set())

    def test_comparison_blocks_new_capture_file_and_model_changes(self):
        self.t.acquire_comparison()
        self.assertTrue(self.t.busy())
        with patch.object(self.t,'_open_stream') as mic:
            with self.assertRaises(RuntimeError):
                self.t.start(None,False)
            mic.assert_not_called()
        with self.assertRaises(RuntimeError):
            self.t.transcribe_file('any.wav')
        with patch.object(app,'download_model') as download:
            with self.assertRaises(RuntimeError):
                self.t.load_model({})
            download.assert_not_called()
        self.t.comparing.clear()
        self.assertFalse(self.t.busy())


class WorkerTimingTests(unittest.TestCase):
    def test_inference_timing_consumes_lazy_generator(self):
        import contextlib
        import io
        import wave
        from types import SimpleNamespace
        from sussurro_models import ModelConfig
        calls = []
        def transcribe(audio, **kwargs):
            calls.append(kwargs)
            is_phrase = len(calls) > 1
            def generate():
                if is_phrase:
                    time.sleep(.06)
                    yield SimpleNamespace(text=' mesma frase ')
            return generate(), None
        model = Mock()
        model.transcribe.side_effect = transcribe
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'phrase.wav'
            with wave.open(str(path),'wb') as wav:
                wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(SAMPLE_RATE)
                wav.writeframes(b'\0\0' * SAMPLE_RATE)
            with patch('faster_whisper.WhisperModel',return_value=model), \
                 patch('faster_whisper.vad.get_speech_timestamps'), \
                 patch('sussurro_models.resolve_model_config',return_value=ModelConfig('base','cpu','int8')), \
                 patch('sussurro_models.model_path',return_value=folder), contextlib.redirect_stdout(output):
                worker('base','cpu','auto',str(path))
        result=json.loads(output.getvalue().splitlines()[-1])
        self.assertEqual(result['text'],'mesma frase')
        self.assertGreaterEqual(result['infer_s'],.05)
        self.assertIsNone(calls[-1]['language'])
        self.assertEqual(calls[-1]['beam_size'],5)
