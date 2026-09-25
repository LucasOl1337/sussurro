"""One immutable clip, isolated model processes, bounded parallelism and cancellation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed

SAMPLE_RATE = 16000
MAX_SECONDS = 30


def comparison_temp_root() -> Path:
    """Use the app cache instead of the quota-limited shared system temp."""
    from sussurro_models import cache_dir
    return cache_dir('compare')


class ClipRecorder:
    """Microphone-only recorder. Never injects text or touches normal dictation history."""
    def __init__(self, device, resampler_factory, audio_backend=None):
        if audio_backend is None:
            import sounddevice as audio_backend
        self.backend = audio_backend
        self.device = device
        self.resampler_factory = resampler_factory
        self.stream = None
        self.parts = []
        self.samples = 0
        self.error = None
        self.lock = threading.Lock()
        self.full = threading.Event()

    def start(self):
        native = int(self.backend.query_devices(self.device, 'input')['default_samplerate'])
        self.resampler = self.resampler_factory(native) if native != SAMPLE_RATE else None
        try:
            self.stream = self.backend.InputStream(device=self.device, samplerate=native,
                                                   channels=1, dtype='float32',
                                                   blocksize=max(1, native // 50), callback=self._capture)
            self.stream.start()
        except Exception:
            if self.stream is not None:
                self.stream.close()
                self.stream = None
            raise

    def _capture(self, data, frames, timing, status):
        if status:
            self.error = str(status)
        block = data[:, 0].copy()
        if self.resampler:
            block = self.resampler.process(block)
        with self.lock:
            remaining = MAX_SECONDS * SAMPLE_RATE - self.samples
            if remaining > 0:
                block = block[:remaining].copy()
                self.parts.append(block)
                self.samples += len(block)
            if self.samples >= MAX_SECONDS * SAMPLE_RATE:
                self.full.set()

    def stop(self):
        import numpy as np
        stream, self.stream = self.stream, None
        if stream is not None:
            try:
                stream.stop()
            finally:
                stream.close()
        with self.lock:
            audio = np.concatenate(self.parts) if self.parts else np.zeros(0, dtype=np.float32)
        if self.error:
            raise RuntimeError(f'Captura interrompida: {self.error}. Grave novamente.')
        return audio


class Comparison:
    """Run models in child processes so every completion releases its model memory."""
    def __init__(self, emit):
        self.emit = emit
        self.cancelled = threading.Event()
        self.processes = set()
        self.lock = threading.Lock()

    def cancel(self):
        with self.lock:
            self.cancelled.set()
            processes = list(self.processes)
        for process in processes:
            try:
                process.terminate()
            except OSError:
                pass

    def run(self, audio, models, device, language, parallel=True):
        import numpy as np
        from sussurro_models import MODEL_LABELS
        problem = None
        try:
            models = list(dict.fromkeys(models))
            if not models or any(m not in MODEL_LABELS or m == 'auto' for m in models):
                raise ValueError('Escolha pelo menos um modelo valido.')
            if device not in ('cpu', 'cuda') or language not in ('pt', 'en', 'auto'):
                raise ValueError('Dispositivo ou idioma invalido.')
            clip = np.asarray(audio, dtype=np.float32).reshape(-1).copy()
            if not 0 < len(clip) <= SAMPLE_RATE * MAX_SECONDS or not np.isfinite(clip).all():
                raise ValueError('Grave uma frase de ate 30 segundos.')
            started = time.perf_counter()
            with tempfile.TemporaryDirectory(prefix='run-', dir=comparison_temp_root()) as folder:
                path = Path(folder) / 'phrase.wav'
                with wave.open(str(path), 'wb') as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(SAMPLE_RATE)
                    wav.writeframes((np.clip(clip, -1, 1) * 32767).astype('<i2').tobytes())
                with ThreadPoolExecutor(max_workers=2 if parallel else 1) as pool:
                    futures = [pool.submit(self._run_model, model, device, language, path, started)
                               for model in models]
                    for future in as_completed(futures):
                        future.result()
        except Exception as error:
            problem = str(error)
        finally:
            self.emit({'event': 'finished', 'cancelled': self.cancelled.is_set(), 'error': problem})

    def _run_model(self, model, device, language, path, started):
        process = None
        try:
            with self.lock:
                if self.cancelled.is_set():
                    self.emit({'event': 'cancelled', 'model': model})
                    return
                self.emit({'event': 'stage', 'model': model, 'stage': 'Preparando...'})
                python = Path(sys.executable)
                if python.name.lower() == 'pythonw.exe':
                    python = python.with_name('python.exe')
                command = [str(python), str(Path(__file__).resolve()), '--worker',
                           model, device, language, str(path)]
                flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                           creationflags=flags)
                self.processes.add(process)
            reported = 0
            cancel_at = None
            final = None
            while True:
                try:
                    out, err = process.communicate(timeout=.2)
                    complete = True
                except subprocess.TimeoutExpired as timeout:
                    out, err = timeout.output or b'', timeout.stderr or b''
                    complete = False
                lines = out.splitlines(keepends=True)
                for line in lines[reported:]:
                    if not line.endswith(b'\n'):
                        break
                    reported += 1
                    try:
                        event = json.loads(line)
                    except (ValueError, UnicodeDecodeError):
                        continue
                    event['model'] = model
                    if event.get('event') == 'result':
                        final = event
                    elif event.get('event') == 'stage':
                        self.emit(event)
                if self.cancelled.is_set():
                    cancel_at = cancel_at or time.perf_counter()
                    if process.poll() is None:
                        if time.perf_counter() - cancel_at > 1:
                            process.kill()
                        else:
                            process.terminate()
                if complete:
                    break
            if self.cancelled.is_set():
                self.emit({'event': 'cancelled', 'model': model})
            elif final is not None:
                final['elapsed_s'] = time.perf_counter() - started
                self.emit(final)
            else:
                detail = err.decode('utf-8', 'replace').strip()[-500:]
                self.emit({'event': 'result', 'model': model, 'error': detail or
                           f'O processo terminou sem resultado (codigo {process.returncode}).'})
        except Exception as error:
            self.emit({'event': 'result', 'model': model, 'error': str(error)})
        finally:
            if process is not None:
                if process.poll() is None:
                    process.kill()
                process.communicate()
                with self.lock:
                    self.processes.discard(process)


def worker(model_name, device, language, path):
    """All timings come from real generator consumption; every model receives the same WAV."""
    def emit(**data):
        print(json.dumps(data), flush=True)
    try:
        prep_start = time.perf_counter()
        from sussurro_cuda import _prepare_cuda_libs
        if device == 'cuda':
            _prepare_cuda_libs()
        import numpy as np
        from faster_whisper import WhisperModel
        from faster_whisper.utils import download_model
        from sussurro_models import resolve_model_config, model_path
        config = resolve_model_config({'whisper_model': model_name, 'whisper_device': device})
        emit(event='stage', stage='Baixando ou lendo cache...')
        if config.engine == 'parakeet':
            import sussurro_parakeet
            weights = sussurro_parakeet.model_dir()
        else:
            weights = model_path(model_name, download_model)
        with wave.open(path) as wav:
            assert wav.getframerate() == SAMPLE_RATE and wav.getnchannels() == 1 and wav.getsampwidth() == 2
            audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype(np.float32) / 32768
        prep_s = time.perf_counter() - prep_start
        emit(event='stage', stage='Carregando e aquecendo...')
        start = time.perf_counter()
        if config.engine == 'parakeet':
            model = sussurro_parakeet.ParakeetModel(weights)
        else:
            model = WhisperModel(weights, device=config.device, compute_type=config.compute_type)
        segments, _ = model.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32),
                                      language='pt', beam_size=5, vad_filter=False)
        list(segments)
        from faster_whisper.vad import get_speech_timestamps, VadOptions
        get_speech_timestamps(np.zeros(SAMPLE_RATE, dtype=np.float32), VadOptions())
        load_s = time.perf_counter() - start
        emit(event='stage', stage='Transcrevendo...')
        start = time.perf_counter()
        segments, _ = model.transcribe(audio, language=None if language == 'auto' else language,
                                      beam_size=5, vad_filter=True)
        text = ' '.join(s.text.strip() for s in segments).strip()
        infer_s = time.perf_counter() - start
        emit(event='result', text=text, infer_s=infer_s, load_s=load_s, prep_s=prep_s,
             audio_s=len(audio) / SAMPLE_RATE, device=config.device, compute_type=config.compute_type)
    except Exception as error:
        emit(event='result', error=f'{type(error).__name__}: {error}')


if __name__ == '__main__':
    if len(sys.argv) == 6 and sys.argv[1] == '--worker':
        worker(*sys.argv[2:])
    else:
        raise SystemExit('Uso interno: sussurro_compare.py --worker modelo dispositivo idioma arquivo.wav')
