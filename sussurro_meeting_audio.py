"""Gravacao da reuniao em disco: duas trilhas via parec, pasta da reuniao e reproducao.

Portado de jankeesvw/omarchy-meeting-recorder (MIT, (c) 2026 Jankees van Woezik).
O microfone (@DEFAULT_SOURCE@) e o monitor da saida padrao (@DEFAULT_MONITOR@) seguem o
sistema: trocar pro fone no meio da call continua funcionando. Enquanto grava, as duas
trilhas vao pro cache em s16le; se o app cair, a proxima abertura acha a gravacao e
oferece salvar. Ao parar, o ffmpeg nivela cada lado e grava Opus na pasta da reuniao.
"""
import collections
import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from sussurro_models import cache_dir

RATE = 48000
CHUNK = RATE // 50 * 2  # 20 ms de s16le mono
HISTORY = 150           # 3 s de picos de 20 ms para o medidor
MIC, PC = "@DEFAULT_SOURCE@", "@DEFAULT_MONITOR@"
MANIFEST = "reuniao.json"
TRANSCRIPT = "transcricao.md"
TRACKS = ".trilhas"


class Source:
    """Um parec por fonte: medidor sempre vivo; grava no arquivo so quando pedido."""

    def __init__(self, device):
        self.device = device
        self._levels = collections.deque([0.0] * HISTORY, maxlen=HISTORY)
        self._file = None
        self._paused = False
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._proc = None
        threading.Thread(target=self._run, name=f"sussurro-parec-{device}", daemon=True).start()

    def _run(self):
        while not self._closed.is_set():
            try:
                self._proc = subprocess.Popen(
                    ["parec", "--raw", "--format=s16le", f"--rate={RATE}", "--channels=1",
                     "--latency-msec=20", "--client-name=Sussurro", "-d", self.device],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            except OSError:
                return  # sem parec (pulseaudio-utils/libpulse): o painel avisa
            chunks = 0
            while not self._closed.is_set():
                data = self._proc.stdout.read(CHUNK)
                if len(data) < CHUNK:
                    break  # dispositivo sumiu: tenta de novo
                chunks += 1
                peak = float(np.max(np.abs(np.frombuffer(data, "<i2")))) / 32768.0
                with self._lock:
                    self._levels.append(peak)
                    if self._file is not None and not self._paused:
                        self._file.write(data)
                        # Queda perde no maximo 1 s; a cada 30 s forca o disco tambem.
                        if chunks % 50 == 0:
                            self._file.flush()
                        if chunks % 1500 == 0:
                            os.fsync(self._file.fileno())
            self._proc.kill()
            self._proc.wait()
            if not self._closed.is_set():
                time.sleep(1)

    def levels(self):
        with self._lock:
            return list(self._levels)

    def start_recording(self, path):
        with self._lock:
            self._file = open(path, "ab")
            self._paused = False

    def set_paused(self, paused):
        with self._lock:
            self._paused = paused

    def stop_recording(self):
        with self._lock:
            if self._file is not None:
                self._file.flush()
                os.fsync(self._file.fileno())
                self._file.close()
                self._file = None

    def close(self):
        self.stop_recording()
        self._closed.set()
        if self._proc is not None and self._proc.poll() is None:
            self._proc.kill()


def to_meter(peak):
    """Pico linear -> 0..1 numa escala de -60 a 0 dB."""
    if peak <= 0:
        return 0.0
    return float(np.clip(1.0 + 20.0 * np.log10(peak) / 60.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Gravacao em andamento (sobrevive a queda do app)

def pending_dir():
    return cache_dir("reuniao", "gravando")


def raw_paths():
    folder = pending_dir()
    return folder / "mic.raw", folder / "pc.raw", folder / "estado.json"


def pending_recording():
    """Estado de uma gravacao que nao terminou direito (queda, logout), ou None."""
    mic, pc, state = raw_paths()
    if not state.is_file() or not (mic.is_file() or pc.is_file()):
        return None
    try:
        info = json.loads(state.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        info = {}
    size = max(mic.stat().st_size if mic.is_file() else 0, pc.stat().st_size if pc.is_file() else 0)
    info["duration_secs"] = size // (RATE * 2)
    return info


def discard_pending():
    shutil.rmtree(pending_dir(), ignore_errors=True)


# ---------------------------------------------------------------------------
# Pasta da reuniao

def meetings_root():
    try:
        docs = subprocess.run(["xdg-user-dir", "DOCUMENTS"], capture_output=True, text=True,
                              timeout=2).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        docs = ""
    root = Path(docs or Path.home() / "Documents") / "Reuniões"
    root.mkdir(parents=True, exist_ok=True)
    return root


def safe_name(title):
    name = re.sub(r"(\d):(\d)", r"\1h\2", title)  # "Reunião 10:20" -> "Reunião 10h20"
    name = re.sub(r"[/\\:\x00-\x1f]", " ", name).strip(" .")
    return re.sub(r"\s+", " ", name)[:80] or "Reunião"


def new_meeting_dir(title, started):
    base = meetings_root() / f"{started:%Y%m%d%H%M} {safe_name(title)}"
    folder, n = base, 2
    while folder.exists():
        folder = base.with_name(f"{base.name} ({n})")
        n += 1
    folder.mkdir(parents=True)
    return folder


def rename_meeting(folder, title):
    """Troca o nome da pasta mantendo o prefixo de data (ordenacao por data)."""
    folder = Path(folder)
    prefix = folder.name.split(" ", 1)[0]
    target = folder.with_name(f"{prefix} {safe_name(title)}")
    if target != folder and not target.exists():
        folder.rename(target)
        return target
    return folder


def list_meetings():
    """Pastas com manifesto, mais recentes primeiro: (pasta, manifesto)."""
    found = []
    for folder in meetings_root().iterdir():
        manifest = folder / MANIFEST
        if manifest.is_file():
            try:
                found.append((folder, json.loads(manifest.read_text(encoding="utf-8"))))
            except (OSError, ValueError):
                continue
    return sorted(found, key=lambda item: item[0].name, reverse=True)


def read_manifest(folder):
    return json.loads((Path(folder) / MANIFEST).read_text(encoding="utf-8"))


def write_manifest(folder, manifest):
    path = Path(folder) / MANIFEST
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


# ---------------------------------------------------------------------------
# Exportacao

TARGET_LEVEL = 0.125   # percentil 95 de quadros de 50 ms, uns -18 dBFS
SILENT_LEVEL = 0.0018  # abaixo disso (~ -55 dBFS) e silencio ou chiado: nao sobe o ruido


def speech_gain_db(raw):
    """Ganho fixo em dB que leva a fala da trilha ao TARGET_LEVEL (o limitador pega os picos)."""
    data = np.fromfile(raw, dtype="<i2") if Path(raw).is_file() else np.zeros(0, "<i2")
    frame = RATE // 20
    frames = data.size // frame
    if frames == 0:
        return 0.0
    blocks = data[:frames * frame].astype(np.float32).reshape(frames, frame) / 32768.0
    p95 = float(np.sort(np.sqrt((blocks * blocks).mean(axis=1)))[(frames - 1) * 95 // 100])
    if p95 < SILENT_LEVEL:
        return 0.0
    return float(np.clip(20 * np.log10(TARGET_LEVEL / p95), -12.0, 24.0))


def _pad_same_length(paths):
    sizes = [p.stat().st_size if p.is_file() else 0 for p in paths]
    longest = max(sizes)
    for path, size in zip(paths, sizes):
        if size < longest:
            with open(path, "ab") as f:
                f.write(b"\0" * (longest - size))


def export(mic_raw, pc_raw, folder):
    """audio.ogg (os dois lados nivelados e misturados) e .trilhas/{mic,pc}.ogg separados,
    que e o que 'transcrever de novo' usa pra manter voce e os outros separados."""
    mic_raw, pc_raw, folder = Path(mic_raw), Path(pc_raw), Path(folder)
    _pad_same_length([mic_raw, pc_raw])
    raw = ["-f", "s16le", "-ar", str(RATE), "-ac", "1", "-i"]
    mic_gain, pc_gain = speech_gain_db(mic_raw), speech_gain_db(pc_raw)
    tracks = folder / TRACKS
    tracks.mkdir(exist_ok=True)
    jobs = [
        [*raw, str(mic_raw), *raw, str(pc_raw), "-filter_complex",
         f"[0]volume={mic_gain:.1f}dB[a];[1]volume={pc_gain:.1f}dB[b];"
         "[a][b]amix=inputs=2:normalize=0,alimiter=limit=0.9:level=disabled",
         "-c:a", "libopus", "-b:a", "64k", str(folder / "audio.ogg")],
        [*raw, str(mic_raw), "-c:a", "libopus", "-b:a", "48k", str(tracks / "mic.ogg")],
        [*raw, str(pc_raw), "-c:a", "libopus", "-b:a", "48k", str(tracks / "pc.ogg")],
    ]
    for args in jobs:
        result = subprocess.run(["ffmpeg", "-y", "-nostdin", "-loglevel", "error", *args],
                                capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg falhou ao salvar a reuniao: {result.stderr.strip()[-300:]}")


def import_audio(source, folder):
    """Copia um arquivo qualquer que o ffmpeg le para audio.ogg da reuniao."""
    result = subprocess.run(["ffmpeg", "-y", "-nostdin", "-loglevel", "error", "-i", str(source),
                             "-ac", "1", "-c:a", "libopus", "-b:a", "64k",
                             str(Path(folder) / "audio.ogg")], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg nao leu {Path(source).name}: {result.stderr.strip()[-300:]}")


# ---------------------------------------------------------------------------
# Reproducao: ffmpeg decodifica direto no pacat (nada alem do que ja grava)

class Player:
    def __init__(self):
        self._procs = []
        self.started_at = None
        self.offset_ms = 0

    def play(self, path, from_ms=0):
        self.stop()
        decoder = subprocess.Popen(["ffmpeg", "-nostdin", "-loglevel", "error", "-ss",
                                    f"{from_ms / 1000:.3f}", "-i", str(path), "-f", "s16le",
                                    "-ac", "1", "-ar", str(RATE), "-"],
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        sink = subprocess.Popen(["pacat", "--raw", "--format=s16le", f"--rate={RATE}",
                                 "--channels=1", "--client-name=Sussurro"],
                                stdin=decoder.stdout, stderr=subprocess.DEVNULL)
        decoder.stdout.close()
        self._procs = [decoder, sink]
        self.started_at, self.offset_ms = time.monotonic(), from_ms

    def position_ms(self):
        if not self.playing():
            return None
        return self.offset_ms + int((time.monotonic() - self.started_at) * 1000)

    def playing(self):
        return bool(self._procs) and self._procs[-1].poll() is None

    def stop(self):
        for proc in self._procs:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
        self._procs = []

