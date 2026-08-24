"""Sussurro — voice-to-text local: mic -> Silero VAD -> faster-whisper large-v3 (CUDA).

HUD Tkinter: gravar/parar, atalho global de mouse (digita onde o cursor estiver),
fonte de captura (microfone, audio do PC via loopback WASAPI ou os dois misturados),
modo de transcricao (simultaneo por trecho ou tudo ao final).
"""

import collections
import ctypes
import ctypes.wintypes as wintypes
import json
import math
import os
import queue
import re
import sys
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
import traceback
import unicodedata
import wave
import winsound
from datetime import date, datetime, timedelta
from pathlib import Path

# Sob pythonw nao existe stdout/stderr; sem streams, print/traceback matam thread calados.
if sys.stdout is None or sys.stderr is None:
    _log = open(Path(__file__).with_name("sussurro.log"), "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stdout or _log
    sys.stderr = sys.stderr or _log

# As DLLs CUDA vem dos wheels da NVIDIA e nao entram sozinhas no caminho de busca
# (sem isto: "Library cublas64_12.dll is not found" na hora de transcrever).
_nvidia_dir = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
for _sub in ("cublas", "cudnn", "cuda_nvrtc"):
    _bin = _nvidia_dir / _sub / "bin"
    if _bin.is_dir():
        os.add_dll_directory(str(_bin))
        os.environ["PATH"] = str(_bin) + os.pathsep + os.environ.get("PATH", "")

import customtkinter as ctk
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk
import sounddevice as sd
from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions, get_speech_timestamps
from pynput import keyboard, mouse

SAMPLE_RATE = 16000
BLOCK_SIZE = 1600  # 100 ms por callback do mic
LEVEL_HZ = 40      # barras de onda por segundo alimentadas pelo mixer
_CANCEL = object()  # sentinela: joga fora o que estava em voo

# Segmentacao (modo simultaneo): corta quando ha fala + >= TAIL_SILENCE_S de silencio.
TAIL_SILENCE_S = 0.7
MAX_SEGMENT_S = 25.0
MAX_IDLE_BUFFER_S = 30.0
VAD_CHECK_EVERY_S = 0.3
VAD_OPTIONS = VadOptions(min_silence_duration_ms=400, speech_pad_ms=200)

# Paleta Asiimov (modo escuro) — skill ~/.claude/skills/asiimov
BG = "#131417"
SURFACE = "#1b1c20"        # card
SURFACE_2 = "#25262b"      # campo
SURFACE_3 = "#2f3036"      # hover de campo
BORDER = "#2f3036"
BORDER_STRONG = "#454750"
INK = "#f2f3f3"
INK_2 = "#c3c6c8"
INK_3 = "#9ba0a4"
ACCENT = "#f0500a"         # nunca carrega texto claro (grafite por cima: 4,96:1)
ACCENT_HOVER = "#d64708"
ACCENT_TEXT = "#f07944"    # laranja-como-texto no escuro
GRAPHITE = "#16181a"       # tinta sobre o laranja
ROW_EVEN = "#232429"
ROW_WASH = "#30211e"       # hover de linha (wash laranja escuro)


def pick_font(candidates, fallback):
    installed = set(tkfont.families())
    for name in candidates:
        if name in installed:
            return name
    return fallback

# Biblioteca, passada fonetica: chave minima e distancia tolerada por tamanho de chave.
# Medido no library.json real — abaixo de 6 letras a troca comeca a pegar palavra
# legitima ("rock" viraria "Grok"); com estes valores "claudio"/"claudia" nao viram
# "Claude" e "nine hauter"/"ninerouter"/"9 rooter" viram "9router".
LIB_FUZZY_MIN = 6
def LIB_FUZZY_DIST(chave: str) -> int:
    return 1 if len(chave) < 8 else (2 if len(chave) < 12 else 3)

SETTINGS_PATH = Path(__file__).with_name("settings.json")
LIBRARY_PATH = Path(__file__).with_name("library.json")
HISTORY_DIR = Path(__file__).with_name("history")
HISTORY_INDEX = HISTORY_DIR / "history.jsonl"
HIST_RENDER_MAX = 200  # linhas desenhadas na aba HISTORICO (a memoria guarda tudo)
DEFAULT_SETTINGS = {
    "mouse_button": "x2",       # middle | x1 | x2
    "trigger_mode": "alternar",  # alternar (clique liga/desliga) | segurar (push-to-talk)
    "device_name": None,
    "capture_mode": "microfone",  # microfone | audio_pc | os_dois
    "loopback_device_name": None,  # None = padrao do sistema
    "transcribe_mode": "simultaneo",  # simultaneo | final
    "language": "pt",
    "inject_method": "colar",   # colar (Ctrl+V, clipboard preservado) | digitar
    "dot_pos": [0.5, 0.94],     # posicao da bolinha, fracao da area util do monitor
}

# Mensagens do hook de mouse do Windows (win32_event_filter do pynput)
WM_MBUTTONDOWN, WM_MBUTTONUP = 0x0207, 0x0208
WM_XBUTTONDOWN, WM_XBUTTONUP = 0x020B, 0x020C
BUTTON_LABELS = {"middle": "botao do meio", "x1": "lateral 1 (tras)", "x2": "lateral 2 (frente)"}

# rotulos do combo FONTE <-> valores persistidos em settings.json
CAPTURE_LABELS = {"microfone": "microfone", "audio_pc": "audio do PC", "os_dois": "os dois"}
CAPTURE_VALUES = {v: k for k, v in CAPTURE_LABELS.items()}


def load_settings() -> dict:
    settings = dict(DEFAULT_SETTINGS)
    if SETTINGS_PATH.exists():
        settings.update(json.loads(SETTINGS_PATH.read_text(encoding="utf-8")))
    return settings


def save_settings(settings: dict):
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")


def list_input_devices() -> dict:
    """Nome -> indice, so entradas do host WASAPI (nomes completos, sem duplicata MME)."""
    wasapi = next(i for i, h in enumerate(sd.query_hostapis()) if "WASAPI" in h["name"])
    return {
        d["name"]: d["index"]
        for d in sd.query_devices()
        if d["max_input_channels"] > 0 and d["hostapi"] == wasapi
    }


def list_loopback_devices() -> dict:
    """Nome -> indice, saidas WASAPI (capturaveis via loopback p/ pegar o audio do PC)."""
    wasapi = next(i for i, h in enumerate(sd.query_hostapis()) if "WASAPI" in h["name"])
    return {
        d["name"]: d["index"]
        for d in sd.query_devices()
        if d["max_output_channels"] > 0 and d["hostapi"] == wasapi
    }


# -- clipboard seguro (backup/restauracao pra colar sem sujar o clipboard) ----
_u32 = ctypes.windll.user32
_k32 = ctypes.windll.kernel32
_u32.GetClipboardData.restype = wintypes.HANDLE
_u32.SetClipboardData.restype = wintypes.HANDLE
_u32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
_k32.GlobalLock.restype = wintypes.LPVOID
_k32.GlobalLock.argtypes = [wintypes.HGLOBAL]
_k32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
_k32.GlobalAlloc.restype = wintypes.HGLOBAL
_k32.GlobalSize.restype = ctypes.c_size_t
_k32.GlobalSize.argtypes = [wintypes.HGLOBAL]

CF_UNICODETEXT = 13
GMEM_MOVEABLE = 0x0002


class MONITORINFO(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
                ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD)]


_u32.MonitorFromPoint.restype = wintypes.HMONITOR
_u32.MonitorFromPoint.argtypes = [wintypes.POINT, wintypes.DWORD]


def monitor_work_area(x: int, y: int):
    """Area util (sem taskbar) do monitor que contem o ponto (x, y)."""
    hmon = _u32.MonitorFromPoint(wintypes.POINT(x, y), 2)  # MONITOR_DEFAULTTONEAREST
    mi = MONITORINFO()
    mi.cbSize = ctypes.sizeof(MONITORINFO)
    _u32.GetMonitorInfoW(hmon, ctypes.byref(mi))
    r = mi.rcWork
    return r.left, r.top, r.right, r.bottom


def cursor_pos():
    pt = wintypes.POINT()
    _u32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y
# formatos cujo dado NAO e HGLOBAL (handles GDI etc.) ou sao privados — nao da pra
# copiar byte a byte; o Windows sintetiza CF_BITMAP a partir de CF_DIB, entao
# imagens/screenshots sobrevivem mesmo pulando estes.
_SKIP_FORMATS = {2, 3, 9, 14}  # CF_BITMAP, CF_METAFILEPICT, CF_PALETTE, CF_ENHMETAFILE


def _open_clipboard(retries: int = 15) -> bool:
    for _ in range(retries):
        if _u32.OpenClipboard(None):
            return True
        time.sleep(0.02)
    return False


def backup_clipboard():
    """Copia todos os formatos HGLOBAL do clipboard. None = nao conseguiu abrir."""
    if not _open_clipboard():
        return None
    data = []
    try:
        fmt = 0
        while (fmt := _u32.EnumClipboardFormats(fmt)):
            if fmt in _SKIP_FORMATS or 0x0080 <= fmt <= 0x008F or 0x0200 <= fmt <= 0x03FF:
                continue  # owner-display / privados / GDI-obj
            handle = _u32.GetClipboardData(fmt)
            if not handle:
                continue
            size = _k32.GlobalSize(handle)
            ptr = _k32.GlobalLock(handle) if size else None
            if not ptr:
                continue
            try:
                data.append((fmt, ctypes.string_at(ptr, size)))
            finally:
                _k32.GlobalUnlock(handle)
    finally:
        _u32.CloseClipboard()
    return data


def restore_clipboard(data) -> None:
    if data is None or not _open_clipboard():
        return
    try:
        _u32.EmptyClipboard()
        for fmt, blob in data:
            handle = _k32.GlobalAlloc(GMEM_MOVEABLE, len(blob))
            ptr = _k32.GlobalLock(handle)
            ctypes.memmove(ptr, blob, len(blob))
            _k32.GlobalUnlock(handle)
            if not _u32.SetClipboardData(fmt, handle):
                _k32.GlobalFree(handle)
    finally:
        _u32.CloseClipboard()


def set_clipboard_text(text: str) -> bool:
    if not _open_clipboard():
        return False
    try:
        _u32.EmptyClipboard()
        blob = text.encode("utf-16-le") + b"\x00\x00"
        handle = _k32.GlobalAlloc(GMEM_MOVEABLE, len(blob))
        ptr = _k32.GlobalLock(handle)
        ctypes.memmove(ptr, blob, len(blob))
        _k32.GlobalUnlock(handle)
        if not _u32.SetClipboardData(CF_UNICODETEXT, handle):
            _k32.GlobalFree(handle)
            return False
        return True
    finally:
        _u32.CloseClipboard()


class StreamResampler:
    """Reamostra blocos continuos pra 16 kHz (interp linear com fase persistente).

    Necessario porque WASAPI so abre na taxa nativa do dispositivo (44.1/48 kHz).
    """

    def __init__(self, src_rate: int, dst_rate: int = SAMPLE_RATE):
        self.step = src_rate / dst_rate
        self.next_t = 0.0   # tempo (em amostras da origem) da proxima amostra de saida
        self.buf = np.zeros(0, dtype=np.float32)
        self.buf_start = 0  # indice absoluto (na origem) de buf[0]

    def process(self, block: np.ndarray) -> np.ndarray:
        self.buf = np.concatenate([self.buf, block])
        end = self.buf_start + self.buf.size - 1
        ts = np.arange(self.next_t, end, self.step)
        if ts.size == 0:
            return np.zeros(0, dtype=np.float32)
        out = np.interp(ts - self.buf_start, np.arange(self.buf.size), self.buf).astype(np.float32)
        self.next_t = ts[-1] + self.step
        keep_from = min(max(int(self.next_t) - self.buf_start, 0), self.buf.size - 1)
        self.buf = self.buf[keep_from:]
        self.buf_start += keep_from
        return out


def fonetica(s: str) -> str:
    """Reduz o texto ao som aproximado em portugues: 'cloude' e 'klaude' viram a mesma chave.

    Sem acento, sem maiuscula, digrafos colapsados (ch->x, lh->l, ph->f), c/g moles
    resolvidos, h mudo fora, letra repetida colapsada. Nao e IPA — e o suficiente pra
    duas grafias do mesmo som cairem a distancia 0 ou 1 uma da outra.
    """
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    for a, b in (("ph", "f"), ("ch", "x"), ("lh", "l"), ("nh", "n"),
                 (r"qu([ei])", r"k\1"), (r"gu([ei])", r"g\1"), ("q", "k"),
                 (r"sc([ei])", r"s\1"), (r"c([ei])", r"s\1"), ("c", "k"),
                 (r"g([ei])", r"j\1"), ("h", ""), ("y", "i"), ("w", "v"), ("z", "s")):
        s = re.sub(a, b, s)
    return re.sub(r"(.)\1+", r"\1", s).replace(" ", "")


def _distancia(a: str, b: str) -> int:
    """Levenshtein simples (as chaves aqui tem poucas letras; nao vale trazer dependencia)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class Library:
    """Biblioteca de palavras: troca o que o whisper escreve errado pelo termo certo.

    Cada entrada e {"certo": "Grok", "erros": ["grock", "groque"]}. A troca e feita no
    texto ja transcrito, uma unica vez, antes de ir pra UI / injecao / historico, em
    duas passadas:

    1. exata — a variante escrita, sem diferenciar maiuscula nem espacamento
       ("nine houter" pega "Nine  Houter"), so em palavra inteira ("grok" nao mexe em
       "grokking");
    2. fonetica — janela de palavras cujo som chega perto de alguma variante ja listada
       ("nine hauter", "ninerouter", "9 rooter" caem em "9router" sem estarem na lista).

    A passada fonetica so vale pra variante com chave de LIB_FUZZY_MIN letras ou mais e
    exige mesmo numero de palavras: em termo curto o risco de trocar palavra legitima
    ("rock" -> "Grok") e maior que o ganho, entao ali so a passada exata trabalha.

    Os termos certos tambem viram `hotwords` do whisper, o que enviesa a decodificacao
    pra escrever o termo certo em vez de inventar ("Nightingale").
    """

    def __init__(self):
        self._rules = None  # (regex, {variante -> certo}); trocado inteiro (outra thread le)
        self._fuzzy = {}    # n_palavras -> [(chave fonetica, certo)]; idem
        self.hotwords = None
        self.entries: list = []
        self.load()

    def load(self):
        if LIBRARY_PATH.exists():
            self.entries = json.loads(LIBRARY_PATH.read_text(encoding="utf-8"))
        self._compile()

    def save(self):
        LIBRARY_PATH.write_text(
            json.dumps(self.entries, indent=2, ensure_ascii=False), encoding="utf-8")

    def _compile(self):
        termos = [e["certo"] for e in self.entries if e["certo"].strip()]
        self.hotwords = ", ".join(termos) if termos else None
        pairs = []
        for entry in self.entries:
            for wrong in entry["erros"]:
                wrong = " ".join(wrong.split())
                if wrong:
                    pairs.append((wrong, entry["certo"]))
        fuzzy = {}
        for wrong, certo in pairs:
            chave = fonetica(wrong)
            if len(chave) >= LIB_FUZZY_MIN:
                fuzzy.setdefault(len(wrong.split()), []).append((chave, certo))
        self._fuzzy = fuzzy
        if not pairs:
            self._rules = None
            return
        pairs.sort(key=lambda p: len(p[0]), reverse=True)  # variante mais longa ganha
        mapping = {}
        for wrong, certo in pairs:
            mapping.setdefault(wrong.lower(), certo)
        # separador [\s-]+: o whisper gosta de hifenizar o que ouviu junto ("9-router")
        alt = "|".join(r"[\s-]+".join(re.escape(w) for w in wrong.split())
                       for wrong, _c in pairs)
        # (?<!\w)/(?!\w) em vez de \b: funciona tambem com variante que comeca/termina
        # em pontuacao, e segue pegando so palavra inteira
        self._rules = (re.compile(rf"(?<!\w)(?:{alt})(?!\w)", re.IGNORECASE), mapping)

    def add(self, certo: str, erros: list) -> int:
        """Junta as variantes na entrada desse termo (cria se nao existir). Devolve quantas entraram."""
        entry = next((e for e in self.entries if e["certo"].lower() == certo.lower()), None)
        if entry is None:
            entry = {"certo": certo, "erros": []}
            self.entries.append(entry)
            self.entries.sort(key=lambda e: e["certo"].lower())
        known = {w.lower() for w in entry["erros"]}
        novos = [w for w in erros if w.lower() not in known and w != entry["certo"]]
        entry["erros"].extend(novos)
        self.save()
        self._compile()
        return len(novos)

    def remove(self, index: int):
        del self.entries[index]
        self.save()
        self._compile()

    def apply(self, text: str) -> tuple:
        """Devolve (texto corrigido, quantas trocas foram feitas) — a contagem alimenta
        a estatistica de correcoes da Biblioteca."""
        rules = self._rules
        if not rules or not text:
            return text, 0
        pattern, mapping = rules
        trocas = 0

        def _troca(m):
            nonlocal trocas
            trocas += 1
            # normaliza espaco E hifen pra achar a variante no mapa ("9-Router" -> "9 router")
            return mapping[" ".join(m.group(0).lower().replace("-", " ").split())]

        text = pattern.sub(_troca, text)
        text, fuzzy_trocas = self._apply_fuzzy(text)
        return text, trocas + fuzzy_trocas

    def _apply_fuzzy(self, text: str) -> tuple:
        """Segunda passada: janela de palavras que SOA como uma variante listada."""
        fuzzy = self._fuzzy
        if not fuzzy:
            return text, 0
        toks = list(re.finditer(r"\w+", text, re.UNICODE))
        tamanhos = sorted(fuzzy, reverse=True)
        trocas, i = [], 0
        while i < len(toks):
            for n in tamanhos:
                if i + n > len(toks):
                    continue
                janela = toks[i:i + n]
                # so junta palavras coladas por espaco ou hifen ("9-router"); outra
                # pontuacao no meio quer dizer que nao sao um termo so
                if any(text[janela[k].end():janela[k + 1].start()].strip(" 	-")
                       for k in range(n - 1)):
                    continue
                # letra solta ("o", "e") nao e parte de termo: e artigo, e o alinhamento
                # o engoliria ("o 9router" fica a 1 edicao de "9 router"). Digito solto
                # ("9 router") e parte do termo e continua valendo.
                sons = [fonetica(t.group(0)) for t in janela]
                if any(len(x) < 2 and x.isalpha() for x in sons):
                    continue
                chave = fonetica(text[janela[0].start():janela[-1].end()])
                if len(chave) < LIB_FUZZY_MIN:
                    continue
                # limite pela chave mais CURTA das duas: senao a janela maior compra
                # folga pra colar uma palavra a mais no termo ("um router" -> "9router")
                num = sorted(c for c in chave if c.isdigit())
                alvo = min(
                    ((_distancia(chave, k), c) for k, c in fuzzy[n]
                     # numero e ancora, nao som: "router" nao pode virar "9houter"
                     if sorted(x for x in k if x.isdigit()) == num
                     and _distancia(chave, k) <= LIB_FUZZY_DIST(min(k, chave, key=len))),
                    default=None, key=lambda p: p[0])
                if alvo is None:
                    continue
                _dist, certo = alvo
                # a passada exata pode ter acabado de escrever o termo certo bem aqui;
                # o fuzzy o reconhece de novo e "trocaria" pelo mesmo texto — isso nao e
                # correcao nenhuma e inflaria a contagem da aba ESTATISTICAS
                if text[janela[0].start():janela[-1].end()] != certo:
                    trocas.append((janela[0].start(), janela[-1].end(), certo))
                i += n
                break
            else:
                i += 1
        if not trocas:
            return text, 0
        out, fim = [], 0
        for ini, stop, certo in trocas:
            out.append(text[fim:ini])
            out.append(certo)
            fim = stop
        out.append(text[fim:])
        return "".join(out), len(trocas)


class Transcriber:
    """Threads de captura, segmentacao (VAD) e transcricao. UI le da text_queue."""

    def __init__(self, text_queue: queue.Queue, status_queue: queue.Queue):
        self.text_queue = text_queue
        self.status_queue = status_queue
        self.model = None
        self.library = Library()
        self.language = "pt"
        self.transcribe_mode = "simultaneo"
        self.inject_method = "colar"
        self.recording = threading.Event()
        self.history_queue: queue.Queue = queue.Queue()
        self._audio_queue: queue.Queue = queue.Queue()
        self._segment_queue: queue.Queue = queue.Queue()
        self._session_parts: list = []
        self._session_started = None
        self._streams: list = []
        self._resamplers: list = []   # um resampler por stream (taxa nativa != 16 kHz)
        self._slot = 0                # indice do stream sendo aberto em start()
        self._mix_lock = threading.Lock()
        self._mix_buffers: list = []  # buffers por stream; o mixer alinha e soma
        self._session_inject = False
        self._session_mode = "simultaneo"
        self._session_had_speech = False
        self._session_id = 0          # cancelar/reiniciar invalida o que ficou em voo
        self._pending = 0             # trechos aceitos e ainda nao entregues
        self._pending_lock = threading.Lock()
        self._drained = True          # o segmentador ja soltou tudo que tinha da sessao
        # nivel RMS recente do audio ja misturado; a barra de overlay le daqui
        self.levels: collections.deque = collections.deque(maxlen=96)
        self._keyboard = keyboard.Controller()
        threading.Thread(target=self._segmenter_loop, daemon=True).start()
        threading.Thread(target=self._transcribe_loop, daemon=True).start()
        threading.Thread(target=self._mixer_loop, daemon=True).start()

    # -- modelo -------------------------------------------------------------
    def load_model(self):
        self.status_queue.put("Carregando large-v3 na GPU...")
        t0 = time.perf_counter()
        self.model = WhisperModel("large-v3", device="cuda", compute_type="float16")
        # aquecimento: primeira inferencia compila kernels e distorce a latencia
        self.model.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), language="pt")
        self.status_queue.put(f"Modelo pronto ({time.perf_counter() - t0:.1f}s). Pode gravar.")

    # -- captura ------------------------------------------------------------
    def _make_callback(self, label: str, slot: int):
        """Callback de um stream: downmix mono, reamostra e guarda no slot dele."""

        def cb(indata, frames, time_info, status):
            if status:
                self.status_queue.put(f"Aviso ({label}): {status}")
            if not self.recording.is_set():
                return
            mono = indata.mean(axis=1)  # downmix p/ mono
            res = self._resamplers[slot]
            data = res.process(mono) if res else mono.copy()
            with self._mix_lock:
                self._mix_buffers[slot].append(data)

        return cb

    def _mixer_loop(self):
        """Junta os buffers dos streams (min len, soma, clip) e alimenta o VAD."""
        while True:
            time.sleep(0.01)
            with self._mix_lock:
                if not self._mix_buffers or any(not b for b in self._mix_buffers):
                    continue
                # descarta cabecas vazias
                for i, buf in enumerate(self._mix_buffers):
                    while buf and buf[0].size == 0:
                        buf.pop(0)
                if any(not b for b in self._mix_buffers):
                    continue
                n = min(b[0].size for b in self._mix_buffers)
                if n <= 0:
                    continue
                mixed = np.zeros(n, dtype=np.float32)
                for i, buf in enumerate(self._mix_buffers):
                    mixed += buf[0][:n]
                    rest = buf[0][n:]
                    self._mix_buffers[i] = ([rest] if rest.size else []) + buf[1:]
            np.clip(mixed, -1.0, 1.0, out=mixed)
            self._push_levels(mixed)
            self._audio_queue.put(mixed)

    def _push_levels(self, chunk: np.ndarray):
        """RMS em fatias de ~25 ms: e o que a barra de overlay desenha como onda."""
        step = max(1, SAMPLE_RATE // LEVEL_HZ)
        for i in range(0, chunk.size, step):
            part = chunk[i:i + step]
            if part.size:
                self.levels.append(float(np.sqrt(np.mean(part * part))))

    def _loopback_loop(self, slot: int, label: str, device_index: int | None, handle):
        """Captura o que o PC esta tocando (WASAPI loopback via soundcard)."""
        try:
            import soundcard as sc
            if device_index is None:
                spk = sc.default_speaker()
            else:
                name = sd.query_devices(device_index)["name"]
                try:
                    spk = sc.get_speaker(name)
                except Exception:
                    spk = sc.default_speaker()
                    self.status_queue.put(f"{label}: canal '{name}' nao achado, usando padrao ({spk.name}).")
            loop = sc.get_microphone(id=spk.id, include_loopback=True)
        except Exception as e:
            self.status_queue.put(f"ERRO ({label}): {e}")
            return
        native = 48000
        res = StreamResampler(native)
        self._resamplers[slot] = res
        while not self.recording.is_set() and not handle.stop_flag.is_set():
            time.sleep(0.02)
        try:
            with loop.recorder(samplerate=native, channels=2) as rec:
                chunk = int(native * 0.1)
                while self.recording.is_set() and not handle.stop_flag.is_set():
                    block = rec.record(numframes=chunk)
                    if block is None or block.size == 0:
                        continue
                    mono = block.mean(axis=1).astype(np.float32)
                    data = res.process(mono)
                    if data.size:
                        with self._mix_lock:
                            self._mix_buffers[slot].append(data)
        except Exception as e:
            self.status_queue.put(f"ERRO ({label}): {e}")

    def _open_stream(self, device_index: int | None, loopback: bool, label: str):
        """Abre mic (sounddevice) ou audio do PC (soundcard loopback)."""
        if loopback:
            class _LoopbackHandle:
                def __init__(self):
                    self.stop_flag = threading.Event()
                def stop(self):
                    self.stop_flag.set()
                def close(self):
                    self.stop_flag.set()
            handle = _LoopbackHandle()
            self._resamplers.append(None)
            threading.Thread(
                target=self._loopback_loop,
                args=(self._slot, label, device_index, handle),
                daemon=True,
            ).start()
            self._streams.append(handle)
            return
        try:
            stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                blocksize=BLOCK_SIZE, device=device_index,
                callback=self._make_callback(label, self._slot),
            )
            stream.start()
            self._resamplers.append(None)
        except sd.PortAudioError:
            query = device_index if device_index is not None else sd.default.device[0]
            native = int(sd.query_devices(query)["default_samplerate"])
            self._resamplers.append(StreamResampler(native))
            stream = sd.InputStream(
                samplerate=native, channels=1, dtype="float32",
                blocksize=int(native * 0.1), device=device_index,
                callback=self._make_callback(label, self._slot),
            )
            stream.start()
        self._streams.append(stream)

    def start(self, device_index: int | None, inject: bool,
              capture_mode: str = "microfone", loopback_index: int | None = None):
        if self.recording.is_set():
            return
        self._session_inject = inject
        self._session_mode = self.transcribe_mode
        self._session_had_speech = False
        self._session_started = datetime.now()
        self._session_id += 1
        self._session_parts = []
        self.levels.clear()
        with self._pending_lock:
            self._pending = 0
        self._streams = []
        self._resamplers = []
        self._mix_buffers = [[], []] if capture_mode == "os_dois" else [[]]
        self._slot = 0
        try:
            if capture_mode != "audio_pc":
                self._open_stream(device_index, False, "mic")
                self._slot += 1
            if capture_mode != "microfone":
                self._open_stream(loopback_index, True, "audio do PC")
        except Exception:
            # falhou um dos streams: fecha o que abriu e propaga com o dispositivo culpado
            for s in self._streams:
                s.stop()
                s.close()
            raise
        self._drained = False  # so depois dos streams de pe: se falhar, nada fica ocupado
        self.recording.set()
        fonte = {"microfone": "mic", "audio_pc": "audio do PC",
                 "os_dois": "mic + audio do PC"}[capture_mode]
        self.status_queue.put(f"Gravando ({fonte}) — pode falar.")

    def stop(self):
        if not self.recording.is_set():
            return
        self.recording.clear()
        # status ANTES do sentinela: o segmentador emite os status finais depois dele,
        # e a ordem na fila e o que impede "Transcrevendo..." de ficar pendurado
        self.status_queue.put("Parado. Transcrevendo...")
        self._audio_queue.put(None)  # sentinela: descarrega o buffer restante
        streams, self._streams = self._streams, []
        for s in streams:
            s.stop()
            s.close()

    def busy(self) -> bool:
        """Ha trabalho da sessao em andamento: gravando, segmentando ou transcrevendo.

        E o que decide a barra de overlay ficar na tela — em vez de adivinhar pelo texto
        do status, que so chega depois de colar e de arquivar.
        """
        with self._pending_lock:
            pendentes = self._pending
        return self.recording.is_set() or not self._drained or pendentes > 0

    def _pending_done(self):
        with self._pending_lock:
            if self._pending:
                self._pending -= 1

    def cancel(self, from_processing: bool = False):
        """Descarta a sessao: nada de transcrever, colar ou gravar no historico.

        O que ja foi transcrito e colado no modo simultaneo nao volta atras — o
        cancelamento vale para o audio e para os trechos ainda em voo.
        """
        if not self.recording.is_set() and not from_processing:
            return
        self.recording.clear()
        self._session_id += 1  # tudo que estiver na fila com o id antigo vira lixo
        streams, self._streams = self._streams, []
        for s in streams:
            s.stop()
            s.close()
        with self._mix_lock:
            self._mix_buffers = [[] for _ in self._mix_buffers]
        for q in (self._audio_queue, self._segment_queue):
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
        self._session_parts = []
        self.levels.clear()
        with self._pending_lock:
            self._pending = 0
        self._drained = True
        self._audio_queue.put(_CANCEL)  # o segmentador larga o buffer que sobrou
        self.status_queue.put("Cancelado.")

    # -- segmentacao --------------------------------------------------------
    def _segmenter_loop(self):
        buffer = np.zeros(0, dtype=np.float32)
        last_check = 0.0
        while True:
            item = self._audio_queue.get()
            try:
                if item is _CANCEL:  # sessao cancelada: o buffer acumulado morre aqui
                    buffer = np.zeros(0, dtype=np.float32)
                    continue
                if item is None:  # fim da gravacao: manda o que sobrou
                    if buffer.size > SAMPLE_RATE // 4 and get_speech_timestamps(buffer, VAD_OPTIONS):
                        self._enqueue_segment(buffer)
                    elif not self._session_had_speech:
                        self.status_queue.put("Parado (sem fala detectada).")
                    else:
                        self.status_queue.put("Parado.")
                    self._drained = True  # nao vem mais trecho desta sessao
                    # marcador de fim de sessao (carimbado: cancelamento o invalida)
                    self._segment_queue.put((None, None, None, self._session_id))
                    buffer = np.zeros(0, dtype=np.float32)
                    continue
                buffer = np.concatenate([buffer, item])
                if self._session_mode == "final":
                    continue  # acumula tudo; transcreve de uma vez no stop
                now = time.monotonic()
                if now - last_check < VAD_CHECK_EVERY_S:
                    continue
                last_check = now

                speech = get_speech_timestamps(buffer, VAD_OPTIONS)
                if not speech:
                    if buffer.size > MAX_IDLE_BUFFER_S * SAMPLE_RATE:
                        buffer = buffer[-SAMPLE_RATE:]
                    continue
                last_end = speech[-1]["end"]
                tail_silence = (buffer.size - last_end) / SAMPLE_RATE
                if tail_silence >= TAIL_SILENCE_S:
                    self._enqueue_segment(buffer[:last_end])
                    buffer = buffer[last_end:]
                elif buffer.size > MAX_SEGMENT_S * SAMPLE_RATE:
                    self._enqueue_segment(buffer)
                    buffer = np.zeros(0, dtype=np.float32)
            except Exception as e:  # falha alto: reporta no status e mantem a thread viva
                traceback.print_exc()
                self.status_queue.put(f"ERRO na segmentacao: {e}")
                buffer = np.zeros(0, dtype=np.float32)

    def _enqueue_segment(self, audio: np.ndarray):
        self._session_had_speech = True
        with self._pending_lock:
            self._pending += 1
        self._segment_queue.put((audio, self._session_inject, self._session_mode,
                                 self._session_id))

    # -- transcricao --------------------------------------------------------
    def _transcribe_loop(self):
        while True:
            audio, inject, mode, sid = self._segment_queue.get()
            # trecho real desta sessao: precisa dar baixa mesmo se a transcricao falhar
            deve_baixar = audio is not None and sid == self._session_id
            try:
                if sid != self._session_id:
                    continue  # sessao cancelada ou substituida: descarta sem colar nada
                if audio is None:  # fim de sessao: grava no historico
                    self._finalize_session()
                    continue
                t0 = time.perf_counter()
                lang = None if self.language == "auto" else self.language
                segments, _info = self.model.transcribe(
                    audio, language=lang, beam_size=5, vad_filter=(mode == "final"),
                    # enviesa a decodificacao pros termos da Biblioteca: e o que evita
                    # o whisper inventar "Nightingale" no lugar de "9router"
                    hotwords=self.library.hotwords,
                )
                text = "".join(s.text for s in segments).strip()
                text, fixes = self.library.apply(text)  # troca da Biblioteca antes de sair daqui
                if sid != self._session_id:
                    continue  # cancelado enquanto este trecho transcrevia
                dt = time.perf_counter() - t0
                self._session_parts.append((audio, text, fixes))
                if text:
                    self.text_queue.put(text)
                # baixa aqui, antes de colar e de arquivar: o texto ja saiu, a barra nao
                # tem mais o que mostrar — colar ainda leva ~0,45 s so restaurando o
                # clipboard, e segurar a barra ate la e o que parecia travamento
                self._pending_done()
                deve_baixar = False
                if text and inject:
                    payload = text + (" " if mode == "simultaneo" else "")
                    if self.inject_method == "colar":
                        self._paste(payload)
                    else:
                        self._keyboard.type(payload)
                state = "Gravando — pode falar." if self.recording.is_set() else "Parado."
                self.status_queue.put(f"{state}  (trecho de {audio.size / SAMPLE_RATE:.1f}s em {dt:.1f}s)")
            except Exception as e:  # falha alto: reporta no status e mantem a thread viva
                traceback.print_exc()
                self.status_queue.put(f"ERRO na transcricao: {e}")
            finally:
                if deve_baixar:
                    self._pending_done()

    def _finalize_session(self):
        parts, self._session_parts = self._session_parts, []
        text = " ".join(t for _a, t, _f in parts if t).strip()
        if not text:
            return
        started = self._session_started or datetime.now()
        audio = np.concatenate([a for a, _t, _f in parts])
        fixes = sum(f for _a, _t, f in parts)
        HISTORY_DIR.mkdir(exist_ok=True)
        wav_name = started.strftime("%Y%m%d_%H%M%S") + ".wav"
        with wave.open(str(HISTORY_DIR / wav_name), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
        # dur e fix alimentam a aba ESTATISTICAS; entrada antiga sem eles usa o wav / zero
        entry = {"ts": started.isoformat(timespec="seconds"), "wav": wav_name, "text": text,
                 "dur": round(audio.size / SAMPLE_RATE, 2), "fix": fixes}
        with HISTORY_INDEX.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.history_queue.put(entry)

    def _paste(self, text: str):
        """Cola via Ctrl+V preservando o clipboard original (imagens, arquivos etc.)."""
        backup = backup_clipboard()
        if not set_clipboard_text(text):
            self._keyboard.type(text)  # clipboard ocupado por outro app: cai pro digitar
            return
        time.sleep(0.05)
        with self._keyboard.pressed(keyboard.Key.ctrl):
            self._keyboard.press("v")
            self._keyboard.release("v")
        time.sleep(0.4)  # o app alvo precisa ler o clipboard antes da restauracao
        restore_clipboard(backup)


class RecorderBar:
    """Barra flutuante de gravacao: cancelar (X), onda ao vivo do audio e confirmar (V).

    Sempre no topo, fora do Alt-Tab, sem roubar foco. Aparece no monitor onde o cursor
    esta (e segue se o cursor trocar de monitor) na posicao relativa salva; arrastavel
    pela area da onda pra reposicionar — a posicao e salva como fracao da area util,
    entao se adapta a qualquer resolucao.

    A onda desenha o RMS recente do audio JA MISTURADO (Transcriber.levels), entao ela
    reflete exatamente a fonte escolhida: microfone, audio do PC ou os dois somados.

    Desenho via PIL com supersampling — o canvas do Tk nao tem antialias e a capsula
    ficaria serrilhada.
    """

    W, H = 152, 40
    PAD = 6
    BTN = 28                       # diametro dos botoes redondos
    BARS = 15
    BAR_W = 2.8
    BAR_MIN, BAR_MAX = 1.6, 11.0   # meia-altura da barra (silencio -> pico)
    SS = 4                         # fator de supersampling do desenho
    FPS_MS = 45
    TRANSPARENT = "#010203"

    PILL_BG = "#17181b"
    PILL_BORDER = "#3a3c43"
    BTN_BG = "#3a3c43"
    BTN_BG_HOVER = "#4a4d55"
    WAVE_IDLE = "#5c6065"

    def __init__(self, root: tk.Tk, get_rel_pos, save_rel_pos,
                 get_levels, on_cancel, on_confirm):
        self.root = root
        self.get_rel_pos = get_rel_pos
        self.save_rel_pos = save_rel_pos
        self.get_levels = get_levels
        self.on_cancel = on_cancel
        self.on_confirm = on_confirm

        self.CY = self.H / 2
        self.LX = self.PAD + self.BTN / 2            # centro do botao cancelar
        self.RX = self.W - self.PAD - self.BTN / 2   # centro do botao confirmar
        self.WX0 = self.LX + self.BTN / 2 + 7        # area da onda
        self.WX1 = self.RX - self.BTN / 2 - 7

        self.win = tk.Toplevel(root)
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        self.win.attributes("-transparentcolor", self.TRANSPARENT)
        self.canvas = tk.Canvas(self.win, width=self.W, height=self.H,
                                bg=self.TRANSPARENT, highlightthickness=0, cursor="fleur")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag_move)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.canvas.bind("<Motion>", self._hover)
        self.canvas.bind("<Leave>", lambda _e: self._set_hover(None))
        self.win.update_idletasks()
        self._set_exstyle()
        self.win.withdraw()

        self._state = None
        self._hovered = None    # "cancel" | "ok" | None
        self._pressed = None
        self._press_xy = (0, 0)
        self._dragging = False
        self._moved = False
        self._drag_off = (0, 0)
        self._phase = 0.0
        self._photo = None
        self._ticking = False

    def _set_exstyle(self):
        GWL_EXSTYLE = -20
        # LAYERED | NOACTIVATE (nao rouba foco) | TOOLWINDOW (fora do Alt-Tab)
        flags = 0x00080000 | 0x08000000 | 0x00000080
        get_long = _u32.GetWindowLongPtrW
        set_long = _u32.SetWindowLongPtrW
        get_long.restype = ctypes.c_longlong
        get_long.argtypes = [wintypes.HWND, ctypes.c_int]
        set_long.restype = ctypes.c_longlong
        set_long.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_longlong]
        hwnd = _u32.GetParent(self.win.winfo_id()) or self.win.winfo_id()
        set_long(hwnd, GWL_EXSTYLE, get_long(hwnd, GWL_EXSTYLE) | flags)

    # -- posicionamento ------------------------------------------------------
    def _target_xy(self):
        cx, cy = cursor_pos()
        left, top, right, bottom = monitor_work_area(cx, cy)
        rx, ry = self.get_rel_pos()
        x = int(left + rx * (right - left - self.W))
        y = int(top + ry * (bottom - top - self.H))
        return x, y

    def _follow(self):
        if not self._dragging:
            x, y = self._target_xy()
            self.win.geometry(f"+{x}+{y}")

    # -- ciclo de vida -------------------------------------------------------
    def show(self, state: str):
        self._state = state
        self._follow()
        self.win.deiconify()
        self.win.attributes("-topmost", True)
        self._draw()
        if not self._ticking:
            self._ticking = True
            self._tick()

    def visivel(self) -> bool:
        return self._state is not None

    def hide(self):
        self._state = None
        self._dragging = False
        self._moved = False
        self._pressed = None
        self._hovered = None
        self.win.withdraw()

    def _tick(self):
        if self._state is None:
            self._ticking = False
            return
        self._phase += self.FPS_MS / 1000.0
        self._draw()
        self._follow()
        self.root.after(self.FPS_MS, self._tick)

    # -- mouse ---------------------------------------------------------------
    def _zone(self, x, y):
        r2 = (self.BTN / 2) ** 2
        if (x - self.LX) ** 2 + (y - self.CY) ** 2 <= r2:
            return "cancel"
        if self._state == "rec" and (x - self.RX) ** 2 + (y - self.CY) ** 2 <= r2:
            return "ok"
        return None

    def _set_hover(self, zone):
        if zone != self._hovered:
            self._hovered = zone
            self.canvas.configure(cursor="hand2" if zone else "fleur")
            self._draw()

    def _hover(self, event):
        self._set_hover(self._zone(event.x, event.y))

    def _press(self, event):
        self._press_xy = (event.x, event.y)
        self._pressed = self._zone(event.x, event.y)
        self._dragging = self._pressed is None
        self._moved = False
        self._drag_off = (event.x, event.y)

    def _drag_move(self, event):
        if not self._dragging:
            # arrastou de dentro de um botao: passa de clique pra reposicionamento
            if abs(event.x - self._press_xy[0]) + abs(event.y - self._press_xy[1]) < 5:
                return
            self._pressed = None
            self._dragging = True
        if abs(event.x - self._press_xy[0]) + abs(event.y - self._press_xy[1]) < 3:
            return  # tremida de clique nao conta como reposicionamento
        self._moved = True
        x = self.win.winfo_pointerx() - self._drag_off[0]
        y = self.win.winfo_pointery() - self._drag_off[1]
        self.win.geometry(f"+{x}+{y}")

    def _release(self, event):
        if self._dragging:
            moved, self._dragging, self._moved = self._moved, False, False
            if moved:  # so salva se saiu do lugar; clique simples no meio nao mexe em nada
                self._save_pos()
            return
        zone, self._pressed = self._pressed, None
        if zone and self._zone(event.x, event.y) == zone:
            (self.on_cancel if zone == "cancel" else self.on_confirm)()

    def _save_pos(self):
        wx, wy = self.win.winfo_x(), self.win.winfo_y()
        cx, cy = wx + self.W // 2, wy + self.H // 2
        left, top, right, bottom = monitor_work_area(cx, cy)
        rx = (wx - left) / max(right - left - self.W, 1)
        ry = (wy - top) / max(bottom - top - self.H, 1)
        self.save_rel_pos([round(min(max(rx, 0.0), 1.0), 4),
                           round(min(max(ry, 0.0), 1.0), 4)])

    # -- desenho -------------------------------------------------------------
    def _draw(self):
        if self._state is None:
            return
        self._photo = ImageTk.PhotoImage(self._render())
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)

    def _render(self) -> Image.Image:
        s = self.SS
        img = Image.new("RGB", (self.W * s, self.H * s), self.TRANSPARENT)
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([0.7 * s, 0.7 * s, (self.W - 0.7) * s, (self.H - 0.7) * s],
                            radius=(self.H / 2 - 0.7) * s,
                            fill=self.PILL_BG, outline=self.PILL_BORDER,
                            width=max(1, round(1.1 * s)))
        self._draw_wave(d)
        self._draw_cancel(d)
        if self._state == "rec":
            self._draw_confirm(d)
        else:
            self._draw_spinner(d)
        return img.resize((self.W, self.H), Image.BOX)

    def _draw_wave(self, d):
        s = self.SS
        slot = (self.WX1 - self.WX0) / self.BARS
        if self._state == "rec":
            levels = list(self.get_levels())[-self.BARS:]
            levels = [0.0] * (self.BARS - len(levels)) + levels
            amps = [min(1.0, lv * 9.0) ** 0.55 for lv in levels]
            colors = [ACCENT] * self.BARS
        else:  # transcrevendo: nao ha entrada pra mostrar, entao onda viajando
            amps, colors = [], []
            head = (self._phase * 9.0) % (self.BARS + 6) - 3
            for i in range(self.BARS):
                bump = math.exp(-((i - head) ** 2) / 4.0)
                amps.append(0.12 + 0.75 * bump)
                colors.append(ACCENT if bump > 0.25 else self.WAVE_IDLE)
        for i, (a, color) in enumerate(zip(amps, colors)):
            x = self.WX0 + slot * (i + 0.5)
            half = self.BAR_MIN + (self.BAR_MAX - self.BAR_MIN) * a
            d.rounded_rectangle([(x - self.BAR_W / 2) * s, (self.CY - half) * s,
                                 (x + self.BAR_W / 2) * s, (self.CY + half) * s],
                                radius=self.BAR_W / 2 * s, fill=color)

    def _circle(self, d, cx, r, fill):
        s = self.SS
        d.ellipse([(cx - r) * s, (self.CY - r) * s, (cx + r) * s, (self.CY + r) * s], fill=fill)

    def _draw_cancel(self, d):
        s = self.SS
        self._circle(d, self.LX, self.BTN / 2,
                     self.BTN_BG_HOVER if self._hovered == "cancel" else self.BTN_BG)
        a = 4.4
        for dx in (a, -a):
            d.line([(self.LX - dx) * s, (self.CY - a) * s,
                    (self.LX + dx) * s, (self.CY + a) * s],
                   fill=INK, width=max(1, round(1.9 * s)))

    def _draw_confirm(self, d):
        s = self.SS
        self._circle(d, self.RX, self.BTN / 2, ACCENT if self._hovered == "ok" else INK)
        pts = [(-4.6, 0.3), (-1.5, 3.4), (4.8, -3.6)]
        d.line([((self.RX + px) * s, (self.CY + py) * s) for px, py in pts],
               fill=GRAPHITE, width=max(1, round(2.3 * s)), joint="curve")

    def _draw_spinner(self, d):
        s = self.SS
        r = self.BTN / 2 - 3
        box = [(self.RX - r) * s, (self.CY - r) * s, (self.RX + r) * s, (self.CY + r) * s]
        w = max(1, round(2.2 * s))
        d.ellipse(box, outline=self.BTN_BG, width=w)
        start = (self._phase * 300.0) % 360.0
        d.arc(box, start, start + 100, fill=ACCENT, width=w)


class MouseHotkey:
    """Hook global de mouse. O botao configurado e suprimido do sistema e vira o atalho.

    Eventos sao empurrados na event_queue ("start"/"stop"/("captured", nome));
    quem consome e a UI, no thread do Tk.
    """

    def __init__(self, event_queue: queue.Queue, button: str, trigger_mode: str):
        self.event_queue = event_queue
        self.button = button
        self.trigger_mode = trigger_mode
        self.capturing = False
        self.active = False  # sessao de gravacao iniciada pelo atalho
        self._listener = mouse.Listener(win32_event_filter=self._filter)
        self._listener.start()

    @staticmethod
    def _decode(msg, data):
        if msg in (WM_MBUTTONDOWN, WM_MBUTTONUP):
            return "middle", msg == WM_MBUTTONDOWN
        if msg in (WM_XBUTTONDOWN, WM_XBUTTONUP):
            xbtn = (data.mouseData >> 16) & 0xFFFF
            return ("x1" if xbtn == 1 else "x2"), msg == WM_XBUTTONDOWN
        return None, None

    def _filter(self, msg, data):
        button, pressed = self._decode(msg, data)
        if button is None:
            return True
        if self.capturing:
            if pressed:
                self.capturing = False
                self.button = button
                self.event_queue.put(("captured", button))
            self._listener.suppress_event()
        if button != self.button:
            return True
        # atalho configurado: nunca deixa vazar pro sistema (ex.: voltar/avancar no browser)
        if self.trigger_mode == "alternar":
            if pressed:
                self.event_queue.put(("stop" if self.active else "start", None))
                self.active = not self.active
        else:  # segurar (push-to-talk)
            if pressed and not self.active:
                self.active = True
                self.event_queue.put(("start", None))
            elif not pressed and self.active:
                self.active = False
                self.event_queue.put(("stop", None))
        self._listener.suppress_event()


# -- estatisticas -----------------------------------------------------------
# Tudo aqui e derivado do history.jsonl: nao existe contador paralelo pra
# dessincronizar. Entrada antiga nao tem "dur" nem "fix" — a duracao sai do
# tamanho do wav e as correcoes contam a partir desta versao.
WAV_HEADER_BYTES = 44       # cabecalho PCM que o modulo wave escreve
TYPING_WPM = 40             # digitacao media, referencia da "economia vs digitar"
HEAT_WEEKS = 26             # semanas mostradas no mapa de atividade
MIN_DUR_PPM = 20.0          # ditado curto demais falseia o recorde de ppm
MES_ABREV = ["jan", "fev", "mar", "abr", "mai", "jun",
             "jul", "ago", "set", "out", "nov", "dez"]
# escala do mapa de atividade: vazio -> laranja cheio (o unico acento da casa)
HEAT_SCALE = ["#2e2f36", "#4a2417", "#7a3312", "#b7440d", ACCENT]
BAR_DIM = "#b7440d"         # barra normal; o pico usa ACCENT
# palavras que nao dizem nada sobre o que voce fala: fora do "mais ditadas"
STOPWORDS = {
    "que", "com", "uma", "para", "por", "dos", "das", "nao", "mais", "mas", "como",
    "isso", "esse", "essa", "este", "esta", "aqui", "ali", "sao", "foi", "ser", "tem",
    "vai", "vou", "voce", "eles", "elas", "nos", "meu", "minha", "seu", "sua", "num",
    "numa", "pra", "pro", "ate", "tao", "tudo", "todo", "toda", "muito", "pouco",
    "entao", "porque", "quando", "onde", "quem", "qual", "ele", "ela", "sem", "sobre",
    "depois", "antes", "agora", "assim", "cada", "coisa", "fazer", "faz", "fica",
    "ficar", "estar", "estao", "tinha", "ter", "dar", "vamos", "bem", "melhor",
    "mesmo", "outro", "outra", "ainda", "aqui", "tambem", "ver", "vez", "dai",
    "the", "and", "for", "you", "that", "this", "with", "from", "your", "have", "was",
}


def _sem_acento(palavra: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", palavra)
                   if not unicodedata.combining(c))


def entry_words(entry: dict) -> int:
    return len(entry["text"].split())


def entry_dur(entry: dict) -> float:
    """Segundos de fala do ditado. Entrada antiga nao tem "dur": mede pelo wav.

    O wav do historico e sempre PCM 16 bits mono a SAMPLE_RATE, escrito aqui mesmo,
    entao o tamanho do arquivo da a duracao sem precisar abrir o audio. O valor
    medido fica no proprio dict em memoria — o history.jsonl nao e reescrito.
    """
    dur = entry.get("dur")
    if dur is None:
        try:
            size = (HISTORY_DIR / entry["wav"]).stat().st_size
        except OSError:  # wav apagado: as palavras contam, mas fora do ppm
            dur = 0.0
        else:
            dur = max(0.0, (size - WAV_HEADER_BYTES) / (2 * SAMPLE_RATE))
        entry["dur"] = dur
    return dur


def _streaks(dias: set) -> tuple:
    """(sequencia atual, maior sequencia) de dias seguidos com ditado."""
    if not dias:
        return 0, 0
    ordenados = sorted(dias)
    recorde = atual = 1
    for anterior, dia in zip(ordenados, ordenados[1:]):
        atual = atual + 1 if (dia - anterior).days == 1 else 1
        recorde = max(recorde, atual)
    # so conta como sequencia viva se chega em hoje ou ontem
    if (date.today() - ordenados[-1]).days > 1:
        return 0, recorde
    return atual, recorde


def compute_stats(entries: list) -> dict:
    """Uma passada sobre o historico inteiro; ~90 entradas custam milissegundos."""
    total_palavras = 0
    total_segundos = 0.0
    correcoes = 0
    maior_ditado = 0
    melhor_ppm = 0.0
    por_dia = collections.Counter()   # date -> palavras
    por_hora = collections.Counter()  # 0..23 -> palavras
    freq = collections.Counter()      # palavra -> vezes
    for e in entries:
        palavras = entry_words(e)
        dur = entry_dur(e)
        quando = datetime.fromisoformat(e["ts"])
        total_palavras += palavras
        total_segundos += dur
        correcoes += e.get("fix", 0)
        maior_ditado = max(maior_ditado, palavras)
        por_dia[quando.date()] += palavras
        por_hora[quando.hour] += palavras
        if dur >= MIN_DUR_PPM:
            melhor_ppm = max(melhor_ppm, palavras / (dur / 60))
        for palavra in re.findall(r"[^\W\d_]+", e["text"].lower(), re.UNICODE):
            if len(palavra) > 2 and _sem_acento(palavra) not in STOPWORDS:
                freq[palavra] += 1
    ppm = total_palavras / (total_segundos / 60) if total_segundos else 0.0
    atual, recorde = _streaks(set(por_dia))
    return {
        "ditados": len(entries),
        "palavras": total_palavras,
        "segundos": total_segundos,
        "ppm": ppm,
        "melhor_ppm": melhor_ppm,
        "correcoes": correcoes,
        "maior_ditado": maior_ditado,
        "streak": atual,
        "recorde": recorde,
        # quanto o teclado levaria pra escrever o mesmo, menos o tempo falado
        "economia": max(0.0, total_palavras / TYPING_WPM * 60 - total_segundos),
        "por_dia": por_dia,
        "por_hora": por_hora,
        "top": freq.most_common(8),
    }


def fmt_int(n) -> str:
    """Numero com ponto de milhar (pt-BR), sem depender do locale do sistema."""
    return f"{int(round(n)):,}".replace(",", ".")


def fmt_dias(n: int) -> str:
    return f"{n} dia" if n == 1 else f"{n} dias"


def fmt_dur(segundos: float) -> str:
    segundos = int(segundos)
    horas, minutos = divmod(segundos // 60, 60)
    if horas:
        return f"{horas}h {minutos:02d}min"
    if minutos:
        return f"{minutos}min {segundos % 60:02d}s"
    return f"{segundos}s"


def _pil_font(size: int):
    for nome in ("segoeui.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(nome, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_heatmap(por_dia: dict, semanas: int = HEAT_WEEKS, cell: int = 13,
                   gap: int = 3, s: int = 2):
    """Mapa de atividade: uma coluna por semana, uma linha por dia (seg..dom).

    Desenhado em PIL e nao em widgets: 7x26 celulas seriam 182 CTkFrames, que
    custam segundos pra montar e mais ainda pra destruir a cada atualizacao.
    Renderiza em 2x (s) pra ficar nitido tambem em tela com escala 200%.
    """
    passo = cell + gap
    esq, topo = 26, 13
    w, h = esq + semanas * passo, topo + 7 * passo
    img = Image.new("RGB", (w * s, h * s), SURFACE_2)
    d = ImageDraw.Draw(img)
    fonte = _pil_font(9 * s)
    hoje = date.today()
    # coluna 0 = segunda-feira da semana mais antiga mostrada
    inicio = hoje - timedelta(days=hoje.weekday(), weeks=semanas - 1)
    teto = max(por_dia.values(), default=0)
    mes_anterior = None
    for col in range(semanas):
        for lin in range(7):
            dia = inicio + timedelta(weeks=col, days=lin)
            if dia > hoje:
                continue
            palavras = por_dia.get(dia, 0)
            faixa = 0 if not palavras else 1 + min(3, int(palavras / teto * 4))
            x, y = esq + col * passo, topo + lin * passo
            d.rounded_rectangle([x * s, y * s, (x + cell) * s, (y + cell) * s],
                                radius=3 * s, fill=HEAT_SCALE[faixa])
        primeira = inicio + timedelta(weeks=col)
        if primeira <= hoje and primeira.month != mes_anterior:
            d.text(((esq + col * passo) * s, 0), MES_ABREV[primeira.month - 1],
                   font=fonte, fill=INK_3)
            mes_anterior = primeira.month
    for lin, nome in ((0, "seg"), (2, "qua"), (4, "sex"), (6, "dom")):
        d.text((0, (topo + lin * passo + 1) * s), nome, font=fonte, fill=INK_3)
    return img, (w, h)


def render_legend(cell: int = 11, gap: int = 3, s: int = 2):
    """Os cinco tons da escala do mapa, de menos pra mais."""
    passo = cell + gap
    w, h = len(HEAT_SCALE) * passo - gap, cell
    img = Image.new("RGB", (w * s, h * s), SURFACE_2)
    d = ImageDraw.Draw(img)
    for i, cor in enumerate(HEAT_SCALE):
        x = i * passo
        d.rounded_rectangle([x * s, 0, (x + cell) * s, cell * s - 1], radius=3 * s, fill=cor)
    return img, (w, h)


def render_hours(por_hora: dict, cell: int = 12, altura: int = 84, s: int = 2):
    """Barras 0h..23h: em que hora do dia voce fala mais. Pico em laranja cheio."""
    base, rodape = altura, 13
    w, h = 24 * cell, altura + rodape
    img = Image.new("RGB", (w * s, h * s), SURFACE_2)
    d = ImageDraw.Draw(img)
    fonte = _pil_font(9 * s)
    teto = max(por_hora.values(), default=0)
    for hora in range(24):
        palavras = por_hora.get(hora, 0)
        alt = round(palavras / teto * (altura - 4)) if teto else 0
        x, largura = hora * cell + 2, cell - 4
        if alt < 2:  # hora vazia: risco na base, pra grade nao sumir
            d.rectangle([x * s, (base - 2) * s, (x + largura) * s, base * s],
                        fill=HEAT_SCALE[0])
            continue
        cor = ACCENT if palavras == teto else BAR_DIM
        d.rounded_rectangle([x * s, (base - alt) * s, (x + largura) * s, base * s],
                            radius=2 * s, fill=cor)
    for hora in (0, 6, 12, 18):
        d.text(((hora * cell + 1) * s, (base + 2) * s), f"{hora}h", font=fonte, fill=INK_3)
    return img, (w, h)


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Sussurro")
        icon_path = Path(__file__).with_name("assets") / "sussurro.ico"
        if icon_path.exists():
            root.iconbitmap(str(icon_path))
        root.geometry("720x660")

        self.settings = load_settings()
        self.devices = list_input_devices()
        self.loopback_devices = list_loopback_devices()

        self.text_queue: queue.Queue = queue.Queue()
        self.status_queue: queue.Queue = queue.Queue()
        self.hotkey_queue: queue.Queue = queue.Queue()
        self.transcriber = Transcriber(self.text_queue, self.status_queue)
        self.transcriber.language = self.settings["language"]
        self.transcriber.transcribe_mode = self.settings["transcribe_mode"]
        self.transcriber.inject_method = self.settings["inject_method"]
        self.hotkey = MouseHotkey(
            self.hotkey_queue, self.settings["mouse_button"], self.settings["trigger_mode"]
        )

        self.FONT_DISPLAY = pick_font(
            ["Bahnschrift SemiBold Condensed", "Bahnschrift SemiBold", "Bahnschrift"],
            "Arial Narrow",
        )
        self.FONT_MONO = pick_font(["Cascadia Mono", "JetBrains Mono"], "Consolas")
        root.configure(fg_color=BG)

        # cabecalho: marca + modelo
        header = ctk.CTkFrame(root, fg_color="transparent")
        header.pack(fill="x", padx=18, pady=(14, 10))
        icon_png = Path(__file__).with_name("assets") / "sussurro.png"
        if icon_png.exists():
            self._brand_img = ctk.CTkImage(Image.open(icon_png), size=(30, 30))
            ctk.CTkLabel(header, image=self._brand_img, text="").pack(side="left")
        ctk.CTkLabel(header, text="SUSSURRO", text_color=INK,
                     font=(self.FONT_DISPLAY, 24)).pack(side="left", padx=(10, 0))
        ctk.CTkLabel(header, text="large-v3 · CUDA", text_color=INK_3,
                     font=(self.FONT_MONO, 12)).pack(side="right")

        # faixa de comando: GRAVAR e o unico laranja da janela
        cmd = ctk.CTkFrame(root, fg_color="transparent")
        cmd.pack(fill="x", padx=18, pady=(0, 10))
        self.record_btn = ctk.CTkButton(
            cmd, text="GRAVAR", command=self.toggle, state="disabled",
            width=132, height=40, corner_radius=10,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            text_color=GRAPHITE, text_color_disabled=GRAPHITE,
            font=(self.FONT_DISPLAY, 17))
        self.record_btn.pack(side="left")
        for label, cb in (("Copiar tudo", self.copy_all), ("Limpar", self.clear)):
            ctk.CTkButton(cmd, text=label, command=cb, width=104, height=40,
                          corner_radius=10, fg_color="transparent", hover_color=SURFACE_2,
                          border_width=1, border_color=BORDER_STRONG, text_color=INK_2,
                          font=("Segoe UI", 13)).pack(side="left", padx=(8, 0))

        # card de configuracao: grade 3 colunas, rotulos caixa alta discretos
        card = ctk.CTkFrame(root, fg_color=SURFACE, corner_radius=12)
        card.pack(fill="x", padx=18, pady=(0, 10))
        for col in range(3):
            card.grid_columnconfigure(col, weight=1, uniform="cfg")

        def cfg_label(text, r, c):
            ctk.CTkLabel(card, text=text, text_color=INK_3, anchor="w", height=14,
                         font=("Segoe UI", 10, "bold")).grid(
                row=r, column=c, sticky="ew", padx=14, pady=((14, 0) if r == 0 else (10, 0)))

        def combo(values, current, command, r, c, bottom=0):
            box = ctk.CTkComboBox(card, values=values, command=command, state="readonly",
                                  height=30, corner_radius=8,
                                  fg_color=SURFACE_2, border_color=BORDER,
                                  button_color=SURFACE_2, button_hover_color=SURFACE_3,
                                  dropdown_fg_color=SURFACE_2, dropdown_hover_color=SURFACE_3,
                                  dropdown_text_color=INK, text_color=INK,
                                  font=("Segoe UI", 12))
            box.set(current)
            box.grid(row=r, column=c, sticky="ew", padx=14, pady=(4, bottom))
            return box

        cfg_label("ATALHO DO MOUSE", 0, 0)
        cfg_label("ACAO", 0, 1)
        cfg_label("MICROFONE", 0, 2)
        hk = ctk.CTkFrame(card, fg_color="transparent")
        hk.grid(row=1, column=0, sticky="ew", padx=14, pady=(4, 0))
        hk.grid_columnconfigure(0, weight=1)
        self.hotkey_var = tk.StringVar(value=BUTTON_LABELS[self.settings["mouse_button"]])
        ctk.CTkEntry(hk, textvariable=self.hotkey_var, state="readonly", height=30,
                     corner_radius=8, fg_color=SURFACE_2, border_color=BORDER,
                     text_color=INK, font=("Segoe UI", 12)).grid(row=0, column=0, sticky="ew")
        self.set_hotkey_btn = ctk.CTkButton(
            hk, text="Setar", command=self.capture_hotkey, width=56, height=30,
            corner_radius=8, fg_color="transparent", hover_color=SURFACE_2,
            border_width=1, border_color=BORDER_STRONG, text_color=INK_2,
            font=("Segoe UI", 12))
        self.set_hotkey_btn.grid(row=0, column=1, padx=(6, 0))
        self.trigger = combo(["alternar", "segurar"], self.settings["trigger_mode"],
                             self._on_trigger, 1, 1)
        names = list(self.devices)
        saved = self.settings["device_name"]
        self.mic = combo(names, saved if saved in self.devices else (names[0] if names else ""),
                         self._on_mic, 1, 2)
        cfg_label("TRANSCRICAO", 2, 0)
        cfg_label("ENVIO", 2, 1)
        cfg_label("IDIOMA", 2, 2)
        self.mode = combo(["simultaneo", "final"], self.settings["transcribe_mode"],
                          self._on_mode, 3, 0, bottom=14)
        self.inject = combo(["colar", "digitar"], self.settings["inject_method"],
                            self._on_inject, 3, 1, bottom=14)
        self.lang = combo(["pt", "en", "auto"], self.settings["language"],
                          self._on_lang, 3, 2, bottom=14)

        # 3a linha: fonte de captura (mic / audio do PC / os dois) + canal do PC
        cfg_label("FONTE", 4, 0)
        cfg_label("CANAL DO PC", 4, 1)
        self.fonte = combo(["microfone", "audio do PC", "os dois"],
                           CAPTURE_LABELS[self.settings["capture_mode"]],
                           self._on_fonte, 5, 0, bottom=14)
        pc_names = ["padrao do sistema"] + list(self.loopback_devices)
        saved_pc = self.settings["loopback_device_name"]
        if saved_pc not in self.loopback_devices:
            saved_pc = None
        self.pc_channel = combo(pc_names, "padrao do sistema" if saved_pc is None else saved_pc,
                                self._on_pc_channel, 5, 1, bottom=14)
        self.pc_channel.configure(
            state="disabled" if self.settings["capture_mode"] == "microfone" else "readonly")
        ctk.CTkLabel(card, text="", height=14).grid(row=4, column=2,
                                                    padx=14, pady=(10, 0))

        # abas: acento fica no GRAVAR; aba ativa marca por chapa mais clara
        tabbar = ctk.CTkFrame(root, fg_color="transparent")
        tabbar.pack(fill="x", padx=18, pady=(0, 6))
        self.tab_btns = {}
        for name, label in (("historico", "HISTORICO"), ("aovivo", "AO VIVO"),
                            ("biblioteca", "BIBLIOTECA"), ("estatisticas", "ESTATISTICAS")):
            btn = ctk.CTkButton(tabbar, text=label, width=118, height=30, corner_radius=8,
                                fg_color="transparent", hover_color=SURFACE_2,
                                text_color=INK_3, font=(self.FONT_DISPLAY, 14),
                                command=lambda n=name: self._show_tab(n))
            btn.pack(side="left", padx=(0, 6))
            self.tab_btns[name] = btn

        # status empacotado antes do conteudo (side=bottom) pra nunca ser espremido pra fora
        status_bar = ctk.CTkFrame(root, fg_color="transparent")
        status_bar.pack(side="bottom", fill="x", padx=18, pady=(2, 8))
        self.status = ctk.CTkLabel(status_bar, text="Iniciando...", text_color=INK_3,
                                   anchor="w", font=(self.FONT_MONO, 11))
        self.status.pack(side="left")
        ctk.CTkLabel(status_bar, text="barra: arraste pelo meio para reposicionar", text_color=INK_3,
                     font=("Segoe UI", 10)).pack(side="right")

        # card de conteudo: historico / ao vivo
        self.content = ctk.CTkFrame(root, fg_color=SURFACE, corner_radius=12)
        self.content.pack(fill="both", expand=True, padx=18, pady=(0, 4))
        self.hist_frame = ctk.CTkScrollableFrame(self.content, fg_color="transparent")
        self.text = ctk.CTkTextbox(self.content, fg_color="transparent", text_color=INK,
                                   font=("Segoe UI", 13), wrap="word", border_width=0)
        self.library = self.transcriber.library
        self.lib_tab = self._build_library_tab()
        self.stats_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        self._stats_dirty = True   # so desenha quando a aba abrir
        self._geom_antes = None    # geometria de fora da aba ESTATISTICAS
        self._tabs = {"historico": self.hist_frame, "aovivo": self.text,
                      "biblioteca": self.lib_tab, "estatisticas": self.stats_frame}
        self._tab = None
        self._show_tab("historico")

        self.entries = self._load_history()  # mais recente primeiro
        self._render_history()
        self._render_library()

        self.bar = RecorderBar(root, lambda: self.settings["dot_pos"], self._save_bar_pos,
                               get_levels=lambda: self.transcriber.levels,
                               on_cancel=self._cancel, on_confirm=self._stop)
        self._playing = None

        threading.Thread(target=self._load_model, daemon=True).start()
        root.after(100, self._poll)

    # -- abas / historico ----------------------------------------------------
    def _show_tab(self, name: str):
        if name == "estatisticas":
            # os graficos custam alguns ms; so valem com alguem olhando pra eles
            if self._stats_dirty:
                self._render_stats()
            if self._geom_antes is None:
                self._geom_antes = self.root.geometry()
        elif self._geom_antes is not None:  # saindo da aba: devolve o tamanho de antes
            self.root.geometry(self._geom_antes)
            self._geom_antes = None
        if name == self._tab:
            return
        self._tab = name
        for n, btn in self.tab_btns.items():
            if n == name:
                btn.configure(fg_color=SURFACE_3, text_color=INK, hover_color=SURFACE_3)
            else:
                btn.configure(fg_color="transparent", text_color=INK_3, hover_color=SURFACE_2)
        for n, widget in self._tabs.items():
            if n != name:
                widget.pack_forget()
        self._tabs[name].pack(fill="both", expand=True, padx=6, pady=6)
        if name == "estatisticas":
            self._fit_stats_window()

    @staticmethod
    def _load_history():
        if not HISTORY_INDEX.exists():
            return []
        entries = []
        for line in HISTORY_INDEX.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entries.append(json.loads(line))
        return list(reversed(entries))  # mais recente primeiro

    @staticmethod
    def _day_label(dt: datetime) -> str:
        return "HOJE" if dt.date() == date.today() else dt.strftime("%d/%m/%Y")

    def _make_day_header(self, day: str):
        hdr = ctk.CTkFrame(self.hist_frame, fg_color="transparent")
        ctk.CTkLabel(hdr, text=day, text_color=INK_3, height=14,
                     font=("Segoe UI", 10, "bold")).pack(side="left")
        ctk.CTkFrame(hdr, fg_color=BORDER, height=1).pack(
            side="left", fill="x", expand=True, padx=(10, 0))
        return hdr

    def _make_row(self, entry: dict, dt: datetime, base: str):
        row = ctk.CTkFrame(self.hist_frame, fg_color=base, corner_radius=8)
        time_lbl = ctk.CTkLabel(row, text=dt.strftime("%H:%M"), text_color=INK_3,
                                width=44, font=(self.FONT_MONO, 12))
        time_lbl.pack(side="left", anchor="n", padx=(8, 6), pady=7)
        wav_path = HISTORY_DIR / entry["wav"]
        btns = ctk.CTkFrame(row, fg_color="transparent")
        btns.pack(side="right", anchor="n", padx=(6, 8), pady=6)
        for glyph, cb in (("▶", lambda p=str(wav_path): self._play(p)),
                          ("⧉", lambda t=entry["text"]: self._copy_entry(t))):
            ctk.CTkButton(btns, text=glyph, command=cb, width=30, height=26,
                          corner_radius=8, fg_color="transparent", hover_color=SURFACE_3,
                          border_width=1, border_color=BORDER_STRONG, text_color=INK_2,
                          font=("Segoe UI", 12)).pack(side="left", padx=(4, 0))
        text_lbl = ctk.CTkLabel(row, text=entry["text"], text_color=INK, wraplength=452,
                                justify="left", anchor="w", cursor="hand2",
                                font=("Segoe UI", 12))
        text_lbl.pack(side="left", fill="x", expand=True, pady=6)
        text_lbl.bind("<Button-1>", lambda _e, t=entry["text"]: self._copy_entry(t))
        enter = lambda _e, r=row: r.configure(fg_color=ROW_WASH)
        leave = lambda _e, r=row, b=base: r.configure(fg_color=b)
        for w in (row, time_lbl, text_lbl):
            w.bind("<Enter>", enter)
            w.bind("<Leave>", leave)
        return row

    def _render_history(self):
        """Reconstroi a lista inteira: so na abertura e ao mexer na Biblioteca.

        Com ~70 entradas isto custa segundos (medido: 3-6 s no thread do Tk), entao
        transcricao nova entra por _add_history, que cria uma linha so.
        """
        for child in self.hist_frame.winfo_children():
            child.destroy()
        self._hist_day = None      # dia do grupo do topo
        self._hist_anchor = None   # linha antes da qual a proxima entrada e empacotada
        self._hist_base = None     # cor da linha do topo, pra manter o zebrado
        prev_day = None
        i = 0
        for entry in self.entries[:HIST_RENDER_MAX]:
            dt = datetime.fromisoformat(entry["ts"])
            day = self._day_label(dt)
            if day != prev_day:
                self._make_day_header(day).pack(fill="x", padx=8, pady=(12, 4))
                prev_day = day
                i = 0
            base = "transparent" if i % 2 == 0 else ROW_EVEN
            i += 1
            row = self._make_row(entry, dt, base)
            row.pack(fill="x", padx=2, pady=1)
            if self._hist_anchor is None:  # primeira linha do grupo mais recente
                self._hist_day, self._hist_anchor, self._hist_base = day, row, base

    def _add_history(self, entry: dict):
        """Insere a transcricao recem-chegada no topo, sem reconstruir a lista."""
        self.entries.insert(0, entry)
        dt = datetime.fromisoformat(entry["ts"])
        day = self._day_label(dt)
        # pack_slaves da a ordem visual; winfo_children da a de criacao, que deixa de
        # bater com a tela logo na primeira insercao no topo
        filhos = self.hist_frame.pack_slaves()
        topo = filhos[0] if filhos else None
        if day != self._hist_day:  # primeira do dia: cabecalho novo acima de tudo
            hdr = self._make_day_header(day)
            if topo is None:
                hdr.pack(fill="x", padx=8, pady=(12, 4))
            else:
                hdr.pack(fill="x", padx=8, pady=(12, 4), before=topo)
            self._hist_day, self._hist_anchor, self._hist_base = day, topo, None
        # alterna com a linha que estava no topo; nao reestampa o grupo inteiro
        base = "transparent" if self._hist_base == ROW_EVEN else ROW_EVEN
        row = self._make_row(entry, dt, base)
        if self._hist_anchor is None:
            row.pack(fill="x", padx=2, pady=1)
        else:
            row.pack(fill="x", padx=2, pady=1, before=self._hist_anchor)
        self._hist_anchor, self._hist_base = row, base
        self._stats_dirty = True
        if self._tab == "estatisticas":
            self._render_stats()

    def _copy_entry(self, text: str):
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.status.configure(text="Transcricao copiada para a area de transferencia.")

    def _play(self, path: str):
        if self._playing == path:
            winsound.PlaySound(None, 0)
            self._playing = None
            return
        winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        self._playing = path

    # -- estatisticas --------------------------------------------------------
    def _fit_stats_window(self):
        """Cresce a janela ate a aba caber inteira, sem barra de rolagem.

        Numero escondido atras de scroll nao serve pra nada, entao quem se ajusta e a
        janela: mede o que os widgets pedem, soma a diferenca e limita a area util do
        monitor (puxando a janela de volta pra dentro se ela passar da borda).
        """
        self.root.update_idletasks()
        falta_w = self.stats_frame.winfo_reqwidth() - self.stats_frame.winfo_width()
        falta_h = self.stats_frame.winfo_reqheight() - self.stats_frame.winfo_height()
        if falta_w <= 0 and falta_h <= 0:
            return
        atual = re.match(r"(\d+)x(\d+)\+(-?\d+)\+(-?\d+)", self.root.geometry())
        if not atual:  # janela ainda sem geometria resolvida: tenta no proximo ciclo
            return
        largura, altura, x, y = (int(v) for v in atual.groups())
        esq, topo, dir_, base = monitor_work_area(x, y)
        largura = min(largura + max(0, falta_w), dir_ - esq)
        altura = min(altura + max(0, falta_h), base - topo)
        x = max(esq, min(x, dir_ - largura))
        y = max(topo, min(y, base - altura))
        self.root.geometry(f"{largura}x{altura}+{x}+{y}")

    def _stat_card(self, parent, rotulo: str, valor: str, nota: str, row: int, col: int):
        card = ctk.CTkFrame(parent, fg_color=SURFACE_2, corner_radius=10)
        card.grid(row=row, column=col, sticky="nsew",
                  padx=(0 if col == 0 else 8, 0), pady=(0, 8))
        ctk.CTkLabel(card, text=rotulo, text_color=INK_3, anchor="w", height=12,
                     font=("Segoe UI", 9, "bold")).pack(fill="x", padx=12, pady=(10, 0))
        ctk.CTkLabel(card, text=valor, text_color=INK, anchor="w",
                     font=(self.FONT_DISPLAY, 25)).pack(fill="x", padx=12, pady=(1, 0))
        ctk.CTkLabel(card, text=nota, text_color=INK_3, anchor="w", height=12,
                     font=("Segoe UI", 10)).pack(fill="x", padx=12, pady=(0, 10))

    def _stats_titulo(self, parent, texto: str, nota: str = ""):
        head = ctk.CTkFrame(parent, fg_color="transparent")
        head.pack(fill="x", padx=12, pady=(10, 2))
        ctk.CTkLabel(head, text=texto, text_color=INK_3, height=12,
                     font=("Segoe UI", 9, "bold")).pack(side="left")
        if nota:
            ctk.CTkLabel(head, text=nota, text_color=INK_3, height=12,
                         font=("Segoe UI", 10)).pack(side="right")

    def _render_stats(self):
        """Redesenha a aba inteira a partir do historico em memoria."""
        for child in self.stats_frame.winfo_children():
            child.destroy()
        self._stats_dirty = False
        if not self.entries:
            ctk.CTkLabel(self.stats_frame, text="Nenhum ditado ainda — grave alguma coisa "
                         "e os numeros aparecem aqui.", text_color=INK_3,
                         font=("Segoe UI", 12)).pack(anchor="w", padx=10, pady=10)
            return
        st = compute_stats(self.entries)

        grade = ctk.CTkFrame(self.stats_frame, fg_color="transparent")
        grade.pack(fill="x")
        media = st["segundos"] / st["ditados"]
        cards = (
            ("PALAVRAS DITADAS", fmt_int(st["palavras"]),
             f"maior: {fmt_int(st['maior_ditado'])} palavras"),
            ("PALAVRAS POR MINUTO", fmt_int(st["ppm"]),
             f"melhor: {fmt_int(st['melhor_ppm'])} ppm"),
            ("SEQUENCIA", fmt_dias(st["streak"]), f"recorde: {fmt_dias(st['recorde'])}"),
            ("TEMPO FALADO", fmt_dur(st["segundos"]),
             f"{fmt_int(st['ditados'])} ditados, media de {fmt_dur(media)}"),
            ("ECONOMIA VS DIGITAR", fmt_dur(st["economia"]),
             f"teclado a {TYPING_WPM} palavras/min"),
            ("CORRECOES DA BIBLIOTECA", fmt_int(st["correcoes"]),
             f"{len(self.library.entries)} termos na lista" if st["correcoes"]
             else "comeca a contar nesta versao"),
        )
        for col in range(len(cards)):
            grade.grid_columnconfigure(col, weight=1, uniform="stat")
        for i, (rotulo, valor, nota) in enumerate(cards):
            self._stat_card(grade, rotulo, valor, nota, 0, i)

        # mapa, horas e palavras dividem a linha de baixo: cresce pro lado, nao pra baixo
        baixo = ctk.CTkFrame(self.stats_frame, fg_color="transparent")
        baixo.pack(fill="both", expand=True)
        baixo.grid_rowconfigure(0, weight=1)
        baixo.grid_columnconfigure(2, weight=1)  # a lista de palavras pega a sobra
        mapa = ctk.CTkFrame(baixo, fg_color=SURFACE_2, corner_radius=10)
        mapa.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        self._stats_titulo(mapa, "ATIVIDADE", f"{fmt_dias(len(st['por_dia']))} com ditado")
        img, tam = render_heatmap(st["por_dia"])
        self._heat_img = ctk.CTkImage(img, size=tam)
        ctk.CTkLabel(mapa, image=self._heat_img, text="").pack(anchor="w", padx=12)
        legenda = ctk.CTkFrame(mapa, fg_color="transparent")
        legenda.pack(anchor="w", padx=12, pady=(4, 12))
        img_leg, tam_leg = render_legend()
        self._leg_img = ctk.CTkImage(img_leg, size=tam_leg)
        ctk.CTkLabel(legenda, text="menos", text_color=INK_3, height=12,
                     font=("Segoe UI", 10)).pack(side="left", padx=(26, 6))
        ctk.CTkLabel(legenda, image=self._leg_img, text="").pack(side="left")
        ctk.CTkLabel(legenda, text="mais", text_color=INK_3, height=12,
                     font=("Segoe UI", 10)).pack(side="left", padx=(6, 0))

        horas = ctk.CTkFrame(baixo, fg_color=SURFACE_2, corner_radius=10)
        horas.grid(row=0, column=1, sticky="nsew", padx=(8, 0), pady=(0, 8))
        pico = max(st["por_hora"], key=st["por_hora"].get)
        self._stats_titulo(horas, "POR HORA DO DIA", f"pico as {pico}h")
        img_h, tam_h = render_hours(st["por_hora"])
        self._hours_img = ctk.CTkImage(img_h, size=tam_h)
        ctk.CTkLabel(horas, image=self._hours_img, text="").pack(padx=12, pady=(2, 12))

        top = ctk.CTkFrame(baixo, fg_color=SURFACE_2, corner_radius=10)
        top.grid(row=0, column=2, sticky="nsew", padx=(8, 0), pady=(0, 8))
        self._stats_titulo(top, "PALAVRAS MAIS DITADAS")  # sem nota: a coluna e estreita
        for palavra, vezes in st["top"]:
            linha = ctk.CTkFrame(top, fg_color="transparent")
            linha.pack(fill="x", padx=12)
            ctk.CTkLabel(linha, text=palavra, text_color=INK_2, anchor="w", height=15,
                         font=("Segoe UI", 12)).pack(side="left")
            ctk.CTkLabel(linha, text=fmt_int(vezes), text_color=INK_3, anchor="e", height=15,
                         font=(self.FONT_MONO, 11)).pack(side="right")
        ctk.CTkLabel(top, text="", height=6).pack()
        if self._tab == "estatisticas":  # ditado novo chegou com a aba aberta
            self._fit_stats_window()

    # -- biblioteca ----------------------------------------------------------
    def _build_library_tab(self):
        wrap = ctk.CTkFrame(self.content, fg_color="transparent")
        form = ctk.CTkFrame(wrap, fg_color="transparent")
        form.pack(fill="x", padx=2, pady=(2, 8))
        form.grid_columnconfigure(0, weight=3)
        form.grid_columnconfigure(1, weight=2)
        ctk.CTkLabel(form, text="SAI ASSIM (separe variantes por virgula)", text_color=INK_3,
                     anchor="w", height=14, font=("Segoe UI", 10, "bold")).grid(
            row=0, column=0, sticky="ew", padx=(6, 0))
        ctk.CTkLabel(form, text="DEVE VIRAR", text_color=INK_3, anchor="w", height=14,
                     font=("Segoe UI", 10, "bold")).grid(row=0, column=1, sticky="ew", padx=(8, 0))

        def field(placeholder, col):
            e = ctk.CTkEntry(form, placeholder_text=placeholder, height=30, corner_radius=8,
                             fg_color=SURFACE_2, border_color=BORDER, text_color=INK,
                             placeholder_text_color=INK_3, font=("Segoe UI", 12))
            e.grid(row=1, column=col, sticky="ew", padx=(6 if col == 0 else 8, 0), pady=(4, 0))
            e.bind("<Return>", lambda _ev: self._lib_add())
            return e

        self.lib_wrong = field("grock, groque, grote", 0)
        self.lib_right = field("Grok", 1)
        ctk.CTkButton(form, text="Adicionar", command=self._lib_add, width=92, height=30,
                      corner_radius=8, fg_color="transparent", hover_color=SURFACE_2,
                      border_width=1, border_color=BORDER_STRONG, text_color=INK_2,
                      font=("Segoe UI", 12)).grid(row=1, column=2, padx=(8, 6), pady=(4, 0))
        self.lib_list = ctk.CTkScrollableFrame(wrap, fg_color="transparent")
        self.lib_list.pack(fill="both", expand=True)
        return wrap

    def _render_library(self):
        for child in self.lib_list.winfo_children():
            child.destroy()
        if not self.library.entries:
            ctk.CTkLabel(self.lib_list, text="Nenhuma palavra na biblioteca ainda.",
                         text_color=INK_3, font=("Segoe UI", 12)).pack(anchor="w", padx=10, pady=10)
            return
        for i, entry in enumerate(self.library.entries):
            base = "transparent" if i % 2 == 0 else ROW_EVEN
            row = ctk.CTkFrame(self.lib_list, fg_color=base, corner_radius=8)
            row.pack(fill="x", padx=2, pady=1)
            ctk.CTkButton(row, text="✕", command=lambda n=i: self._lib_remove(n),
                          width=30, height=26, corner_radius=8, fg_color="transparent",
                          hover_color=SURFACE_3, border_width=1, border_color=BORDER_STRONG,
                          text_color=INK_2, font=("Segoe UI", 12)).pack(
                side="right", anchor="n", padx=(6, 8), pady=6)
            ctk.CTkLabel(row, text=entry["certo"], text_color=INK, width=130, anchor="w",
                         font=("Segoe UI", 12, "bold")).pack(side="left", padx=(10, 6), pady=6)
            ctk.CTkLabel(row, text="⟵  " + ", ".join(entry["erros"]), text_color=INK_3,
                         anchor="w", justify="left", wraplength=380,
                         font=("Segoe UI", 12)).pack(side="left", fill="x", expand=True, pady=6)

    def _lib_add(self):
        certo = self.lib_right.get().strip()
        erros = [w.strip() for w in self.lib_wrong.get().split(",") if w.strip()]
        if not certo or not erros:
            self.status.configure(text="Biblioteca: preencha o que sai errado e o termo certo.")
            return
        novos = self.library.add(certo, erros)
        self.lib_wrong.delete(0, "end")
        self.lib_right.delete(0, "end")
        self._render_library()
        self._stats_dirty = True
        self.status.configure(
            text=f"Biblioteca: {novos} variante(s) viram \"{certo}\"."
            if novos else f"Biblioteca: essas variantes ja estavam em \"{certo}\".")

    def _lib_remove(self, index: int):
        certo = self.library.entries[index]["certo"]
        self.library.remove(index)
        self._render_library()
        self._stats_dirty = True
        self.status.configure(text=f"Biblioteca: \"{certo}\" removido.")

    # -- callbacks de configuracao ------------------------------------------
    def _save(self):
        save_settings(self.settings)

    def _on_lang(self, _e):
        self.settings["language"] = self.lang.get()
        self.transcriber.language = self.lang.get()
        self._save()

    def _on_trigger(self, _e):
        self.settings["trigger_mode"] = self.trigger.get()
        self.hotkey.trigger_mode = self.trigger.get()
        self._save()

    def _on_mic(self, _e):
        self.settings["device_name"] = self.mic.get()
        self._save()

    def _on_fonte(self, _e):
        mode = CAPTURE_VALUES[self.fonte.get()]
        self.settings["capture_mode"] = mode
        self.pc_channel.configure(
            state="disabled" if mode == "microfone" else "readonly")
        self._save()

    def _on_pc_channel(self, _e):
        name = self.pc_channel.get()
        self.settings["loopback_device_name"] = None if name == "padrao do sistema" else name
        self._save()

    def _on_mode(self, _e):
        self.settings["transcribe_mode"] = self.mode.get()
        self.transcriber.transcribe_mode = self.mode.get()
        self._save()

    def _on_inject(self, _e):
        self.settings["inject_method"] = self.inject.get()
        self.transcriber.inject_method = self.inject.get()
        self._save()

    def _save_bar_pos(self, pos):
        self.settings["dot_pos"] = pos
        self._save()
        self.status.configure(text="Posicao da barra salva.")

    def capture_hotkey(self):
        self.hotkey.capturing = True
        self.status.configure(text="Clique o botao do mouse desejado (meio ou laterais)...")

    # -- gravacao -----------------------------------------------------------
    def _device_index(self):
        return self.devices.get(self.mic.get())

    def _pc_device_index(self):
        name = self.settings["loopback_device_name"]
        if name not in self.loopback_devices:
            return None  # None + loopback = saida padrao do sistema
        return self.loopback_devices[name]

    def _start(self, inject: bool):
        if self.transcriber.model is None:
            self.status.configure(text="Modelo ainda carregando — aguarde.")
            return False
        try:
            self.transcriber.start(
                self._device_index(), inject,
                capture_mode=self.settings["capture_mode"],
                loopback_index=self._pc_device_index(),
            )
        except Exception as e:
            fonte = self.settings["capture_mode"]
            if fonte == "audio_pc":
                alvo = "o audio do PC"
            elif fonte == "os_dois":
                alvo = "o microfone ou o audio do PC"
            else:
                alvo = "o microfone"
            self.status.configure(text=f"ERRO ao abrir {alvo}: {e}")
            return False
        self.record_btn.configure(text="PARAR")
        self.bar.show("rec")
        return True

    def _stop(self):
        if not self.transcriber.recording.is_set():
            return  # ja parado (ex.: confirmou na barra e soltou o atalho depois)
        self.transcriber.stop()
        self.hotkey.active = False  # parar pela UI nao pode deixar o atalho invertido
        self.record_btn.configure(text="GRAVAR")
        self.bar.show("proc")

    def _cancel(self):
        """X da barra: joga a sessao fora — nao transcreve, nao cola, nao arquiva."""
        self.transcriber.cancel(from_processing=True)
        self.hotkey.active = False
        self.record_btn.configure(text="GRAVAR")
        self.bar.hide()

    def toggle(self):
        if self.transcriber.recording.is_set():
            self._stop()
        else:
            self._start(inject=False)

    def copy_all(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.text.get("1.0", "end-1c"))
        self.status.configure(text="Copiado para a area de transferencia.")

    def clear(self):
        self.text.delete("1.0", "end")

    # -- loop de UI ---------------------------------------------------------
    def _load_model(self):
        try:
            self.transcriber.load_model()
            self.record_btn.configure(state="normal")
        except Exception as e:
            self.status_queue.put(f"ERRO ao carregar modelo: {e}")

    def _poll(self):
        # primeiro de tudo: a barra so fica enquanto ha trabalho. Esperar o status certo
        # a punha pra sair depois de colar e de redesenhar o historico — e e isso que o
        # olho le como travamento.
        if self.bar.visivel() and not self.transcriber.busy():
            self.bar.hide()
        try:
            while True:
                event, payload = self.hotkey_queue.get_nowait()
                if event == "captured":
                    self.settings["mouse_button"] = payload
                    self.hotkey_var.set(BUTTON_LABELS[payload])
                    self._save()
                    self.status.configure(text=f"Atalho definido: {BUTTON_LABELS[payload]}.")
                elif event == "start":
                    if not self._start(inject=True):
                        self.hotkey.active = False  # falhou: nao deixa o estado do atalho preso
                elif event == "stop":
                    self._stop()
        except queue.Empty:
            pass
        try:
            while True:
                chunk = self.text_queue.get_nowait()
                self.text.insert("end", chunk + " ")
                self.text.see("end")
        except queue.Empty:
            pass
        try:
            while True:
                entry = self.transcriber.history_queue.get_nowait()
                self._add_history(entry)
        except queue.Empty:
            pass
        try:
            while True:
                msg = self.status_queue.get_nowait()
                self.status.configure(text=msg)
                if msg.startswith("ERRO"):
                    self.bar.hide()
        except queue.Empty:
            pass
        self.root.after(100, self._poll)


if __name__ == "__main__":
    # identidade propria na taskbar: sem isto o Windows agrupa sob o pythonw
    # generico e mostra o icone do Python em vez do sussurro.ico da janela
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Lucas.Sussurro")
    ctk.set_appearance_mode("dark")
    root = ctk.CTk()
    App(root)
    root.mainloop()
