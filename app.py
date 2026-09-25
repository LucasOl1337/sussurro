"""Sussurro — voice-to-text local: mic -> Silero VAD -> faster-whisper (CPU ou CUDA).

HUD Tkinter: gravar/parar, atalho global de mouse (digita onde o cursor estiver),
fonte de captura (microfone, audio do PC via loopback ou os dois misturados),
modo de transcricao (simultaneo por trecho ou tudo ao final).

Windows e Linux (X11/Pulse ou PipeWire). CPU ou GPU NVIDIA opcional.
"""

import sys
from sussurro_ipc import IPC_SOCK, cli as _cli, ipc_send
from sussurro_hypr import Hypr
import sussurro_devices as devmod

# O atalho sai ANTES de carregar interface, audio e bibliotecas CUDA.
if __name__ == "__main__" and _cli(sys.argv):
    raise SystemExit(0)

import collections
import gc
import ctypes
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
import queue
import re
import shutil
import socket
import subprocess
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
import traceback
import unicodedata
import wave
from datetime import date, datetime, timedelta
from pathlib import Path

IS_WIN = sys.platform == "win32"

if IS_WIN:
    import ctypes.wintypes as wintypes

# Sob pythonw nao existe stdout/stderr; sem streams, print/traceback matam thread calados.
if sys.stdout is None or sys.stderr is None:
    _log = open(Path(__file__).with_name("sussurro.log"), "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stdout or _log
    sys.stderr = sys.stderr or _log


from sussurro_cuda import _prepare_cuda_libs


_prepare_cuda_libs()

import customtkinter as ctk
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk
from sussurro_compare_ui import ComparisonPanel
if not IS_WIN:
    from sussurro_meeting_ui import MeetingPanel
import sounddevice as sd
from faster_whisper import WhisperModel
from faster_whisper.utils import download_model
from sussurro_audio import prepare_input_devices
from sussurro_hardware import (detect_hardware, execution_device_labels,
                               execution_hardware_note)
from sussurro_models import (MODEL_LABELS, DEFAULT_MODEL_SETTINGS,
                             normalize_model_settings, resolve_model_config, model_path)
import sussurro_parakeet
from sussurro_meeting import drop_hallucinations
from faster_whisper.vad import VadOptions, get_speech_timestamps
from pynput import keyboard, mouse

SAMPLE_RATE = 16000
BLOCK_SIZE = 1600 if IS_WIN else 320  # Linux: 20 ms; Windows: 100 ms
UI_POLL_MS = 20
LEVEL_HZ = 40      # barras de onda por segundo alimentadas pelo mixer
_CANCEL = object()  # sentinela: joga fora o que estava em voo

# Segmentacao (modo simultaneo): corta quando ha fala + >= TAIL_SILENCE_S de silencio.
TAIL_SILENCE_S = 0.7
# Pausa longa vira paragrafo. 0,7 s e respiracao — nao quebra o texto.
PARAGRAPH_SILENCE_S = 1.5
MAX_SEGMENT_S = 25.0
MAX_IDLE_BUFFER_S = 30.0
VAD_CHECK_EVERY_S = 0.3
VAD_OPTIONS = VadOptions(min_silence_duration_ms=400, speech_pad_ms=200)

# Clausula nova que o whisper capitaliza sem ponto. Texto intacto; so entra o ponto.
_DISCOURSE_START = (
    "Então", "Entao", "Depois", "Porque", "Agora", "Bom", "Olha",
    "Além", "Alem", "Indo", "Enfim", "Inclusive", "Porém", "Porem",
    "Portanto", "Beleza", "Respondendo",
)
_DISCOURSE_RE = re.compile(
    r"([A-Za-záéíóúãõâêôçàèìòùÁÉÍÓÚÃÕÂÊÔÇÀÈÌÒÙ0-9])\s+(?=(?:%s)\b)"
    % "|".join(_DISCOURSE_START)
)
# Quebra de linha antes da ancora falada; a ancora continua no texto.
_LIST_ANCHOR_RE = re.compile(
    r"(?<!\n)\s+(?=(?:Pergunta|Questão|Questao)\s+\d+|(?:Primeiro|Segundo|Terceiro)\b)"
)
_SENTENCE_END = frozenset(".!?:;…")

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
# Medidas locais, sem audio nem texto ditado, com rotacao de arquivos.
_perf_log = logging.getLogger("sussurro.performance")
_perf_log.setLevel(logging.INFO)
_perf_log.propagate = False
if not _perf_log.handlers:
    _handler = RotatingFileHandler(
        Path(__file__).with_name("sussurro-performance.log"),
        maxBytes=1_000_000, backupCount=2, encoding="utf-8",
    )
    _handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    _perf_log.addHandler(_handler)


def _perf(event: str, **fields):
    _perf_log.info(json.dumps({"event": event, **fields}, ensure_ascii=False))


def _is_wayland() -> bool:
    return (not IS_WIN) and bool(os.environ.get("WAYLAND_DISPLAY"))


HISTORY_DIR = Path(__file__).with_name("history")
HISTORY_INDEX = HISTORY_DIR / "history.jsonl"
HIST_RENDER_MAX = 200  # linhas desenhadas na aba HISTORICO (a memoria guarda tudo)
DEFAULT_SETTINGS = {
    **DEFAULT_MODEL_SETTINGS,
    "mouse_button": "x2",       # middle | x1 | x2
    "trigger_mode": "alternar",  # alternar (clique liga/desliga) | segurar (push-to-talk)
    "device_name": None,
    "capture_mode": "microfone",  # microfone | audio_pc | os_dois
    "loopback_device_name": None,  # None = padrao do sistema
    "transcribe_mode": "simultaneo",  # simultaneo | final
    "language": "pt",
    "inject_method": "colar",   # colar (clipboard + atalho do app focado) | digitar
    "dot_pos": [0.5, 0.94],     # posicao da bolinha, fracao da area util do monitor
    "devices": dict(devmod.DEFAULTS),  # gestos de dispositivos (aba OMARCHY, so Linux)
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
    settings.update(normalize_model_settings(settings))
    settings["devices"] = {**devmod.DEFAULTS, **(settings.get("devices") or {})}
    return settings


def save_settings(settings: dict):
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")


def _hostapi_index() -> int:
    """Host PortAudio do sistema: WASAPI no Windows; Pulse/ALSA/JACK no Linux."""
    apis = list(sd.query_hostapis())
    prefer = ("WASAPI",) if IS_WIN else ("PulseAudio", "Pulse", "ALSA", "JACK")
    for needle in prefer:
        for i, h in enumerate(apis):
            if needle.lower() in h["name"].lower():
                return i
    default = sd.default.hostapi
    if isinstance(default, int) and 0 <= default < len(apis):
        return default
    if not apis:
        raise RuntimeError("PortAudio nao achou nenhum host de audio")
    return 0


def list_input_devices() -> dict:
    """Nome nativo -> entrada com rotulo humano, sem quebrar preferencias salvas."""
    host = _hostapi_index()
    devices = [
        d for d in sd.query_devices()
        if d["max_input_channels"] > 0 and d["hostapi"] == host
    ]
    system_microphone = ""
    if not IS_WIN:
        system_microphone = devmod.pretty_source(devmod.default_source())
        if system_microphone == "—":
            system_microphone = ""
    return prepare_input_devices(devices, system_microphone)


def list_loopback_devices() -> dict:
    """Nome -> indice das fontes de audio do PC (saida WASAPI ou monitor Pulse/PipeWire)."""
    host = _hostapi_index()
    if IS_WIN:
        return {
            d["name"]: d["index"]
            for d in sd.query_devices()
            if d["max_output_channels"] > 0 and d["hostapi"] == host
        }
    found = {
        d["name"]: d["index"]
        for d in sd.query_devices()
        if d["max_input_channels"] > 0 and d["hostapi"] == host
        and re.search(r"monitor|loopback", d["name"], re.I)
    }
    if found:
        return found
    return {
        d["name"]: d["index"]
        for d in sd.query_devices()
        if d["max_output_channels"] > 0 and d["hostapi"] == host
    }


# -- clipboard, cursor, area util --------------------------------------------
if IS_WIN:
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
    _SKIP_FORMATS = {2, 3, 9, 14}  # CF_BITMAP, CF_METAFILEPICT, CF_PALETTE, CF_ENHMETAFILE
else:
    _u32 = _k32 = None
    _CLIP_READ = _CLIP_WRITE = None
    if shutil.which("wl-copy") and shutil.which("wl-paste"):
        _CLIP_READ, _CLIP_WRITE = ["wl-paste", "-n"], ["wl-copy"]
    elif shutil.which("xclip"):
        _CLIP_READ = ["xclip", "-selection", "clipboard", "-o"]
        _CLIP_WRITE = ["xclip", "-selection", "clipboard"]
    elif shutil.which("xsel"):
        _CLIP_READ = ["xsel", "--clipboard", "--output"]
        _CLIP_WRITE = ["xsel", "--clipboard", "--input"]


_HYPR = None


def _hypr() -> Hypr:
    """Cliente do socket do Hyprland (Omarchy); `available` e False fora dele."""
    global _HYPR
    if _HYPR is None:
        _HYPR = Hypr()
    return _HYPR


# Terminais cujo colar nativo e Ctrl+Shift+V. O Codex TUI trata Ctrl+V como
# colar imagem; se o terminal encaminha a tecla, aparece "Failed to paste image"
# mesmo com texto no clipboard.
_TERMINAL_PASTE_CLASSES = frozenset({
    "foot", "kitty", "alacritty", "ghostty", "com.mitchellh.ghostty",
    "org.wezfurlong.wezterm", "wezterm", "org.gnome.terminal",
    "gnome-terminal-server",
})
def paste_strategy(window_class: str | None) -> str:
    """Atalho de colagem do app focado: terminal | ctrl_v.

    O modo colar sempre usa o clipboard, inclusive no Codex/ChatGPT.
    Digitar e uma escolha separada nas configuracoes.
    """
    cls = (window_class or "").strip().lower()
    if not cls:
        return "ctrl_v"
    if cls in _TERMINAL_PASTE_CLASSES:
        return "terminal"
    return "ctrl_v"


def focused_window_class() -> str | None:
    if IS_WIN:
        return None
    h = _hypr()
    if not h.available:
        return None
    win = h.activewindow()
    if not isinstance(win, dict):
        return None
    return win.get("class") or win.get("initialClass")


def _ydotool_env() -> dict:
    env = os.environ.copy()
    if env.get("YDOTOOL_SOCKET"):
        return env
    runtime = Path(env.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}")
    for name in ("ponte-input.sock", "ydotool.socket", ".ydotool_socket"):
        sock = runtime / name
        if sock.exists():
            env["YDOTOOL_SOCKET"] = str(sock)
            break
    return env


# Keycodes evdev (linux/input-event-codes.h) usados pelo ydotool.
_KEY_LEFTCTRL, _KEY_LEFTSHIFT, _KEY_ENTER, _KEY_V = 29, 42, 28, 47


def _ydotool_keys(*events: str) -> bool:
    """Injeta teclas pelo ydotoold (uinput), nao pelo wtype.

    O wtype cria um teclado virtual Wayland com keymap proprio; com o fcitx5
    no meio, cada evento faz o Hyprland trocar o teclado ativo e reenviar o
    keymap para todos os clientes (Xwayland recompila via xkbcomp a cada um).
    Uma colagem virava ~30 broadcasts e travava a sessao. O device uinput do
    ydotoold usa o mesmo keymap do Hyprland, entao nao ha reenvio.
    """
    if IS_WIN or not shutil.which("ydotool"):
        return False
    env = _ydotool_env()
    if not env.get("YDOTOOL_SOCKET"):
        return False
    try:
        r = subprocess.run(
            ["ydotool", "key", "-d", "12", *events],
            timeout=2, check=False, capture_output=True, env=env,
        )
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def click_at_cursor() -> bool:
    """Clica no ponteiro para focar o campo (web/GTK) sem mover o mouse.

    No Wayland o wtype manda Ctrl+V para a janela com foco de teclado; em
    Chromium o campo so recebe a colagem se o input estiver focado. O click
    usa o ydotoold ja no ar (socket ponte-input). Falha silenciosa se o
    daemon nao estiver acessivel.
    """
    if IS_WIN or not shutil.which("ydotool"):
        return False
    try:
        result = subprocess.run(
            ["ydotool", "click", "0xC0"],
            timeout=1, check=False, capture_output=True, env=_ydotool_env(),
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def prepare_paste_target():
    """Garante janela + campo sob o mouse antes do Ctrl+V.

    Devolve a classe da janela alvo para escolher o atalho (terminal vs ctrl_v).
    """
    if IS_WIN:
        return None
    h = _hypr()
    if not h.available:
        return focused_window_class()
    win = h.focus_at_cursor()
    time.sleep(0.04)
    cls = None
    if isinstance(win, dict):
        cls = win.get("class") or win.get("initialClass")
    strategy = paste_strategy(cls or focused_window_class())
    if strategy == "ctrl_v":
        click_at_cursor()
        time.sleep(0.03)
    return cls or focused_window_class()


def monitor_work_area(x: int, y: int):
    """Area util do monitor que contem o ponto (x, y).

    Windows: MonitorFromPoint. Hyprland: socket (o Tk em XWayland so ve a uniao
    dos monitores, e por isso a barra caia sempre no monitor do meio). Outro X11:
    tela Tk inteira.
    """
    if IS_WIN:
        hmon = _u32.MonitorFromPoint(wintypes.POINT(x, y), 2)  # MONITOR_DEFAULTTONEAREST
        mi = MONITORINFO()
        mi.cbSize = ctypes.sizeof(MONITORINFO)
        _u32.GetMonitorInfoW(hmon, ctypes.byref(mi))
        r = mi.rcWork
        return r.left, r.top, r.right, r.bottom
    h = _hypr()
    if h.available:
        area = h.work_area_at(x, y)
        if area:
            return area
    root = tk._default_root
    if root is None:
        raise RuntimeError("monitor_work_area precisa da janela Tk")
    return 0, 0, root.winfo_screenwidth(), root.winfo_screenheight()


def cursor_pos():
    if IS_WIN:
        pt = wintypes.POINT()
        _u32.GetCursorPos(ctypes.byref(pt))
        return pt.x, pt.y
    h = _hypr()
    if h.available:  # o ponteiro do XWayland congela fora de janelas X
        pos = h.cursorpos()
        if pos:
            return pos
    root = tk._default_root
    if root is None:
        raise RuntimeError("cursor_pos precisa da janela Tk")
    return root.winfo_pointerx(), root.winfo_pointery()


def _open_clipboard(retries: int = 15) -> bool:
    for _ in range(retries):
        if _u32.OpenClipboard(None):
            return True
        time.sleep(0.02)
    return False


def backup_clipboard():
    """Windows: todos os formatos HGLOBAL. Linux: texto via wl-clipboard/xclip/xsel."""
    if not IS_WIN:
        if not _CLIP_READ:
            return None
        try:
            r = subprocess.run(_CLIP_READ, capture_output=True, timeout=1, check=False)
            return r.stdout if r.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None
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
    if data is None:
        return
    if not IS_WIN:
        if not _CLIP_WRITE:
            return
        blob = data if isinstance(data, (bytes, bytearray)) else str(data).encode("utf-8")
        try:
            subprocess.run(_CLIP_WRITE, input=blob, timeout=2, check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.TimeoutExpired):
            pass
        return
    if not _open_clipboard():
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
    if not IS_WIN:
        if not _CLIP_WRITE:
            return False
        try:
            # wl-copy/xclip deixam um filho servindo o clipboard. Capturar stdout/stderr
            # espera EOF desses filhos e causa timeout mesmo quando a copia funcionou.
            subprocess.run(
                _CLIP_WRITE, input=text.encode("utf-8"), timeout=2, check=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return True
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False
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

    Necessario porque WASAPI/ALSA/Pulse muitas vezes so abrem na taxa nativa (44.1/48 kHz).
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


def _fecha_frase(s: str) -> str:
    """Ponto no fim se ainda nao ha pontuacao; virgula nao vira ponto."""
    if not s or s[-1] in _SENTENCE_END or s[-1] in ",;":
        return s
    return s + "."


def load_audio_16k_mono(path: Path) -> "np.ndarray":
    """Decodifica qualquer audio (ogg/opus do WhatsApp, mp3, m4a, wav) em float32 mono 16 kHz.

    O ffmpeg faz a conversao porque o whisper aqui sempre recebe o mesmo formato do
    microfone; assim o arquivo entra pelo mesmo caminho do ditado.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg nao encontrado no PATH")
    proc = subprocess.run(
        [ffmpeg, "-nostdin", "-v", "error", "-i", str(path),
         "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
    if proc.returncode != 0:
        detalhe = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or ["falha"]
        raise RuntimeError(f"ffmpeg: {detalhe[0]}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def format_transcript(segments) -> str:
    """Pontua e quebra o texto do whisper sem reescrever.

    Pausa >= PARAGRAPH_SILENCE_S entre segmentos vira paragrafo. Segmento
    seguinte com maiuscula ganha ponto se o anterior nao termina frase.
    Depois, ponto antes de marcador de clausula e newline antes de ancora
    de lista falada (Pergunta N, Questao N, Primeiro/Segundo/Terceiro).
    """
    pieces = []
    prev_end = None
    for seg in segments:
        t = (seg.text or "").strip()
        if not t:
            continue
        start, end = seg.start, seg.end
        if not pieces:
            pieces.append(t)
            prev_end = end
            continue
        gap = start - prev_end if prev_end is not None else 0.0
        if gap >= PARAGRAPH_SILENCE_S:
            pieces[-1] = _fecha_frase(pieces[-1])
            pieces.append("\n\n" + t)
        else:
            if t[0].isupper() and pieces[-1][-1] not in _SENTENCE_END and pieces[-1][-1] not in ",;":
                pieces[-1] = _fecha_frase(pieces[-1])
            pieces.append(" " + t)
        prev_end = end
    text = "".join(pieces)
    if not text:
        return ""
    text = _DISCOURSE_RE.sub(r"\1. ", text)
    text = _LIST_ANCHOR_RE.sub("\n", text)
    return text


def _join_session_text(parts) -> str:
    """Junta trechos ja formatados. para=True (pausa longa) vira paragrafo."""
    out = []
    for item in parts:
        t = item[1]
        if not t:
            continue
        para = item[3] if len(item) > 3 else False
        if not out:
            out.append(t)
        elif para:
            out.append("\n\n" + t)
        else:
            out.append(" " + t)
    return "".join(out).strip()


def _open_loopback_mic(device_index: int | None, status_queue, label: str):
    """Microfone virtual que escuta o que o PC toca. WASAPI no Windows, monitor no Linux."""
    import soundcard as sc
    if device_index is None:
        if IS_WIN:
            spk = sc.default_speaker()
            return sc.get_microphone(id=spk.id, include_loopback=True)
        mics = sc.all_microphones(include_loopback=True)
        spk = sc.default_speaker()
        for mic in mics:
            if spk.id in mic.id or (spk.name and spk.name in mic.name):
                return mic
        monitors = [m for m in mics if "monitor" in m.id.lower() or "monitor" in m.name.lower()]
        if monitors:
            return monitors[0]
        raise RuntimeError("nenhum monitor de audio do PC (PulseAudio/PipeWire)")
    name = sd.query_devices(device_index)["name"]
    try:
        spk = sc.get_speaker(name)
        return sc.get_microphone(id=spk.id, include_loopback=True)
    except Exception:
        try:
            return sc.get_microphone(name, include_loopback=True)
        except Exception:
            spk = sc.default_speaker()
            status_queue.put(f"{label}: canal '{name}' nao achado, usando padrao ({spk.name}).")
            if IS_WIN:
                return sc.get_microphone(id=spk.id, include_loopback=True)
            mics = sc.all_microphones(include_loopback=True)
            for mic in mics:
                if spk.id in mic.id or (spk.name and spk.name in mic.name):
                    return mic
            raise RuntimeError(f"canal '{name}' nao achado e sem monitor padrao")


class Transcriber:
    """Threads de captura, segmentacao (VAD) e transcricao. UI le da text_queue."""

    def __init__(self, text_queue: queue.Queue, status_queue: queue.Queue):
        self.text_queue = text_queue
        self.status_queue = status_queue
        self.model = None
        self.model_config = None
        self.model_loading = threading.Event()
        self.comparing = threading.Event()
        self.library = Library()
        self.language = "pt"
        self.transcribe_mode = "simultaneo"
        self.inject_method = "colar"
        self.recording = threading.Event()
        self.history_queue: queue.Queue = queue.Queue()
        self._audio_queue: queue.Queue = queue.Queue()
        self._segment_queue: queue.Queue = queue.Queue()
        self._session_parts: list = []
        self._session_audio: list = []
        self._session_errors: list[str] = []
        self._session_started = None
        self._stop_requested_at = None
        self._streams: list = []
        self._resamplers: list = []   # um resampler por stream (taxa nativa != 16 kHz)
        self._slot = 0                # indice do stream sendo aberto em start()
        self._mix_lock = threading.Lock()
        self._model_lock = threading.Lock()  # serializa ditado e transcricao de arquivo (IPC)
        self._history_lock = threading.Lock()
        self._retrying = threading.Event()
        self._mix_buffers: list = []  # buffers por stream; o mixer alinha e soma
        self._session_inject = False
        self._session_mode = "simultaneo"
        self._session_had_speech = False
        self._session_emitted = False  # ja saiu texto nesta sessao (colar / ao vivo)
        self._session_auto_enter = False  # fone: aperta Enter depois da ultima colagem
        self._session_id = 0          # cancelar/reiniciar invalida o que ficou em voo
        self._file_jobs = 0
        self._meeting_jobs = 0        # reuniao salvando/transcrevendo: segura a troca de modelo
        self._pending = 0             # trechos aceitos e ainda nao entregues
        self._pending_lock = threading.Lock()
        self._drained = True          # todos os trechos foram entregues e arquivados
        # nivel RMS recente do audio ja misturado; a barra de overlay le daqui
        self.levels: collections.deque = collections.deque(maxlen=96)
        self._keyboard = keyboard.Controller()
        self._clipboard_lock = threading.RLock()
        threading.Thread(target=self._segmenter_loop, daemon=True).start()
        threading.Thread(target=self._transcribe_loop, daemon=True).start()
        threading.Thread(target=self._mixer_loop, daemon=True).start()

    # -- modelo -------------------------------------------------------------
    def load_model(self, settings=None):
        """Download first; replace weights under the same lock used by inference."""
        self.model_loading.set()
        try:
            with self._model_lock:
                with self._pending_lock:
                    if (self.comparing.is_set() or self._file_jobs or self._meeting_jobs or self._pending
                            or self.recording.is_set() or not self._drained or self._retrying.is_set()):
                        raise RuntimeError("Aguarde o trabalho atual terminar antes de trocar o modelo.")
                config = resolve_model_config(settings or DEFAULT_MODEL_SETTINGS)
                self.status_queue.put(f"Preparando {config.model}: primeiro uso pode baixar o modelo...")
                # Network failure leaves the current model usable.
                path = self._weights(config)
                previous = self.model_config
                if previous is not None and previous.engine == "parakeet" and self.model is not None:
                    self.model.close()  # sessoes do onnxruntime nao somem so com gc
                self.model = None
                self.model_config = None
                gc.collect()
                try:
                    self._load_model_config(config, path)
                except Exception:
                    # Do not hold two models in VRAM. Recover from the local cache only.
                    if previous is not None:
                        try:
                            self._load_model_config(previous, self._weights(previous, local_only=True))
                        except Exception:
                            self.model = None
                            self.model_config = None
                    raise
        finally:
            self.model_loading.clear()

    def _weights(self, config, local_only=False):
        if config.engine == "parakeet":
            return sussurro_parakeet.model_dir(download=not local_only, status=self.status_queue.put)
        if local_only:
            return download_model(config.model, local_files_only=True)
        return model_path(config.model, download_model)

    def _load_model_config(self, config, path):
        self.status_queue.put(f"Carregando {config.label}...")
        t0 = time.perf_counter()
        if config.engine == "parakeet":
            model = sussurro_parakeet.ParakeetModel(path)
        else:
            model = WhisperModel(path, device=config.device, compute_type=config.compute_type)
        try:
            silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
            segments, _ = model.transcribe(silence, language="pt", beam_size=5)
            collections.deque(segments, maxlen=0)
            get_speech_timestamps(silence, VAD_OPTIONS)
        except Exception:
            # Explicitly free failed weights before attempting to recover the old model.
            if config.engine == "parakeet":
                model.close()
            else:
                model.model.unload_model()
            raise
        self.model = model
        self.model_config = config
        _perf("model_ready", device=config.device, compute_type=config.compute_type,
              model=config.model, load_s=round(time.perf_counter() - t0, 3))
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
                if self.recording.is_set():
                    self._mix_buffers[slot].append(data)

        return cb

    def _mixer_loop(self):
        while True:
            time.sleep(0.01)
            with self._mix_lock:
                self._drain_mix_locked()

    def _drain_mix_locked(self, flush: bool = False):
        """Enfileira sob o lock: nenhum bloco ultrapassa o marcador de stop."""
        while self._mix_buffers:
            for buf in self._mix_buffers:
                while buf and not buf[0].size:
                    buf.pop(0)
            active = [buf for buf in self._mix_buffers if buf]
            if not active or (not flush and len(active) != len(self._mix_buffers)):
                return
            n = min(buf[0].size for buf in active)
            mixed = np.zeros(n, dtype=np.float32)
            for buf in active:
                mixed += buf[0][:n]
                buf[0] = buf[0][n:]
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
        """Captura o que o PC esta tocando (WASAPI loopback / monitor PulsePipeWire)."""
        try:
            loop = _open_loopback_mic(device_index, self.status_queue, label)
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
                            if self.recording.is_set():
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
        query = device_index if device_index is not None else sd.default.device[0]
        native = int(sd.query_devices(query)["default_samplerate"])
        # ALSA/USB costuma recusar 16 kHz. Abrir a taxa nativa evita uma tentativa
        # com erro em cada clique (o FIFINE desta maquina captura a 48 kHz).
        rates = dict.fromkeys((SAMPLE_RATE, native) if IS_WIN else (native, SAMPLE_RATE))
        for rate in rates:
            stream = None
            self._resamplers.append(StreamResampler(rate) if rate != SAMPLE_RATE else None)
            try:
                stream = sd.InputStream(
                    samplerate=rate, channels=1, dtype="float32",
                    blocksize=round(rate * BLOCK_SIZE / SAMPLE_RATE), device=device_index,
                    latency=None if IS_WIN else "low",
                    callback=self._make_callback(label, self._slot),
                )
                stream.start()
            except sd.PortAudioError:
                if stream is not None:
                    stream.close()
                self._resamplers.pop()
                if rate == list(rates)[-1]:
                    raise
            else:
                self._streams.append(stream)
                return

    def start(self, device_index: int | None, inject: bool,
              capture_mode: str = "microfone", loopback_index: int | None = None,
              auto_enter: bool = False):
        if self.comparing.is_set():
            raise RuntimeError("Comparacao de modelos em andamento.")
        if self.model_loading.is_set():
            raise RuntimeError("Aguarde a troca do modelo terminar.")
        if self.recording.is_set():
            return
        if self.busy():
            raise RuntimeError("Aguarde o ditado anterior terminar.")
        self._session_inject = inject
        self._session_auto_enter = auto_enter
        self._session_mode = self.transcribe_mode
        self._session_had_speech = False
        self._session_emitted = False
        self._session_started = datetime.now()
        self._stop_requested_at = None
        self._session_id += 1
        self._session_parts = []
        self._session_audio = []
        self._session_errors = []
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
        self._stop_requested_at = time.perf_counter()
        # status ANTES do sentinela: o segmentador emite os status finais depois dele,
        # e a ordem na fila e o que impede "Transcrevendo..." de ficar pendurado
        self.status_queue.put("Parado. Transcrevendo...")
        with self._mix_lock:
            self.recording.clear()
            self._drain_mix_locked(flush=True)
            self._audio_queue.put(None)  # depois do ultimo bloco, inclusive os do mixer
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
            pendentes = self._pending + self._file_jobs
        return (self.recording.is_set() or not self._drained or pendentes > 0
                or self._retrying.is_set() or self.model_loading.is_set()
                or self._model_lock.locked() or self.comparing.is_set())

    def acquire_comparison(self):
        with self._pending_lock:
            if (self.comparing.is_set() or self.recording.is_set() or not self._drained
                    or self._pending or self._file_jobs or self._meeting_jobs or self._retrying.is_set()
                    or self.model_loading.is_set() or self._model_lock.locked()):
                raise RuntimeError("Aguarde o ditado, arquivo, reuniao ou carregamento atual terminar.")
            self.comparing.set()

    def begin_meeting_job(self):
        """A reuniao usa o modelo do ditado (sob o _model_lock so no whisper); a troca de
        modelo e a comparacao esperam, o ditado segue entre um lado e outro."""
        with self._pending_lock:
            if self.model is None or self.model_loading.is_set():
                raise RuntimeError("O modelo ainda esta carregando.")
            if self.comparing.is_set():
                raise RuntimeError("Comparacao de modelos em andamento.")
            self._meeting_jobs += 1

    def end_meeting_job(self):
        with self._pending_lock:
            self._meeting_jobs -= 1

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
        self._session_audio = []
        self._session_errors = []
        self.levels.clear()
        with self._pending_lock:
            self._pending = 0
        self._drained = True
        self._audio_queue.put(_CANCEL)  # o segmentador larga o buffer que sobrou
        self.status_queue.put("Cancelado.")

    # -- segmentacao --------------------------------------------------------
    def _segmenter_loop(self):
        buffer = np.zeros(0, dtype=np.float32)
        final_blocks = []
        last_check = 0.0
        while True:
            item = self._audio_queue.get()
            try:
                if item is _CANCEL:  # sessao cancelada: o buffer acumulado morre aqui
                    buffer = np.zeros(0, dtype=np.float32)
                    final_blocks.clear()
                    continue
                if item is None:  # fim da gravacao: manda o que sobrou
                    if self._session_mode == "final":
                        buffer = (np.concatenate(final_blocks) if final_blocks
                                  else np.zeros(0, dtype=np.float32))
                        final_blocks.clear()
                        # O Whisper ja aplica VAD neste modo; nao varrer o audio duas vezes.
                        speech = [{"start": 0}] if buffer.size > SAMPLE_RATE // 4 else []
                    else:
                        speech = (get_speech_timestamps(buffer, VAD_OPTIONS)
                                  if buffer.size > SAMPLE_RATE // 4 else [])
                    if speech:
                        lead_s = speech[0]["start"] / SAMPLE_RATE
                        self._enqueue_segment(buffer, lead_s)
                    elif not self._session_had_speech:
                        self.status_queue.put("Parado (sem fala detectada).")
                    else:
                        self.status_queue.put("Parado.")
                    # marcador de fim de sessao (carimbado: cancelamento o invalida)
                    self._segment_queue.put((None, None, None, self._session_id, 0.0))
                    buffer = np.zeros(0, dtype=np.float32)
                    continue
                if self._session_mode == "final":
                    final_blocks.append(item)
                    continue  # acumula tudo; transcreve de uma vez no stop
                buffer = np.concatenate([buffer, item])
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
                lead_s = speech[0]["start"] / SAMPLE_RATE
                tail_silence = (buffer.size - last_end) / SAMPLE_RATE
                if tail_silence >= TAIL_SILENCE_S:
                    self._enqueue_segment(buffer[:last_end], lead_s)
                    buffer = buffer[last_end:]
                elif buffer.size > MAX_SEGMENT_S * SAMPLE_RATE:
                    self._enqueue_segment(buffer, lead_s)
                    buffer = np.zeros(0, dtype=np.float32)
            except Exception as e:  # falha alto: reporta no status e mantem a thread viva
                traceback.print_exc()
                self.status_queue.put(f"ERRO na segmentacao: {e}")
                buffer = np.zeros(0, dtype=np.float32)
                final_blocks.clear()

    def _enqueue_segment(self, audio: np.ndarray, lead_s: float = 0.0):
        self._session_had_speech = True
        # O audio pertence ao historico mesmo se o Whisper falhar. Antes ele so
        # sobrevivia quando a transcricao chegava ate _session_parts.
        self._session_audio.append(audio)
        with self._pending_lock:
            self._pending += 1
        self._segment_queue.put((audio, self._session_inject, self._session_mode,
                                 self._session_id, lead_s))

    # -- transcricao --------------------------------------------------------
    def _transcribe_loop(self):
        while True:
            audio, inject, mode, sid, lead_s = self._segment_queue.get()
            # trecho real desta sessao: precisa dar baixa mesmo se a transcricao falhar
            deve_baixar = audio is not None and sid == self._session_id
            try:
                if sid != self._session_id:
                    continue  # sessao cancelada ou substituida: descarta sem colar nada
                if audio is None:  # fim de sessao: grava no historico
                    try:
                        # o sentinela chega com inject=None: usar o estado da sessao, nao o campo da fila
                        if (self._session_auto_enter and self._session_emitted
                                and self._session_inject and not self._session_errors):
                            self._press_enter()
                        self._session_auto_enter = False
                        self._finalize_session()
                    finally:
                        if sid == self._session_id:
                            self._drained = True
                    continue
                t0 = time.perf_counter()
                lang = None if self.language == "auto" else self.language
                segments, _info = self._transcribe_locked(
                    audio, language=lang, vad_filter=(mode == "final"))
                text = format_transcript(segments)
                text, fixes = self.library.apply(text)  # troca da Biblioteca antes de sair daqui
                if sid != self._session_id:
                    continue  # cancelado enquanto este trecho transcrevia
                dt = time.perf_counter() - t0
                para = self._session_emitted and lead_s >= PARAGRAPH_SILENCE_S
                self._session_parts.append((audio, text, fixes, para))
                if text:
                    if self._session_emitted:
                        payload = ("\n\n" if para else " ") + text
                    else:
                        payload = text
                    self.text_queue.put(payload)
                    self._session_emitted = True
                if text and inject:
                    if self.inject_method == "colar":
                        self._paste(payload)
                    else:
                        self._type_fallback(payload)
                _perf("segment_done", session=sid, audio_s=round(audio.size / SAMPLE_RATE, 3),
                      inference_ms=round(dt * 1000, 1),
                      delivery_ms=round((time.perf_counter() - t0 - dt) * 1000, 1),
                      stop_to_delivery_ms=(round((time.perf_counter() - self._stop_requested_at) * 1000, 1)
                                           if self._stop_requested_at is not None else None),
                      injected=bool(text and inject))
                state = "Gravando — pode falar." if self.recording.is_set() else "Parado."
                self.status_queue.put(f"{state}  (trecho de {audio.size / SAMPLE_RATE:.1f}s em {dt:.1f}s)")
            except Exception as e:  # falha alto: reporta no status e mantem a thread viva
                traceback.print_exc()
                if deve_baixar and sid == self._session_id:
                    self._session_errors.append(f"{type(e).__name__}: {e}")
                self.status_queue.put(f"ERRO na transcricao: {e}")
            finally:
                if deve_baixar and sid == self._session_id:
                    self._pending_done()

    def _transcribe_locked(self, audio, *, language, vad_filter):
        """Inferencia serializada: o ditado e a transcricao de arquivo (IPC) dividem um
        unico modelo na GPU, entao duas chamadas simultaneas competem pela mesma VRAM.

        hotwords enviesa a decodificacao pros termos da Biblioteca: e o que evita o
        whisper inventar "Nightingale" no lugar de "9router". Consome o gerador aqui
        dentro para a inferencia acontecer com o lock ainda tomado.
        """
        with self._model_lock:
            if self.model_loading.is_set() or self.model is None:
                raise RuntimeError("Modelo indisponivel ou sendo trocado. Tente novamente quando estiver pronto.")
            segments, info = self.model.transcribe(
                audio, language=language, beam_size=5, vad_filter=vad_filter,
                hotwords=self.library.hotwords,
            )
            return drop_hallucinations(list(segments), audio), info

    def transcribe_file(self, path: str) -> dict:
        """Transcreve um arquivo de audio e arquiva no historico, sem microfone nem colagem.

        Usado pelo IPC (`sussurro transcribe <arquivo>`), que e como o Hermes manda os
        audios do WhatsApp. Retorna o mesmo dict que vai pro history.jsonl.
        """
        with self._pending_lock:
            if self.model_loading.is_set() or self.comparing.is_set():
                raise RuntimeError("Modelo sendo trocado ou comparacao em andamento; aguarde.")
            self._file_jobs += 1
        try:
            if self.model is None:
                raise RuntimeError("modelo ainda nao esta pronto")
            src = Path(path).expanduser()
            if not src.is_file():
                raise FileNotFoundError(str(src))
            audio = load_audio_16k_mono(src)
            if audio.size == 0:
                raise ValueError("audio vazio")
            t0 = time.perf_counter()
            lang = None if self.language == "auto" else self.language
            segments, _info = self._transcribe_locked(audio, language=lang, vad_filter=True)
            text = format_transcript(segments)
            text, fixes = self.library.apply(text)
            if not text:
                raise ValueError("nenhuma fala reconhecida")
            entry = self._archive_audio(audio, text, fixes, datetime.now())
            _perf("file_done", source=src.name, audio_s=round(audio.size / SAMPLE_RATE, 3),
                  inference_ms=round((time.perf_counter() - t0) * 1000, 1))
            self.status_queue.put(f"Arquivo transcrito ({audio.size / SAMPLE_RATE:.1f}s).")
            return entry
        finally:
            with self._pending_lock:
                self._file_jobs -= 1


    def _archive_audio(self, audio, text: str, fixes: int, started: datetime, *,
                       failed: bool = False, error: str | None = None) -> dict:
        """Grava o WAV e seu registro, inclusive quando a transcricao falhou."""
        HISTORY_DIR.mkdir(exist_ok=True)
        wav_name = started.strftime("%Y%m%d_%H%M%S") + ".wav"
        with wave.open(str(HISTORY_DIR / wav_name), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
        # dur e fix alimentam a aba ESTATISTICAS; entrada antiga sem eles usa o wav / zero
        entry = {"ts": started.isoformat(timespec="seconds"), "wav": wav_name,
                 "text": text, "dur": round(audio.size / SAMPLE_RATE, 2), "fix": fixes}
        if failed:
            entry["failed"] = True
            entry["error"] = error or "Nenhuma fala reconhecida."
        with self._history_lock:
            with HISTORY_INDEX.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.history_queue.put(entry)
        return entry

    def _finalize_session(self):
        parts, self._session_parts = self._session_parts, []
        audio_parts, self._session_audio = self._session_audio, []
        errors, self._session_errors = self._session_errors, []
        if not audio_parts:
            return
        text = _join_session_text(parts)
        started = self._session_started or datetime.now()
        audio = np.concatenate(audio_parts)
        fixes = sum(f for _a, _t, f, *_ in parts)
        failed = bool(errors) or not text
        error = errors[-1] if errors else ("Nenhuma fala reconhecida." if not text else None)
        self._archive_audio(audio, text, fixes, started, failed=failed, error=error)

    def _replace_history_entry(self, entry: dict) -> None:
        """Atualiza uma linha pelo nome do WAV sem arriscar truncar o historico."""
        with self._history_lock:
            lines = []
            replaced = False
            if HISTORY_INDEX.exists():
                for raw in HISTORY_INDEX.read_text(encoding="utf-8").splitlines():
                    if not raw.strip():
                        continue
                    current = json.loads(raw)
                    if current.get("wav") == entry.get("wav"):
                        current = entry
                        replaced = True
                    lines.append(json.dumps(current, ensure_ascii=False))
            if not replaced:
                lines.append(json.dumps(entry, ensure_ascii=False))
            temporary = HISTORY_INDEX.with_suffix(".jsonl.tmp")
            temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
            os.replace(temporary, HISTORY_INDEX)

    def retry_history_async(self, entry: dict) -> None:
        """Tenta novamente um WAV falho e atualiza o registro existente em background."""
        if self.model is None:
            raise RuntimeError("Modelo ainda carregando — aguarde.")
        if self.busy():
            raise RuntimeError("Aguarde o trabalho atual terminar.")
        wav_name = str(entry.get("wav", ""))
        if not wav_name or Path(wav_name).name != wav_name:
            raise ValueError("arquivo de historico invalido")
        path = HISTORY_DIR / wav_name
        if not path.is_file():
            raise FileNotFoundError(str(path))
        self._retrying.set()
        self.status_queue.put("Tentando transcrever o audio novamente...")
        threading.Thread(target=self._retry_history_worker, args=(dict(entry), path),
                         name="sussurro-history-retry", daemon=True).start()

    def _retry_history_worker(self, entry: dict, path: Path) -> None:
        try:
            audio = load_audio_16k_mono(path)
            if audio.size == 0:
                raise ValueError("audio vazio")
            started = time.perf_counter()
            lang = None if self.language == "auto" else self.language
            segments, _info = self._transcribe_locked(audio, language=lang, vad_filter=True)
            text = format_transcript(segments)
            text, fixes = self.library.apply(text)
            if not text:
                raise ValueError("nenhuma fala reconhecida")
            updated = {**entry, "text": text, "dur": round(audio.size / SAMPLE_RATE, 2),
                       "fix": fixes}
            updated.pop("failed", None)
            updated.pop("error", None)
            updated.pop("retrying", None)
            self._replace_history_entry(updated)
            self.history_queue.put(updated)
            _perf("history_retry_done", wav=path.name,
                  audio_s=round(audio.size / SAMPLE_RATE, 3),
                  inference_ms=round((time.perf_counter() - started) * 1000, 1))
            self.status_queue.put("Audio recuperado e transcrito.")
        except Exception as e:
            traceback.print_exc()
            failed = {**entry, "failed": True, "error": f"{type(e).__name__}: {e}"}
            failed.pop("retrying", None)
            try:
                self._replace_history_entry(failed)
                self.history_queue.put(failed)
            except Exception:
                traceback.print_exc()
            self.status_queue.put(f"ERRO ao tentar novamente: {e}")
        finally:
            self._retrying.clear()

    def _paste(self, text: str):
        """Cola no app focado; no Linux deixa o ditado no clipboard para recolar."""
        if not IS_WIN:
            return self._paste_linux(text)
        backup = backup_clipboard()
        if not set_clipboard_text(text):
            self._type_fallback(text)
            return
        time.sleep(0.05)
        if self._send_paste_key():
            time.sleep(0.4)
            restore_clipboard(backup)
            return
        with self._keyboard.pressed(keyboard.Key.ctrl):
            self._keyboard.press("v")
            self._keyboard.release("v")
        time.sleep(0.4)  # o app alvo precisa ler o clipboard antes da restauracao
        restore_clipboard(backup)

    def _paste_linux(self, text: str):
        with self._clipboard_lock:
            target_class = prepare_paste_target()
            strategy = paste_strategy(target_class)
            if not set_clipboard_text(text):
                if _is_wayland() and _hypr().available:
                    raise RuntimeError("Falha no clipboard; texto preservado no historico para copiar novamente.")
                self._type_fallback(text)
                return
            time.sleep(0.05)
            if not self._send_paste_key(strategy):
                if _is_wayland():
                    raise RuntimeError("Falha ao enviar o atalho de colagem; texto preservado no historico.")
                modifiers = [keyboard.Key.ctrl]
                if strategy == "terminal":
                    modifiers.append(keyboard.Key.shift)
                with self._keyboard.pressed(*modifiers):
                    self._keyboard.press("v")
                    self._keyboard.release("v")
            _perf("paste_dispatched", target_class=target_class, strategy=strategy)
            # Wayland does not acknowledge that the target consumed this paste.
            # Do not race a slow client by restoring an old image/text on a timer.
            # Leave this dictation available until the next explicit clipboard write.

    def arm_auto_enter(self):
        """Gesto do fone chegou no meio da sessao (ex.: parou por ele): confirma com Enter no fim."""
        self._session_auto_enter = True

    def _press_enter(self):
        """Modo fone: confirma o envio da frase com Enter depois da ultima colagem."""
        time.sleep(0.15)  # o app alvo precisa processar o Ctrl+V antes do Enter
        if _ydotool_keys(f"{_KEY_ENTER}:1", f"{_KEY_ENTER}:0"):
            return
        if _is_wayland() and _hypr().available:
            raise RuntimeError("Entrada uinput indisponivel; Enter automatico cancelado.")
        if _is_wayland() and shutil.which("wtype"):
            try:
                r = subprocess.run(["wtype", "-k", "Return"], timeout=2, check=False, capture_output=True)
                if r.returncode == 0:
                    return
            except (OSError, subprocess.TimeoutExpired):
                pass
        self._keyboard.press(keyboard.Key.enter)
        self._keyboard.release(keyboard.Key.enter)

    def _send_paste_key(self, strategy: str = "ctrl_v") -> bool:
        """Use uinput on Hyprland; wtype remains available on other compositors.

        `terminal` usa Ctrl+Shift+V (colar nativo do foot/kitty/ghostty) para o
        Codex TUI nao receber Ctrl+V como colar-imagem.
        """
        if not _is_wayland():
            return False
        if strategy == "terminal":
            ok = _ydotool_keys(
                f"{_KEY_LEFTCTRL}:1", f"{_KEY_LEFTSHIFT}:1", f"{_KEY_V}:1",
                f"{_KEY_V}:0", f"{_KEY_LEFTSHIFT}:0", f"{_KEY_LEFTCTRL}:0",
            )
        else:
            ok = _ydotool_keys(
                f"{_KEY_LEFTCTRL}:1", f"{_KEY_V}:1", f"{_KEY_V}:0", f"{_KEY_LEFTCTRL}:0",
            )
        if ok:
            return True
        if _hypr().available:
            return False  # wtype's separate keymap can trigger a broadcast storm.
        if not shutil.which("wtype"):
            return False
        if strategy == "terminal":
            cmd = ["wtype", "-M", "ctrl", "-M", "shift", "-k", "v", "-m", "shift", "-m", "ctrl"]
        else:
            cmd = ["wtype", "-M", "ctrl", "-k", "v", "-m", "ctrl"]
        try:
            r = subprocess.run(cmd, timeout=2, check=False, capture_output=True)
            return r.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    def _type_fallback(self, text: str):
        if _is_wayland() and shutil.which("wtype"):
            try:
                subprocess.run(["wtype", "--", text], timeout=8, check=False, capture_output=True)
                return
            except (OSError, subprocess.TimeoutExpired):
                pass
        self._keyboard.type(text)


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

        # No Hyprland a barra e uma janela gerenciada (classe SussurroBar): a regra em
        # contrib/omarchy/sussurro.lua a deixa flutuante, fixa, sem foco e com rounding,
        # que recorta os cantos — o Tk nao tem transparencia por pixel no X11.
        self.hypr = (not IS_WIN) and _hypr().available
        self.win = tk.Toplevel(root, class_="SussurroBar")
        self.win.title("Sussurro — barra")
        if not self.hypr:
            self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        if IS_WIN:
            self.win.attributes("-transparentcolor", self.TRANSPARENT)
            canvas_bg = self.TRANSPARENT
        else:
            if not self.hypr:
                try:
                    self.win.wm_attributes("-type", "dock")
                except tk.TclError:
                    pass
            self.TRANSPARENT = self.PILL_BG  # fora da capsula fica a cor dela; o rounding corta
            canvas_bg = self.PILL_BG
        self.canvas = tk.Canvas(self.win, width=self.W, height=self.H,
                                bg=canvas_bg, highlightthickness=0, cursor="fleur")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag_move)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.canvas.bind("<Motion>", self._hover)
        self.canvas.bind("<Leave>", lambda _e: self._set_hover(None))
        self.win.update_idletasks()
        if IS_WIN:
            self._set_exstyle()
        else:
            self._apply_shape()
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
        self._follow_n = 0

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

    def _apply_shape(self):
        """X11 sem Hyprland: recorta a janela na forma da capsula (extensao SHAPE).

        O Tk nao tem transparencia por pixel; sem isto os cantos ficam um retangulo
        escuro. No Hyprland o rounding da regra de janela ja faz esse recorte.
        """
        if self.hypr:
            return
        try:
            from Xlib import display as xdisplay
            from Xlib.ext import shape
        except ImportError:
            return
        try:
            d = xdisplay.Display()
            if not d.has_extension("SHAPE"):
                d.close()
                return
            xwin = d.create_resource_object("window", self.win.winfo_id())
            pm = xwin.create_pixmap(self.W, self.H, 1)
            gc = pm.create_gc(foreground=0, background=0)
            pm.fill_rectangle(gc, 0, 0, self.W, self.H)
            gc.change(foreground=1)
            r = self.H // 2
            pm.fill_arc(gc, 0, 0, self.H - 1, self.H - 1, 0, 360 * 64)
            pm.fill_arc(gc, self.W - self.H, 0, self.H - 1, self.H - 1, 0, 360 * 64)
            pm.fill_rectangle(gc, r, 0, self.W - 2 * r, self.H)
            xwin.shape_mask(shape.SO.Set, shape.SK.Bounding, 0, 0, pm)
            d.sync()
            gc.free()
            pm.free()
            d.close()
        except Exception:  # noqa: BLE001 — sem SHAPE a barra so fica retangular
            pass

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
            if not self.hypr or not _hypr().place_bar(x, y):
                self.win.geometry(f"+{x}+{y}")

    # -- ciclo de vida -------------------------------------------------------
    def show(self, state: str):
        if self.hypr:
            _hypr().reset_bar_placement()
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
        self._follow_n += 1
        if self._follow_n % 4 == 0:  # ~180 ms: segue o cursor de monitor sem martelar o socket
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


def _mouse_button_id(button) -> str | None:
    name = getattr(button, "name", None)
    if name in BUTTON_LABELS:
        return name
    value = getattr(button, "value", None)
    if value == 2:
        return "middle"
    if value in (8, 6):
        return "x1"
    if value in (9, 7):
        return "x2"
    return None


class IpcServer(threading.Thread):
    """Unix socket: Hyprland (e `sussurro toggle`) ligam/desligam gravacao sem foco na janela."""

    daemon = True

    def __init__(self, event_queue: queue.Queue, path: Path, transcriber=None):
        super().__init__(name="sussurro-ipc")
        self.event_queue = event_queue
        self.transcriber = transcriber
        self.path = path
        self._sock = None
        self._stop = threading.Event()

    def run(self):
        try:
            if self.path.exists():
                try:
                    self.path.unlink()
                except OSError:
                    pass
            self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._sock.bind(str(self.path))
            os.chmod(self.path, 0o600)
            self._sock.listen(4)
            self._sock.settimeout(0.5)
            while not self._stop.is_set():
                try:
                    conn, _ = self._sock.accept()
                except socket.timeout:
                    continue
                with conn:
                    conn.settimeout(1)
                    raw = conn.recv(4096).decode("utf-8", "replace").strip()
                    if raw.split(" ", 1)[0].lower() == "transcribe":
                        # transcricao de arquivo leva segundos: responder aqui deixaria o
                        # atalho do mouse esperando. Sai da thread do accept.
                        threading.Thread(target=self._serve_slow, args=(conn.dup(), raw),
                                         name="sussurro-ipc-file", daemon=True).start()
                        continue
                    conn.sendall(self._handle(raw).encode("utf-8"))
        except Exception:
            traceback.print_exc()
        finally:
            try:
                if self._sock is not None:
                    self._sock.close()
                if self.path.exists():
                    self.path.unlink()
            except OSError:
                pass

    def _serve_slow(self, conn, raw: str) -> None:
        """Responde um `transcribe` fora da thread do accept, que precisa ficar livre."""
        with conn:
            conn.settimeout(900)
            try:
                conn.sendall(self._handle(raw).encode("utf-8"))
            except OSError:
                pass  # cliente desistiu de esperar

    def _handle(self, data: str) -> str:
        verb, _, arg = data.partition(" ")
        if verb.lower() == "transcribe":
            return self._handle_transcribe(arg.strip())
        data = data.lower()
        # "toggle-enter"/"start-enter"/"stop-enter": veio do fone (daemon x9-sussurro);
        # ao terminar de colar, o Sussurro aperta Enter para confirmar o envio.
        base, _, flag = data.partition("-")
        if base == "meeting" and flag in ("start", "stop", "pause"):
            self.event_queue.put(("meeting", flag))
            return "ok\n"
        if base in ("toggle", "start", "stop") and flag in ("", "enter"):
            _perf("activation_request", source="ipc", command=data)
            self.event_queue.put((base, {"enter": True} if flag == "enter" else None))
            return "ok\n"
        if data == "status":
            if self.transcriber is None:
                return "ok\n"
            t = self.transcriber
            config = t.model_config
            return json.dumps({"ready": t.model is not None and not t.model_loading.is_set(),
                               "recording": t.recording.is_set(), "busy": t.busy(),
                               "loading": t.model_loading.is_set(),
                               "comparing": t.comparing.is_set(),
                               "model": config.model if config else None,
                               "device": config.device if config else None,
                               "compute_type": config.compute_type if config else None}) + "\n"

        return "err unknown\n"

    def _handle_transcribe(self, path: str) -> str:
        """`transcribe <arquivo>`: transcreve, arquiva no historico e devolve JSON."""
        if not path:
            return json.dumps({"ok": False, "error": "uso: transcribe <arquivo>"}) + "\n"
        if self.transcriber is None:
            return json.dumps({"ok": False, "error": "transcritor indisponivel"}) + "\n"
        try:
            entry = self.transcriber.transcribe_file(path)
        except Exception as e:
            traceback.print_exc()
            return json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"},
                              ensure_ascii=False) + "\n"
        return json.dumps({"ok": True, **entry}, ensure_ascii=False) + "\n"


def _wants_enter(payload) -> bool:
    """Evento veio do fone (IPC *-enter): confirmar a frase com Enter ao final."""
    return bool(payload) and isinstance(payload, dict) and bool(payload.get("enter"))


class MouseHotkey:
    """Hook global de mouse. No Windows o botao e suprimido; no Linux o clique tambem chega ao app debaixo.

    Eventos sao empurrados na event_queue ("start"/"stop"/("captured", nome));
    quem consome e a UI, no thread do Tk.
    """

    def __init__(self, event_queue: queue.Queue, button: str, trigger_mode: str):
        self.event_queue = event_queue
        self.button = button
        self.trigger_mode = trigger_mode
        self.capturing = False
        self.active = False  # sessao de gravacao iniciada pelo atalho
        self._listener = None
        # Wayland: pynput nao ve clique global se a janela nao tem foco (e pode
        # SIGSEGV). O Hyprland manda toggle/start/stop pelo socket.
        if _is_wayland():
            return
        try:
            if IS_WIN:
                self._listener = mouse.Listener(win32_event_filter=self._filter)
            else:
                self._listener = mouse.Listener(on_click=self._on_click)
            self._listener.start()
        except Exception as e:
            self._listener = None
            self.event_queue.put(("error", f"atalho de mouse indisponivel: {e}"))

    @staticmethod
    def _decode(msg, data):
        if msg in (WM_MBUTTONDOWN, WM_MBUTTONUP):
            return "middle", msg == WM_MBUTTONDOWN
        if msg in (WM_XBUTTONDOWN, WM_XBUTTONUP):
            xbtn = (data.mouseData >> 16) & 0xFFFF
            return ("x1" if xbtn == 1 else "x2"), msg == WM_XBUTTONDOWN
        return None, None

    def _handle(self, button: str, pressed: bool, suppress: bool):
        if self.capturing:
            if pressed:
                self.capturing = False
                self.button = button
                self.event_queue.put(("captured", button))
            if suppress:
                self._listener.suppress_event()
            return
        if button != self.button:
            return
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
        if suppress:
            self._listener.suppress_event()

    def _filter(self, msg, data):
        button, pressed = self._decode(msg, data)
        if button is None:
            return True
        self._handle(button, pressed, suppress=True)
        if button != self.button and not self.capturing:
            return True
        return True

    def _on_click(self, _x, _y, button, pressed):
        name = _mouse_button_id(button)
        if name is None:
            return
        self._handle(name, pressed, suppress=False)


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
    entries = [entry for entry in entries if not entry.get("failed")]
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
    for nome in (
        "segoeui.ttf", "arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ):
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


class HistoryList(ctk.CTkFrame):
    """Historico num canvas so. CTkFrame por linha trava o Tk uns 3s no restore
    (Configure -> _draw em cada canvas); item de canvas pinta na hora."""

    WHEEL_PX = 96        # pixels por clique da roda (uns 2 ditados curtos)
    SCROLL_FRAME_MS = 12  # ~80 fps na animacao do scroll

    def __init__(self, master, font_mono, font_ui, day_label, on_play, on_copy, on_retry):
        super().__init__(master, fg_color="transparent", width=1, height=1)
        self.font_mono = font_mono
        self.font_ui = font_ui
        self.day_label = day_label
        self.on_play = on_play
        self.on_copy = on_copy
        self.on_retry = on_retry
        self._entries = []
        self._width = 0
        self._hits = []
        self._hover = None
        self._bg_ids = {}
        self._icon_ids = {}
        self._content_h = 0
        self._scroll_target = None
        self._scroll_job = None

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(self, bg=SURFACE, highlightthickness=0, bd=0,
                                yscrollincrement=1)
        self.sb = ctk.CTkScrollbar(self, orientation="vertical", command=self._on_scrollbar,
                                   fg_color="transparent", width=12)
        self.canvas.configure(yscrollcommand=self.sb.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.sb.grid(row=0, column=1, sticky="ns", padx=(0, 2), pady=2)

        self.canvas.bind("<Configure>", self._on_cfg)
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Leave>", self._on_leave)
        self.canvas.bind("<Button-1>", self._on_click)
        top = self.winfo_toplevel()
        top.bind_all("<MouseWheel>", self._on_wheel, add=True)
        top.bind_all("<Button-4>", self._on_wheel, add=True)
        top.bind_all("<Button-5>", self._on_wheel, add=True)

    def set_entries(self, entries):
        self._entries = list(entries)
        if self._width >= 40:
            self._redraw()

    def _s(self, v):
        return int(round(self._apply_widget_scaling(v)))

    def _on_cfg(self, event):
        if event.width == self._width or event.width < 40:
            return
        self._width = event.width
        self._redraw()

    def _pointer_over_canvas(self):
        try:
            x, y = self.winfo_pointerxy()
            return self.winfo_containing(x, y) is self.canvas
        except tk.TclError:
            return False

    # -- scroll ---------------------------------------------------------------
    def _max_scroll(self) -> int:
        return max(0, self._content_h - self.canvas.winfo_height())

    def _scroll_pos(self) -> float:
        return self.canvas.yview()[0] * self._content_h

    def _on_scrollbar(self, *args):
        # arrasto da barra: pula direto, sem animacao no meio do caminho
        self._cancel_scroll_anim()
        self.canvas.yview(*args)

    def _cancel_scroll_anim(self):
        if self._scroll_job is not None:
            self.after_cancel(self._scroll_job)
            self._scroll_job = None
        self._scroll_target = None

    def _on_wheel(self, event):
        if not self._pointer_over_canvas():
            return
        if self._max_scroll() <= 0:
            return "break"
        # Tk 9 manda <MouseWheel> com delta em multiplos de 120 no X11/Wayland;
        # Tk 8 no X11 manda Button-4/5. Touchpad chega com delta fracionado.
        delta = getattr(event, "delta", 0) or 0
        num = getattr(event, "num", 0) or 0
        if num == 4:
            notches = -1.0
        elif num == 5:
            notches = 1.0
        elif delta:
            notches = -delta / 120.0
        else:
            return "break"
        self._scroll_by(notches * self._s(self.WHEEL_PX))
        return "break"

    def _scroll_by(self, px: float):
        base = self._scroll_pos() if self._scroll_target is None else self._scroll_target
        self._scroll_target = min(self._max_scroll(), max(0.0, base + px))
        if self._scroll_job is None:
            self._scroll_step()

    def _scroll_step(self):
        self._scroll_job = None
        if self._scroll_target is None or not self._content_h:
            return
        pos = self._scroll_pos()
        diff = self._scroll_target - pos
        if abs(diff) < 0.75:
            self.canvas.yview_moveto(self._scroll_target / self._content_h)
            self._scroll_target = None
            return
        pos += diff * 0.3  # ease-out: fecha 30% da distancia por quadro
        self.canvas.yview_moveto(pos / self._content_h)
        self._scroll_job = self.after(self.SCROLL_FRAME_MS, self._scroll_step)

    # -- hit test / hover -------------------------------------------------------
    def _hit(self, x, y):
        for h in self._hits:
            if not (h["y0"] <= y < h["y1"]):
                continue
            px0, py0, px1, py1 = h["play"]
            cx0, cy0, cx1, cy1 = h["copy"]
            if px0 <= x <= px1 and py0 <= y <= py1:
                return h, "play"
            if cx0 <= x <= cx1 and cy0 <= y <= cy1:
                return h, "copy"
            tx0, ty0, tx1, ty1 = h["text"]
            if tx0 <= x <= tx1 and ty0 <= y <= ty1:
                return h, "text"
            return h, "row"
        return None, None

    def _set_hover(self, idx):
        if idx == self._hover:
            return
        if self._hover is not None and self._hover in self._bg_ids:
            self.canvas.itemconfigure(self._bg_ids[self._hover], fill=SURFACE)
            failed = self._entries[self._hover].get("failed")
            for kind, iid in self._icon_ids.get(self._hover, ()):
                self.canvas.itemconfigure(
                    iid, fill=ACCENT_TEXT if (failed and kind == "copy") else INK_3)
        self._hover = idx
        if idx is not None and idx in self._bg_ids:
            self.canvas.itemconfigure(self._bg_ids[idx], fill=SURFACE_2)
            failed = self._entries[idx].get("failed")
            for kind, iid in self._icon_ids.get(idx, ()):
                self.canvas.itemconfigure(
                    iid, fill=ACCENT_TEXT if (failed and kind == "copy") else INK)

    def _on_motion(self, event):
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        h, zone = self._hit(x, y)
        failed = h is not None and self._entries[h["i"]].get("failed")
        clickable = zone in ("play", "copy", "text") or (failed and zone == "row")
        self.canvas.configure(cursor="hand2" if clickable else "")
        self._set_hover(None if h is None else h["i"])

    def _on_leave(self, _event):
        self.canvas.configure(cursor="")
        self._set_hover(None)

    def _on_click(self, event):
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        h, zone = self._hit(x, y)
        if h is None:
            return
        entry = self._entries[h["i"]]
        if zone == "play":
            self.on_play(str(HISTORY_DIR / entry["wav"]))
        elif entry.get("failed"):
            self.on_retry(entry)
        elif zone in ("copy", "text"):
            self.on_copy(entry["text"])

    # -- desenho --------------------------------------------------------------
    def _redraw(self):
        frac = self.canvas.yview()[0]
        self._cancel_scroll_anim()
        self.canvas.delete("all")
        self._hits = []
        self._bg_ids = {}
        self._icon_ids = {}
        self._hover = None
        w = self._width
        if w < 40:
            return
        s = self._s
        padx, pady = s(14), s(9)
        btn, gap = s(28), s(2)
        btns_w = btn * 2 + gap + padx + s(6)
        y = s(6)
        prev_day = None
        font_day = (self.font_ui, 10, "bold")
        font_time = (self.font_mono, 11)
        # largura da hora medida na fonte real: com Xft a mono e mais larga que os 44 px fixos
        time_w = tkfont.Font(font=font_time).measure("00:00") + s(12)
        font_text = (self.font_ui, 13)
        font_btn = (self.font_ui, 12)
        if not self._entries:
            self.canvas.create_text(padx, y + s(8), text="Nenhum ditado ainda.",
                                    fill=INK_3, anchor="nw", font=font_text)
            self._content_h = y + s(40)
            self.canvas.configure(scrollregion=(0, 0, w, self._content_h))
            return
        for i, entry in enumerate(self._entries):
            dt = datetime.fromisoformat(entry["ts"])
            day = self.day_label(dt)
            if day != prev_day:
                if prev_day is not None:
                    y += s(14)
                hid = self.canvas.create_text(padx, y + s(4), text=day.upper(), fill=INK_3,
                                              anchor="nw", font=font_day)
                hb = self.canvas.bbox(hid)
                mid = (hb[1] + hb[3]) / 2
                self.canvas.create_line(hb[2] + s(10), mid, w - padx, mid, fill=BORDER)
                y = hb[3] + s(6)
                prev_day = day
            failed = bool(entry.get("failed"))
            retrying = bool(entry.get("retrying"))
            shown_text = ("Tentando transcrever novamente..." if retrying else
                          "A transcrição falhou. Clique para tentar novamente.") if failed else entry["text"]
            text_x = padx + time_w
            text_w = max(s(80), w - text_x - btns_w)
            tid = self.canvas.create_text(
                text_x, y + pady, text=shown_text, fill=ACCENT_TEXT if failed else INK, anchor="nw",
                width=text_w, font=font_text, justify="left")
            tb = self.canvas.bbox(tid)
            th = tb[3] - tb[1]
            row_h = max(th, btn) + pady * 2
            bg = self.canvas.create_rectangle(
                s(4), y, w - s(4), y + row_h, fill=SURFACE, outline="", width=0)
            self.canvas.tag_lower(bg, tid)
            # hora alinhada na primeira linha do texto (fonte menor: desce 1-2 px)
            self.canvas.create_text(
                padx, y + pady + s(2), text=dt.strftime("%H:%M"), fill=INK_3,
                anchor="nw", font=font_time)
            copy_x1 = w - padx - s(4)
            copy_x0 = copy_x1 - btn
            play_x1 = copy_x0 - gap
            play_x0 = play_x1 - btn
            by = y + pady
            play_id = self.canvas.create_text(
                (play_x0 + play_x1) / 2, by + btn / 2, text="▶", fill=INK_3,
                font=font_btn, anchor="center")
            copy_id = self.canvas.create_text(
                (copy_x0 + copy_x1) / 2, by + btn / 2,
                text="…" if retrying else ("↻" if failed else "⧉"),
                fill=ACCENT_TEXT if failed else INK_3,
                font=font_btn, anchor="center")
            # separador fino entre ditados; some no ultimo de cada dia
            self.canvas.create_line(padx, y + row_h, w - padx, y + row_h, fill=BORDER)
            self._hits.append({
                "i": i, "y0": y, "y1": y + row_h,
                "play": (play_x0, y, play_x1, y + row_h),
                "copy": (copy_x0, y, copy_x1, y + row_h),
                "text": (text_x, y, play_x0 - s(4), y + row_h),
            })
            self._bg_ids[i] = bg
            self._icon_ids[i] = (("play", play_id), ("copy", copy_id))
            y += row_h + 1
        self._content_h = y + s(10)
        self.canvas.configure(scrollregion=(0, 0, w, self._content_h))
        self.canvas.yview_moveto(frac)


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Sussurro")
        assets = Path(__file__).with_name("assets")
        ico, png = assets / "sussurro.ico", assets / "sussurro.png"
        if IS_WIN and ico.exists():
            root.iconbitmap(str(ico))
        if png.exists():
            photo = tk.PhotoImage(file=str(png))
            root.iconphoto(True, photo)
            root._sussurro_icon = photo
        root.geometry(f"760x{min(800, max(600, root.winfo_screenheight() - 80))}")

        self.settings = load_settings()
        self.devices = list_input_devices()
        self.loopback_devices = list_loopback_devices()
        self.hardware = detect_hardware()
        self.device_labels = execution_device_labels(self.hardware)

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
        self._ipc = IpcServer(self.hotkey_queue, IPC_SOCK, self.transcriber)
        self._ipc.start()
        self.devcfg = self.settings["devices"]
        self.gestures = None
        if not IS_WIN:
            self.gestures = devmod.DeviceGestures(self.devcfg, self._on_device_gesture)
            if self.devcfg.get("enabled"):
                self.gestures.start()

        self.FONT_UI = pick_font(
            ["Segoe UI", "Noto Sans", "DejaVu Sans", "Ubuntu", "Liberation Sans"],
            "TkDefaultFont",
        )
        self.FONT_DISPLAY = pick_font(
            ["Bahnschrift SemiBold Condensed", "Bahnschrift SemiBold", "Bahnschrift",
             "Noto Sans Condensed", "DejaVu Sans"],
            "Arial",
        )
        self.FONT_MONO = pick_font(
            ["Cascadia Mono", "JetBrains Mono", "DejaVu Sans Mono", "Noto Sans Mono", "Ubuntu Mono"],
            "TkFixedFont",
        )
        root.configure(fg_color=BG)
        PADX = self.PADX = 16

        # cabecalho: marca + modelo
        header = ctk.CTkFrame(root, fg_color="transparent")
        header.pack(fill="x", padx=PADX, pady=(14, 12))
        icon_png = Path(__file__).with_name("assets") / "sussurro.png"
        if icon_png.exists():
            self._brand_img = ctk.CTkImage(Image.open(icon_png), size=(28, 28))
            ctk.CTkLabel(header, image=self._brand_img, text="").pack(side="left")
        ctk.CTkLabel(header, text="SUSSURRO", text_color=INK,
                     font=(self.FONT_DISPLAY, 22)).pack(side="left", padx=(10, 0))
        self.model_label = ctk.CTkLabel(header, text="Preparando modelo...", text_color=INK_3,
                                        font=(self.FONT_MONO, 11))
        self.model_label.pack(side="right")

        # faixa de comando: GRAVAR e o unico laranja da janela
        cmd = self.command_bar = ctk.CTkFrame(root, fg_color="transparent")
        cmd.pack(fill="x", padx=PADX, pady=(0, 12))
        self.record_btn = ctk.CTkButton(
            cmd, text="GRAVAR", command=self.toggle, state="disabled",
            width=136, height=38, corner_radius=8,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            text_color=GRAPHITE, text_color_disabled=GRAPHITE,
            font=(self.FONT_DISPLAY, 16))
        self.record_btn.pack(side="left")
        for label, cb in (("Copiar tudo", self.copy_all), ("Limpar", self.clear)):
            self._secondary(cmd, label, cb, width=104, height=38).pack(side="left", padx=(8, 0))

        # card de configuracao: grade 3 colunas, rotulos caixa alta discretos
        card = self.config_card = ctk.CTkFrame(root, fg_color=SURFACE, corner_radius=12)
        card.pack(fill="x", padx=PADX, pady=(0, 12))
        for col in range(3):
            card.grid_columnconfigure(col, weight=1, uniform="cfg")
        CPAD, GROUP = 16, 16   # margem interna do card; respiro entre grupos de linhas

        def cfg_label(text, r, c):
            ctk.CTkLabel(card, text=text, text_color=INK_3, anchor="w", height=14,
                         font=(self.FONT_UI, 10, "bold")).grid(
                row=r, column=c, sticky="ew", padx=CPAD, pady=((CPAD, 0) if r == 0 else (GROUP, 0)))

        def combo(values, current, command, r, c):
            box = ctk.CTkComboBox(card, values=values, command=command, state="readonly",
                                  height=32, corner_radius=8,
                                  fg_color=SURFACE_2, border_color=BORDER,
                                  button_color=SURFACE_2, button_hover_color=SURFACE_3,
                                  dropdown_fg_color=SURFACE_2, dropdown_hover_color=SURFACE_3,
                                  dropdown_text_color=INK, text_color=INK,
                                  font=(self.FONT_UI, 12))
            box.set(current)
            box.grid(row=r, column=c, sticky="ew", padx=CPAD, pady=(6, 0))
            return box

        cfg_label("ATALHO DO MOUSE", 0, 0)
        cfg_label("AÇÃO", 0, 1)
        cfg_label("MICROFONE", 0, 2)
        hk = ctk.CTkFrame(card, fg_color="transparent")
        hk.grid(row=1, column=0, sticky="ew", padx=CPAD, pady=(6, 0))
        hk.grid_columnconfigure(0, weight=1)
        self.hotkey_var = tk.StringVar(value=BUTTON_LABELS[self.settings["mouse_button"]])
        ctk.CTkEntry(hk, textvariable=self.hotkey_var, state="readonly", height=32,
                     corner_radius=8, fg_color=SURFACE_2, border_color=BORDER,
                     text_color=INK, font=(self.FONT_UI, 12)).grid(row=0, column=0, sticky="ew")
        self.set_hotkey_btn = self._secondary(hk, "Setar", self.capture_hotkey, width=60)
        self.set_hotkey_btn.grid(row=0, column=1, padx=(6, 0))
        self.trigger = combo(["alternar", "segurar"], self.settings["trigger_mode"],
                             self._on_trigger, 1, 1)
        self.mic_by_label = {device.label: device for device in self.devices.values()}
        names = list(self.mic_by_label)
        saved = self.settings["device_name"]
        saved_device = self.devices.get(saved)
        current_mic = saved_device.label if saved_device else (names[0] if names else "")
        self.mic = combo(names, current_mic,
                         self._on_mic, 1, 2)
        cfg_label("TRANSCRIÇÃO", 2, 0)
        cfg_label("ENVIO", 2, 1)
        cfg_label("IDIOMA", 2, 2)
        self.mode = combo(["simultaneo", "final"], self.settings["transcribe_mode"],
                          self._on_mode, 3, 0)
        self.inject = combo(["colar", "digitar"], self.settings["inject_method"],
                            self._on_inject, 3, 1)
        self.lang = combo(["pt", "en", "auto"], self.settings["language"],
                          self._on_lang, 3, 2)

        # 3a linha: fonte de captura (mic / audio do PC / os dois) + canal do PC
        cfg_label("FONTE", 4, 0)
        cfg_label("CANAL DO PC", 4, 1)
        self.fonte = combo(["microfone", "audio do PC", "os dois"],
                           CAPTURE_LABELS[self.settings["capture_mode"]],
                           self._on_fonte, 5, 0)
        pc_names = ["padrao do sistema"] + list(self.loopback_devices)
        saved_pc = self.settings["loopback_device_name"]
        if saved_pc not in self.loopback_devices:
            saved_pc = None
        self.pc_channel = combo(pc_names, "padrao do sistema" if saved_pc is None else saved_pc,
                                self._on_pc_channel, 5, 1)
        self.pc_channel.configure(
            state="disabled" if self.settings["capture_mode"] == "microfone" else "readonly")
        cfg_label("O QUE ESTA ENTRADA FAZ", 4, 2)
        self.mic_help = ctk.CTkLabel(
            card, text="", text_color=INK_3, font=(self.FONT_UI, 11),
            anchor="nw", justify="left", wraplength=340)
        self.mic_help.grid(row=5, column=2, sticky="new", padx=CPAD, pady=(8, 0))
        self._update_mic_help()

        cfg_label("MODELO", 6, 0)
        cfg_label("EXECUTAR EM", 6, 1)
        self.model_choice = combo(list(MODEL_LABELS.values()),
                                  MODEL_LABELS[self.settings["whisper_model"]], lambda _: None, 7, 0)
        self.device_choice = combo(list(self.device_labels.values()),
                                   self.device_labels[self.settings["whisper_device"]], lambda _: None, 7, 1)
        self.apply_model_btn = self._secondary(card, "Aplicar modelo", self._apply_model)
        self.apply_model_btn.grid(row=7, column=2, sticky="ew", padx=CPAD, pady=(6, 0))
        ctk.CTkLabel(card, text="CPU básico: Base · CPU moderno: Small · GPU NVIDIA: Turbo · "
                               "Large-v3 prioriza precisão.\nParakeet (só GPU): o mais rápido, sem idioma fixo; "
                               "em português às vezes escorrega pro inglês.\n" + execution_hardware_note(self.hardware),
                     text_color=INK_3, font=(self.FONT_UI, 11), anchor="w", justify="left").grid(
            row=8, column=0, columnspan=3, sticky="ew", padx=CPAD, pady=(12, CPAD))

        # abas: acento fica no GRAVAR; aba ativa marca por chapa mais clara
        tabbar = self.tabbar = ctk.CTkFrame(root, fg_color="transparent")
        tabbar.pack(fill="x", padx=PADX, pady=(0, 8))
        self.tab_btns = {}
        tabs = [("historico", "HISTÓRICO"), ("aovivo", "AO VIVO")]
        if not IS_WIN:
            tabs.append(("reuniao", "REUNIÃO"))  # parec/pacat: PulseAudio ou PipeWire
        tabs += [("biblioteca", "BIBLIOTECA"), ("estatisticas", "ESTATÍSTICAS"),
                 ("comparar", "COMPARAR")]
        if self.gestures is not None:
            tabs.append(("omarchy", "OMARCHY"))
        tab_font = (self.FONT_DISPLAY, 12)  # sete abas cabem nos 760 px da janela
        tab_measure = tkfont.Font(font=tab_font)
        for name, label in tabs:
            btn = ctk.CTkButton(tabbar, text=label, height=30, corner_radius=6,
                                width=tab_measure.measure(label) + 16,
                                fg_color="transparent", hover_color=SURFACE_2,
                                text_color=INK_3, font=tab_font,
                                command=lambda n=name: self._show_tab(n))
            btn.pack(side="left", padx=(0, 2))
            self.tab_btns[name] = btn

        # status empacotado antes do conteudo (side=bottom) pra nunca ser espremido pra fora
        status_bar = ctk.CTkFrame(root, fg_color="transparent")
        status_bar.pack(side="bottom", fill="x", padx=PADX, pady=(4, 10))
        self.status = ctk.CTkLabel(status_bar, text="Iniciando...", text_color=INK_3,
                                   anchor="w", font=(self.FONT_MONO, 11))
        self.status.pack(side="left")
        ctk.CTkLabel(status_bar, text="barra: arraste pelo meio para reposicionar", text_color=INK_3,
                     font=(self.FONT_UI, 10)).pack(side="right")

        # card de conteudo: historico / ao vivo
        self.content = ctk.CTkFrame(root, fg_color=SURFACE, corner_radius=12)
        self.content.pack(fill="both", expand=True, padx=PADX, pady=(0, 4))
        self._playing = None
        self.hist_frame = HistoryList(
            self.content, font_mono=self.FONT_MONO, font_ui=self.FONT_UI,
            day_label=self._day_label, on_play=self._play, on_copy=self._copy_entry,
            on_retry=self._retry_entry)
        self.text = ctk.CTkTextbox(self.content, fg_color="transparent", text_color=INK,
                                   font=(self.FONT_UI, 13), wrap="word", border_width=0)
        self.library = self.transcriber.library
        self.lib_tab = self._build_library_tab()
        self.stats_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        self._stats_dirty = True   # so desenha quando a aba abrir
        self._geom_antes = None    # geometria de fora da aba ESTATISTICAS
        self._tabs = {"historico": self.hist_frame, "aovivo": self.text,
                      "biblioteca": self.lib_tab, "estatisticas": self.stats_frame}
        if self.gestures is not None:
            self._tabs["omarchy"] = self._build_devices_tab()
        self.compare_panel = ComparisonPanel(self.content, self, StreamResampler)
        self._tabs["comparar"] = self.compare_panel
        self.meeting_panel = None
        if not IS_WIN:
            self.meeting_panel = MeetingPanel(self.content, self)
            self._tabs["reuniao"] = self.meeting_panel
        root.protocol("WM_DELETE_WINDOW", self._close)
        self._tab = None
        self._show_tab("historico")

        self.entries = self._load_history()  # mais recente primeiro
        self._render_history()
        self._render_library()

        self.bar = RecorderBar(root, lambda: self.settings["dot_pos"], self._save_bar_pos,
                               get_levels=lambda: self.transcriber.levels,
                               on_cancel=self._cancel, on_confirm=self._stop)

        # Tk 9 no Linux SIGSEGV se um widget for tocado fora da thread do mainloop
        # (CTkButton.configure desenha via canvas create_text -> TkpGetColor).
        # No Windows isso "funcionava"; aqui tudo que mexe na HUD passa por esta fila.
        self._ui_queue: queue.Queue = queue.Queue()
        self._begin_model_load(dict(self.settings), persist=False)
        root.after(UI_POLL_MS, self._poll)

    def _secondary(self, master, text, command, width=None, height=32, **kw):
        """Botao neutro padrao: chapa sutil, borda fina, sem acento (o acento e do GRAVAR)."""
        opts = dict(text=text, command=command, height=height, corner_radius=8,
                    fg_color=SURFACE_2, hover_color=SURFACE_3, border_width=1,
                    border_color=BORDER, text_color=INK_2, font=(self.FONT_UI, 12))
        if width is not None:
            opts["width"] = width
        opts.update(kw)
        return ctk.CTkButton(master, **opts)

    def _close(self):
        self.compare_panel.close()
        if self.meeting_panel is not None:
            self.meeting_panel.close()
        self.root.destroy()

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
        # abas de tela cheia: sem a faixa do GRAVAR e sem o card de configuracao do ditado
        full = ("comparar", "reuniao")
        if name in full and self._tab not in full:
            self.command_bar.pack_forget()
            self.config_card.pack_forget()
        elif name not in full and self._tab in full:
            self.config_card.pack(fill="x", padx=self.PADX, pady=(0, 12), before=self.tabbar)
            self.command_bar.pack(fill="x", padx=self.PADX, pady=(0, 12), before=self.config_card)
        if self.meeting_panel is not None:
            if name == "reuniao":
                self.meeting_panel.shown()
            elif self._tab == "reuniao":
                self.meeting_panel.hidden()
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
        if name == "omarchy":
            self._devices_tick()

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

    def _render_history(self):
        self.hist_frame.set_entries(self.entries[:HIST_RENDER_MAX])

    def _add_history(self, entry: dict):
        for i, current in enumerate(self.entries):
            if current.get("wav") == entry.get("wav"):
                self.entries[i] = entry
                break
        else:
            self.entries.insert(0, entry)
        self.hist_frame.set_entries(self.entries[:HIST_RENDER_MAX])
        self._stats_dirty = True
        if self._tab == "estatisticas":
            self._render_stats()

    def _copy_entry(self, text: str):
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.status.configure(text="Transcricao copiada para a area de transferencia.")

    def _retry_entry(self, entry: dict):
        if entry.get("retrying"):
            return
        try:
            self.transcriber.retry_history_async(entry)
        except Exception as e:
            self.status.configure(text=f"ERRO ao tentar novamente: {e}")
            return
        entry["retrying"] = True
        self._render_history()

    def _play(self, path: str):
        if self._playing == path:
            sd.stop()
            self._playing = None
            return
        with wave.open(path, "rb") as w:
            sr = w.getframerate()
            nch = w.getnchannels()
            raw = w.readframes(w.getnframes())
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if nch > 1:
            audio = audio.reshape(-1, nch)
        sd.play(audio, sr)
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
                     font=(self.FONT_UI, 9, "bold")).pack(fill="x", padx=12, pady=(10, 0))
        ctk.CTkLabel(card, text=valor, text_color=INK, anchor="w",
                     font=(self.FONT_DISPLAY, 25)).pack(fill="x", padx=12, pady=(1, 0))
        ctk.CTkLabel(card, text=nota, text_color=INK_3, anchor="w", height=12,
                     font=(self.FONT_UI, 10)).pack(fill="x", padx=12, pady=(0, 10))

    def _stats_titulo(self, parent, texto: str, nota: str = ""):
        head = ctk.CTkFrame(parent, fg_color="transparent")
        head.pack(fill="x", padx=12, pady=(10, 2))
        ctk.CTkLabel(head, text=texto, text_color=INK_3, height=12,
                     font=(self.FONT_UI, 9, "bold")).pack(side="left")
        if nota:
            ctk.CTkLabel(head, text=nota, text_color=INK_3, height=12,
                         font=(self.FONT_UI, 10)).pack(side="right")

    def _render_stats(self):
        """Redesenha a aba inteira a partir do historico em memoria."""
        for child in self.stats_frame.winfo_children():
            child.destroy()
        self._stats_dirty = False
        if not self.entries:
            ctk.CTkLabel(self.stats_frame, text="Nenhum ditado ainda — grave alguma coisa "
                         "e os numeros aparecem aqui.", text_color=INK_3,
                         font=(self.FONT_UI, 12)).pack(anchor="w", padx=10, pady=10)
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
                     font=(self.FONT_UI, 10)).pack(side="left", padx=(26, 6))
        ctk.CTkLabel(legenda, image=self._leg_img, text="").pack(side="left")
        ctk.CTkLabel(legenda, text="mais", text_color=INK_3, height=12,
                     font=(self.FONT_UI, 10)).pack(side="left", padx=(6, 0))

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
                         font=(self.FONT_UI, 12)).pack(side="left")
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
                     anchor="w", height=14, font=(self.FONT_UI, 10, "bold")).grid(
            row=0, column=0, sticky="ew", padx=(6, 0))
        ctk.CTkLabel(form, text="DEVE VIRAR", text_color=INK_3, anchor="w", height=14,
                     font=(self.FONT_UI, 10, "bold")).grid(row=0, column=1, sticky="ew", padx=(8, 0))

        def field(placeholder, col):
            e = ctk.CTkEntry(form, placeholder_text=placeholder, height=32, corner_radius=8,
                             fg_color=SURFACE_2, border_color=BORDER, text_color=INK,
                             placeholder_text_color=INK_3, font=(self.FONT_UI, 12))
            e.grid(row=1, column=col, sticky="ew", padx=(6 if col == 0 else 8, 0), pady=(4, 0))
            e.bind("<Return>", lambda _ev: self._lib_add())
            return e

        self.lib_wrong = field("grock, groque, grote", 0)
        self.lib_right = field("Grok", 1)
        self._secondary(form, "Adicionar", self._lib_add, width=96).grid(
            row=1, column=2, padx=(8, 6), pady=(4, 0))
        self.lib_list = ctk.CTkScrollableFrame(wrap, fg_color="transparent")
        self.lib_list.pack(fill="both", expand=True)
        return wrap

    def _render_library(self):
        for child in self.lib_list.winfo_children():
            child.destroy()
        if not self.library.entries:
            ctk.CTkLabel(self.lib_list, text="Nenhuma palavra na biblioteca ainda.",
                         text_color=INK_3, font=(self.FONT_UI, 12)).pack(anchor="w", padx=10, pady=10)
            return
        for i, entry in enumerate(self.library.entries):
            row = ctk.CTkFrame(self.lib_list, fg_color="transparent", corner_radius=8)
            row.pack(fill="x", padx=2, pady=1)
            ctk.CTkButton(row, text="✕", command=lambda n=i: self._lib_remove(n),
                          width=28, height=26, corner_radius=6, fg_color="transparent",
                          hover_color=SURFACE_3, text_color=INK_3,
                          font=(self.FONT_UI, 12)).pack(
                side="right", anchor="n", padx=(6, 8), pady=5)
            ctk.CTkLabel(row, text=entry["certo"], text_color=INK, width=130, anchor="w",
                         font=(self.FONT_UI, 12, "bold")).pack(side="left", padx=(10, 6), pady=6)
            ctk.CTkLabel(row, text="⟵  " + ", ".join(entry["erros"]), text_color=INK_3,
                         anchor="w", justify="left", wraplength=380,
                         font=(self.FONT_UI, 12)).pack(side="left", fill="x", expand=True, pady=6)

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

    # -- aba OMARCHY: gestos de dispositivos (so Linux) ------------------------
    def _build_devices_tab(self):
        g = self.gestures
        wrap = ctk.CTkScrollableFrame(self.content, fg_color="transparent")
        ctk.CTkLabel(wrap, text=f"GESTOS DO HEADSET · {g.profile.label.upper()}", text_color=INK,
                     anchor="w", font=(self.FONT_DISPLAY, 16)).pack(fill="x", padx=8, pady=(6, 0))
        ctk.CTkLabel(wrap, text=("Aciona o Sussurro pelo fone, sem tocar no PC: gire a roda de volume "
                                 "pra cima e pra baixo rapido, ou de um toque duplo no mute. "
                                 "Ao terminar de colar, o Sussurro aperta Enter."),
                     text_color=INK_3, anchor="w", justify="left", wraplength=620,
                     font=(self.FONT_UI, 12)).pack(fill="x", padx=8, pady=(2, 8))

        # status: fone / acesso / mic do fone / mic padrao
        st = ctk.CTkFrame(wrap, fg_color=SURFACE_2, corner_radius=10)
        st.pack(fill="x", padx=6, pady=(0, 8))
        self.dev_status_labels = {}
        for key, title in (("headset", "FONE"), ("access", "ACESSO AOS BOTOES"),
                           ("mic", "MIC DO FONE"), ("default", "MIC PADRAO AGORA")):
            col = ctk.CTkFrame(st, fg_color="transparent")
            col.pack(side="left", expand=True, fill="x", padx=12, pady=10)
            ctk.CTkLabel(col, text=title, text_color=INK_3, anchor="w", height=14,
                         font=(self.FONT_UI, 10, "bold")).pack(fill="x")
            lab = ctk.CTkLabel(col, text="—", text_color=INK, anchor="w", font=(self.FONT_UI, 12))
            lab.pack(fill="x")
            self.dev_status_labels[key] = lab
        self.dev_udev_btn = self._secondary(st, "Instalar regra udev", self._dev_install_udev,
                                            width=150)

        # liga/desliga
        sw = ctk.CTkFrame(wrap, fg_color="transparent")
        sw.pack(fill="x", padx=6, pady=(0, 4))
        sw.grid_columnconfigure((0, 1), weight=1)
        self.dev_switches = {}
        self.dev_switch_labels = {
            "enabled": "Gestos do fone ativos",
            "auto_enter": "Enter automatico ao terminar",
            "wheel_gesture": "Gesto: roda de volume invertida",
            "mute_gesture": "Gesto: toque duplo no mute",
            "mic_follow": "Mic padrao segue o fone",
        }
        for i, (key, label) in enumerate(self.dev_switch_labels.items()):
            var = tk.BooleanVar(value=bool(self.devcfg.get(key)))
            ctk.CTkSwitch(sw, text=label, variable=var, command=lambda k=key: self._on_dev_switch(k),
                          progress_color=ACCENT, button_color=INK_2, button_hover_color=INK,
                          text_color=INK, font=(self.FONT_UI, 12)).grid(
                row=i // 2, column=i % 2, sticky="w", padx=8, pady=6)
            self.dev_switches[key] = var

        # parametros
        pr = ctk.CTkFrame(wrap, fg_color="transparent")
        pr.pack(fill="x", padx=6, pady=(4, 4))
        pr.grid_columnconfigure(1, weight=1)
        self.dev_param_labels = {}
        self._dev_save_job = None

        def slider(row, key, title, lo, hi, steps, fmt):
            ctk.CTkLabel(pr, text=title, text_color=INK_3, anchor="w",
                         font=(self.FONT_UI, 10, "bold")).grid(row=row, column=0, sticky="w",
                                                                padx=(8, 10), pady=4)
            sl = ctk.CTkSlider(pr, from_=lo, to=hi, number_of_steps=steps, progress_color=ACCENT,
                               button_color=INK_2, button_hover_color=INK,
                               command=lambda v, k=key, f=fmt: self._on_dev_param(k, v, f))
            sl.set(float(self.devcfg.get(key)))
            sl.grid(row=row, column=1, sticky="ew", pady=4)
            lab = ctk.CTkLabel(pr, text=fmt(float(self.devcfg.get(key))), text_color=INK, width=64,
                               anchor="e", font=(self.FONT_MONO, 11))
            lab.grid(row=row, column=2, padx=(8, 8))
            self.dev_param_labels[key] = lab

        slider(0, "reversal_window", "JANELA DA INVERSAO DA RODA", 0.2, 1.0, 16, lambda v: f"{v:.2f} s")
        slider(1, "tap_max", "TOQUE DUPLO NO MUTE · MAXIMO", 2.0, 10.0, 16, lambda v: f"{v:.1f} s")
        slider(2, "idle_switch", "FONE PARADO ATE TROCAR O MIC", 3.0, 30.0, 27, lambda v: f"{v:.0f} s")

        ctk.CTkLabel(pr, text="MIC RESERVA (QUANDO O FONE ESTA MUDO OU DESLIGADO)", text_color=INK_3,
                     anchor="w", font=(self.FONT_UI, 10, "bold")).grid(
            row=3, column=0, columnspan=3, sticky="w", padx=8, pady=(8, 0))
        self.dev_sources = devmod.list_sources()
        self.dev_source_labels = ["automatico (primeiro que nao e o fone)"] + [
            devmod.pretty_source(x) for x in self.dev_sources]
        cur = self.devcfg.get("fallback_source")
        current = (devmod.pretty_source(cur) if cur in self.dev_sources
                   else self.dev_source_labels[0])
        self.dev_fallback = ctk.CTkComboBox(
            pr, values=self.dev_source_labels, command=self._on_dev_fallback, state="readonly",
            height=30, corner_radius=8, fg_color=SURFACE_2, border_color=BORDER,
            button_color=SURFACE_2, button_hover_color=SURFACE_3, dropdown_fg_color=SURFACE_2,
            dropdown_hover_color=SURFACE_3, dropdown_text_color=INK, text_color=INK,
            font=(self.FONT_UI, 12))
        self.dev_fallback.set(current)
        self.dev_fallback.grid(row=4, column=0, columnspan=3, sticky="ew", padx=8, pady=(4, 4))

        # eventos
        ctk.CTkLabel(wrap, text="EVENTOS", text_color=INK_3, anchor="w",
                     font=(self.FONT_UI, 10, "bold")).pack(fill="x", padx=14, pady=(8, 0))
        self.dev_feed = ctk.CTkTextbox(wrap, fg_color=SURFACE_2, text_color=INK_2,
                                       font=(self.FONT_MONO, 11), height=150, corner_radius=10,
                                       border_width=0)
        self.dev_feed.pack(fill="x", padx=6, pady=(4, 8))
        self.dev_feed.configure(state="disabled")
        self._dev_feed_key = None
        return wrap

    def _devices_tick(self):
        """Atualiza status e eventos a cada segundo enquanto a aba OMARCHY esta aberta."""
        if self._tab != "omarchy" or self.gestures is None:
            return
        g = self.gestures
        st = g.status()
        lab = self.dev_status_labels
        lab["headset"].configure(text="detectado" if st["headset"] else "nao encontrado",
                                 text_color=INK if st["headset"] else ACCENT_TEXT)
        if st["access"]:
            lab["access"].configure(text="ok", text_color=INK)
            self.dev_udev_btn.pack_forget()
        else:
            lab["access"].configure(text="sem regra udev" if not st["udev"] else "sem permissao",
                                    text_color=ACCENT_TEXT)
            if not self.dev_udev_btn.winfo_ismapped():
                self.dev_udev_btn.pack(side="right", padx=12, pady=10)
        lab["mic"].configure(text=st["mic_state"] if st["running"] else "gestos desligados")
        lab["default"].configure(text=devmod.pretty_source(st["default"]))
        key = (len(g.events), g.events[0] if g.events else None)
        if key != self._dev_feed_key:
            self._dev_feed_key = key
            self.dev_feed.configure(state="normal")
            self.dev_feed.delete("1.0", "end")
            self.dev_feed.insert("end", "\n".join(f"{t}  {m}" for t, m in list(g.events)[:40])
                                 or "Nenhum evento ainda. Gire a roda pra cima e pra baixo rapido.")
            self.dev_feed.configure(state="disabled")
        self.root.after(1000, self._devices_tick)

    def _on_device_gesture(self, _why: str):
        """Gesto do fone: liga/desliga o ditado; com Enter no fim se auto_enter estiver ligado."""
        _perf("activation_request", source="headset", reason=_why)
        self.hotkey_queue.put(("toggle", {"enter": bool(self.devcfg.get("auto_enter", True))}))

    def _on_dev_switch(self, key: str):
        val = bool(self.dev_switches[key].get())
        self.devcfg[key] = val
        self._save()
        if key == "enabled":
            (self.gestures.start if val else self.gestures.stop)()
        self.status.configure(
            text=f"Omarchy: {self.dev_switch_labels[key]} — {'ligado' if val else 'desligado'}.")

    def _on_dev_param(self, key: str, value, fmt):
        self.devcfg[key] = round(float(value), 2)
        self.dev_param_labels[key].configure(text=fmt(float(value)))
        if self._dev_save_job is not None:  # o slider dispara dezenas de vezes por arraste
            self.root.after_cancel(self._dev_save_job)
        self._dev_save_job = self.root.after(400, self._save)

    def _on_dev_fallback(self, label: str):
        idx = self.dev_source_labels.index(label) if label in self.dev_source_labels else 0
        self.devcfg["fallback_source"] = None if idx == 0 else self.dev_sources[idx - 1]
        self._save()

    def _dev_install_udev(self):
        self.dev_udev_btn.configure(state="disabled", text="Instalando...")

        def work():
            ok, msg = devmod.install_udev_rule()
            self._ui_queue.put((self._dev_install_done, (ok, msg), {}))

        threading.Thread(target=work, daemon=True).start()

    def _dev_install_done(self, ok: bool, msg: str):
        self.dev_udev_btn.configure(state="normal", text="Instalar regra udev")
        self.status.configure(text=("Omarchy: " if ok else "ERRO udev: ") + msg)
        if ok and self.gestures.running():
            self.gestures.restart()

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
        device = self.mic_by_label.get(self.mic.get())
        self.settings["device_name"] = device.name if device else None
        self._update_mic_help()
        self._save()

    def _update_mic_help(self):
        device = self.mic_by_label.get(self.mic.get())
        self.mic_help.configure(
            text=device.description if device else "Nenhuma entrada de microfone encontrada.")

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
        device = self.mic_by_label.get(self.mic.get())
        return device.index if device else None

    def _pc_device_index(self):
        name = self.settings["loopback_device_name"]
        if name not in self.loopback_devices:
            return None  # None + loopback = saida padrao do sistema
        return self.loopback_devices[name]

    def _start(self, inject: bool, auto_enter: bool = False):
        if self.transcriber.comparing.is_set():
            self.status.configure(text="Comparacao em andamento — termine ou cancele na aba COMPARAR.")
            return False
        if self.transcriber.model is None or self.transcriber.model_loading.is_set():
            self.status.configure(text="Modelo ainda carregando — aguarde.")
            return False
        try:
            self.transcriber.start(
                self._device_index(), inject,
                capture_mode=self.settings["capture_mode"],
                loopback_index=self._pc_device_index(),
                auto_enter=auto_enter,
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
    def _ui(self, fn, *args, **kwargs):
        """Agenda chamada de widget para a thread do Tk. Nao usar Tk daqui de outra thread."""
        self._ui_queue.put((fn, args, kwargs))

    def _apply_model(self):
        if self.transcriber.busy():
            self.status.configure(text="Aguarde o ditado ou arquivo atual terminar antes de trocar o modelo.")
            return
        selection = {
            "whisper_model": next(k for k, v in MODEL_LABELS.items() if v == self.model_choice.get()),
            "whisper_device": next(k for k, v in self.device_labels.items() if v == self.device_choice.get()),
        }
        self._begin_model_load(selection, persist=True)

    def _begin_model_load(self, selection, *, persist):
        # Set the gate on Tk's thread before launching work: queued hotkeys cannot race it.
        self.transcriber.model_loading.set()
        self.record_btn.configure(state="disabled")
        self.apply_model_btn.configure(state="disabled")
        self.model_choice.configure(state="disabled")
        self.device_choice.configure(state="disabled")
        self.model_label.configure(text="Preparando modelo...")
        threading.Thread(target=self._load_model, args=(selection, persist), daemon=True).start()

    def _load_model(self, selection, persist):
        error = None
        try:
            self.transcriber.load_model(selection)
        except Exception as e:
            traceback.print_exc()
            error = str(e)
        self._ui(self._model_load_done, selection, persist, error)

    def _model_load_done(self, selection, persist, error):
        config = self.transcriber.model_config
        ready = self.transcriber.model is not None
        if error is None and persist:
            self.settings.update(normalize_model_settings(selection))
            try:
                self._save()
            except OSError as e:
                self.status_queue.put(f"Modelo ativo, mas nao foi possivel salvar a preferencia: {e}")
        self.model_label.configure(text=config.label if config else "Modelo indisponivel")
        self.record_btn.configure(state="normal" if ready else "disabled")
        self.apply_model_btn.configure(state="normal")
        self.model_choice.configure(state="readonly")
        self.device_choice.configure(state="readonly")
        if error:
            self.status_queue.put("ERRO ao carregar: " + error +
                                  (" — modelo anterior mantido." if ready else " — escolha modelo menor ou CPU e aplique."))

    def _poll(self):
        try:
            while True:
                fn, args, kwargs = self._ui_queue.get_nowait()
                fn(*args, **kwargs)
        except queue.Empty:
            pass
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
                elif event == "error":
                    self.status.configure(text=f"ERRO: {payload}")
                elif event == "start":
                    if not self._start(inject=True, auto_enter=_wants_enter(payload)):
                        self.hotkey.active = False  # falhou: nao deixa o estado do atalho preso
                elif event == "stop":
                    if _wants_enter(payload):
                        self.transcriber.arm_auto_enter()
                    self._stop()
                elif event == "meeting":
                    if self.meeting_panel is not None:
                        self.meeting_panel.command(payload)
                elif event == "toggle":
                    if self.transcriber.recording.is_set():
                        if _wants_enter(payload):
                            self.transcriber.arm_auto_enter()
                        self._stop()
                    elif not self._start(inject=True, auto_enter=_wants_enter(payload)):
                        self.hotkey.active = False
        except queue.Empty:
            pass
        try:
            while True:
                chunk = self.text_queue.get_nowait()
                self.text.insert("end", chunk)
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
        self.root.after(UI_POLL_MS, self._poll)


if __name__ == "__main__":
    if IS_WIN:
        # identidade propria na taskbar: sem isto o Windows agrupa sob o pythonw
        # generico e mostra o icone do Python em vez do sussurro.ico da janela
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Lucas.Sussurro")
    ctk.set_appearance_mode("dark")
    # className no Linux casa com StartupWMClass=Sussurro do .desktop (no Windows o
    # AppUserModelID ja cuida do agrupamento na taskbar).
    root = ctk.CTk() if IS_WIN else ctk.CTk(className="Sussurro")
    App(root)
    root.mainloop()
