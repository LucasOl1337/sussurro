"""Sussurro — voice-to-text local: mic -> Silero VAD -> faster-whisper (CPU ou CUDA).

HUD Tkinter: gravar/parar, atalho global de mouse (digita onde o cursor estiver),
fonte de captura (microfone, audio do PC via loopback ou os dois misturados),
modo de transcricao (simultaneo por trecho ou tudo ao final).

Windows e Linux (X11/Pulse ou PipeWire). CPU ou GPU NVIDIA opcional.
"""

import sys
from sussurro_ipc import IPC_SOCK, cli as _cli, ipc_send, already_running
from sussurro_hypr import Hypr
import sussurro_devices as devmod

# O atalho sai ANTES de carregar interface, audio e bibliotecas CUDA.
if __name__ == "__main__" and _cli(sys.argv):
    raise SystemExit(0)

if __name__ == "__main__":
    try:
        running = already_running()
    except OSError as e:
        print(f"Nao foi possivel verificar o Sussurro existente: {e}", file=sys.stderr)
        raise SystemExit(1)
    if running:
        try:
            ipc_send("show")
        except OSError as e:
            print(f"Nao foi possivel reabrir o Sussurro: {e}", file=sys.stderr)
            raise SystemExit(1)
        print("Sussurro ja esta rodando.")
        raise SystemExit(0)

import collections
import gc
import ctypes
import errno
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
import tempfile
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox, simpledialog
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
import sussurro_icons as icons
if not IS_WIN:
    from sussurro_meeting_ui import MeetingPanel
import sounddevice as sd
from faster_whisper import WhisperModel
from faster_whisper.utils import download_model
from sussurro_audio import pipewire_inputs, prepare_input_devices
from sussurro_hardware import (detect_hardware, execution_device_labels,
                               execution_hardware_note)
from sussurro_models import (LOCAL_DEFAULT, MEETING_LOCAL, MODEL_LABELS, DEFAULT_MODEL_SETTINGS, engine_label,
                             fallback_config, normalize_meeting_engine, normalize_model_settings,
                             resolve_model_config, model_path)
from sussurro_cloud import ELEVENLABS, PROVIDER_NAMES, XAI, CloudError, CloudModel, is_cloud, require_credentials
from sussurro_usage import fmt_credits, fmt_minutes, fmt_usd, shared_ledger, spend_line
import sussurro_sounds
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

# Paleta clara no molde do Hub do Wispr Flow: papel quente, tinta quase preta, um so
# cinza de chapa (#eeebe3) para hover, item ativo e botao neutro.
BG = "#f5f4f0"             # fundo do app e da barra lateral
SURFACE = "#fcfcfb"        # cartao principal
SURFACE_2 = "#f5f4f0"      # cartao interno (estatisticas, configuracoes)
SURFACE_3 = "#eeebe3"      # hover, item ativo, botao neutro
SURFACE_4 = "#e4e0d5"      # hover do botao neutro
WHITE = "#ffffff"          # campo (dropdown, entrada)
BORDER = "#eeebe3"
BORDER_STRONG = "#dedad0"
INK = "#1a1a1a"
INK_2 = "#3f3f3c"
INK_3 = "#71716e"
ACCENT = "#1a1a1a"         # botao principal: chapa escura, texto claro
ACCENT_HOVER = "#383835"
ACCENT_TEXT = "#b4531a"    # aviso/erro como texto sobre o claro
COPIED_BG = "#e6f2e8"      # linha recem-copiada no historico (pisca e volta)
COPIED_INK = "#1f7a3a"
SELECT_BG = "#d9e4f2"      # trecho selecionado com o mouse no historico
HOVER_ROW = "#f7f6f2"      # linha do historico sob o mouse
GRAPHITE = "#fcfcfb"       # tinta sobre o botao principal
TEAL = "#1a5c5c"           # graficos
TEAL_2 = "#369489"
# barra de gravacao e indicador de reuniao continuam pilulas escuras
PILL_INK = "#f2f3f3"
PILL_INK_2 = "#c3c6c8"
PILL_INK_3 = "#9ba0a4"
ENGINE_COLORS = {"ok": "#1f7a3a", "aviso": ACCENT_TEXT, "off": "#c0392b"}  # EM USO AGORA


def pick_font(candidates, fallback):
    installed = set(tkfont.families())
    for name in candidates:
        if name in installed:
            return name
    return fallback


FONTS_DIR = Path(__file__).with_name("assets") / "fonts"


def register_bundled_fonts():
    """Figtree e EB Garamond (OFL) vem em assets/fonts. Registra antes de abrir o Tk:
    no Windows so para este processo; no Linux copia pro diretorio de fontes do usuario
    (o Xft do Tk le o fontconfig uma vez, na primeira fonte usada)."""
    files = sorted(FONTS_DIR.glob("*.ttf"))
    if not files:
        return
    if IS_WIN:
        for f in files:
            try:
                ctypes.windll.gdi32.AddFontResourceExW(str(f), 0x10, 0)  # FR_PRIVATE
            except (OSError, AttributeError):
                pass
        return
    dest = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "fonts" / "sussurro"
    try:
        dest.mkdir(parents=True, exist_ok=True)
        changed = False
        for f in files:
            alvo = dest / f.name
            if not alvo.exists() or alvo.stat().st_size != f.stat().st_size:
                shutil.copyfile(f, alvo)
                changed = True
        if changed and shutil.which("fc-cache"):
            subprocess.run(["fc-cache", "-f", str(dest)], capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        pass

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
# Falha CUDA repetida descarta o modelo; o app recarrega sozinho (VRAM costuma voltar em segundos).
CUDA_DISCARD_STATUS = "ERRO: modelo descartado apos falha CUDA. Recarregando sozinho em instantes."
MODEL_RECOVERY_DELAY_MS = 5000
# Nuvem caiu: os proximos trechos vao direto pro Turbo local por um minuto (sem esperar o
# timeout de novo), e o Turbo emprestado sai da VRAM depois de dois minutos parado.
CLOUD_RETRY_S = 60.0
LOCAL_IDLE_S = 120.0

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
    "meeting_pos": {"monitor": None, "rel": [0.5, 1.0]},  # indicador de reuniao
    "feedback_sounds": False,
    "meeting_engine": LOCAL_DEFAULT,  # qualquer modelo do seletor (aba REUNIAO)
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
    settings["meeting_engine"] = normalize_meeting_engine(settings.get("meeting_engine"))
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
        found = pipewire_inputs(devices, _pipewire_sources(), devmod.default_source())
        if found:
            return found
        system_microphone = devmod.pretty_source(devmod.default_source())
        if system_microphone == "—":
            system_microphone = ""
    return prepare_input_devices(devices, system_microphone)


def _pipewire_sources() -> list:
    """`pactl -f json list sources`; lista vazia se o pactl faltar ou nao souber JSON."""
    try:
        r = subprocess.run(["pactl", "-f", "json", "list", "sources"], capture_output=True,
                           text=True, timeout=3)
        sources = json.loads(r.stdout) if r.returncode == 0 else []
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return []
    return sources if isinstance(sources, list) else []


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


class OwnWindowTarget(RuntimeError):
    """Ditado com o proprio Sussurro em foco: so guarda no historico, como antes."""

    def __init__(self):
        super().__init__("O proprio Sussurro estava em foco: texto salvo no historico, sem colar.")


def prepare_paste_target(target=None):
    """Garante janela + campo sob o mouse antes do Ctrl+V.

    Devolve a classe da janela alvo para escolher o atalho (terminal vs ctrl_v).
    """
    if IS_WIN:
        return None
    h = _hypr()
    if not h.available:
        cls = focused_window_class()
        if cls in {"Sussurro", "SussurroBar"}:
            raise OwnWindowTarget()
        return cls
    win = h.focus_at_cursor(target) if target else h.focus_at_cursor()
    time.sleep(0.04)
    cls = None
    if isinstance(win, dict):
        cls = win.get("class") or win.get("initialClass")
    cls = cls or focused_window_class()
    if cls in {"Sussurro", "SussurroBar"}:
        raise OwnWindowTarget()
    strategy = paste_strategy(cls)
    if strategy == "ctrl_v":
        pos = h.native_cursorpos()
        hit = h.window_at(*pos, include_bar=True) if pos else None
        # O foco ja foi confirmado. Clicar na barra (ou num novo alvo) o roubaria.
        if win and hit and hit.get("address") == win.get("address"):
            click_at_cursor()
            time.sleep(0.03)
    return cls


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


def _library_words(s: str) -> list:
    return [w for w in re.split(r"[\s-]+", s) if w]


def _library_key(s: str) -> str:
    """Normaliza variante da Biblioteca: minuscula, hifen vira espaco."""
    return " ".join(_library_words(s.lower()))


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
        alts = []
        for wrong, certo in pairs:
            mapping.setdefault(_library_key(wrong), certo)
            words = _library_words(wrong)
            if words:
                # separador [\s-]+: o whisper gosta de hifenizar o que ouviu junto ("9-router")
                alts.append(r"[\s-]+".join(re.escape(w) for w in words))
        if not alts:
            self._rules = None
            return
        alt = "|".join(alts)
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
            certo = mapping.get(_library_key(m.group(0)))
            if certo is None:
                return m.group(0)
            trocas += 1
            return certo

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
        self._local = None            # Turbo emprestado: nuvem caiu ou reuniao com modelo de nuvem
        self._local_config = None
        self._local_used = 0.0
        self._fallback_device = "auto"
        self._cloud_down_until = 0.0
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
        self._delivery_queue: queue.Queue = queue.Queue()
        self._session_parts: list = []
        self.last_engine = None       # quem transcreveu o ultimo trecho (engine_label)
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
        self._session_lock = threading.RLock()  # cancel vs aceite/publicacao do ditado
        self._retrying = threading.Event()
        self._mix_buffers: list = []  # buffers por stream; o mixer alinha e soma
        self._slot_last_block: list = []
        self._slot_labels: list = []
        self._loopback_slots: set = set()
        self._dead_slots: set = set()
        self._session_inject = False
        self._session_target = None
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
        threading.Thread(target=self._delivery_loop, daemon=True).start()
        threading.Thread(target=self._mixer_loop, daemon=True).start()
        threading.Thread(target=self._local_reaper, daemon=True).start()

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
                settings = normalize_model_settings(settings or DEFAULT_MODEL_SETTINGS)
                config = resolve_model_config(settings)
                self._replace_model_locked(config)
                self._fallback_device = settings["whisper_device"]
                self._cloud_down_until = 0.0
        finally:
            self.model_loading.clear()

    def engine_status(self) -> tuple[str, str]:
        """O que transcreve o proximo trecho e se e o escolhido: ("texto", ok|aviso|off).
        So le atributos: o poll da UI chama sem tomar o _model_lock."""
        if self.model_loading.is_set():
            return "carregando o modelo...", "aviso"
        config, model = self.model_config, self.model
        if model is None or config is None:
            return "nenhum modelo carregado", "off"
        if config.engine != "cloud":
            return engine_label(config.model, config.device), "ok"
        chosen = PROVIDER_NAMES[config.model]
        left = self._cloud_down_until - time.monotonic()
        if left > 0:
            local = self._local_config
            turbo = (engine_label(local.model, local.device) if local
                     else engine_label(LOCAL_DEFAULT, self._fallback_device))
            return f"{turbo} — {chosen} fora, tenta de novo em {math.ceil(left)} s", "aviso"
        used = getattr(model, "last_provider", None)
        if used and used != config.model:
            return f"{PROVIDER_NAMES[used]} · nuvem — {chosen} falhou no último trecho", "aviso"
        return engine_label(config.model), "ok"

    def _is_cloud_locked(self):
        return self.model_config is not None and self.model_config.engine == "cloud"

    def _clear_model_locked(self):
        if self._is_cloud_locked() and self.model is not None:
            self.model.close()  # fecha o pool HTTP
        self.model = None
        self.model_config = None
        gc.collect()

    def _release_local_locked(self):
        if self._local is not None:
            self._local = None
            self._local_config = None
            gc.collect()
            _perf("local_fallback_released")

    def _borrow_local_locked(self):
        """Turbo local para quando o modelo escolhido e de nuvem: carrega na hora e fica
        ate LOCAL_IDLE_S sem uso (o _local_reaper solta a VRAM)."""
        if self._local is None:
            config = fallback_config({"whisper_device": self._fallback_device})
            self.status_queue.put(f"Carregando {config.model} local...")
            t0 = time.perf_counter()
            self._local = WhisperModel(model_path(config.model, download_model), device=config.device,
                                       compute_type=config.compute_type)
            self._local_config = config
            _perf("local_fallback_loaded", model=config.model, device=config.device,
                  load_s=round(time.perf_counter() - t0, 3))
        self._local_used = time.monotonic()
        return self._local

    def _local_reaper(self):
        while True:
            time.sleep(10)
            if (self._local is None or self._meeting_jobs
                    or time.monotonic() - self._local_used < LOCAL_IDLE_S):
                continue
            if self._model_lock.acquire(blocking=False):
                try:
                    if not self._meeting_jobs and time.monotonic() - self._local_used >= LOCAL_IDLE_S:
                        self._release_local_locked()
                finally:
                    self._model_lock.release()

    def meeting_model(self, engine=MEETING_LOCAL):
        """Motor da reuniao. Grok/ElevenLabs: um CloudModel proprio, fora do _model_lock (o
        provedor separa as vozes e o ditado segue livre); o chamador fecha com close().
        Turbo/Large-v3: o modelo do ditado quando e o mesmo, o Turbo emprestado quando o
        ditado e de nuvem, ou um modelo so desta reuniao, que vai embora com ela (sem o
        _model_lock: `shares_dictation_model` diz quando o lock vale). `MEETING_LOCAL`: o
        modelo do ditado, ou o Turbo emprestado se o ditado for de nuvem.
        Chamar com o job de reuniao aberto."""
        if is_cloud(engine):
            model = CloudModel(engine)
            model.check()  # sem credencial: erro claro antes de subir audio
            return model, resolve_model_config({"whisper_model": engine})
        with self._model_lock:
            dictation = self.model_config.model if self.model_config is not None else None
            if engine == MEETING_LOCAL or (engine == LOCAL_DEFAULT and self._is_cloud_locked()):
                if self._is_cloud_locked():
                    return self._borrow_local_locked(), self._local_config
                return self.model, self.model_config
            if engine == dictation:
                return self.model, self.model_config
            if self._local_config is not None and self._local_config.model == engine:
                self._local_used = time.monotonic()
                return self._local, self._local_config
        config = resolve_model_config({"whisper_model": engine, "whisper_device": self._fallback_device})
        t0 = time.perf_counter()
        model = WhisperModel(model_path(config.model, download_model), device=config.device,
                             compute_type=config.compute_type)
        _perf("meeting_model_loaded", model=config.model, device=config.device,
              load_s=round(time.perf_counter() - t0, 3))
        return model, config

    def shares_dictation_model(self, model):
        """O modelo da reuniao e o do ditado (ou o Turbo emprestado): usa sob o _model_lock."""
        return model is not None and (model is self.model or model is self._local)

    def _replace_model_locked(self, config):
        """Chamador segura _model_lock; nunca mantem dois modelos na VRAM."""
        if config.engine != "cloud":
            self.status_queue.put(f"Preparando {config.model}: primeiro uso pode baixar o modelo...")
        path = self._weights(config)  # falha no download preserva o modelo atual
        previous = self.model_config
        if config.engine == "cloud":
            require_credentials(config.model)  # sem credencial: mantem o modelo atual
        self._clear_model_locked()
        self._release_local_locked()
        try:
            self._load_model_config(config, path)
        except Exception:
            if previous is not None:
                try:
                    self._load_model_config(previous, self._weights(previous, local_only=True))
                except Exception:
                    self.model = None
                    self.model_config = None
            raise

    def _weights(self, config, local_only=False):
        if config.engine == "cloud":
            return None
        if local_only:
            return download_model(config.model, local_files_only=True)
        return model_path(config.model, download_model)

    def _load_model_config(self, config, path):
        self.status_queue.put(f"Carregando {config.label}...")
        t0 = time.perf_counter()
        if config.engine == "cloud":
            model = CloudModel(config.model)
            model.check()
            get_speech_timestamps(np.zeros(SAMPLE_RATE, dtype=np.float32), VAD_OPTIONS)
            self.model = model
            self.model_config = config
            _perf("model_ready", device=config.device, model=config.model, load_s=0)
            self.status_queue.put(f"{config.label} pronto. Pode gravar. "
                                  "O audio do ditado vai pra nuvem; reuniao continua local.")
            return
        model = WhisperModel(path, device=config.device, compute_type=config.compute_type)
        try:
            silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
            segments, _ = model.transcribe(silence, language="pt", beam_size=5)
            collections.deque(segments, maxlen=0)
            get_speech_timestamps(silence, VAD_OPTIONS)
        except Exception:
            # Explicitly free failed weights before attempting to recover the old model.
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
                if self.recording.is_set() and data.size:
                    self._slot_last_block[slot] = time.monotonic()
                    self._dead_slots.discard(slot)
                    self._mix_buffers[slot].append(data)

        return cb

    def _mixer_loop(self):
        while True:
            time.sleep(0.01)
            with self._mix_lock:
                if self.recording.is_set():
                    now = time.monotonic()
                    for slot, last in enumerate(self._slot_last_block):
                        if now - last > 2.0 and slot not in self._dead_slots:
                            self._dead_slots.add(slot)
                            # Monitor pode ficar sem blocos no silencio. Libera o
                            # mixer, mas so a saida da thread prova sua morte.
                            if slot not in self._loopback_slots:
                                self.status_queue.put(
                                    f"ERRO: {self._slot_labels[slot]} parou de enviar áudio")
                self._drain_mix_locked()

    def _drain_mix_locked(self, flush: bool = False):
        """Enfileira sob o lock: nenhum bloco ultrapassa o marcador de stop."""
        while self._mix_buffers:
            for buf in self._mix_buffers:
                while buf and not buf[0].size:
                    buf.pop(0)
            active = [buf for buf in self._mix_buffers if buf]
            waiting = any(not buf and slot not in self._dead_slots
                          for slot, buf in enumerate(self._mix_buffers))
            if not active or (not flush and waiting):
                return
            n = min(buf[0].size for buf in active)
            mixed = np.zeros(n, dtype=np.float32)
            for buf in active:
                mixed += buf[0][:n]
                buf[0] = buf[0][n:]
            np.clip(mixed, -1.0, 1.0, out=mixed)
            self._push_levels(mixed)
            self._audio_queue.put((self._session_id, mixed))

    def _push_levels(self, chunk: np.ndarray):
        """RMS em fatias de ~25 ms: e o que a barra de overlay desenha como onda."""
        step = max(1, SAMPLE_RATE // LEVEL_HZ)
        for i in range(0, chunk.size, step):
            part = chunk[i:i + step]
            if part.size:
                self.levels.append(float(np.sqrt(np.mean(part * part))))

    def _loopback_loop(self, slot: int, label: str, device_index: int | None, handle):
        try:
            self._capture_loopback(slot, label, device_index, handle)
        finally:
            while not self.recording.is_set() and not handle.stop_flag.is_set():
                time.sleep(0.02)
            with self._mix_lock:
                if self.recording.is_set() and not handle.stop_flag.is_set():
                    self._dead_slots.add(slot)
                    self.status_queue.put("ERRO: áudio do PC parou de enviar áudio")

    def _capture_loopback(self, slot: int, label: str, device_index: int | None, handle):
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
                                self._slot_last_block[slot] = time.monotonic()
                                self._dead_slots.discard(slot)
                                self._mix_buffers[slot].append(data)
        except Exception as e:
            self.status_queue.put(f"ERRO ({label}): {e}")

    def _open_stream(self, device_index: int | None, loopback: bool, label: str,
                     source: str | None = None):
        """Abre mic (sounddevice) ou audio do PC (soundcard loopback). `source` grava uma
        fonte PipeWire especifica pela rota "pulse": o libpulse le PULSE_SOURCE ao abrir."""
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
            previous_source = os.environ.get("PULSE_SOURCE")
            try:
                if source:
                    os.environ["PULSE_SOURCE"] = source
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
            finally:
                if source:
                    if previous_source is None:
                        os.environ.pop("PULSE_SOURCE", None)
                    else:
                        os.environ["PULSE_SOURCE"] = previous_source

    def start(self, device_index: int | None, inject: bool,
              capture_mode: str = "microfone", loopback_index: int | None = None,
              auto_enter: bool = False, mic_source: str | None = None):
        if self.comparing.is_set():
            raise RuntimeError("Comparacao de modelos em andamento.")
        if self.model_loading.is_set():
            raise RuntimeError("Aguarde a troca do modelo terminar.")
        if self.recording.is_set():
            return
        if self.busy():
            raise RuntimeError("Aguarde o ditado anterior terminar.")
        self._session_inject = inject
        self._session_target = None
        if inject and not IS_WIN:
            h = _hypr()
            pos = h.native_cursorpos() if h.available else None
            if pos:
                win = h.window_at(*pos)
                if win and win.get("address"):
                    self._session_target = {"address": win["address"]}
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
        with self._mix_lock:
            self._mix_buffers = [[], []] if capture_mode == "os_dois" else [[]]
            self._slot_labels = {"microfone": ["microfone"], "audio_pc": ["áudio do PC"],
                                 "os_dois": ["microfone", "áudio do PC"]}[capture_mode]
            self._slot_last_block = [time.monotonic()] * len(self._mix_buffers)
            self._loopback_slots = ({len(self._mix_buffers) - 1}
                                    if capture_mode != "microfone" else set())
            self._dead_slots.clear()
        self._slot = 0
        try:
            if capture_mode != "audio_pc":
                self._open_stream(device_index, False, "mic", source=mic_source)
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
        with self._mix_lock:
            self._slot_last_block = [time.monotonic()] * len(self._mix_buffers)
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
            self._audio_queue.put((self._session_id, None))  # depois do ultimo bloco do mixer
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

    def begin_meeting_job(self, cloud=False):
        """A reuniao usa o modelo do ditado (sob o _model_lock so no whisper); a troca de
        modelo e a comparacao esperam, o ditado segue entre um lado e outro. Reuniao na
        nuvem (`cloud`) nao toca no modelo do ditado e nao espera ele."""
        with self._pending_lock:
            if cloud:
                return  # modelo proprio: nao segura troca de modelo, comparacao nem historico
            if self.model is None or self.model_loading.is_set():
                raise RuntimeError("O modelo ainda esta carregando.")
            if self.comparing.is_set():
                raise RuntimeError("Comparacao de modelos em andamento.")
            if self._retrying.is_set():
                raise RuntimeError("Aguarde a transcricao do historico terminar.")
            self._meeting_jobs += 1

    def end_meeting_job(self, cloud=False):
        if cloud:
            return
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
        with self._mix_lock:
            self.recording.clear()
            with self._session_lock:
                self._session_id += 1  # tudo que estiver na fila com o id antigo vira lixo
            self._mix_buffers = [[] for _ in self._mix_buffers]
        streams, self._streams = self._streams, []
        for s in streams:
            s.stop()
            s.close()
        for q in (self._audio_queue, self._segment_queue, self._delivery_queue):
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
        buffer_sid = None
        while True:
            item = self._audio_queue.get()
            try:
                if item is _CANCEL:  # sessao cancelada: o buffer acumulado morre aqui
                    buffer = np.zeros(0, dtype=np.float32)
                    final_blocks.clear()
                    continue
                sid, item = item
                if sid != self._session_id:
                    continue
                if sid != buffer_sid:
                    buffer = np.zeros(0, dtype=np.float32)
                    final_blocks.clear()
                    last_check = 0.0
                    buffer_sid = sid
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
                    if sid != self._session_id:
                        continue
                    if speech:
                        lead_s = speech[0]["start"] / SAMPLE_RATE
                        self._enqueue_segment(buffer, lead_s, sid=sid)
                    elif not self._session_had_speech:
                        self.status_queue.put("Parado (sem fala detectada).")
                    else:
                        self.status_queue.put("Parado.")
                    # marcador de fim de sessao (carimbado: cancelamento o invalida)
                    self._segment_queue.put((None, None, None, sid, 0.0))
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
                if sid != self._session_id:
                    continue
                if not speech:
                    if buffer.size > MAX_IDLE_BUFFER_S * SAMPLE_RATE:
                        buffer = buffer[-SAMPLE_RATE:]
                    continue
                last_end = speech[-1]["end"]
                lead_s = speech[0]["start"] / SAMPLE_RATE
                tail_silence = (buffer.size - last_end) / SAMPLE_RATE
                if tail_silence >= TAIL_SILENCE_S:
                    self._enqueue_segment(buffer[:last_end], lead_s, sid=sid)
                    buffer = buffer[last_end:]
                elif buffer.size > MAX_SEGMENT_S * SAMPLE_RATE:
                    # A ultima fala ainda pode estar em andamento: aproveita a pausa anterior.
                    cut = speech[-2]["end"] if len(speech) > 1 else buffer.size
                    self._enqueue_segment(buffer[:cut], lead_s, sid=sid)
                    buffer = buffer[cut:]
            except Exception as e:  # falha alto: reporta no status e mantem a thread viva
                traceback.print_exc()
                self.status_queue.put(f"ERRO na segmentacao: {e}")
                buffer = np.zeros(0, dtype=np.float32)
                final_blocks.clear()

    def _enqueue_segment(self, audio: np.ndarray, lead_s: float = 0.0, *, sid=None):
        with self._session_lock:
            sid = self._session_id if sid is None else sid
            if sid != self._session_id:
                return
            self._session_had_speech = True
            # O audio pertence ao historico mesmo se o Whisper falhar.
            self._session_audio.append(audio)
            with self._pending_lock:
                self._pending += 1
            self._segment_queue.put((audio, self._session_inject, self._session_mode,
                                     sid, lead_s))

    # -- transcricao --------------------------------------------------------
    def _transcribe_loop(self):
        while True:
            audio, inject, mode, sid, lead_s = self._segment_queue.get()
            # trecho real desta sessao: precisa dar baixa mesmo se a transcricao falhar
            deve_baixar = audio is not None and sid == self._session_id
            try:
                if sid != self._session_id:
                    continue  # sessao cancelada ou substituida: descarta sem colar nada
                if audio is None:  # finaliza so depois de entregar todos os trechos
                    self._delivery_queue.put((None, None, None, sid, None, None, False))
                    continue
                t0 = time.perf_counter()
                lang = None if self.language == "auto" else self.language
                self.last_engine = None
                segments, _info = self._transcribe_locked(
                    audio, language=lang, vad_filter=(mode == "final"))
                engine = self.last_engine
                text = format_transcript(segments)
                text, fixes = self.library.apply(text)  # troca da Biblioteca antes de sair daqui
                if sid != self._session_id:
                    continue  # cancelado enquanto este trecho transcrevia
                dt = time.perf_counter() - t0
                payload = ""
                para = self._session_emitted and lead_s >= PARAGRAPH_SILENCE_S
                self._session_parts.append((audio, text, fixes, para, engine))
                if text:
                    if self._session_emitted:
                        payload = ("\n\n" if para else " ") + text
                    else:
                        payload = text
                    self.text_queue.put(payload)
                    self._session_emitted = True
                self._delivery_queue.put((audio.size / SAMPLE_RATE, payload, self.inject_method,
                                          sid, t0 + dt, dt, bool(text and inject), engine))
                deve_baixar = False  # a entrega agora e responsavel por dar baixa
            except Exception as e:  # falha alto: reporta no status e mantem a thread viva
                traceback.print_exc()
                if deve_baixar and sid == self._session_id:
                    self._session_errors.append(f"{type(e).__name__}: {e}")
                self.status_queue.put(f"ERRO na transcricao: {e}")
            finally:
                if deve_baixar and sid == self._session_id:
                    self._pending_done()

    def _delivery_loop(self):
        """Uma fila por sessao: paste/type, Enter e historico nunca ultrapassam um trecho."""
        while True:
            audio_s, payload, method, sid, ready_at, dt, inject, *extra = self._delivery_queue.get()
            engine = extra[0] if extra else None
            try:
                if sid != self._session_id:
                    continue
                if audio_s is None:
                    try:
                        if (self._session_auto_enter and self._session_emitted
                                and self._session_inject and not self._session_errors):
                            self._press_enter(sid)
                        if sid == self._session_id:
                            self._session_auto_enter = False
                            self._finalize_session(sid=sid)
                    finally:
                        if sid == self._session_id:
                            self._drained = True
                    continue
                if inject:
                    if method == "colar":
                        self._paste(payload)
                    else:
                        self._type_fallback(payload)
                if sid != self._session_id:
                    continue  # cancelado durante o envio: nao altera a proxima sessao
                _perf("segment_done", session=sid, audio_s=round(audio_s, 3),
                      inference_ms=round(dt * 1000, 1),
                      delivery_ms=round((time.perf_counter() - ready_at) * 1000, 1),
                      stop_to_delivery_ms=(round((time.perf_counter() - self._stop_requested_at) * 1000, 1)
                                           if self._stop_requested_at is not None else None),
                      injected=inject)
                state = "Gravando — pode falar." if self.recording.is_set() else "Parado."
                via = f" com {engine}" if engine else ""
                self.status_queue.put(f"{state}  (trecho de {audio_s:.1f}s em {dt:.1f}s{via})")
            except OwnWindowTarget as e:
                if sid == self._session_id:
                    self._session_auto_enter = False  # Enter na janela do Sussurro apertaria botao
                    self.status_queue.put(str(e))
            except Exception as e:
                traceback.print_exc()
                if audio_s is not None and sid == self._session_id:
                    self._session_errors.append(f"{type(e).__name__}: {e}")
                if sid == self._session_id:
                    self.status_queue.put(f"ERRO na entrega: {e}")
            finally:
                if audio_s is not None and sid == self._session_id:
                    self._pending_done()

    def _transcribe_locked(self, audio, *, language, vad_filter, config=None):
        """Inferencia serializada: o ditado e a transcricao de arquivo (IPC) dividem um
        unico modelo na GPU, entao duas chamadas simultaneas competem pela mesma VRAM.

        hotwords enviesa a decodificacao pros termos da Biblioteca: e o que evita o
        whisper inventar "Nightingale" no lugar de "9router". Consome o gerador aqui
        dentro para a inferencia acontecer com o lock ainda tomado. Quando config
        e informado, usa esse modelo so nesta chamada e restaura o anterior.
        """
        with self._model_lock:
            if self.model_loading.is_set() or self.model is None:
                raise RuntimeError("Modelo indisponivel ou sendo trocado. Tente novamente quando estiver pronto.")
            previous = self.model_config
            temporary = config is not None and config != previous
            if temporary:
                self._replace_model_locked(config)
            unavailable = False
            try:
                if self._is_cloud_locked():
                    return self._transcribe_cloud_locked(audio, language, vad_filter)
                for attempt in range(2):
                    try:
                        segments, info = self.model.transcribe(
                            audio, language=language, beam_size=5, vad_filter=vad_filter,
                            hotwords=self.library.hotwords,
                        )
                        result = drop_hallucinations(list(segments), audio), info
                        config = self.model_config
                        self.last_engine = engine_label(config.model, config.device) if config else None
                        return result
                    except RuntimeError as error:
                        message = str(error).lower()
                        cuda = ("cuda" in message or
                                (self.model_config is not None and self.model_config.device == "cuda"
                                 and any(word in message for word in ("out of memory", "invalid device"))))
                        if not cuda and not attempt:
                            raise
                        choice = self.model_config
                        segments = None  # libera o gerador antes de descartar os pesos
                        error.__traceback__ = None
                        if not attempt and choice is not None:
                            self.status_queue.put("Erro CUDA. Recarregando o modelo para repetir o trecho...")
                            try:
                                self._clear_model_locked()
                                self._load_model_config(choice, self._weights(choice, local_only=True))
                            except Exception as reload_error:
                                error = reload_error
                            else:
                                continue
                        unavailable = True
                        self._clear_model_locked()
                        status = CUDA_DISCARD_STATUS
                        self.status_queue.put(status)
                        raise RuntimeError(status) from error
            finally:
                if temporary and not unavailable:
                    self._clear_model_locked()
                    try:
                        self._load_model_config(previous, self._weights(previous, local_only=True))
                    except Exception as error:
                        raise RuntimeError("Nao foi possivel restaurar o modelo do ditado. "
                                           "Escolha o modelo de novo para carrega-lo.") from error

    def _transcribe_cloud_locked(self, audio, language, vad_filter):
        """Nuvem primeiro; se os dois provedores falharem, o mesmo trecho sai no Turbo local
        e os proximos vao direto pra ele por CLOUD_RETRY_S (sem pagar o timeout de novo)."""
        if time.monotonic() >= self._cloud_down_until:
            try:
                segments, info = self.model.transcribe(audio, language=language,
                                                       hotwords=self.library.hotwords)
                if self._cloud_down_until:
                    self._cloud_down_until = 0.0
                    self.status_queue.put(f"{self.model_config.label} voltou.")
                _perf("cloud_segment", provider=self.model.last_provider,
                      audio_s=round(audio.size / SAMPLE_RATE, 3))
                used = getattr(self.model, "last_provider", None)
                if used not in PROVIDER_NAMES:
                    used = self.model_config.model
                self.last_engine = engine_label(used)
                if used != self.model_config.model:
                    self.last_engine += f" (reserva: {PROVIDER_NAMES[self.model_config.model]} falhou)"
                return drop_hallucinations(list(segments), audio), info
            except CloudError as error:
                self._cloud_down_until = time.monotonic() + CLOUD_RETRY_S
                _perf("cloud_failed", error=str(error)[:200])
                self.status_queue.put(f"Nuvem indisponivel ({error}). Usando o Turbo local por enquanto.")
        local = self._borrow_local_locked()
        try:
            segments, info = local.transcribe(audio, language=language, beam_size=5, vad_filter=vad_filter,
                                              hotwords=self.library.hotwords)
            result = drop_hallucinations(list(segments), audio), info
            self.last_engine = engine_label(self._local_config.model, self._local_config.device) + " (nuvem caiu)"
            return result
        finally:
            self._local_used = time.monotonic()

    @staticmethod
    def _audio_diagnosis(audio):
        # Audio normalizado [-1, 1]: ate um passo de PCM16 nao ha sinal util;
        # RMS abaixo de -50 dBFS indica entrada muito baixa, nao falha do modelo.
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if audio.size else 0.0
        level = {"peak": round(peak, 8), "rms": round(rms, 8)}
        if peak <= 1 / 32768:
            return level, "Microfone não enviou sinal (confira a entrada)."
        if rms < 10 ** (-50 / 20):
            return level, "Áudio muito baixo."
        return level, "Fala não reconhecida. Tente Refazer com outro modelo."

    def transcribe_file(self, path: str) -> dict:
        """Transcreve um arquivo de audio e arquiva no historico, sem microfone nem colagem.

        Usado pelo IPC (`sussurro transcribe <arquivo>`), que e como o Hermes manda os
        audios do WhatsApp. Retorna o mesmo dict que vai pro history.jsonl.
        """
        with self._pending_lock:
            if self.recording.is_set() or not self._drained or self._pending:
                raise RuntimeError("Modelo ocupado com ditado; aguarde o ditado terminar e tente novamente.")
            if self.model_loading.is_set() or self.comparing.is_set() or self._retrying.is_set():
                raise RuntimeError("Modelo ocupado com troca, comparacao ou historico; aguarde.")
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
                _level, diagnosis = self._audio_diagnosis(audio)
                raise ValueError(diagnosis)
            entry = self._archive_audio(audio, text, fixes, datetime.now(), engine=self.last_engine)
            _perf("file_done", source=src.name, audio_s=round(audio.size / SAMPLE_RATE, 3),
                  inference_ms=round((time.perf_counter() - t0) * 1000, 1))
            self.status_queue.put(f"Arquivo transcrito ({audio.size / SAMPLE_RATE:.1f}s).")
            return entry
        finally:
            with self._pending_lock:
                self._file_jobs -= 1


    def _archive_audio(self, audio, text: str, fixes: int, started: datetime, *,
                       failed: bool = False, error: str | None = None, sid=None,
                       engine: str | None = None) -> dict | None:
        """Grava o WAV e seu registro, inclusive quando a transcricao falhou."""
        if failed:
            level, diagnosis = self._audio_diagnosis(audio)
            if not error or error.lower().rstrip(".") == "nenhuma fala reconhecida":
                error = diagnosis
        HISTORY_DIR.mkdir(exist_ok=True)
        wav_name = started.strftime("%Y%m%d_%H%M%S") + ".wav"
        temporary = None
        target = HISTORY_DIR / wav_name
        if sid is not None:
            fd, name = tempfile.mkstemp(prefix=".ditado-", suffix=".wav", dir=HISTORY_DIR)
            os.close(fd)
            temporary = Path(name)
        try:
            with wave.open(str(temporary or target), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SAMPLE_RATE)
                w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
            # Cancel segue livre durante a escrita. So a publicacao e atomica com ele.
            with self._session_lock:
                if sid is not None and sid != self._session_id:
                    return None
                if temporary is not None:
                    os.replace(temporary, target)
                # dur e fix alimentam a aba ESTATISTICAS; entrada antiga usa wav / zero
                entry = {"ts": started.isoformat(timespec="seconds"), "wav": wav_name,
                         "text": text, "dur": round(audio.size / SAMPLE_RATE, 2), "fix": fixes}
                if engine:
                    entry["engine"] = engine
                if failed:
                    entry["failed"] = True
                    entry["error"] = error
                    entry["audio_level"] = level
                with self._history_lock:
                    with HISTORY_INDEX.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                self.history_queue.put(entry)
                if failed and sid is not None:
                    self.status_queue.put(f"ERRO: {error}")
                return entry
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _finalize_session(self, *, sid=None):
        with self._session_lock:
            sid = self._session_id if sid is None else sid
            if sid != self._session_id:
                return
            parts, self._session_parts = self._session_parts, []
            audio_parts, self._session_audio = self._session_audio, []
            errors, self._session_errors = self._session_errors, []
            started = self._session_started or datetime.now()
        if not audio_parts:
            return
        text = _join_session_text(parts)
        audio = np.concatenate(audio_parts)
        fixes = sum(f for _a, _t, f, *_ in parts)
        failed = bool(errors) or not text
        error = errors[-1] if errors else ("Nenhuma fala reconhecida." if not text else None)
        engines = list(dict.fromkeys(p[4] for p in parts if len(p) > 4 and p[4]))
        self._archive_audio(audio, text, fixes, started, failed=failed, error=error, sid=sid,
                            engine=" + ".join(engines) or None)

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

    def prune_history_audio(self, days: int, *, confirmed: bool = False, now=None, only_files=None) -> dict:
        """Previa por padrao; apaga so WAV antigo com texto, nunca o indice."""
        if type(days) is not int or not 1 <= days <= 36500:
            raise ValueError("Escolha entre 1 e 36500 dias.")
        cutoff = (now or datetime.now()) - timedelta(days=days)
        allowed = set(only_files) if only_files is not None else None
        with self._pending_lock:
            if (self.recording.is_set() or not self._drained or self._pending or self._file_jobs
                    or self._meeting_jobs or self._retrying.is_set()):
                raise RuntimeError("Aguarde o trabalho atual terminar antes de liberar espaço.")
            with self._history_lock:
                entries = [json.loads(line) for line in HISTORY_INDEX.read_text(encoding="utf-8").splitlines()
                           if line.strip()] if HISTORY_INDEX.exists() else []
                protected = {entry.get("wav") for entry in entries if not entry.get("text", "").strip()}
                for entry in entries:
                    try:
                        if datetime.fromisoformat(entry["ts"]) >= cutoff:
                            protected.add(entry.get("wav"))
                    except (KeyError, ValueError, TypeError):
                        protected.add(entry.get("wav"))
                result = {"files": [], "bytes": 0, "deleted": 0, "errors": []}
                for entry in entries:
                    name = str(entry.get("wav", ""))
                    if (not name or Path(name).name != name or Path(name).suffix.lower() != ".wav"
                            or name in protected or name in result["files"]
                            or (allowed is not None and name not in allowed)):
                        continue
                    try:
                        old = datetime.fromisoformat(entry["ts"]) < cutoff
                    except (KeyError, ValueError, TypeError):
                        continue
                    path = HISTORY_DIR / name
                    if not old or path.is_symlink() or not path.is_file():
                        continue
                    try:
                        size = path.stat().st_size
                        if confirmed:
                            path.unlink()
                            result["deleted"] += 1
                        result["files"].append(name)
                        result["bytes"] += size
                    except OSError as error:
                        result["errors"].append(f"{name}: {error}")
                return result

    def retry_history_async(self, entry: dict, selection=None, language=None) -> None:
        """Refaz qualquer WAV salvo; as escolhas valem so para esta tentativa."""
        language = self.language if language is None else language
        if language not in ("pt", "en", "auto"):
            raise ValueError("Idioma invalido.")
        wav_name = str(entry.get("wav", ""))
        if not wav_name or Path(wav_name).name != wav_name:
            raise ValueError("arquivo de historico invalido")
        path = HISTORY_DIR / wav_name
        if not path.is_file():
            raise FileNotFoundError("O audio salvo nao foi encontrado. Para refazer, "
                                    "o WAV precisa estar na pasta do historico.")
        with self._pending_lock:
            if self.model is None:
                raise RuntimeError("Modelo ainda carregando; aguarde.")
            if (self.recording.is_set() or not self._drained or self._pending or self._file_jobs
                    or self._meeting_jobs or self._retrying.is_set() or self.comparing.is_set()
                    or self.model_loading.is_set() or self._model_lock.locked()):
                raise RuntimeError("Aguarde o trabalho atual terminar.")
            self._retrying.set()
        self.status_queue.put("Refazendo a transcricao do audio salvo...")
        try:
            threading.Thread(target=self._retry_history_worker,
                             args=(dict(entry), path, dict(selection) if selection else None, language),
                             name="sussurro-history-retry", daemon=True).start()
        except Exception:
            self._retrying.clear()
            raise

    def _retry_history_worker(self, entry: dict, path: Path, selection=None, language=None) -> None:
        result = dict(entry)
        result.pop("retrying", None)
        try:
            audio = load_audio_16k_mono(path)
            if audio.size == 0:
                raise ValueError("audio vazio")
            started = time.perf_counter()
            language = self.language if language is None else language
            lang = None if language == "auto" else language
            config = resolve_model_config(selection) if selection else self.model_config
            segments, _info = self._transcribe_locked(audio, language=lang, vad_filter=True, config=config)
            text = format_transcript(segments)
            text, fixes = self.library.apply(text)
            if not text:
                level, diagnosis = self._audio_diagnosis(audio)
                if entry.get("failed") or not entry.get("text"):
                    result["audio_level"] = level
                raise ValueError(diagnosis)
            updated = {**entry, "text": text, "dur": round(audio.size / SAMPLE_RATE, 2),
                       "fix": fixes, "language": language}
            if config is not None:
                updated["model"] = config.model
                updated["device"] = config.device
            if self.last_engine:
                updated["engine"] = self.last_engine
            updated.pop("failed", None)
            updated.pop("error", None)
            updated.pop("retrying", None)
            updated.pop("retry_error", None)
            self._replace_history_entry(updated)
            result = updated
            _perf("history_retry_done", wav=path.name,
                  audio_s=round(audio.size / SAMPLE_RATE, 3),
                  inference_ms=round((time.perf_counter() - started) * 1000, 1))
            self.status_queue.put("Transcricao refeita e salva no historico.")
        except Exception as e:
            traceback.print_exc()
            result["retry_error"] = f"{type(e).__name__}: {e}"
            if entry.get("failed") or not entry.get("text"):
                result.update(failed=True, error=result["retry_error"])
            try:
                self._replace_history_entry(result)
            except Exception:
                traceback.print_exc()
            self.status_queue.put(f"ERRO ao refazer: {e}. A transcricao anterior foi mantida.")
        finally:
            self._retrying.clear()
            self.history_queue.put(result)

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
            target_class = (prepare_paste_target(self._session_target) if self._session_target
                            else prepare_paste_target())
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

    def _press_enter(self, sid=None):
        """Modo fone: confirma o envio da frase com Enter depois da ultima colagem."""
        time.sleep(0.15)  # o app alvo precisa processar o Ctrl+V antes do Enter
        if sid is not None and sid != self._session_id:
            return  # cancelado enquanto aguardava o alvo processar a colagem
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


def _native_drag_start(cls: str):
    """Ponto de partida do arraste no Hyprland: janela e cursor em coordenadas nativas."""
    h = _hypr()
    cur = h.native_cursorpos()
    win = h.own_window(cls)
    if not cur or not win or not win.get("address"):
        return None
    return (win["address"], cur, tuple(win.get("at") or (0, 0)), tuple(win.get("size") or (0, 0)))


def _native_drag_move(drag) -> bool:
    address, (cx0, cy0), (ax, ay), _size = drag
    cur = _hypr().native_cursorpos()
    if not cur:
        return False
    return _hypr().drag_to(address, ax + cur[0] - cx0, ay + cur[1] - cy0)


def _native_drag_end(drag):
    """Onde a janela parou: (monitor, [rx, ry]) como fracao da area util desse monitor."""
    h = _hypr()
    address, (cx0, cy0), (ax, ay), (w, hgt) = drag
    cur = h.native_cursorpos()
    if not cur:
        return None
    x, y = ax + cur[0] - cx0, ay + cur[1] - cy0
    mon = h.native_monitor_at(x + w / 2, y + hgt / 2)
    if mon is None:
        return None
    left, top, right, bottom = h.native_work_area(mon)
    rx = (x - left) / max(right - left - w, 1)
    ry = (y - top) / max(bottom - top - hgt, 1)
    return mon["name"], [round(min(max(rx, 0.0), 1.0), 4), round(min(max(ry, 0.0), 1.0), 4)]


class RecorderBar:
    """Barra flutuante de gravacao no desenho da pilula do Wispr Flow: cancelar (X),
    dez barras que pulsam com o nivel do audio e confirmar (V); transcrevendo, as
    barras apagam e um spinner toma o lugar do confirmar (o X volta com o mouse em cima).

    Sempre no topo, fora do Alt-Tab, sem roubar foco. Aparece no monitor onde o cursor
    esta (e segue se o cursor trocar de monitor) na posicao relativa salva; arrastavel
    pela area da onda pra reposicionar — a posicao e salva como fracao da area util,
    entao se adapta a qualquer resolucao.

    As barras seguem o RMS recente do audio JA MISTURADO (Transcriber.levels), entao
    refletem exatamente a fonte escolhida: microfone, audio do PC ou os dois somados.

    Desenho via PIL com supersampling — o canvas do Tk nao tem antialias e a capsula
    ficaria serrilhada.

    No Hyprland o arraste move a janela pelo compositor (coordenadas nativas): o Tk
    em XWayland pede `geometry` e o Hyprland ignora pedido de posicao de janela
    gerenciada, entao a barra ficava presa onde nasceu.
    """

    CLS = "SussurroBar"
    # Medidas da pilula de ditado do Wispr Flow (medidas no app rodando, 20/11/2026):
    # capsula 103x30 preta, botoes redondos de 18 px a 6 px da borda, 10 barras de 2 px
    # com passo de 4 px que pulsam juntas pelo nivel do audio, spinner de 8 tracos.
    W, H = 103, 30
    PAD = 6
    BTN = 18                       # diametro dos botoes redondos
    BARS = 10
    BAR_W = 2.0
    BAR_PITCH = 4.0
    BAR_BASE = 2.0                 # altura da barra com escala 1 (silencio)
    AUDIO_MAX = 5.5                # escala maxima pelo audio: 2 px * 5.5 * 1.5 cabe em 18 px
    # pulso de cada barra: (fracao do ciclo de 1 s, multiplicador), ease-in-out entre pontos
    PULSE = ((0.0, 1.0), (0.2, 1.2), (0.4, 1.5), (0.8, 1.1), (0.9, 1.3), (1.0, 1.0))
    PULSE_DELAYS = (0.0, 0.1, 0.2, 0.3, 0.4, -0.5, -0.4, -0.3, -0.2, -0.1)
    SPIN_S = 1.1                   # volta do spinner
    FADE_S = 0.28                  # entrada da capsula
    SS = 4                         # fator de supersampling do desenho
    FPS_MS = 33
    TRANSPARENT = "#010203"

    PILL_BG = "#000000"
    PILL_BORDER = "#30302f"
    BTN_BG = "#4d4a42"             # cancelar
    BTN_INK = "#fcfcfb"
    OK_BG = "#fcfcfb"              # confirmar
    OK_INK = "#1a1a1a"
    WAVE_ON = "#ffffff"
    WAVE_DIM = "#666666"           # branco a 40% sobre o preto
    SPIN_RGB = (163, 148, 199)

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
        span = (self.BARS - 1) * self.BAR_PITCH + self.BAR_W
        self.WX0 = (self.W - span) / 2               # borda esquerda da primeira barra
        self.WX1 = self.WX0 + span
        # transcrevendo, os botoes somem: barras puxam pra esquerda, spinner a direita
        self.PX0 = 21.0
        self.SX = self.W - 21.0

        # No Hyprland a barra e uma janela gerenciada (classe SussurroBar): a regra em
        # contrib/omarchy/sussurro.lua a deixa flutuante, fixa, sem foco e com rounding,
        # que recorta os cantos — o Tk nao tem transparencia por pixel no X11.
        self.hypr = (not IS_WIN) and _hypr().available
        self.win = tk.Toplevel(root, class_=self.CLS)
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
        # Com escala fracionaria (xwayland:force_zero_scaling) a regra de 103x30 logicos
        # vira 155x45 px no X11: a capsula acompanha o tamanho real da janela, senao a
        # sobra aparece no cinza padrao do Tk.
        self._k = 1.0
        self._px = (self.W, self.H)
        self.win.configure(bg=canvas_bg)
        self.canvas = tk.Canvas(self.win, width=self.W, height=self.H,
                                bg=canvas_bg, highlightthickness=0, cursor="fleur")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._resized)
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
        self._feedback = None  # tipo, motivo, prazo monotonic
        self._reason_font = _pil_font(8 * self.SS)
        self._hovered = None    # "cancel" | "ok" | None
        self._pressed = None
        self._press_xy = (0, 0)
        self._dragging = False
        self._moved = False
        self._drag_off = (0, 0)
        self._native_drag = None  # (address, cursor0, at0, size) no Hyprland
        self._phase = 0.0
        self._audio = 1.0       # escala das barras pelo audio, suavizada
        self._shown_at = 0.0
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
        x = int(left + rx * (right - left - self._px[0]))
        y = int(top + ry * (bottom - top - self._px[1]))
        return x, y

    def _follow(self):
        if not self._dragging:
            x, y = self._target_xy()
            if not self.hypr or not _hypr().place_bar(x, y):
                self.win.geometry(f"+{x}+{y}")

    # -- ciclo de vida -------------------------------------------------------
    def show(self, state: str):
        self._feedback = None
        if self.hypr:
            _hypr().reset_bar_placement(self.CLS)
        if self._state not in ("rec", "proc"):
            self._shown_at = time.monotonic()
            self._audio = 1.0
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

    def flash(self, kind: str, reason: str, duration_ms: int):
        active = self._state in ("rec", "proc")
        if not active:
            self.show(kind)
        self._feedback = (kind, reason, time.monotonic() + duration_ms / 1000)
        self._phase = 0.0
        self._draw()

    def finish(self):
        """O trabalho acabou, mas um aviso recente precisa terminar de aparecer."""
        if self._feedback and time.monotonic() < self._feedback[2]:
            self._state = self._feedback[0]
        else:
            self.hide()

    def hide(self):
        self._state = None
        self._feedback = None
        self._dragging = False
        self._moved = False
        self._pressed = None
        self._hovered = None
        self._native_drag = None
        self.win.withdraw()

    def _tick(self):
        if self._feedback and time.monotonic() >= self._feedback[2]:
            self._feedback = None
            if self._state in ("busy", "error"):
                self.finish()
        if self._state is None:
            self._ticking = False
            return
        self._phase += self.FPS_MS / 1000.0
        self._audio = self._next_audio(self._audio)
        self._draw()
        self._follow_n += 1
        if self._follow_n % 4 == 0:  # ~180 ms: segue o cursor de monitor sem martelar o socket
            self._follow()
        self.root.after(self.FPS_MS, self._tick)

    # -- mouse ---------------------------------------------------------------
    def _resized(self, event):
        k = min(event.width / self.W, event.height / self.H)
        if k <= 0 or abs(k - self._k) < 0.01:
            return
        self._k = k
        self._px = (max(1, round(self.W * k)), max(1, round(self.H * k)))
        self._draw()

    def _zone(self, x, y):
        x, y = x / self._k, y / self._k
        if self._state in ("busy", "error"):
            return "dismiss" if 0 <= x < self.W and 0 <= y < self.H else None
        r2 = (self.BTN / 2) ** 2
        if (x - self.LX) ** 2 + (y - self.CY) ** 2 <= r2:
            return "cancel"  # transcrevendo, o X so aparece com o mouse em cima
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
        self._native_drag = _native_drag_start(self.CLS) if self.hypr else None

    def _drag_move(self, event):
        dist = abs(event.x - self._press_xy[0]) + abs(event.y - self._press_xy[1])
        if not self._dragging:
            # arrastou de dentro de um botao: passa de clique pra reposicionamento
            if dist < 5:
                return
            self._pressed = None
            self._dragging = True
        if not self._moved and dist < 3:
            return  # tremida de clique nao conta como reposicionamento
        self._moved = True
        if self._native_drag and _native_drag_move(self._native_drag):
            return
        x = self.win.winfo_pointerx() - self._drag_off[0]
        y = self.win.winfo_pointery() - self._drag_off[1]
        self.win.geometry(f"+{x}+{y}")

    def _release(self, event):
        if self._dragging:
            moved, self._dragging, self._moved = self._moved, False, False
            if moved:  # so salva se saiu do lugar; clique simples no meio nao mexe em nada
                self._save_pos()
            self._native_drag = None
            return
        self._native_drag = None
        zone, self._pressed = self._pressed, None
        if zone and self._zone(event.x, event.y) == zone:
            if zone == "dismiss":
                self.hide()
            else:
                (self.on_cancel if zone == "cancel" else self.on_confirm)()

    def _save_pos(self):
        if self._native_drag:
            saved = _native_drag_end(self._native_drag)
            if saved is not None:
                _hypr().reset_bar_placement(self.CLS)
                self.save_rel_pos(saved[1])
                return
        # winfo_x ainda e a posicao de antes do ultimo `geometry`: usa o ponteiro
        wx = self.win.winfo_pointerx() - self._drag_off[0]
        wy = self.win.winfo_pointery() - self._drag_off[1]
        pw, ph = self._px
        cx, cy = wx + pw // 2, wy + ph // 2
        left, top, right, bottom = monitor_work_area(cx, cy)
        rx = (wx - left) / max(right - left - pw, 1)
        ry = (wy - top) / max(bottom - top - ph, 1)
        self.save_rel_pos([round(min(max(rx, 0.0), 1.0), 4),
                           round(min(max(ry, 0.0), 1.0), 4)])

    # -- desenho -------------------------------------------------------------
    def _draw(self):
        if self._state is None:
            return
        self._photo = ImageTk.PhotoImage(self._render())
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)

    def _next_audio(self, cur: float) -> float:
        """Nivel do audio -> escala das barras, com subida e descida suaves."""
        if self._state != "rec":
            return 1.0
        recent = list(self.get_levels())[-4:]  # ~100 ms
        level = sum(recent) / len(recent) if recent else 0.0
        target = 1.0 + (self.AUDIO_MAX - 1.0) * min(1.0, level * 9.0) ** 0.55
        keep = 0.85 ** (self.FPS_MS / (1000 / 60))
        return cur * keep + target * (1 - keep)

    @classmethod
    def _pulse(cls, t: float) -> float:
        t %= 1.0
        for (t0, v0), (t1, v1) in zip(cls.PULSE, cls.PULSE[1:]):
            if t <= t1:
                u = (t - t0) / (t1 - t0)
                return v0 + (v1 - v0) * u * u * (3 - 2 * u)
        return cls.PULSE[-1][1]

    def _render(self) -> Image.Image:
        s = self.SS
        img = Image.new("RGB", (self.W * s, self.H * s), self.TRANSPARENT)
        d = ImageDraw.Draw(img)
        error = bool(self._feedback and self._feedback[0] == "error")
        d.rounded_rectangle([0.5 * s, 0.5 * s, (self.W - 0.5) * s, (self.H - 0.5) * s],
                            radius=(self.H / 2 - 0.5) * s, fill=self.PILL_BG,
                            outline="#f08080" if error else self.PILL_BORDER,
                            width=max(1, round(1.0 * s)))
        active = self._state in ("rec", "proc")
        if self._feedback:
            x0, x1 = (self.LX + self.BTN / 2 + 4, self.RX - self.BTN / 2 - 4) if active \
                else (12, self.W - 12)
            self._draw_notice(img, x0, x1, "#f08080" if error else PILL_INK_2)
            if not active:
                return self._finish(img)
        else:
            self._draw_wave(d)
        if self._state == "rec":
            self._draw_cancel(d)
            self._draw_confirm(d)
        else:
            self._draw_spinner(d)
            if self._hovered == "cancel":
                self._draw_cancel(d)
        return self._finish(img)

    def _finish(self, img):
        img = img.resize(self._px, Image.BOX)
        k = (time.monotonic() - self._shown_at) / self.FADE_S
        if 0 <= k < 1:  # entrada: o fundo e preto, entao clarear do preto e o fade
            k = 1 - (1 - k) ** 3
            img = Image.blend(Image.new("RGB", img.size, self.PILL_BG), img, k)
        return img

    def _draw_notice(self, img, x0, x1, color):
        s = self.SS
        reason = self._feedback[1] if self._feedback else ""
        w = max(1, round((x1 - x0) * s))
        strip = Image.new("RGB", (w, (self.H - 8) * s), self.PILL_BG)
        text = ImageDraw.Draw(strip)
        width = text.textlength(reason, font=self._reason_font)
        mid = (self.H - 8) * s / 2
        if width > w:  # nao cabe: mostra o comeco, espera um pouco e rola
            offset = int(max(0.0, self._phase - 0.8) * 30 * s) % max(1, int(width + w / 2))
            text.text((-offset, mid), reason, font=self._reason_font, fill=color, anchor="lm")
        else:
            text.text((w / 2, mid), reason, font=self._reason_font, fill=color, anchor="mm")
        img.paste(strip, (round(x0 * s), 4 * s))

    def _draw_wave(self, d):
        s = self.SS
        rec = self._state == "rec"
        x0 = self.WX0 if rec else self.PX0
        mid = (self.BARS - 1) / 2
        for i in range(self.BARS):
            if rec:
                shape = max(0.0, 1 - (mid - i) ** 2 / 48)
                pulse = self._pulse(self._phase + self.PULSE_DELAYS[i])
                h = max(1.0, self.BAR_BASE * self._audio * shape * pulse)
            else:
                h = self.BAR_BASE
            x = x0 + i * self.BAR_PITCH
            d.rounded_rectangle([x * s, (self.CY - h / 2) * s,
                                 (x + self.BAR_W) * s, (self.CY + h / 2) * s],
                                radius=0.5 * s, fill=self.WAVE_ON if rec else self.WAVE_DIM)

    def _circle(self, d, cx, r, fill):
        s = self.SS
        d.ellipse([(cx - r) * s, (self.CY - r) * s, (cx + r) * s, (self.CY + r) * s], fill=fill)

    def _stroke(self, d, pts, cx, color, w):
        s = self.SS
        xy = [((cx + px) * s, (self.CY + py) * s) for px, py in pts]
        d.line(xy, fill=color, width=max(1, round(w * s)), joint="curve")
        r = w / 2 * s
        for x, y in (xy[0], xy[-1]):  # pontas redondas
            d.ellipse([x - r, y - r, x + r, y + r], fill=color)

    def _draw_cancel(self, d):
        self._circle(d, self.LX, self.BTN / 2, self.BTN_BG)
        a = 3.0
        self._stroke(d, [(-a, -a), (a, a)], self.LX, self.BTN_INK, 1.6)
        self._stroke(d, [(-a, a), (a, -a)], self.LX, self.BTN_INK, 1.6)

    def _draw_confirm(self, d):
        self._circle(d, self.RX, self.BTN / 2, self.OK_BG)
        self._stroke(d, [(-4.0, 0.0), (-1.5, 2.5), (4.0, -3.0)], self.RX, self.OK_INK, 1.6)

    def _draw_spinner(self, d):
        s = self.SS
        r_, g_, b_ = self.SPIN_RGB
        for i in range(8):
            delay = -0.1375 * (7 - i)
            k = ((self._phase - delay) / self.SPIN_S) % 1.0
            alpha = 0.88 - (0.88 - 0.15) * k
            color = (round(r_ * alpha), round(g_ * alpha), round(b_ * alpha))
            ang = math.radians(45 * i)
            ux, uy = math.sin(ang), -math.cos(ang)
            # traco de 2x5 px com centro a 6 px do meio
            p0 = (self.SX + ux * 4.5, self.CY + uy * 4.5)
            p1 = (self.SX + ux * 7.5, self.CY + uy * 7.5)
            d.line([(p0[0] * s, p0[1] * s), (p1[0] * s, p1[1] * s)], fill=color, width=2 * s)
            for x, y in (p0, p1):
                d.ellipse([(x - 1) * s, (y - 1) * s, (x + 1) * s, (y + 1) * s], fill=color)


class MeetingPill:
    """Indicador discreto de reuniao gravando, no estilo da pilula do Wispr Flow.

    Uma capsula pequena com ponto e tempo; os dois botoes (pausar/retomar e esconder)
    ficam apagados e acendem quando o mouse passa. Arrastavel pelo resto da capsula.
    Fica onde foi solta, no monitor em que foi solta: ao contrario da barra de ditado,
    nao persegue o cursor. Esconder vale ate a reuniao acabar; a aba REUNIAO traz
    de volta. Janela propria (classe SussurroMeeting) pra o Hyprland dar tamanho e
    cantos dela, sem brigar com a barra de ditado, que pode aparecer junto.
    """

    CLS = "SussurroMeeting"
    W, H = 112, 26
    SS = 4
    BTN_R = 9
    TICK_MS = 250

    PILL_BG = RecorderBar.PILL_BG
    PILL_BORDER = "#2c2e33"
    REC = "#e5534b"
    ICON_IDLE = "#6b6f76"
    ICON_HOVER = PILL_INK
    HOVER_BG = "#30323a"

    def __init__(self, root: tk.Tk, get_pos, save_pos, on_pause, on_dismiss=None):
        self.root = root
        self.get_pos = get_pos          # {"monitor": nome|None, "rel": [rx, ry]}
        self.save_pos = save_pos
        self.on_pause = on_pause
        self.on_dismiss = on_dismiss
        self.CY = self.H / 2
        self.PX = self.W - 36           # centro do pausar
        self.HX = self.W - 14           # centro do esconder
        self.hypr = (not IS_WIN) and _hypr().available
        self.win = tk.Toplevel(root, class_=self.CLS)
        self.win.title("Sussurro — reunião")
        if not self.hypr:
            self.win.overrideredirect(True)
            if not IS_WIN:
                try:
                    self.win.wm_attributes("-type", "dock")
                except tk.TclError:
                    pass
        self.win.attributes("-topmost", True)
        bg = RecorderBar.TRANSPARENT if IS_WIN else self.PILL_BG
        self._bg = bg
        if IS_WIN:
            self.win.attributes("-transparentcolor", bg)
        self._k = 1.0
        self._px = (self.W, self.H)
        self.win.configure(bg=bg)
        self.canvas = tk.Canvas(self.win, width=self.W, height=self.H, bg=bg,
                                highlightthickness=0, cursor="fleur")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._resized)
        self.canvas.bind("<Button-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag_move)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.canvas.bind("<Motion>", lambda e: self._set_hover(self._zone(e.x, e.y)))
        self.canvas.bind("<Leave>", lambda _e: self._set_hover(None))
        self.win.update_idletasks()
        if IS_WIN:
            RecorderBar._set_exstyle(self)
        else:
            RecorderBar._apply_shape(self)  # X11 sem Hyprland: cantos da capsula
        self.win.withdraw()

        self._shown = False
        self._dismissed = False
        self._meeting = None            # (pausada, segundos)
        self._drawn = None
        self._hovered = None
        self._pressed = None
        self._press_xy = (0, 0)
        self._dragging = False
        self._moved = False
        self._drag_off = (0, 0)
        self._native_drag = None
        self._photo = None
        self._font = _pil_font(11 * self.SS)

    # -- estado --------------------------------------------------------------
    def update(self, recording: bool, paused: bool, seconds: float):
        if not recording:
            self._meeting = None
            self._dismissed = False   # proxima reuniao volta a mostrar
            if self._shown:
                self._hide()
            return
        self._meeting = (bool(paused), int(seconds))
        if self._dismissed:
            return
        if not self._shown:
            self._show()
        self._draw()

    def visible(self) -> bool:
        return self._shown

    def reveal(self):
        """Traz de volta um indicador escondido (botao da aba REUNIAO)."""
        self._dismissed = False
        if self._meeting is not None and not self._shown:
            self._show()
            self._draw(force=True)

    def dismiss(self):
        self._dismissed = True
        self._hide()
        if self.on_dismiss:
            self.on_dismiss()

    def _show(self):
        self._shown = True
        self._drawn = None
        self.win.deiconify()
        self.win.attributes("-topmost", True)
        self._place()

    def _hide(self):
        self._shown = False
        self._dragging = self._moved = False
        self._pressed = self._hovered = self._native_drag = None
        self.win.withdraw()

    # -- posicao -------------------------------------------------------------
    def _place(self):
        pos = self.get_pos() or {}
        rx, ry = (pos.get("rel") or [0.5, 1.0])[:2]
        if self.hypr:
            h = _hypr()
            h.reset_bar_placement(self.CLS)
            mon = h.native_monitor(pos["monitor"]) if pos.get("monitor") else None
            if mon is None:
                cur = h.native_cursorpos()
                mon = h.native_monitor_at(*cur) if cur else None
            if mon is not None:
                left, top, right, bottom = h.native_work_area(mon)
                gx = round(left + rx * (right - left - self.W))
                gy = round(top + ry * (bottom - top - self.H))
                if h.place_native(self.CLS, mon["name"], gx, gy):
                    return
        cx, cy = cursor_pos()
        left, top, right, bottom = monitor_work_area(cx, cy)
        x = int(left + rx * (right - left - self._px[0]))
        y = int(top + ry * (bottom - top - self._px[1]))
        self.win.geometry(f"+{x}+{y}")

    # -- mouse ---------------------------------------------------------------
    def _resized(self, event):
        k = min(event.width / self.W, event.height / self.H)
        if k <= 0 or abs(k - self._k) < 0.01:
            return
        self._k = k
        self._px = (max(1, round(self.W * k)), max(1, round(self.H * k)))
        self._draw(force=True)

    def _zone(self, x, y):
        x, y = x / self._k, y / self._k
        r2 = (self.BTN_R + 2) ** 2
        if (x - self.PX) ** 2 + (y - self.CY) ** 2 <= r2:
            return "pause"
        if (x - self.HX) ** 2 + (y - self.CY) ** 2 <= r2:
            return "hide"
        return None

    def _set_hover(self, zone):
        if zone != self._hovered:
            self._hovered = zone
            self.canvas.configure(cursor="hand2" if zone else "fleur")
            self._draw()

    def _press(self, event):
        self._press_xy = (event.x, event.y)
        self._pressed = self._zone(event.x, event.y)
        self._dragging = self._pressed is None
        self._moved = False
        self._drag_off = (event.x, event.y)
        self._native_drag = _native_drag_start(self.CLS) if self.hypr else None

    def _drag_move(self, event):
        dist = abs(event.x - self._press_xy[0]) + abs(event.y - self._press_xy[1])
        if not self._dragging:
            if dist < 5:
                return
            self._pressed = None
            self._dragging = True
        if not self._moved and dist < 3:
            return
        self._moved = True
        if self._native_drag and _native_drag_move(self._native_drag):
            return
        x = self.win.winfo_pointerx() - self._drag_off[0]
        y = self.win.winfo_pointery() - self._drag_off[1]
        self.win.geometry(f"+{x}+{y}")

    def _release(self, event):
        drag, self._native_drag = self._native_drag, None
        if self._dragging:
            moved, self._dragging, self._moved = self._moved, False, False
            if moved:
                self._save(drag)
            return
        zone, self._pressed = self._pressed, None
        if zone and self._zone(event.x, event.y) == zone:
            if zone == "pause":
                self.on_pause()
            else:
                self.dismiss()

    def _save(self, drag):
        if drag:
            saved = _native_drag_end(drag)
            if saved is not None:
                _hypr().reset_bar_placement(self.CLS)
                self.save_pos({"monitor": saved[0], "rel": saved[1]})
                return
        wx = self.win.winfo_pointerx() - self._drag_off[0]
        wy = self.win.winfo_pointery() - self._drag_off[1]
        pw, ph = self._px
        left, top, right, bottom = monitor_work_area(wx + pw // 2, wy + ph // 2)
        rx = (wx - left) / max(right - left - pw, 1)
        ry = (wy - top) / max(bottom - top - ph, 1)
        self.save_pos({"monitor": None, "rel": [round(min(max(rx, 0.0), 1.0), 4),
                                                round(min(max(ry, 0.0), 1.0), 4)]})

    # -- desenho -------------------------------------------------------------
    def _draw(self, force: bool = False):
        if not self._shown or self._meeting is None:
            return
        paused, seconds = self._meeting
        blink = paused or seconds % 2 == 0
        key = (paused, seconds, blink, self._hovered, self._px)
        if key == self._drawn and not force:
            return
        self._drawn = key
        self._photo = ImageTk.PhotoImage(self._render(paused, seconds, blink))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)

    def _render(self, paused: bool, seconds: int, blink: bool) -> Image.Image:
        s = self.SS
        img = Image.new("RGB", (self.W * s, self.H * s), self._bg)
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([0.6 * s, 0.6 * s, (self.W - 0.6) * s, (self.H - 0.6) * s],
                            radius=(self.H / 2 - 0.6) * s, fill=self.PILL_BG,
                            outline=self.PILL_BORDER, width=max(1, round(1.0 * s)))
        # ponto: vermelho gravando (pisca a cada segundo), cinza vazado em pausa
        r = 3.6
        cx = 13
        box = [(cx - r) * s, (self.CY - r) * s, (cx + r) * s, (self.CY + r) * s]
        if paused:
            d.ellipse(box, outline=self.ICON_IDLE, width=max(1, round(1.4 * s)))
        else:
            d.ellipse(box, fill=self.REC if blink else "#8f3a35")
        h, rem = divmod(seconds, 3600)
        text = f"{h}:{rem // 60:02d}:{rem % 60:02d}" if h else f"{rem // 60:02d}:{rem % 60:02d}"
        d.text((22 * s, self.CY * s), text, font=self._font,
               fill=PILL_INK_3 if paused else PILL_INK_2, anchor="lm")
        for zone, x in (("pause", self.PX), ("hide", self.HX)):
            hover = self._hovered == zone
            if hover:
                d.ellipse([(x - self.BTN_R) * s, (self.CY - self.BTN_R) * s,
                           (x + self.BTN_R) * s, (self.CY + self.BTN_R) * s], fill=self.HOVER_BG)
            color = self.ICON_HOVER if hover else self.ICON_IDLE
            w = max(1, round(1.6 * s))
            if zone == "hide":
                a = 3.2
                for dx in (a, -a):
                    d.line([(x - dx) * s, (self.CY - a) * s, (x + dx) * s, (self.CY + a) * s],
                           fill=color, width=w)
            elif paused:  # retomar: triangulo
                d.polygon([((x - 2.6) * s, (self.CY - 4) * s), ((x - 2.6) * s, (self.CY + 4) * s),
                           ((x + 4) * s, self.CY * s)], fill=color)
            else:         # pausar: duas barras
                for dx in (-2.2, 2.2):
                    d.rounded_rectangle([(x + dx - 1) * s, (self.CY - 4) * s,
                                         (x + dx + 1) * s, (self.CY + 4) * s],
                                        radius=1 * s, fill=color)
        return img.resize(self._px, Image.BOX)


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
        owned_inode = None
        try:
            if self.path.exists():
                if already_running(self.path):
                    return  # nunca roubar o socket de uma instancia viva
                try:
                    self.path.unlink()
                except OSError:
                    pass
            self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._sock.bind(str(self.path))
            owned_inode = self.path.stat().st_ino
            os.chmod(self.path, 0o600)
            self._sock.listen(4)
            self._sock.settimeout(0.5)
            while not self._stop.is_set():
                try:
                    conn, _ = self._sock.accept()
                except socket.timeout:
                    continue
                except OSError as e:
                    if self._stop.is_set():
                        break
                    if e.errno in (errno.EBADF, errno.ENOTSOCK, errno.EINVAL):
                        raise
                    traceback.print_exc()
                    self._stop.wait(0.1)  # erro transitorio nao vira loop quente
                    continue
                try:
                    with conn:
                        conn.settimeout(1)
                        raw = conn.recv(4096).decode("utf-8", "replace").strip()
                        if not raw:
                            continue
                        if raw.split(" ", 1)[0].lower() == "transcribe" or raw.lower().startswith("meeting-"):
                            # Arquivo/ACK de reuniao aguardam fora do accept do atalho.
                            threading.Thread(target=self._serve_slow, args=(conn.dup(), raw),
                                             name="sussurro-ipc-file", daemon=True).start()
                            continue
                        conn.sendall(self._handle(raw).encode("utf-8"))
                except (OSError, socket.timeout):
                    continue  # cliente lento/fechado nao mata o atalho
                except Exception:
                    traceback.print_exc()
        except Exception as e:
            traceback.print_exc()
            self.event_queue.put(("error", f"atalho IPC indisponivel: {e}"))
        finally:
            try:
                if self._sock is not None:
                    self._sock.close()
                if (self._stop.is_set() and owned_inode is not None and self.path.exists()
                        and self.path.stat().st_ino == owned_inode):
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
        if data in ("cancel", "show", "quit"):
            self.event_queue.put((data, None))
            return "ok\n"
        # "toggle-enter"/"start-enter"/"stop-enter": veio do fone (daemon x9-sussurro);
        # ao terminar de colar, o Sussurro aperta Enter para confirmar o envio.
        base, _, flag = data.partition("-")
        if base == "meeting" and flag in ("start", "stop", "pause"):
            reply = queue.Queue(maxsize=1)
            self.event_queue.put(("meeting", (flag, reply, time.monotonic() + 1.0)))
            try:
                return reply.get(timeout=1.0)
            except queue.Empty:
                return "err Reuniao nao confirmou o comando a tempo, confira o estado antes de repetir.\n"
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
HEAT_SCALE = ["#eeebe3", "#cfe1dc", "#86bdb3", TEAL_2, TEAL]
BAR_DIM = TEAL_2            # barra normal; o pico usa TEAL
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


def fmt_compact(n) -> str:
    """137.812 -> "137,8K"; abaixo de mil, o numero inteiro."""
    n = int(round(n))
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1000:.1f}K".replace(".", ",").replace(",0K", "K")
    return f"{n / 1_000_000:.1f}M".replace(".", ",").replace(",0M", "M")


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
        cor = TEAL if palavras == teto else BAR_DIM
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
    DRAG_PX = 4           # arrasto menor que isso ainda e clique (copia o ditado todo)
    COPIED_MS = 1400      # quanto tempo a linha fica marcada como copiada

    def __init__(self, master, font_mono, font_ui, day_label, on_play, on_copy, on_retry,
                 font_semi=None):
        super().__init__(master, fg_color="transparent", width=1, height=1)
        self.font_mono = font_mono
        self.font_ui = font_ui
        self.font_semi = font_semi or font_ui
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
        self._text_ids = {}
        self._content_h = 0
        self._scroll_target = None
        self._scroll_job = None
        self._press = None      # (x, y, linha, indice do caractere) do botao apertado
        self._dragging = False
        self._flash_job = None
        self._flash_row = None  # linha verde de "copiado": o hover nao pinta por cima

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(self, bg=SURFACE, highlightthickness=0, bd=0,
                                yscrollincrement=1, selectbackground=SELECT_BG,
                                selectforeground=INK, takefocus=1)
        self.sb = ctk.CTkScrollbar(self, orientation="vertical", command=self._on_scrollbar,
                                   fg_color="transparent", width=12)
        self.canvas.configure(yscrollcommand=self.sb.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.sb.grid(row=0, column=1, sticky="ns", padx=(4, 0), pady=2)

        self.canvas.bind("<Configure>", self._on_cfg)
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Leave>", self._on_leave)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Control-c>", self._on_copy_key)
        self.canvas.bind("<Control-C>", self._on_copy_key)
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
            rx0, ry0, rx1, ry1 = h["retry"]
            if px0 <= x <= px1 and py0 <= y <= py1:
                return h, "play"
            if cx0 <= x <= cx1 and cy0 <= y <= cy1:
                return h, "copy"
            if rx0 <= x <= rx1 and ry0 <= y <= ry1:
                return h, "retry"
            tx0, ty0, tx1, ty1 = h["text"]
            if tx0 <= x <= tx1 and ty0 <= y <= ty1:
                return h, "text"
            return h, "row"
        return None, None

    def _set_hover(self, idx):
        if idx == self._hover:
            return
        if self._hover is not None and self._hover in self._bg_ids and self._hover != self._flash_row:
            self.canvas.itemconfigure(self._bg_ids[self._hover], fill=SURFACE)
            for kind, iid in self._icon_ids.get(self._hover, ()):
                if not self._icon_pinned(self._hover, kind):
                    self.canvas.itemconfigure(iid, state="hidden")
        self._hover = idx
        if idx is not None and idx in self._bg_ids and idx != self._flash_row:
            self.canvas.itemconfigure(self._bg_ids[idx], fill=HOVER_ROW)
            for _kind, iid in self._icon_ids.get(idx, ()):
                self.canvas.itemconfigure(iid, state="normal")

    def _icon_pinned(self, i, kind):
        """Acoes da linha so aparecem no hover, como no Hub; falha mantem o Refazer."""
        entry = self._entries[i]
        return kind == "retry" and bool(entry.get("failed") or entry.get("retrying"))

    def _on_motion(self, event):
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        h, zone = self._hit(x, y)
        failed = h is not None and self._entries[h["i"]].get("failed")
        clickable = zone in ("play", "copy", "text", "retry") or (failed and zone == "row")
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
        elif zone == "retry" or (entry.get("failed") and zone in ("text", "row")):
            self.on_retry(entry)
        elif zone in ("copy", "text") and entry.get("text"):
            self.on_copy(entry["text"], h["i"])

    # -- selecao de trecho ----------------------------------------------------
    def _on_press(self, event):
        """Apertar no texto arma um arrasto; soltar sem arrastar e o clique de sempre."""
        self.canvas.focus_set()  # Ctrl+C chega no canvas
        self.canvas.select_clear()
        self._dragging = False
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        h, zone = self._hit(x, y)
        self._press = None
        if h is not None and zone == "text" and not self._entries[h["i"]].get("failed"):
            tid = self._text_ids.get(h["i"])
            if tid is not None:
                self._press = (event.x, event.y, h["i"], self.canvas.index(tid, f"@{int(x)},{int(y)}"))

    def _on_drag(self, event):
        if self._press is None:
            return
        x0, y0, i, start = self._press
        if not self._dragging and abs(event.x - x0) + abs(event.y - y0) < self.DRAG_PX:
            return
        self._dragging = True
        tid = self._text_ids[i]
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        end = self.canvas.index(tid, f"@{int(x)},{int(y)}")
        self.canvas.select_from(tid, start)
        self.canvas.select_to(tid, end)

    def _on_release(self, event):
        if self._dragging:
            self._dragging = False
            return
        self._press = None
        self._on_click(event)

    def selected_text(self):
        """Trecho selecionado com o mouse e a linha dele, ou (None, None)."""
        if self._press is None or not self.canvas.select_item():
            return None, None
        i = self._press[2]
        tid = self._text_ids.get(i)
        if tid is None:
            return None, None
        try:
            first = self.canvas.index(tid, "sel.first")
            last = self.canvas.index(tid, "sel.last")
        except tk.TclError:
            return None, None
        text = self.canvas.itemcget(tid, "text")[first:last + 1]
        return (text, i) if text.strip() else (None, None)

    def _on_copy_key(self, _event):
        text, i = self.selected_text()
        if text:
            self.on_copy(text, i, partial=True)
        return "break"

    def flash_copied(self, i):
        """Marca a linha como copiada por um instante: fundo verde e o icone vira ✓."""
        if self._flash_job is not None:
            self.after_cancel(self._flash_job)
            self._flash_job = None
            self._redraw()
        bg = self._bg_ids.get(i)
        icons = dict(self._icon_ids.get(i, ()))
        if bg is None:
            return
        self._flash_row = i
        self.canvas.itemconfigure(bg, fill=COPIED_BG)
        if "play" in icons:
            self.canvas.itemconfigure(icons["play"], state="hidden")  # o aviso ocupa o lugar dele
        if "copy" in icons:
            self.canvas.itemconfigure(icons["copy"], state="hidden")
            box = next(h["copy"] for h in self._hits if h["i"] == i)
            self.canvas.create_text(box[2] - self._s(2), self.canvas.coords(icons["copy"])[1],
                                    text="✓ Copiado", fill=COPIED_INK, anchor="e",
                                    font=(self.font_semi, -self._s(13)))
        def restore():
            self._flash_job = self._flash_row = None
            self.canvas.select_clear()
            self._press = None
            self._redraw()
        self._flash_job = self.after(self.COPIED_MS, restore)

    # -- desenho --------------------------------------------------------------
    def _redraw(self):
        frac = self.canvas.yview()[0]
        self._cancel_scroll_anim()
        self.canvas.delete("all")
        self._hits = []
        self._bg_ids = {}
        self._icon_ids = {}
        self._text_ids = {}
        self._hover = None
        self._flash_row = None
        self._press = None
        w = self._width
        if w < 40:
            return
        s = self._s
        # medidas do Hub: cartao com borda de 1 px e raio 16; hora a 16 px da borda,
        # texto a 114 px, linhas com 16 px de respiro em cima e embaixo
        x0, x1 = 1, w - 1
        radius = s(16)
        pady = s(16)
        btn, gap = s(30), s(2)
        btns_w = btn * 2 + gap + s(28)  # tocar e copiar; Refazer usa o mesmo lugar
        font_day = (self.font_semi, -s(13))
        font_time = (self.font_ui, -s(13))
        font_text = (self.font_ui, -s(15))
        font_btn = (self.font_ui, -s(15))
        time_x = x0 + s(16)
        text_x = x0 + max(s(114), tkfont.Font(font=font_time).measure("00:00") + s(40))
        text_w = max(s(80), min(w - text_x - btns_w, s(620)))
        y = 0
        if not self._entries:
            self.canvas.create_text(time_x, y + s(8), text="Nenhum ditado ainda. Segure o atalho "
                                    "e fale: o texto aparece aqui.",
                                    fill=INK_3, anchor="nw", font=font_text)
            self._content_h = y + s(40)
            self.canvas.configure(scrollregion=(0, 0, w, self._content_h))
            return
        prev_day = None
        group_top = None

        def close_group(bottom):
            self._group_frame(x0, group_top, x1, bottom, radius)

        for i, entry in enumerate(self._entries):
            dt = datetime.fromisoformat(entry["ts"])
            day = self.day_label(dt)
            if day != prev_day:
                if prev_day is not None:
                    close_group(y)
                    y += s(28)
                hid = self.canvas.create_text(x0, y, text=day.upper(), fill=INK_3,
                                              anchor="nw", font=font_day)
                y = self.canvas.bbox(hid)[3] + s(10)
                group_top = y
                prev_day = day
            elif i:
                self.canvas.create_line(x0 + 1, y, x1 - 1, y, fill=BORDER)
                y += 1
            failed = bool(entry.get("failed"))
            retrying = bool(entry.get("retrying"))
            audio_removed = bool(entry.get("audio_removed"))
            shown_text = ("A transcrição falhou. Clique em Refazer para tentar novamente."
                          if failed else entry["text"])
            if retrying:
                shown_text = "Refazendo transcrição..." + ("\n\n" + entry["text"] if entry.get("text") else "")
            elif audio_removed:
                shown_text += "\nÁudio removido. Texto mantido."
            tid = self.canvas.create_text(
                text_x, y + pady, text=shown_text, fill=ACCENT_TEXT if failed else INK, anchor="nw",
                width=text_w, font=font_text, justify="left")
            tb = self.canvas.bbox(tid)
            row_h = max(tb[3] - tb[1], btn) + pady * 2
            bg = self.canvas.create_rectangle(
                x0 + 1, y, x1 - 1, y + row_h, fill=SURFACE, outline="", width=0)
            self.canvas.tag_lower(bg, tid)
            mid = y + row_h / 2
            self.canvas.create_text(time_x, mid, text=dt.strftime("%H:%M"), fill=INK_3,
                                    anchor="w", font=font_time)
            copy_x1 = x1 - s(14)
            copy_x0 = copy_x1 - btn
            play_x1 = copy_x0 - gap
            play_x0 = play_x1 - btn
            nada = (-1, -1, -1, -1)
            if failed or retrying:  # sem texto pra copiar: o lugar das acoes vira Refazer
                retry_id = self.canvas.create_text(
                    (play_x0 + copy_x1) / 2, mid, text="…" if retrying else "Refazer",
                    fill=ACCENT_TEXT if failed else INK_2, font=(self.font_semi, -s(13)),
                    anchor="center")
                icons_row = (("retry", retry_id),)
                boxes = {"play": nada, "copy": nada, "retry": (play_x0, y, copy_x1, y + row_h)}
            else:
                if audio_removed:  # nada pra tocar: um ponto no lugar do botao
                    play_id = self.canvas.create_text(
                        (play_x0 + play_x1) / 2, mid, text="·", fill=INK_3,
                        font=font_btn, anchor="center", state="hidden")
                else:
                    play_id = self.canvas.create_image((play_x0 + play_x1) / 2, mid,
                                                       image=self._row_icon("play"), state="hidden")
                copy_id = self.canvas.create_image((copy_x0 + copy_x1) / 2, mid,
                                                   image=self._row_icon("copy"), state="hidden")
                icons_row = (("play", play_id), ("copy", copy_id))
                boxes = {"play": (play_x0, y, play_x1, y + row_h),
                         "copy": (copy_x0, y, copy_x1, y + row_h), "retry": nada}
            self._hits.append({"i": i, "y0": y, "y1": y + row_h,
                               "text": (text_x, y, play_x0 - s(4), y + row_h), **boxes})
            self._bg_ids[i] = bg
            self._text_ids[i] = tid
            self._icon_ids[i] = icons_row
            y += row_h
        close_group(y)
        self._content_h = y + s(12)
        self.canvas.configure(scrollregion=(0, 0, w, self._content_h))
        self.canvas.yview_moveto(frac)

    def _row_icon(self, name):
        """PhotoImage do icone de acao da linha, no tamanho da escala atual (cacheado)."""
        size = self._s(16)
        cache = self.__dict__.setdefault("_row_icons", {})
        if (name, size) not in cache:
            img = icons.icon_image(name, INK_2, size * 2, 1.8).resize((size, size), Image.LANCZOS)
            cache[(name, size)] = ImageTk.PhotoImage(img, master=self.canvas)
        return cache[(name, size)]

    def _group_frame(self, x0, y0, x1, y1, r):
        """Moldura arredondada do dia: mascara os cantos das linhas e traca a borda."""
        c = self.canvas
        r = min(r, (y1 - y0) / 2)
        steps = 8
        for cx, cy, a0 in ((x0 + r, y0 + r, 90), (x1 - r, y0 + r, 0),
                           (x1 - r, y1 - r, 270), (x0 + r, y1 - r, 180)):
            # canto de fora do arco pintado com o fundo (a linha do hover nao vaza)
            corner_x = x0 if cx < (x0 + x1) / 2 else x1
            corner_y = y0 if cy < (y0 + y1) / 2 else y1
            pts = [corner_x, corner_y]
            for k in range(steps + 1):
                ang = math.radians(a0 + 90 * k / steps)
                pts += [cx + r * math.cos(ang), cy - r * math.sin(ang)]
            c.create_polygon(pts, fill=SURFACE, outline="")
            c.create_arc(cx - r, cy - r, cx + r, cy + r, start=a0, extent=90,
                         style="arc", outline=BORDER, width=1)
        c.create_line(x0 + r, y0, x1 - r, y0, fill=BORDER)
        c.create_line(x0 + r, y1, x1 - r, y1, fill=BORDER)
        c.create_line(x0, y0 + r, x0, y1 - r, fill=BORDER)
        c.create_line(x1, y0 + r, x1, y1 - r, fill=BORDER)


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
        root.geometry(f"{min(1180, max(900, root.winfo_screenwidth() - 120))}x"
                      f"{min(800, max(600, root.winfo_screenheight() - 80))}")

        self.settings = load_settings()
        self.devices = list_input_devices()
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
            ["Figtree", "Segoe UI", "Noto Sans", "DejaVu Sans", "Ubuntu", "Liberation Sans"],
            "TkDefaultFont",
        )
        # pesos do Hub: 550 nos itens da lateral, 600 em titulo e botao
        self.FONT_MED = pick_font(["Figtree Medium", "Segoe UI Semibold"], self.FONT_UI)
        self.FONT_SEMI = pick_font(["Figtree SemiBold", "Segoe UI Semibold"], self.FONT_UI)
        self.FONT_SERIF = pick_font(
            ["EB Garamond", "Garamond", "Noto Serif", "Georgia", "Liberation Serif", "DejaVu Serif"],
            "TkDefaultFont",
        )
        self.FONT_DISPLAY = self.FONT_SEMI
        self.FONT_MONO = pick_font(
            ["Cascadia Mono", "JetBrains Mono", "DejaVu Sans Mono", "Noto Sans Mono", "Ubuntu Mono"],
            "TkFixedFont",
        )
        root.configure(fg_color=BG)
        root.minsize(860, 560)
        PADX = self.PADX = 40
        root.grid_columnconfigure(1, weight=1)
        root.grid_rowconfigure(0, weight=1)

        # -- barra lateral: marca, navegacao, motor em uso, configuracoes ------
        side = self.sidebar = ctk.CTkFrame(root, fg_color=BG, corner_radius=0, width=228)
        side.grid(row=0, column=0, sticky="ns")
        side.pack_propagate(False)
        brand = ctk.CTkFrame(side, fg_color="transparent")
        brand.pack(fill="x", padx=(21, 12), pady=(26, 22))
        self._brand_img = ctk.CTkImage(icons.brand_image(INK, 44), size=icons.brand_image(INK, 22).size)
        ctk.CTkLabel(brand, image=self._brand_img, text="").pack(side="left")
        ctk.CTkLabel(brand, text="Sussurro", text_color=INK,
                     font=self._font(21, "semibold")).pack(side="left", padx=(8, 0))

        self.tab_btns = {}
        self._icon_imgs = {}
        tabs = [("historico", "Ditado"), ("aovivo", "Ao vivo")]
        if not IS_WIN:
            tabs.append(("reuniao", "Reunião"))  # parec/pacat: PulseAudio ou PipeWire
        tabs += [("estatisticas", "Estatísticas"), ("biblioteca", "Dicionário"),
                 ("comparar", "Comparar")]
        if self.gestures is not None:
            tabs.append(("omarchy", "Omarchy"))
        self.PAGE_TITLES = dict(tabs)
        self.PAGE_TITLES["historico"] = "Bem-vindo de volta"
        self.PAGE_TITLES["config"] = "Configurações"

        def nav(master, name, label, **pack):
            img = icons.icon_image(name, INK, 36, 1.7)
            self._icon_imgs[name] = ctk.CTkImage(img, size=(18, 18))
            btn = ctk.CTkButton(master, text=label, image=self._icon_imgs[name], compound="left",
                                anchor="w", height=36, corner_radius=6, border_spacing=8,
                                fg_color="transparent", hover_color=SURFACE_3, text_color=INK,
                                font=self._font(15, "medium"),
                                command=lambda n=name: self._show_tab(n))
            btn.pack(fill="x", padx=(13, 11), pady=2, **pack)
            self.tab_btns[name] = btn
            return btn

        for name, label in tabs:
            nav(side, name, label)

        nav(side, "config", "Configurações", side="bottom", ipady=0).pack_configure(pady=(2, 14))
        tk.Frame(side, bg=BORDER_STRONG, height=1, bd=0, highlightthickness=0).pack(
            side="bottom", fill="x", padx=(13, 11), pady=(10, 6))
        # cartao "em uso agora": o que transcreve o proximo trecho (inclusive quando a
        # nuvem caiu e o Turbo local assumiu) e o gasto do mes com modelo de nuvem
        now = ctk.CTkFrame(side, fg_color=WHITE, corner_radius=12, border_width=1,
                           border_color=BORDER)
        now.pack(side="bottom", fill="x", padx=(13, 11), pady=(0, 0))
        ctk.CTkLabel(now, text="Em uso agora", text_color=INK, anchor="w",
                     font=self._font(13, "semibold")).pack(fill="x", padx=16, pady=(14, 2))
        now_cell = ctk.CTkFrame(now, fg_color="transparent")
        now_cell.pack(fill="x", padx=16, pady=(0, 12))
        self.engine_now = ctk.CTkLabel(now_cell, text="● carregando o modelo...",
                                       text_color=ENGINE_COLORS["aviso"], anchor="w",
                                       justify="left", wraplength=168, font=self._font(13))
        self.engine_now.pack(fill="x")
        self.cloud_spend = ctk.CTkLabel(now_cell, text="", text_color=INK_3, anchor="w",
                                        justify="left", wraplength=168, font=self._font(12))
        self._spend_shown = None
        self._engine_shown = None
        # rotulo do modelo carregado: o texto vive no cartao acima; fica pra quem o atualiza
        self.model_label = ctk.CTkLabel(now, text="Preparando modelo...")

        # -- cartao principal: titulo da pagina, acoes, conteudo, status --------
        card = self.card = ctk.CTkFrame(root, fg_color=SURFACE, corner_radius=16,
                                        border_width=1, border_color=BORDER)
        card.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)
        head = ctk.CTkFrame(card, fg_color="transparent")
        head.pack(fill="x", padx=PADX, pady=(30, 18))
        self.page_title = ctk.CTkLabel(head, text="", text_color=INK, anchor="w",
                                       font=self._font(24, "semibold"))
        self.page_title.pack(side="left")
        cmd = self.command_bar = ctk.CTkFrame(head, fg_color="transparent")
        self.record_btn = ctk.CTkButton(
            cmd, text="Gravar", command=self.toggle, state="disabled",
            width=104, height=38, corner_radius=8,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            text_color=GRAPHITE, text_color_disabled="#a8a7a2",
            font=self._font(15, "semibold"))
        for label, cb in (("Limpar", self.clear), ("Copiar tudo", self.copy_all)):
            self._secondary(cmd, label, cb, height=38).pack(side="right", padx=(8, 0))
        self.record_btn.pack(side="right", padx=(8, 0))

        # status empacotado antes do conteudo (side=bottom) pra nunca ser espremido pra fora
        self.status = ctk.CTkLabel(card, text="Iniciando...", text_color=INK_3, anchor="w",
                                   font=self._font(13))
        self.status.pack(side="bottom", fill="x", padx=PADX, pady=(6, 14))

        self.content = ctk.CTkFrame(card, fg_color="transparent")
        self.content.pack(fill="both", expand=True, padx=PADX, pady=(0, 4))

        # pagina Ditado: historico com a hora na coluna da esquerda + numeros na direita
        self.home = ctk.CTkFrame(self.content, fg_color="transparent")
        self.home.grid_columnconfigure(0, weight=1)
        self.home.grid_rowconfigure(0, weight=1)
        self._playing = None
        self.hist_frame = HistoryList(
            self.home, font_mono=self.FONT_MONO, font_ui=self.FONT_UI, font_semi=self.FONT_SEMI,
            day_label=self._day_label, on_play=self._play, on_copy=self._copy_entry,
            on_retry=self._retry_entry)
        self.hist_frame.grid(row=0, column=0, sticky="nsew")
        self._secondary(self.hist_frame, "Liberar espaço", self._prune_history_audio).grid(
            row=1, column=0, columnspan=2, sticky="e", pady=(8, 0))
        self._build_home_stats(self.home).grid(row=0, column=1, sticky="n", padx=(24, 0))

        self.text = ctk.CTkTextbox(self.content, fg_color=WHITE, text_color=INK,
                                   font=self._font(15), wrap="word", border_width=1,
                                   border_color=BORDER, corner_radius=16, border_spacing=14)
        self.library = self.transcriber.library
        self.lib_tab = self._build_library_tab()
        self.stats_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        self._stats_dirty = True   # so desenha quando a aba abrir
        self._geom_antes = None    # geometria de fora da aba ESTATISTICAS
        self._tabs = {"historico": self.home, "aovivo": self.text,
                      "biblioteca": self.lib_tab, "estatisticas": self.stats_frame,
                      "config": self._build_settings_page()}
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
        self.meeting_pill = None
        if self.meeting_panel is not None:
            self.meeting_pill = MeetingPill(
                root, lambda: self.settings.get("meeting_pos"), self._save_meeting_pos,
                on_pause=self.meeting_panel.pause, on_dismiss=self.meeting_panel.pill_dismissed)

        # Tk 9 no Linux SIGSEGV se um widget for tocado fora da thread do mainloop
        # (CTkButton.configure desenha via canvas create_text -> TkpGetColor).
        # No Windows isso "funcionava"; aqui tudo que mexe na HUD passa por esta fila.
        self._ui_queue: queue.Queue = queue.Queue()
        self._begin_model_load(dict(self.settings), persist=False)
        root.after(UI_POLL_MS, self._poll)

    def _font(self, size, weight="regular"):
        """Fonte em pixels (o CTk escala). Sem Figtree, os pesos viram negrito."""
        family = {"regular": self.FONT_UI, "medium": self.FONT_MED,
                  "semibold": self.FONT_SEMI}[weight]
        if weight != "regular" and family == self.FONT_UI:
            return (family, size, "bold")
        return (family, size)

    def _secondary(self, master, text, command, width=None, height=36, **kw):
        """Botao neutro do Hub: chapa #eeebe3, sem borda, texto 15/600."""
        opts = dict(text=text, command=command, height=height, corner_radius=8,
                    width=40 if width is None else width, border_spacing=12,
                    fg_color=SURFACE_3, hover_color=SURFACE_4, border_width=0,
                    text_color=INK, text_color_disabled=INK_3, font=self._font(15, "semibold"))
        opts.update(kw)
        return ctk.CTkButton(master, **opts)

    def _combo(self, master, values, current, command, width=220):
        """Dropdown do Hub: branco, borda fina, cantos de 8."""
        box = ctk.CTkComboBox(master, values=values, command=command, state="readonly",
                              width=width, height=38, corner_radius=8, border_width=1,
                              fg_color=WHITE, border_color=BORDER_STRONG,
                              button_color=WHITE, button_hover_color=SURFACE_3,
                              dropdown_fg_color=WHITE, dropdown_hover_color=SURFACE_3,
                              dropdown_text_color=INK, text_color=INK,
                              font=self._font(14), dropdown_font=self._font(14))
        box.set(current)
        return box

    # -- pagina Ditado: cartao de numeros ao lado do historico ----------------
    def _build_home_stats(self, master):
        box = ctk.CTkFrame(master, fg_color=SURFACE_2, corner_radius=16, width=250)
        self._home_nums = {}
        top = ctk.CTkFrame(box, fg_color="transparent")
        top.pack(fill="x", padx=24, pady=(20, 18))
        for key in ("palavras", "ppm", "streak"):
            row = ctk.CTkFrame(top, fg_color="transparent")
            row.pack(fill="x", pady=2)
            num = ctk.CTkLabel(row, text="0", text_color=INK, anchor="w",
                               font=(self.FONT_SERIF, 30))
            num.pack(side="left")
            lab = ctk.CTkLabel(row, text="", text_color=INK, anchor="w", font=self._font(16))
            lab.pack(side="left", padx=(8, 0), pady=(6, 0))
            self._home_nums[key] = (num, lab)
        tk.Frame(box, bg=BORDER_STRONG, height=1, bd=0, highlightthickness=0).pack(fill="x")
        bottom = ctk.CTkFrame(box, fg_color="transparent")
        bottom.pack(fill="x", padx=24, pady=(18, 22))
        ctk.CTkLabel(bottom, text="Tempo poupado", text_color=INK, anchor="w",
                     font=self._font(17, "semibold")).pack(fill="x")
        self._home_saved = ctk.CTkLabel(bottom, text="", text_color=INK_2, anchor="w",
                                        justify="left", wraplength=200, font=self._font(15))
        self._home_saved.pack(fill="x", pady=(2, 0))
        return box

    def _render_home_stats(self):
        if not getattr(self, "_home_nums", None):
            return
        st = compute_stats(self.entries)
        valores = {
            "palavras": (fmt_compact(st["palavras"]), "palavras"),
            "ppm": (fmt_int(st["ppm"]), "ppm"),
            "streak": (str(st["streak"]), "dia seguido" if st["streak"] == 1 else "dias seguidos"),
        }
        for key, (num, lab) in self._home_nums.items():
            valor, rotulo = valores[key]
            num.configure(text=valor)
            lab.configure(text=rotulo)
        self._home_saved.configure(
            text=f"{fmt_dur(st['economia'])} a menos que digitando a {TYPING_WPM} palavras/min"
            if st["palavras"] else "Os números aparecem depois do primeiro ditado.")

    # -- pagina Configuracoes: linhas com titulo, descricao e controle --------
    def _build_settings_page(self):
        page = ctk.CTkScrollableFrame(self.content, fg_color="transparent")
        card = ctk.CTkFrame(page, fg_color=SURFACE_2, corner_radius=16)
        card.pack(fill="x", pady=(0, 16))
        card.grid_columnconfigure(0, weight=1)
        rows = []

        def row(title, desc=None, var=None):
            r = len(rows) * 2
            if rows:  # separador entre linhas, recuado como no Hub
                tk.Frame(card, bg=BORDER_STRONG, height=1, bd=0, highlightthickness=0).grid(
                    row=r - 1, column=0, columnspan=2, sticky="ew", padx=20)
            cell = ctk.CTkFrame(card, fg_color="transparent")
            cell.grid(row=r, column=0, sticky="ew", padx=(20, 12), pady=18)
            ctk.CTkLabel(cell, text=title, text_color=INK, anchor="w",
                         font=self._font(15, "semibold")).pack(fill="x")
            lab = None
            if desc is not None or var is not None:
                lab = ctk.CTkLabel(cell, text=desc or "", textvariable=var, text_color=INK_2,
                                   anchor="w", justify="left", wraplength=440,
                                   font=self._font(15))
                lab.pack(fill="x", pady=(4, 0))
            rows.append(cell)
            return r, lab

        def place(widget, r):
            widget.grid(row=r, column=1, sticky="e", padx=(0, 20), pady=18)
            return widget

        self.hotkey_var = tk.StringVar(value=BUTTON_LABELS[self.settings["mouse_button"]])
        r, _ = row("Atalho do mouse", var=self.hotkey_var)
        self.set_hotkey_btn = place(self._secondary(card, "Alterar", self.capture_hotkey,
                                                    width=164), r)
        r, _ = row("Ação do atalho",
                   "Alternar: um clique liga, outro desliga. Segurar: grava enquanto o botão "
                   "estiver apertado.")
        self.trigger = place(self._combo(card, ["alternar", "segurar"],
                                         self.settings["trigger_mode"], self._on_trigger), r)
        self.mic_by_label = {device.label: device for device in self.devices.values()}
        names = list(self.mic_by_label)
        saved_device = self.devices.get(self.settings["device_name"])
        current_mic = saved_device.label if saved_device else (names[0] if names else "")
        r, self.mic_help = row("Microfone", "")
        self.mic = place(self._combo(card, names, current_mic, self._on_mic, width=260), r)
        self._update_mic_help()
        r, _ = row("Fonte", "O que o ditado escuta. O áudio do PC é sempre a saída padrão "
                            "do sistema.")
        self.fonte = place(self._combo(card, ["microfone", "audio do PC", "os dois"],
                                       CAPTURE_LABELS[self.settings["capture_mode"]],
                                       self._on_fonte), r)
        r, _ = row("Transcrição", "Simultâneo escreve enquanto você fala; final transcreve "
                                  "tudo ao terminar.")
        self.mode = place(self._combo(card, ["simultaneo", "final"],
                                      self.settings["transcribe_mode"], self._on_mode), r)
        r, _ = row("Envio", "Colar usa a área de transferência; digitar simula o teclado.")
        self.inject = place(self._combo(card, ["colar", "digitar"],
                                        self.settings["inject_method"], self._on_inject), r)
        r, _ = row("Idioma", "Português, inglês ou detecção automática.")
        self.lang = place(self._combo(card, ["pt", "en", "auto"], self.settings["language"],
                                      self._on_lang), r)
        r, _ = row("Modelo", "Turbo e Large-v3 rodam no PC e nenhum áudio sai daqui. A nuvem "
                             "(ElevenLabs, Grok) acerta mais termos, mas manda o áudio do "
                             "ditado pro provedor. Se ela cair, o trecho sai no Turbo local.")
        self.model_choice = place(self._combo(card, list(MODEL_LABELS.values()),
                                              MODEL_LABELS[self.settings["whisper_model"]],
                                              lambda _: self._apply_model(), width=260), r)
        r, _ = row("Executar em", execution_hardware_note(self.hardware))
        self.device_choice = place(self._combo(card, list(self.device_labels.values()),
                                               self.device_labels[self.settings["whisper_device"]],
                                               lambda _: self._apply_model(), width=260), r)
        self.sounds_var = tk.BooleanVar(value=bool(self.settings["feedback_sounds"]))
        r, _ = row("Sons", "Toque curto ao começar, terminar e quando algo dá errado.")
        place(ctk.CTkSwitch(card, text="", variable=self.sounds_var, command=self._on_sounds,
                            width=44, switch_width=40, switch_height=22,
                            fg_color=BORDER_STRONG, progress_color=INK,
                            button_color=WHITE, button_hover_color=WHITE), r)
        ctk.CTkLabel(page, text="A barra de gravação pode ser arrastada pelo meio para outro "
                                "lugar. Reunião transcreve sempre no PC.",
                     text_color=INK_3, anchor="w", justify="left", wraplength=640,
                     font=self._font(13)).pack(fill="x", padx=4)
        return page

    def _close(self):
        # fechar a janela (Super+W, X) encerra o app, como qualquer programa
        self._quit()

    def _show(self):
        self.root.deiconify()
        self.root.lift()

    def _quit(self):
        self.compare_panel.close()
        if self.meeting_panel is not None:
            self.meeting_panel.close()
        if not IS_WIN:
            self._ipc._stop.set()
            if self.hotkey._listener is not None:
                self.hotkey._listener.stop()
            if self.gestures is not None:
                self.gestures.stop()
            self.transcriber.cancel(from_processing=True)
            self.bar.hide()
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
        if self.meeting_panel is not None:
            if name == "reuniao":
                self.meeting_panel.shown()
            elif self._tab == "reuniao":
                self.meeting_panel.hidden()
        self._tab = name
        for n, btn in self.tab_btns.items():
            btn.configure(fg_color=SURFACE_3 if n == name else "transparent")
        # Configuracoes leva titulo serifado, como o painel de ajustes do Hub
        self.page_title.configure(
            text=self.PAGE_TITLES.get(name, name),
            font=(self.FONT_SERIF, 34) if name == "config" else self._font(24, "semibold"))
        if name in ("historico", "aovivo"):
            self.command_bar.pack(side="right")
        else:
            self.command_bar.pack_forget()
        for n, widget in self._tabs.items():
            if n != name:
                widget.pack_forget()
        self._tabs[name].pack(fill="both", expand=True)
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
        dias = (date.today() - dt.date()).days
        if dias == 0:
            return "HOJE"
        if dias == 1:
            return "ONTEM"
        return dt.strftime("%d/%m/%Y")

    def _render_history(self):
        shown = [{**entry, "audio_removed": not (HISTORY_DIR / entry["wav"]).is_file()}
                 for entry in self.entries[:HIST_RENDER_MAX]]
        self.hist_frame.set_entries(shown)
        self._render_home_stats()

    def _add_history(self, entry: dict):
        for i, current in enumerate(self.entries):
            if current.get("wav") == entry.get("wav"):
                if current.get("retrying"):
                    self._model_load_done({}, False, None)
                self.entries[i] = entry
                break
        else:
            self.entries.insert(0, entry)
        self._render_history()
        self._stats_dirty = True
        if self._tab == "estatisticas":
            self._render_stats()

    def _copy_text(self, text: str, done=None):
        """Copia por wl-copy/xclip numa thread. O clipboard do Tk (XWayland) deixava a janela
        dona da selecao, servindo cada leitor (os wl-paste --watch do Omarchy) no laco da
        interface, e as vezes travava o app."""
        def work():
            ok = set_clipboard_text(text)
            if done is not None:
                self._ui(done, ok)
        threading.Thread(target=work, name="sussurro-copiar", daemon=True).start()

    def _copy_entry(self, text: str, row=None, partial=False):
        what = "Trecho copiado" if partial else "Transcrição copiada"
        def done(ok):
            if not ok:
                self.status.configure(text="Não deu pra copiar: wl-copy/xclip falhou.")
                return
            self.status.configure(text=f"{what} ({len(text)} caracteres).")
            if row is not None:
                self.hist_frame.flash_copied(row)
        self._copy_text(text, done)

    def _retry_entry(self, entry: dict):
        # a lista desenha copias (com audio_removed): a marca de "refazendo" tem que ir na
        # entrada de self.entries, senao _add_history nunca destrava o MODELO e o GRAVAR
        entry = next((e for e in self.entries if e.get("wav") == entry.get("wav")), entry)
        if not (HISTORY_DIR / entry["wav"]).is_file():
            self.status.configure(text="Áudio não disponível. O texto do histórico foi mantido.")
            return
        if entry.get("retrying"):
            return
        try:
            self.transcriber.retry_history_async(entry)
        except Exception as error:
            self.status.configure(text=str(error))
            return
        entry["retrying"] = True
        self.record_btn.configure(state="disabled")
        self._set_model_picker("disabled")
        self._render_history()

    def _play(self, path: str):
        if not Path(path).is_file():
            self.status.configure(text="Áudio não disponível. O texto do histórico foi mantido.")
            return
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

    def _prune_history_audio(self):
        if self._playing is not None:
            self.status.configure(text="Pare a reprodução antes de liberar espaço.")
            return
        days = simpledialog.askinteger("Liberar espaço", "Apagar áudio de ditados com texto com mais de quantos dias?",
                                      initialvalue=30, minvalue=1, maxvalue=36500, parent=self.root)
        if days is None:
            return
        try:
            preview = self.transcriber.prune_history_audio(days)
            if not preview["files"]:
                self.status.configure(text="Nenhum áudio antigo com texto para remover.")
                return
            if not messagebox.askyesno(
                    "Apagar áudio antigo?",
                    f"Apagar {len(preview['files'])} WAVs com mais de {days} dias "
                    f"({preview['bytes'] / 1024 / 1024:.1f} MiB)?\n\n"
                    "Os textos ficam no histórico. Não será possível ouvir ou Refazer esses áudios.\n"
                    "Áudio sem texto é preservado. Esta remoção não pode ser desfeita.",
                    parent=self.root, default=messagebox.NO):
                return
            result = self.transcriber.prune_history_audio(days, confirmed=True, only_files=preview["files"])
        except Exception as error:
            self.status.configure(text=f"Não foi possível liberar espaço: {error}")
            return
        self.status.configure(text=f"{result['deleted']} áudios removidos, {result['bytes'] / 1024 / 1024:.1f} MiB liberados. "
                                   "Textos mantidos." + (f" {len(result['errors'])} falhas ao apagar." if result["errors"] else ""))
        self._render_history()

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
        card = ctk.CTkFrame(parent, fg_color=SURFACE_2, corner_radius=16)
        card.grid(row=row, column=col, sticky="nsew",
                  padx=(0 if col == 0 else 12, 0), pady=(0, 12))
        ctk.CTkLabel(card, text=rotulo, text_color=INK_3, anchor="w",
                     font=self._font(13, "semibold")).pack(fill="x", padx=18, pady=(14, 0))
        ctk.CTkLabel(card, text=valor, text_color=INK, anchor="w",
                     font=(self.FONT_SERIF, 30)).pack(fill="x", padx=18, pady=(0, 0))
        ctk.CTkLabel(card, text=nota, text_color=INK_3, anchor="w", justify="left",
                     wraplength=150, font=self._font(12)).pack(fill="x", padx=18, pady=(0, 14))

    def _stats_titulo(self, parent, texto: str, nota: str = ""):
        head = ctk.CTkFrame(parent, fg_color="transparent")
        head.pack(fill="x", padx=18, pady=(14, 4))
        ctk.CTkLabel(head, text=texto, text_color=INK,
                     font=self._font(15, "semibold")).pack(side="left")
        if nota:
            ctk.CTkLabel(head, text=nota, text_color=INK_3,
                         font=self._font(13)).pack(side="right", padx=(12, 0))

    def _render_cloud_spend(self):
        """Linha de gasto na nuvem: aparece depois do primeiro ditado na nuvem."""
        ledger = shared_ledger()
        el, grok = ledger.month(ELEVENLABS), ledger.month(XAI)
        if not (ledger.total(ELEVENLABS)["calls"] or ledger.total(XAI)["calls"]):
            return
        ledger.refresh_balance()
        if ledger.balance and ledger.balance["limit"]:
            b = ledger.balance
            reset = (datetime.fromtimestamp(b["reset"]).strftime("renova em %d/%m")
                     if b.get("reset") else "")
            saldo = (fmt_credits(max(0, b["limit"] - b["used"])),
                     f"de {fmt_credits(b['limit'])} créditos" + (f" · {reset}" if reset else ""))
        else:
            saldo = ("—", ledger.balance_error or "buscando...")
        cards = (
            ("ElevenLabs este mês", f"{fmt_credits(el['credits'])} créditos",
             f"{fmt_usd(el['usd'])} · {fmt_minutes(el['audio_s'])} em {el['calls']} pedidos"),
            ("Saldo ElevenLabs", *saldo),
            ("Grok este mês", fmt_minutes(grok["audio_s"]),
             f"{grok['calls']} pedidos · a xAI não informa custo"),
        )
        grade = ctk.CTkFrame(self.stats_frame, fg_color="transparent")
        grade.pack(fill="x")
        for col in range(len(cards)):
            grade.grid_columnconfigure(col, weight=1, uniform="nuvem")
        for i, (rotulo, valor, nota) in enumerate(cards):
            self._stat_card(grade, rotulo, valor, nota, 0, i)

    def _render_stats(self):
        """Redesenha a aba inteira a partir do historico em memoria."""
        for child in self.stats_frame.winfo_children():
            child.destroy()
        self._stats_dirty = False
        if not self.entries:
            ctk.CTkLabel(self.stats_frame, text="Nenhum ditado ainda — grave alguma coisa "
                         "e os números aparecem aqui.", text_color=INK_3,
                         font=self._font(15)).pack(anchor="w", padx=10, pady=10)
            return
        st = compute_stats(self.entries)

        grade = ctk.CTkFrame(self.stats_frame, fg_color="transparent")
        grade.pack(fill="x")
        media = st["segundos"] / st["ditados"]
        cards = (
            ("Palavras ditadas", fmt_int(st["palavras"]),
             f"maior: {fmt_int(st['maior_ditado'])} palavras"),
            ("Palavras por minuto", fmt_int(st["ppm"]),
             f"melhor: {fmt_int(st['melhor_ppm'])} ppm"),
            ("Sequência", fmt_dias(st["streak"]), f"recorde: {fmt_dias(st['recorde'])}"),
            ("Tempo falado", fmt_dur(st["segundos"]),
             f"{fmt_int(st['ditados'])} ditados, media de {fmt_dur(media)}"),
            ("Economia vs digitar", fmt_dur(st["economia"]),
             f"teclado a {TYPING_WPM} palavras/min"),
            ("Correções do dicionário", fmt_int(st["correcoes"]),
             f"{len(self.library.entries)} termos na lista" if st["correcoes"]
             else "comeca a contar nesta versao"),
        )
        for col in range(len(cards)):
            grade.grid_columnconfigure(col, weight=1, uniform="stat")
        for i, (rotulo, valor, nota) in enumerate(cards):
            self._stat_card(grade, rotulo, valor, nota, 0, i)

        self._render_cloud_spend()

        # mapa, horas e palavras dividem a linha de baixo: cresce pro lado, nao pra baixo
        baixo = ctk.CTkFrame(self.stats_frame, fg_color="transparent")
        baixo.pack(fill="both", expand=True)
        baixo.grid_rowconfigure(0, weight=1)
        baixo.grid_columnconfigure(2, weight=1)  # a lista de palavras pega a sobra
        mapa = ctk.CTkFrame(baixo, fg_color=SURFACE_2, corner_radius=16)
        mapa.grid(row=0, column=0, sticky="nsew", pady=(0, 12))
        self._stats_titulo(mapa, "Atividade", f"{fmt_dias(len(st['por_dia']))} com ditado")
        img, tam = render_heatmap(st["por_dia"])
        self._heat_img = ctk.CTkImage(img, size=tam)
        ctk.CTkLabel(mapa, image=self._heat_img, text="").pack(anchor="w", padx=18)
        legenda = ctk.CTkFrame(mapa, fg_color="transparent")
        legenda.pack(anchor="w", padx=18, pady=(4, 14))
        img_leg, tam_leg = render_legend()
        self._leg_img = ctk.CTkImage(img_leg, size=tam_leg)
        ctk.CTkLabel(legenda, text="menos", text_color=INK_3, height=12,
                     font=self._font(12)).pack(side="left", padx=(26, 6))
        ctk.CTkLabel(legenda, image=self._leg_img, text="").pack(side="left")
        ctk.CTkLabel(legenda, text="mais", text_color=INK_3, height=12,
                     font=self._font(12)).pack(side="left", padx=(6, 0))

        horas = ctk.CTkFrame(baixo, fg_color=SURFACE_2, corner_radius=16)
        horas.grid(row=0, column=1, sticky="nsew", padx=(12, 0), pady=(0, 12))
        pico = max(st["por_hora"], key=st["por_hora"].get)
        self._stats_titulo(horas, "Por hora do dia", f"pico as {pico}h")
        img_h, tam_h = render_hours(st["por_hora"])
        self._hours_img = ctk.CTkImage(img_h, size=tam_h)
        ctk.CTkLabel(horas, image=self._hours_img, text="").pack(padx=18, pady=(2, 14))

        top = ctk.CTkFrame(baixo, fg_color=SURFACE_2, corner_radius=16)
        top.grid(row=0, column=2, sticky="nsew", padx=(12, 0), pady=(0, 12))
        self._stats_titulo(top, "Mais ditadas")  # sem nota: a coluna e estreita
        for palavra, vezes in st["top"]:
            linha = ctk.CTkFrame(top, fg_color="transparent")
            linha.pack(fill="x", padx=18)
            ctk.CTkLabel(linha, text=palavra, text_color=INK, anchor="w", height=20,
                         font=self._font(14)).pack(side="left")
            ctk.CTkLabel(linha, text=fmt_int(vezes), text_color=INK_3, anchor="e", height=20,
                         font=self._font(13)).pack(side="right")
        ctk.CTkLabel(top, text="", height=6).pack()
        if self._tab == "estatisticas":  # ditado novo chegou com a aba aberta
            self._fit_stats_window()

    # -- biblioteca ----------------------------------------------------------
    def _build_library_tab(self):
        wrap = ctk.CTkFrame(self.content, fg_color="transparent")
        form = ctk.CTkFrame(wrap, fg_color="transparent")
        form.pack(fill="x", pady=(0, 8))
        form.grid_columnconfigure(0, weight=3)
        form.grid_columnconfigure(1, weight=2)
        ctk.CTkLabel(form, text="Sai assim (separe variantes por vírgula)", text_color=INK_3,
                     anchor="w", font=self._font(13, "semibold")).grid(
            row=0, column=0, sticky="ew", padx=(2, 0))
        ctk.CTkLabel(form, text="Deve virar", text_color=INK_3, anchor="w",
                     font=self._font(13, "semibold")).grid(row=0, column=1, sticky="ew", padx=(10, 0))

        def field(placeholder, col):
            e = ctk.CTkEntry(form, placeholder_text=placeholder, height=38, corner_radius=8,
                             border_width=1, fg_color=WHITE, border_color=BORDER_STRONG,
                             text_color=INK, placeholder_text_color=INK_3, font=self._font(15))
            e.grid(row=1, column=col, sticky="ew", padx=(0 if col == 0 else 10, 0), pady=(6, 0))
            e.bind("<Return>", lambda _ev: self._lib_add())
            return e

        self.lib_wrong = field("grock, groque, grote", 0)
        self.lib_right = field("Grok", 1)
        self._secondary(form, "Adicionar", self._lib_add, height=38).grid(
            row=1, column=2, padx=(10, 0), pady=(6, 0))
        self.lib_list = ctk.CTkScrollableFrame(wrap, fg_color=WHITE, corner_radius=16,
                                               border_width=1, border_color=BORDER)
        self.lib_list.pack(fill="both", expand=True, pady=(8, 0))
        return wrap

    def _render_library(self):
        for child in self.lib_list.winfo_children():
            child.destroy()
        if not self.library.entries:
            ctk.CTkLabel(self.lib_list, text="Nenhuma palavra no dicionário ainda.",
                         text_color=INK_3, font=self._font(15)).pack(anchor="w", padx=16, pady=14)
            return
        for i, entry in enumerate(self.library.entries):
            if i:
                tk.Frame(self.lib_list, bg=BORDER, height=1, bd=0, highlightthickness=0).pack(fill="x")
            row = ctk.CTkFrame(self.lib_list, fg_color="transparent", corner_radius=0)
            row.pack(fill="x")
            ctk.CTkButton(row, text="✕", command=lambda n=i: self._lib_remove(n),
                          width=30, height=30, corner_radius=6, fg_color="transparent",
                          hover_color=SURFACE_3, text_color=INK_3,
                          font=self._font(13)).pack(side="right", anchor="n", padx=(6, 10), pady=10)
            ctk.CTkLabel(row, text=entry["certo"], text_color=INK, width=150, anchor="w",
                         font=self._font(15, "semibold")).pack(side="left", padx=(16, 8), pady=14)
            ctk.CTkLabel(row, text=", ".join(entry["erros"]), text_color=INK_3,
                         anchor="w", justify="left", wraplength=420,
                         font=self._font(15)).pack(side="left", fill="x", expand=True, pady=14)

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
        ctk.CTkLabel(wrap, text=f"Gestos do headset · {g.profile.label}", text_color=INK,
                     anchor="w", font=self._font(17, "semibold")).pack(fill="x", padx=2, pady=(0, 0))
        ctk.CTkLabel(wrap, text=("Aciona o Sussurro pelo fone, sem tocar no PC: gire a roda de volume "
                                 "pra cima e pra baixo rapido, ou de um toque duplo no mute. "
                                 "Ao terminar de colar, o Sussurro aperta Enter."),
                     text_color=INK_2, anchor="w", justify="left", wraplength=620,
                     font=self._font(15)).pack(fill="x", padx=2, pady=(4, 12))

        # status: fone / acesso / mic do fone / mic padrao
        st = ctk.CTkFrame(wrap, fg_color=SURFACE_2, corner_radius=16)
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
                          fg_color=BORDER_STRONG, progress_color=INK, button_color=WHITE,
                          button_hover_color=WHITE, text_color=INK, font=self._font(15)).grid(
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
            sl = ctk.CTkSlider(pr, from_=lo, to=hi, number_of_steps=steps, progress_color=INK,
                               fg_color=BORDER_STRONG, button_color=INK, button_hover_color=INK_2,
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
            height=38, corner_radius=8, border_width=1, fg_color=WHITE, border_color=BORDER_STRONG,
            button_color=WHITE, button_hover_color=SURFACE_3, dropdown_fg_color=WHITE,
            dropdown_hover_color=SURFACE_3, dropdown_text_color=INK, text_color=INK,
            font=self._font(14))
        self.dev_fallback.set(current)
        self.dev_fallback.grid(row=4, column=0, columnspan=3, sticky="ew", padx=8, pady=(4, 4))

        # eventos
        ctk.CTkLabel(wrap, text="EVENTOS", text_color=INK_3, anchor="w",
                     font=(self.FONT_UI, 10, "bold")).pack(fill="x", padx=14, pady=(8, 0))
        self.dev_feed = ctk.CTkTextbox(wrap, fg_color=SURFACE_2, text_color=INK_2,
                                       font=(self.FONT_MONO, 12), height=150, corner_radius=16,
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

    def _on_sounds(self):
        self.settings["feedback_sounds"] = bool(self.sounds_var.get())
        self._save()

    def _sound(self, event):
        sussurro_sounds.play(event, enabled=bool(self.settings.get("feedback_sounds", False)))

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

    def _save_meeting_pos(self, pos):
        self.settings["meeting_pos"] = pos
        self._save()
        self.status.configure(text="Posicao do indicador de reuniao salva.")

    def show_meeting_pill(self):
        if self.meeting_pill is not None:
            self.meeting_pill.reveal()

    def capture_hotkey(self):
        self.hotkey.capturing = True
        self.status.configure(text="Clique o botao do mouse desejado (meio ou laterais)...")

    # -- gravacao -----------------------------------------------------------
    def _device_index(self):
        device = self.mic_by_label.get(self.mic.get())
        return device.index if device else None

    def _device_source(self):
        device = self.mic_by_label.get(self.mic.get())
        return device.source if device else None

    def _start(self, inject: bool, auto_enter: bool = False):
        if self.transcriber.recording.is_set():
            return True
        if self.transcriber.comparing.is_set():
            self.status.configure(text="Comparacao em andamento — termine ou cancele na aba COMPARAR.")
            self.bar.flash("busy", "Comparacao em andamento.", 1200)
            return False
        if self.transcriber.model is None or self.transcriber.model_loading.is_set():
            self._recover_model()
            self.status.configure(text="Modelo ainda carregando — aguarde.")
            self.bar.flash("busy", "Modelo ainda carregando.", 1200)
            return False
        if not self.transcriber.recording.is_set() and self.transcriber.busy():
            self.status.configure(text="Aguarde o ditado ou arquivo anterior terminar.")
            self.bar.flash("busy", "Aguarde o trabalho anterior.", 1200)
            return False
        try:
            self.transcriber.start(
                self._device_index(), inject,
                capture_mode=self.settings["capture_mode"],
                loopback_index=None,  # audio do PC = saida padrao do sistema
                auto_enter=auto_enter,
                mic_source=self._device_source(),
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
            self.bar.flash("error", f"ERRO ao abrir {alvo}: {e}", 2000)
            self._sound("error")
            return False
        self.record_btn.configure(text="Parar")
        self.bar.show("rec")
        self._sound("start")
        return True

    def _stop(self):
        if not self.transcriber.recording.is_set():
            return  # ja parado (ex.: confirmou na barra e soltou o atalho depois)
        self.transcriber.stop()
        self.hotkey.active = False  # parar pela UI nao pode deixar o atalho invertido
        self.record_btn.configure(text="Gravar")
        self.bar.show("proc")
        self._sound("stop")

    def _cancel(self):
        """X da barra: joga a sessao fora — nao transcreve, nao cola, nao arquiva."""
        self.transcriber.cancel(from_processing=True)
        self.hotkey.active = False
        self.record_btn.configure(text="Gravar")
        self.bar.hide()

    def toggle(self):
        if self.transcriber.recording.is_set():
            self._stop()
        else:
            self._start(inject=False)

    def copy_all(self):
        text = self.text.get("1.0", "end-1c")
        self._copy_text(text, lambda ok: self.status.configure(
            text=f"Copiado ({len(text)} caracteres)." if ok else "Não deu pra copiar: wl-copy/xclip falhou."))

    def clear(self):
        self.text.delete("1.0", "end")

    # -- loop de UI ---------------------------------------------------------
    def _ui(self, fn, *args, **kwargs):
        """Agenda chamada de widget para a thread do Tk. Nao usar Tk daqui de outra thread."""
        self._ui_queue.put((fn, args, kwargs))

    def _apply_model(self):
        """Escolher no MODELO ou no EXECUTAR EM ja troca; recusado, o combo volta pro ativo."""
        if self.transcriber.busy():
            self.status.configure(text="Aguarde o ditado ou arquivo atual terminar antes de trocar o modelo.")
            self._show_model_selection()
            return
        selection = {
            "whisper_model": next(k for k, v in MODEL_LABELS.items() if v == self.model_choice.get()),
            "whisper_device": next(k for k, v in self.device_labels.items() if v == self.device_choice.get()),
        }
        current = normalize_model_settings(self.settings)
        if selection == current and self.transcriber.model is not None:
            return  # mesma escolha ja carregada
        self._begin_model_load(selection, persist=True)

    def _show_model_selection(self):
        """Combos mostram a escolha salva, que e a carregada (ou a que falhou e foi revertida)."""
        current = normalize_model_settings(self.settings)
        self.model_choice.set(MODEL_LABELS[current["whisper_model"]])
        self.device_choice.set(self.device_labels[current["whisper_device"]])

    def _set_model_picker(self, state):
        """Trava os combos de modelo durante ditado, comparacao, Refazer ou carga."""
        state = "readonly" if state == "normal" else state
        self.model_choice.configure(state=state)
        self.device_choice.configure(state=state)

    def _update_engine_now(self):
        if getattr(self, "engine_now", None) is None:
            return  # card de config ainda nao montado
        text, level = self.transcriber.engine_status()
        if (text, level) == self._engine_shown:
            return
        self._engine_shown = (text, level)
        mark = {"ok": "●", "aviso": "◐", "off": "○"}[level]
        self.engine_now.configure(text=f"{mark} {text}", text_color=ENGINE_COLORS[level])
        self.model_label.configure(text=text)

    def _update_cloud_spend(self):
        """Gasto do mes embaixo do EM USO AGORA, so com modelo de nuvem escolhido."""
        if getattr(self, "cloud_spend", None) is None:
            return
        ledger = shared_ledger()
        if self._tab == "estatisticas" and ledger.version != getattr(self, "_spend_version", None):
            self._spend_version = ledger.version
            self._render_stats()  # gasto ou saldo novo com a aba aberta
        model = normalize_model_settings(self.settings)["whisper_model"]
        if model == ELEVENLABS:
            ledger.refresh_balance()
        text = spend_line(shared_ledger(), model) if is_cloud(model) else ""
        if text == self._spend_shown:
            return
        self._spend_shown = text
        if text:
            self.cloud_spend.configure(text=text)
            self.cloud_spend.pack(fill="x", pady=(0, 2))
        else:
            self.cloud_spend.pack_forget()

    def _recover_model(self):
        """Sem modelo e sem carga em curso (falha CUDA, VRAM cheia): tenta de novo o modelo salvo."""
        if (self.transcriber.model is not None or self.transcriber.model_loading.is_set()
                or self.transcriber.busy()):
            return
        self._begin_model_load(dict(self.settings), persist=False)

    def _begin_model_load(self, selection, *, persist):
        # Set the gate on Tk's thread before launching work: queued hotkeys cannot race it.
        self.transcriber.model_loading.set()
        self.record_btn.configure(state="disabled")
        self._set_model_picker("disabled")
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
        self._set_model_picker("normal")
        self._engine_shown = None  # o proximo poll reescreve EM USO e o cabecalho
        if error:
            self._show_model_selection()
            self.status_queue.put("ERRO ao carregar: " + error +
                                  (" — modelo anterior mantido." if ready else " — escolha modelo menor ou CPU e aplique."))

    def _poll(self):
        try:
            while True:
                fn, args, kwargs = self._ui_queue.get_nowait()
                fn(*args, **kwargs)
        except queue.Empty:
            pass
        panel = self.meeting_panel
        if panel is not None:
            seconds = panel.elapsed
            if panel.recording and not panel.paused:
                seconds += time.monotonic() - panel.resumed_at
            if self.meeting_pill is not None:
                self.meeting_pill.update(panel.recording, panel.paused, seconds)
        # primeiro de tudo: a barra so fica enquanto ha trabalho. Esperar o status certo
        # a punha pra sair depois de colar e de redesenhar o historico — e e isso que o
        # olho le como travamento.
        if self.bar.visivel() and not self.transcriber.busy():
            self.bar.finish()
        self._update_engine_now()
        self._update_cloud_spend()
        try:
            while True:
                event, payload = self.hotkey_queue.get_nowait()
                if event == "show":
                    self._show()
                elif event == "quit":
                    self._quit()
                    return  # nenhum widget/after depois de destruir a raiz
                elif event == "captured":
                    self.settings["mouse_button"] = payload
                    self.hotkey_var.set(BUTTON_LABELS[payload])
                    self._save()
                    self.status.configure(text=f"Atalho definido: {BUTTON_LABELS[payload]}.")
                elif event == "error":
                    self.status.configure(text=f"ERRO: {payload}")
                    self.bar.flash("error", f"ERRO: {payload}", 2000)
                    self._sound("error")
                elif event == "start":
                    if not self._start(inject=True, auto_enter=_wants_enter(payload)):
                        self.hotkey.active = False  # falhou: nao deixa o estado do atalho preso
                elif event == "stop":
                    if _wants_enter(payload):
                        self.transcriber.arm_auto_enter()
                    self._stop()
                elif event == "cancel":
                    if self.transcriber.recording.is_set() or not self.transcriber._drained:
                        self._cancel()
                elif event == "meeting":
                    verb, reply, deadline = payload
                    if time.monotonic() >= deadline:
                        continue  # cliente desistiu antes de comecar: nao gravar depois do err
                    try:
                        response = (self.meeting_panel.command(verb) if self.meeting_panel is not None
                                    else "err Reuniao indisponivel nesta plataforma.\n")
                    except Exception as error:
                        response = "err " + str(error).replace("\n", " ") + "\n"
                    reply.put_nowait(response)
                    if response.startswith("err"):
                        self.status.configure(text=response.strip())
                        self.bar.flash("error", response.strip(), 2000)
                        self._sound("error")
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
                    self.bar.flash("error", msg, 2000)
                    self._sound("error")
                if msg == CUDA_DISCARD_STATUS:
                    self.root.after(MODEL_RECOVERY_DELAY_MS, self._recover_model)
        except queue.Empty:
            pass
        self.root.after(UI_POLL_MS, self._poll)


if __name__ == "__main__":
    if IS_WIN:
        # identidade propria na taskbar: sem isto o Windows agrupa sob o pythonw
        # generico e mostra o icone do Python em vez do sussurro.ico da janela
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Lucas.Sussurro")
    ctk.set_appearance_mode("light")
    register_bundled_fonts()
    # className no Linux casa com StartupWMClass=Sussurro do .desktop (no Windows o
    # AppUserModelID ja cuida do agrupamento na taskbar).
    root = ctk.CTk() if IS_WIN else ctk.CTk(className="Sussurro")
    if not IS_WIN:
        # Sem cursor proprio, frame e label herdam o cursor padrao do X, que no
        # XWayland do Hyprland (Omarchy 4.5) e uma imagem 24x24 transparente: o
        # ponteiro sumia no painel de configuracoes. Seta explicito na raiz e em
        # toda Toplevel; Text/Entry seguem com o xterm proprio.
        tk.Tk.configure(root, cursor="left_ptr")
        root.option_add("*Toplevel.cursor", "left_ptr")
    App(root)
    root.mainloop()
