"""Sussurro — voice-to-text local: mic -> Silero VAD -> faster-whisper large-v3 (CUDA).

HUD Tkinter: gravar/parar, atalho global de mouse (digita onde o cursor estiver),
escolha de microfone, modo de transcricao (simultaneo por trecho ou tudo ao final).
"""

import ctypes
import ctypes.wintypes as wintypes
import json
import os
import queue
import sys
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
import traceback
import wave
import winsound
from datetime import date, datetime
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
from PIL import Image
import sounddevice as sd
from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions, get_speech_timestamps
from pynput import keyboard, mouse

SAMPLE_RATE = 16000
BLOCK_SIZE = 1600  # 100 ms por callback do mic

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

SETTINGS_PATH = Path(__file__).with_name("settings.json")
HISTORY_DIR = Path(__file__).with_name("history")
HISTORY_INDEX = HISTORY_DIR / "history.jsonl"
DEFAULT_SETTINGS = {
    "mouse_button": "x2",       # middle | x1 | x2
    "trigger_mode": "alternar",  # alternar (clique liga/desliga) | segurar (push-to-talk)
    "device_name": None,
    "transcribe_mode": "simultaneo",  # simultaneo | final
    "language": "pt",
    "inject_method": "colar",   # colar (Ctrl+V, clipboard preservado) | digitar
    "dot_pos": [0.5, 0.94],     # posicao da bolinha, fracao da area util do monitor
}

# Mensagens do hook de mouse do Windows (win32_event_filter do pynput)
WM_MBUTTONDOWN, WM_MBUTTONUP = 0x0207, 0x0208
WM_XBUTTONDOWN, WM_XBUTTONUP = 0x020B, 0x020C
BUTTON_LABELS = {"middle": "botao do meio", "x1": "lateral 1 (tras)", "x2": "lateral 2 (frente)"}


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


class Transcriber:
    """Threads de captura, segmentacao (VAD) e transcricao. UI le da text_queue."""

    def __init__(self, text_queue: queue.Queue, status_queue: queue.Queue):
        self.text_queue = text_queue
        self.status_queue = status_queue
        self.model = None
        self.language = "pt"
        self.transcribe_mode = "simultaneo"
        self.inject_method = "colar"
        self.recording = threading.Event()
        self.history_queue: queue.Queue = queue.Queue()
        self._audio_queue: queue.Queue = queue.Queue()
        self._segment_queue: queue.Queue = queue.Queue()
        self._session_parts: list = []
        self._session_started = None
        self._stream = None
        self._resampler = None
        self._session_inject = False
        self._session_mode = "simultaneo"
        self._session_had_speech = False
        self._keyboard = keyboard.Controller()
        threading.Thread(target=self._segmenter_loop, daemon=True).start()
        threading.Thread(target=self._transcribe_loop, daemon=True).start()

    # -- modelo -------------------------------------------------------------
    def load_model(self):
        self.status_queue.put("Carregando large-v3 na GPU...")
        t0 = time.perf_counter()
        self.model = WhisperModel("large-v3", device="cuda", compute_type="float16")
        # aquecimento: primeira inferencia compila kernels e distorce a latencia
        self.model.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), language="pt")
        self.status_queue.put(f"Modelo pronto ({time.perf_counter() - t0:.1f}s). Pode gravar.")

    # -- captura ------------------------------------------------------------
    def _mic_callback(self, indata, frames, time_info, status):
        if status:
            self.status_queue.put(f"Aviso do mic: {status}")
        if self.recording.is_set():
            data = indata[:, 0]
            data = self._resampler.process(data) if self._resampler else data.copy()
            if data.size:
                self._audio_queue.put(data)

    def start(self, device_index: int | None, inject: bool):
        if self.recording.is_set():
            return
        self._session_inject = inject
        self._session_mode = self.transcribe_mode
        self._session_had_speech = False
        self._session_started = datetime.now()
        self._resampler = None
        try:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                blocksize=BLOCK_SIZE, device=device_index, callback=self._mic_callback,
            )
            self._stream.start()
        except sd.PortAudioError:
            # dispositivo nao aceita 16 kHz (WASAPI): abre na taxa nativa e reamostra
            query = device_index if device_index is not None else sd.default.device[0]
            native = int(sd.query_devices(query)["default_samplerate"])
            self._resampler = StreamResampler(native)
            self._stream = sd.InputStream(
                samplerate=native, channels=1, dtype="float32",
                blocksize=int(native * 0.1), device=device_index, callback=self._mic_callback,
            )
            self._stream.start()
        self.recording.set()
        self.status_queue.put("Gravando — pode falar.")

    def stop(self):
        if not self.recording.is_set():
            return
        self.recording.clear()
        # status ANTES do sentinela: o segmentador emite os status finais depois dele,
        # e a ordem na fila e o que impede "Transcrevendo..." de ficar pendurado
        self.status_queue.put("Parado. Transcrevendo...")
        self._audio_queue.put(None)  # sentinela: descarrega o buffer restante
        stream, self._stream = self._stream, None
        stream.stop()
        stream.close()

    # -- segmentacao --------------------------------------------------------
    def _segmenter_loop(self):
        buffer = np.zeros(0, dtype=np.float32)
        last_check = 0.0
        while True:
            item = self._audio_queue.get()
            try:
                if item is None:  # fim da gravacao: manda o que sobrou
                    if buffer.size > SAMPLE_RATE // 4 and get_speech_timestamps(buffer, VAD_OPTIONS):
                        self._enqueue_segment(buffer)
                    elif not self._session_had_speech:
                        self.status_queue.put("Parado (sem fala detectada).")
                    else:
                        self.status_queue.put("Parado.")
                    self._segment_queue.put((None, None, None))  # marcador de fim de sessao
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
        self._segment_queue.put((audio, self._session_inject, self._session_mode))

    # -- transcricao --------------------------------------------------------
    def _transcribe_loop(self):
        while True:
            audio, inject, mode = self._segment_queue.get()
            try:
                if audio is None:  # fim de sessao: grava no historico
                    self._finalize_session()
                    continue
                t0 = time.perf_counter()
                lang = None if self.language == "auto" else self.language
                segments, _info = self.model.transcribe(
                    audio, language=lang, beam_size=5, vad_filter=(mode == "final"),
                )
                text = "".join(s.text for s in segments).strip()
                dt = time.perf_counter() - t0
                self._session_parts.append((audio, text))
                if text:
                    self.text_queue.put(text)
                    if inject:
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

    def _finalize_session(self):
        parts, self._session_parts = self._session_parts, []
        text = " ".join(t for _a, t in parts if t).strip()
        if not text:
            return
        started = self._session_started or datetime.now()
        audio = np.concatenate([a for a, _t in parts])
        HISTORY_DIR.mkdir(exist_ok=True)
        wav_name = started.strftime("%Y%m%d_%H%M%S") + ".wav"
        with wave.open(str(HISTORY_DIR / wav_name), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
        entry = {"ts": started.isoformat(timespec="seconds"), "wav": wav_name, "text": text}
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


class DotIndicator:
    """Bolinha de overlay: laranja com anel osso = gravando, invertida = transcrevendo.

    Sempre no topo, fora do Alt-Tab, sem roubar foco. Aparece no monitor onde o cursor
    esta (e segue se o cursor trocar de monitor) na posicao relativa salva; arrastavel
    com o mouse pra reposicionar — a posicao e salva como fracao da area util, entao se
    adapta a qualquer resolucao.
    """

    SIZE = 22
    # anel osso garante leitura sobre qualquer fundo (regra medida do icone sobre #202020)
    COLORS = {"rec": (ACCENT, INK), "proc": (INK, ACCENT)}
    TRANSPARENT = "#010203"

    def __init__(self, root: tk.Tk, get_rel_pos, save_rel_pos):
        self.root = root
        self.get_rel_pos = get_rel_pos
        self.save_rel_pos = save_rel_pos
        self.win = tk.Toplevel(root)
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        self.win.attributes("-transparentcolor", self.TRANSPARENT)
        self.canvas = tk.Canvas(self.win, width=self.SIZE, height=self.SIZE,
                                bg=self.TRANSPARENT, highlightthickness=0, cursor="fleur")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._drag_start)
        self.canvas.bind("<B1-Motion>", self._drag_move)
        self.canvas.bind("<ButtonRelease-1>", self._drag_end)
        self.win.update_idletasks()
        self._set_exstyle()
        self.win.withdraw()
        self._state = None
        self._big = False
        self._dragging = False
        self._drag_off = (0, 0)

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

    def _target_xy(self):
        cx, cy = cursor_pos()
        left, top, right, bottom = monitor_work_area(cx, cy)
        rx, ry = self.get_rel_pos()
        x = int(left + rx * (right - left - self.SIZE))
        y = int(top + ry * (bottom - top - self.SIZE))
        return x, y

    def _follow(self):
        if not self._dragging:
            x, y = self._target_xy()
            self.win.geometry(f"+{x}+{y}")

    def show(self, state: str):
        first = self._state is None
        self._state = state
        self._follow()
        self.win.deiconify()
        self.win.attributes("-topmost", True)
        self._draw()
        if first:
            self._pulse()

    def hide(self):
        self._state = None
        self._dragging = False
        self.win.withdraw()

    # -- arrastar pra reposicionar ------------------------------------------
    def _drag_start(self, event):
        self._dragging = True
        self._drag_off = (event.x, event.y)

    def _drag_move(self, _event):
        x = self.win.winfo_pointerx() - self._drag_off[0]
        y = self.win.winfo_pointery() - self._drag_off[1]
        self.win.geometry(f"+{x}+{y}")

    def _drag_end(self, _event):
        self._dragging = False
        wx, wy = self.win.winfo_x(), self.win.winfo_y()
        cx, cy = wx + self.SIZE // 2, wy + self.SIZE // 2
        left, top, right, bottom = monitor_work_area(cx, cy)
        rx = (wx - left) / max(right - left - self.SIZE, 1)
        ry = (wy - top) / max(bottom - top - self.SIZE, 1)
        self.save_rel_pos([round(min(max(rx, 0.0), 1.0), 4),
                           round(min(max(ry, 0.0), 1.0), 4)])

    def _draw(self):
        if self._state is None:
            return
        self.canvas.delete("all")
        r = 8 if self._big else 6
        c = self.SIZE // 2
        fill, ring = self.COLORS[self._state]
        self.canvas.create_oval(c - r, c - r, c + r, c + r,
                                fill=fill, outline=ring, width=2)

    def _pulse(self):
        if self._state is None:
            return
        self._big = not self._big
        self._draw()
        self._follow()
        self.root.after(450, self._pulse)


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

        # abas: acento fica no GRAVAR; aba ativa marca por chapa mais clara
        tabbar = ctk.CTkFrame(root, fg_color="transparent")
        tabbar.pack(fill="x", padx=18, pady=(0, 6))
        self.tab_btns = {}
        for name, label in (("historico", "HISTORICO"), ("aovivo", "AO VIVO")):
            btn = ctk.CTkButton(tabbar, text=label, width=108, height=30, corner_radius=8,
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
        ctk.CTkLabel(status_bar, text="bolinha: arraste para reposicionar", text_color=INK_3,
                     font=("Segoe UI", 10)).pack(side="right")

        # card de conteudo: historico / ao vivo
        self.content = ctk.CTkFrame(root, fg_color=SURFACE, corner_radius=12)
        self.content.pack(fill="both", expand=True, padx=18, pady=(0, 4))
        self.hist_frame = ctk.CTkScrollableFrame(self.content, fg_color="transparent")
        self.text = ctk.CTkTextbox(self.content, fg_color="transparent", text_color=INK,
                                   font=("Segoe UI", 13), wrap="word", border_width=0)
        self._tab = None
        self._show_tab("historico")

        self.entries = self._load_history()  # mais recente primeiro
        self._render_history()

        self.dot = DotIndicator(root, lambda: self.settings["dot_pos"], self._save_dot_pos)
        self._playing = None

        threading.Thread(target=self._load_model, daemon=True).start()
        root.after(100, self._poll)

    # -- abas / historico ----------------------------------------------------
    def _show_tab(self, name: str):
        if name == self._tab:
            return
        self._tab = name
        for n, btn in self.tab_btns.items():
            if n == name:
                btn.configure(fg_color=SURFACE_3, text_color=INK, hover_color=SURFACE_3)
            else:
                btn.configure(fg_color="transparent", text_color=INK_3, hover_color=SURFACE_2)
        if name == "historico":
            self.text.pack_forget()
            self.hist_frame.pack(fill="both", expand=True, padx=6, pady=6)
        else:
            self.hist_frame.pack_forget()
            self.text.pack(fill="both", expand=True, padx=6, pady=6)

    @staticmethod
    def _load_history():
        if not HISTORY_INDEX.exists():
            return []
        entries = []
        for line in HISTORY_INDEX.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entries.append(json.loads(line))
        return list(reversed(entries[-200:]))  # mais recente primeiro

    def _render_history(self):
        for child in self.hist_frame.winfo_children():
            child.destroy()
        prev_day = None
        i = 0
        for entry in self.entries:
            dt = datetime.fromisoformat(entry["ts"])
            day = "HOJE" if dt.date() == date.today() else dt.strftime("%d/%m/%Y")
            if day != prev_day:
                hdr = ctk.CTkFrame(self.hist_frame, fg_color="transparent")
                hdr.pack(fill="x", padx=8, pady=(12, 4))
                ctk.CTkLabel(hdr, text=day, text_color=INK_3, height=14,
                             font=("Segoe UI", 10, "bold")).pack(side="left")
                ctk.CTkFrame(hdr, fg_color=BORDER, height=1).pack(
                    side="left", fill="x", expand=True, padx=(10, 0))
                prev_day = day
                i = 0
            base = "transparent" if i % 2 == 0 else ROW_EVEN
            i += 1
            row = ctk.CTkFrame(self.hist_frame, fg_color=base, corner_radius=8)
            row.pack(fill="x", padx=2, pady=1)
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

    def _on_mode(self, _e):
        self.settings["transcribe_mode"] = self.mode.get()
        self.transcriber.transcribe_mode = self.mode.get()
        self._save()

    def _on_inject(self, _e):
        self.settings["inject_method"] = self.inject.get()
        self.transcriber.inject_method = self.inject.get()
        self._save()

    def _save_dot_pos(self, pos):
        self.settings["dot_pos"] = pos
        self._save()
        self.status.configure(text="Posicao da bolinha salva.")

    def capture_hotkey(self):
        self.hotkey.capturing = True
        self.status.configure(text="Clique o botao do mouse desejado (meio ou laterais)...")

    # -- gravacao -----------------------------------------------------------
    def _device_index(self):
        return self.devices.get(self.mic.get())

    def _start(self, inject: bool):
        if self.transcriber.model is None:
            self.status.configure(text="Modelo ainda carregando — aguarde.")
            return False
        try:
            self.transcriber.start(self._device_index(), inject)
        except Exception as e:
            self.status.configure(text=f"ERRO ao abrir o microfone: {e}")
            return False
        self.record_btn.configure(text="PARAR")
        self.dot.show("rec")
        return True

    def _stop(self):
        self.transcriber.stop()
        self.record_btn.configure(text="GRAVAR")
        self.dot.show("proc")

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
                self.entries.insert(0, entry)
                self._render_history()
        except queue.Empty:
            pass
        try:
            while True:
                msg = self.status_queue.get_nowait()
                self.status.configure(text=msg)
                done = msg.startswith("Parado") and "Transcrevendo" not in msg
                if done or msg.startswith("ERRO"):
                    self.dot.hide()
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
