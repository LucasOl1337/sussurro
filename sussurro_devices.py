"""Gestos de dispositivos (Linux/Omarchy): aciona o Sussurro pelo headset, sem tocar no PC.

Hoje o unico perfil e o MCHOSE X9 (dongle 2.4 GHz, chip C-Media, USB 3837:6045). O que
o fone entrega ao PC e pouco: volume +/-, play/pause e o audio. O botao de mute do mic
nao gera evento; ele apenas zera o audio (silencio digital exato) e toca um aviso de voz.

Gestos reconhecidos:
  * Roda de volume invertida rapido (vol+ e vol- em menos de `reversal_window` s)
    -> gesto. Girar em uma direcao continua sendo volume.
  * Toque duplo no mute (mic zerado por ate `tap_max` s) -> gesto. O aviso de voz do
    fone gera um segundo zero de ~1,5 s logo depois, ignorado pela assinatura.

Alem disso, o mic padrao do sistema segue o fone: X9 quando esta com audio, o mic
reserva quando o X9 fica mutado/desligado por `idle_switch` s. O Sussurro usa o mic
padrao quando o microfone escolhido e "pipewire"/"default".

Precisa de acesso de leitura ao /dev/input do fone: regra udev em contrib/omarchy.
"""

import collections
import os
import re
import select
import shutil
import struct
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

UDEV_RULE = Path(__file__).with_name("contrib") / "omarchy" / "70-mchose-x9.rules"
UDEV_TARGET = "/etc/udev/rules.d/70-mchose-x9.rules"

KEY_VOLUMEDOWN, KEY_VOLUMEUP = 114, 115
RATE = 16000
CHUNK_MS = 25
CHUNK_BYTES = RATE * 2 * CHUNK_MS // 1000  # s16 mono

TAP_MIN = 0.05        # s: ZERO mais curto que isso e ruido de leitura
MERGE_GAP = 0.3       # s: ZEROs separados por menos que isso sao o mesmo gesto
BLIP_MAX_GAP = 1.3    # s: o aviso de voz gera um 2o ZERO que comeca ate isso apos o anterior...
BLIP_MAX_DUR = 2.0    # s: ...e dura no maximo isso
DEBOUNCE = 2.0        # s: intervalo minimo entre dois gestos
ACTIVE_SWITCH = 0.5   # s: fone com som por tanto tempo -> mic padrao = fone
NODATA_TIMEOUT = 0.5  # s: sem audio do parec = fone desligado/dormindo


@dataclass(frozen=True)
class HeadsetProfile:
    key: str
    label: str
    evdev_name: str      # nome do input "Consumer Control" em /proc/bus/input/devices
    source_match: str    # trecho do nome da fonte PulseAudio/PipeWire do mic do fone
    usb_id: str


PROFILES = {
    "mchose_x9": HeadsetProfile(
        key="mchose_x9", label="MCHOSE X9 (dongle 2.4 GHz)",
        evdev_name="C-Media Electronics Inc MCHOSE X9 Consumer Control",
        source_match="MCHOSE_X9", usb_id="3837:6045"),
}

DEFAULTS = {
    "enabled": False,
    "profile": "mchose_x9",
    "auto_enter": True,        # ao terminar de colar, aperta Enter (ditar longe do PC)
    "wheel_gesture": True,
    "mute_gesture": True,
    "reversal_window": 0.5,    # s
    "tap_max": 7.0,            # s
    "mic_follow": True,
    "fallback_source": None,   # None = primeiro mic que nao e o fone
    "idle_switch": 10.0,       # s
}


def list_sources() -> list[str]:
    """Nomes das fontes de captura (sem monitores) do PulseAudio/PipeWire."""
    if not shutil.which("pactl"):
        return []
    try:
        r = subprocess.run(["pactl", "list", "short", "sources"], capture_output=True,
                           text=True, timeout=3)
    except (OSError, subprocess.TimeoutExpired):
        return []
    out = []
    for line in r.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and ".monitor" not in parts[1]:
            out.append(parts[1])
    return out


def pretty_source(name: str | None) -> str:
    """'alsa_input.usb-3142_FIFINE_Microphone-00.mono-fallback' -> 'FIFINE Microphone'."""
    if not name:
        return "—"
    s = name.split("usb-", 1)[1] if "usb-" in name else name.split(".", 1)[-1]
    s = re.split(r"-\d\d\.", s)[0]                 # tira '-01.mono-fallback'
    s = re.sub(r"_[0-9A-Fa-f]{6,}$", "", s)         # serial no fim
    s = re.sub(r"^[0-9A-Fa-f]{4}_", "", s)          # id do fabricante no comeco
    return s.replace("_", " ")[:40]


def find_event_device(name: str) -> str | None:
    try:
        blocks = Path("/proc/bus/input/devices").read_text().split("\n\n")
    except OSError:
        return None
    for b in blocks:
        if f'N: Name="{name}"' in b:
            for tok in b.split():
                if tok.startswith("event"):
                    return "/dev/input/" + tok
    return None


def evdev_readable(dev: str | None) -> bool:
    return bool(dev) and os.access(dev, os.R_OK)


def udev_rule_installed() -> bool:
    return Path(UDEV_TARGET).exists()


def install_udev_rule() -> tuple[bool, str]:
    """Instala a regra udev com pkexec (app grafico: nao ha terminal pra senha do sudo)."""
    if not UDEV_RULE.exists():
        return False, f"regra nao encontrada em {UDEV_RULE}"
    if not shutil.which("pkexec"):
        return False, "pkexec nao encontrado; instale manualmente (veja contrib/omarchy/README.md)"
    script = (f"install -Dm644 '{UDEV_RULE}' '{UDEV_TARGET}' && udevadm control --reload "
              f"&& udevadm trigger --subsystem-match=input --action=change")
    try:
        r = subprocess.run(["pkexec", "sh", "-c", script], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, str(e)
    if r.returncode != 0:
        return False, (r.stderr or r.stdout).strip() or f"pkexec saiu com {r.returncode}"
    return True, "regra udev instalada; o fone pode ser lido sem root"


def default_source() -> str:
    try:
        return subprocess.run(["pactl", "get-default-source"], capture_output=True,
                              text=True, timeout=3).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


class DeviceGestures:
    """Threads de leitura do fone. `cfg` e o dict `settings["devices"]`, lido ao vivo."""

    def __init__(self, cfg: dict, on_gesture, log=None):
        self.cfg = cfg
        self.on_gesture = on_gesture
        self._log_cb = log
        self.events: collections.deque = collections.deque(maxlen=80)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._last_gesture = 0.0
        self._lock = threading.Lock()
        self.mic_state = "—"
        self.evdev_path = None
        self.parec = None
        self._default_cache = {"name": None, "at": 0.0}

    # -- util ----------------------------------------------------------------
    @property
    def profile(self) -> HeadsetProfile:
        return PROFILES.get(self.cfg.get("profile"), PROFILES["mchose_x9"])

    def log(self, msg: str):
        stamp = time.strftime("%H:%M:%S")
        self.events.appendleft((stamp, msg))
        if self._log_cb:
            self._log_cb(f"{stamp} {msg}")

    def running(self) -> bool:
        return any(t.is_alive() for t in self._threads)

    def headset_source(self) -> str | None:
        for s in list_sources():
            if self.profile.source_match in s:
                return s
        return None

    def fallback_source(self) -> str | None:
        wanted = self.cfg.get("fallback_source")
        sources = list_sources()
        if wanted and wanted in sources:
            return wanted
        for s in sources:
            if self.profile.source_match not in s:
                return s
        return None

    def status(self) -> dict:
        dev = find_event_device(self.profile.evdev_name)
        return {
            "running": self.running(),
            "headset": dev is not None,
            "evdev": dev,
            "access": evdev_readable(dev),
            "udev": udev_rule_installed(),
            "source": self.headset_source(),
            "fallback": self.fallback_source(),
            "default": default_source(),
            "mic_state": self.mic_state,
        }

    def gesture(self, why: str):
        now = time.time()
        with self._lock:
            if now - self._last_gesture < DEBOUNCE:
                self.log(f"{why}: ignorado (debounce)")
                return
            self._last_gesture = now
        self.log(f"{why} -> Sussurro")
        try:
            self.on_gesture(why)
        except Exception as e:  # noqa: BLE001
            self.log(f"erro ao acionar: {e}")

    # -- ciclo de vida --------------------------------------------------------
    def start(self):
        if self.running():
            return
        self._stop.clear()
        self._threads = [
            threading.Thread(target=self._wheel_loop, name="dev-wheel", daemon=True),
            threading.Thread(target=self._mic_loop, name="dev-mic", daemon=True),
        ]
        for t in self._threads:
            t.start()
        self.log(f"gestos ativos ({self.profile.label})")

    def stop(self):
        self._stop.set()
        p = self.parec
        if p is not None:
            try:
                p.kill()
            except OSError:
                pass
        self.mic_state = "—"
        self.log("gestos desligados")

    def restart(self):
        self.stop()
        for t in self._threads:
            t.join(timeout=2)
        self.start()

    # -- roda de volume ------------------------------------------------------
    def _wheel_loop(self):
        while not self._stop.is_set():
            dev = find_event_device(self.profile.evdev_name)
            self.evdev_path = dev
            if not dev or not evdev_readable(dev):
                self._stop.wait(5)
                continue
            try:
                fd = os.open(dev, os.O_RDONLY | os.O_NONBLOCK)
            except OSError as e:
                self.log(f"roda: {e}")
                self._stop.wait(10)
                continue
            self.log("roda de volume: escutando")
            last_dir, last_t = None, 0.0
            try:
                while not self._stop.is_set():
                    r, _, _ = select.select([fd], [], [], 0.5)
                    if not r:
                        continue
                    try:
                        d = os.read(fd, 24 * 64)
                    except BlockingIOError:
                        continue
                    if not d:
                        break
                    for i in range(0, len(d) - 23, 24):
                        _s, _us, typ, code, val = struct.unpack("qqHHi", d[i:i + 24])
                        if typ != 1 or val != 1 or code not in (KEY_VOLUMEDOWN, KEY_VOLUMEUP):
                            continue
                        now = time.time()
                        window = float(self.cfg.get("reversal_window", 0.5))
                        if (self.cfg.get("wheel_gesture", True) and last_dir is not None
                                and code != last_dir and now - last_t < window):
                            self.gesture(f"roda invertida em {now - last_t:.2f}s")
                        last_dir, last_t = code, now
            except OSError as e:
                self.log(f"roda: {e}")
            finally:
                os.close(fd)
            self._stop.wait(3)

    # -- mic: toque duplo no mute + mic padrao ---------------------------------
    def _set_default_source(self, name: str | None):
        if not name:
            return
        now = time.time()
        c = self._default_cache
        if c["name"] is None or now - c["at"] > 10:
            c["name"], c["at"] = default_source(), now
        if c["name"] == name:
            return
        subprocess.run(["pactl", "set-default-source", name], capture_output=True, timeout=3)
        c["name"], c["at"] = name, now
        self.log(f"mic padrao -> {pretty_source(name)}")

    def _mic_loop(self):
        while not self._stop.is_set():
            src = self.headset_source()
            if not src or not shutil.which("parec"):
                self.mic_state = "sem fone"
                self._stop.wait(5)
                continue
            try:
                self.parec = subprocess.Popen(
                    ["parec", f"--device={src}", "--format=s16le", f"--rate={RATE}",
                     "--channels=1", "--raw", "--latency-msec=20"],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            except OSError as e:
                self.log(f"mic: {e}")
                self._stop.wait(5)
                continue
            self.log("mic do fone: escutando")
            self._mic_session(self.parec.stdout.fileno(), src)
            try:
                self.parec.kill()
            except OSError:
                pass
            self.parec = None
            self._stop.wait(3)

    def _mic_session(self, fd: int, src: str):
        os.set_blocking(fd, False)
        state, state_since = None, time.time()
        buf = b""
        zero_start = None       # ZERO em andamento
        pending = None          # (inicio, fim) de um ZERO encerrado aguardando MERGE_GAP
        last_zero_end = 0.0
        while not self._stop.is_set():
            r, _, _ = select.select([fd], [], [], NODATA_TIMEOUT)
            now = time.time()
            if r:
                data = os.read(fd, 65536)
                if not data:
                    self.log("mic do fone: fluxo encerrou")
                    return
                buf += data
                new = None
                while len(buf) >= CHUNK_BYTES:
                    chunk, buf = buf[:CHUNK_BYTES], buf[CHUNK_BYTES:]
                    new = "SOM" if any(chunk) else "ZERO"
                if new is None:
                    continue
            else:
                new = "NODATA"

            if new != state:
                if new == "ZERO":
                    if pending and now - pending[1] < MERGE_GAP:
                        zero_start, pending = pending[0], None
                    else:
                        zero_start = now
                elif state == "ZERO" and zero_start is not None:
                    pending, zero_start = (zero_start, now), None
                state, state_since = new, now
                self.mic_state = {"SOM": "com som", "ZERO": "mudo (zerado)",
                                  "NODATA": "sem dados"}[new]

            if pending and now - pending[1] >= MERGE_GAP:
                start, end = pending
                pending = None
                dur, since_prev = end - start, start - last_zero_end
                last_zero_end = end
                tap_max = float(self.cfg.get("tap_max", 7.0))
                if since_prev < BLIP_MAX_GAP and dur <= BLIP_MAX_DUR:
                    pass  # aviso de voz do fone
                elif dur > tap_max:
                    self.log(f"mute longo ({dur:.1f}s): nada a fazer")
                elif dur >= TAP_MIN and self.cfg.get("mute_gesture", True):
                    self.gesture(f"toque duplo no mute ({dur:.2f}s)")

            if self.cfg.get("mic_follow", True):
                held = now - state_since
                if state == "SOM" and held >= ACTIVE_SWITCH:
                    self._set_default_source(src)
                elif state in ("ZERO", "NODATA") and held >= float(self.cfg.get("idle_switch", 10.0)):
                    self._set_default_source(self.fallback_source())
