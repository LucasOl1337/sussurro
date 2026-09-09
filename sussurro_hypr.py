"""Hyprland (Omarchy): monitores e posicao do cursor pelo socket de controle.

No Wayland o Tk roda em XWayland e so enxerga uma "tela" unica que e a uniao de todos
os monitores; alem disso `winfo_pointerx` fica congelado quando o ponteiro esta sobre
janelas nativas. Por isso a barra de gravacao pergunta ao Hyprland onde o cursor esta e
qual e a area util do monitor que o contem. Sem Hyprland, tudo aqui devolve None e o
chamador cai no Tk.

Coordenadas: o Hyprland e o XWayland podem ordenar os monitores de forma diferente,
mesmo em escala 1. A API publica deste modulo devolve coordenadas X para o Tk;
a correspondencia com o cursor nativo usa o nome do conector, nunca a ordem.
"""

import json
import os
import re
import socket
import subprocess
import time
from pathlib import Path

_MON_TTL = 3.0  # s: cache da lista de monitores


def _socket_path() -> Path | None:
    run = os.environ.get("XDG_RUNTIME_DIR")
    if not run:
        return None
    sig = os.environ.get("HYPRLAND_INSTANCE_SIGNATURE")
    if sig:
        p = Path(run) / "hypr" / sig / ".socket.sock"
        if p.exists():
            return p
    # app iniciado por um servico sem a variavel: pega a instancia mais recente
    cands = sorted(Path(run).glob("hypr/*/.socket.sock"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


class Hypr:
    def __init__(self):
        self.sock = _socket_path()
        self.available = self.sock is not None
        self._mons = []
        self._native_mons = []
        self._placement = None
        self._mons_at = 0.0

    def _query(self, cmd: str):
        """`j/<cmd>` pelo socket (rapido: sem processo); cai no hyprctl se falhar."""
        if not self.available:
            return None
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                s.connect(str(self.sock))
                s.sendall(f"j/{cmd}".encode())
                buf = b""
                while True:
                    chunk = s.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
            return json.loads(buf.decode("utf-8", "replace"))
        except (OSError, ValueError):
            try:
                r = subprocess.run(["hyprctl", "-j", cmd], capture_output=True, text=True, timeout=1)
                return json.loads(r.stdout) if r.returncode == 0 else None
            except (OSError, ValueError, subprocess.TimeoutExpired):
                return None

    def _x_monitors(self):
        try:
            result = subprocess.run(["xrandr", "--query"], capture_output=True,
                                    text=True, timeout=1)
        except (OSError, subprocess.TimeoutExpired):
            return {}
        if result.returncode:
            return {}
        pattern = r"^(\S+) connected(?: primary)? (\d+)x(\d+)\+(-?\d+)\+(-?\d+)"
        return {m[1]: tuple(map(int, m.groups()[1:]))
                for m in re.finditer(pattern, result.stdout, re.MULTILINE)}

    def monitors(self) -> list:
        now = time.monotonic()
        if now - self._mons_at > _MON_TTL:
            mons = self._query("monitors")
            if isinstance(mons, list) and mons:
                self._native_mons = mons
                outputs = self._x_monitors()
                self._mons = []
                for m in mons:
                    w, h, x, y = outputs.get(m["name"], (m["width"], m["height"], m["x"], m["y"]))
                    scale = m.get("scale", 1)
                    sx, sy = w / (m["width"] / scale), h / (m["height"] / scale)
                    reserved = m.get("reserved") or [0, 0, 0, 0]
                    self._mons.append({**m, "width": w, "height": h, "x": x, "y": y,
                                       "reserved": [round(v * (sx if i % 2 == 0 else sy))
                                                    for i, v in enumerate(reserved)]})
            self._mons_at = now
        return self._mons

    def cursorpos(self):
        pos = self._query("cursorpos")
        if isinstance(pos, dict) and "x" in pos:
            x, y = int(pos["x"]), int(pos["y"])
            mapped = {m["name"]: m for m in self.monitors()}
            for m in self._native_mons:
                scale = m.get("scale", 1)
                width, height = m["width"] / scale, m["height"] / scale
                if m["x"] <= x < m["x"] + width and m["y"] <= y < m["y"] + height:
                    target = mapped[m["name"]]
                    return (round(target["x"] + (x - m["x"]) * target["width"] / width),
                            round(target["y"] + (y - m["y"]) * target["height"] / height))
            return x, y
        return None

    def monitor_at(self, x: int, y: int):
        mons = self.monitors()
        for m in mons:
            if m["x"] <= x < m["x"] + m["width"] and m["y"] <= y < m["y"] + m["height"]:
                return m
        return mons[0] if mons else None

    def work_area_at(self, x: int, y: int):
        """(left, top, right, bottom) do monitor que contem (x, y), sem a barra/reservas."""
        m = self.monitor_at(x, y)
        if not m:
            return None
        rl, rt, rr, rb = (m.get("reserved") or [0, 0, 0, 0])[:4]
        return (m["x"] + rl, m["y"] + rt, m["x"] + m["width"] - rr, m["y"] + m["height"] - rb)

    def activewindow(self):
        """Janela com foco, ou None se o Hyprland nao responder."""
        win = self._query("activewindow")
        return win if isinstance(win, dict) and win.get("class") else None

    def place_bar(self, x: int, y: int) -> bool:
        target = self.monitor_at(x, y)
        if target is None:
            return False
        native = next(m for m in self._native_mons if m["name"] == target["name"])
        scale = native.get("scale", 1)
        gx = round(native["x"] + (x - target["x"]) * native["width"] / scale / target["width"])
        gy = round(native["y"] + (y - target["y"]) * native["height"] / scale / target["height"])
        placement = (target["name"], gx, gy)
        if placement == self._placement:
            return True
        code = ('for _,w in ipairs(hl.get_windows()) do '
                f'if w.pid=={os.getpid()} and w.class=="SussurroBar" then '
                'local moved=hl.dispatch(hl.dsp.window.move({window=w,monitor='
                + json.dumps(target["name"]) + ',follow=false})); '
                f'local placed=hl.dispatch(hl.dsp.window.move({{window=w,x={gx},y={gy},relative=false}})); '
                'return moved.ok and placed.ok end end; return false')
        try:
            result = subprocess.run(["hyprctl", "eval", code], capture_output=True,
                                    text=True, timeout=1)
        except (OSError, subprocess.TimeoutExpired):
            return False
        if result.returncode == 0 and result.stdout.strip() == "true":
            self._placement = placement
            return True
        return False

    def reset_bar_placement(self):
        self._placement = None
