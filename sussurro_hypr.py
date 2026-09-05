"""Hyprland (Omarchy): monitores e posicao do cursor pelo socket de controle.

No Wayland o Tk roda em XWayland e so enxerga uma "tela" unica que e a uniao de todos
os monitores; alem disso `winfo_pointerx` fica congelado quando o ponteiro esta sobre
janelas nativas. Por isso a barra de gravacao pergunta ao Hyprland onde o cursor esta e
qual e a area util do monitor que o contem. Sem Hyprland, tudo aqui devolve None e o
chamador cai no Tk.

Coordenadas: o Hyprland responde em coordenadas logicas; com `xwayland:force_zero_scaling`
(padrao do Omarchy) e monitores em escala 1 elas coincidem com as do X.
"""

import json
import os
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

    def monitors(self) -> list:
        now = time.monotonic()
        if now - self._mons_at > _MON_TTL:
            mons = self._query("monitors")
            if isinstance(mons, list) and mons:
                self._mons = mons
            self._mons_at = now
        return self._mons

    def cursorpos(self):
        pos = self._query("cursorpos")
        if isinstance(pos, dict) and "x" in pos:
            return int(pos["x"]), int(pos["y"])
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
