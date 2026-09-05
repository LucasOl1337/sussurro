"""Cliente do atalho: somente biblioteca padrao, sem carregar Tk, audio ou CUDA."""

import os
import socket
import sys
from pathlib import Path

IPC_SOCK = Path(os.environ.get("XDG_RUNTIME_DIR") or "/tmp") / "sussurro.sock"


def ipc_send(cmd: str, timeout: float = 1.5) -> str:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect(str(IPC_SOCK))
        sock.sendall((cmd.strip() + "\n").encode("utf-8"))
        return sock.recv(4096).decode("utf-8", "replace")


def cli(argv: list[str]) -> bool:
    if len(argv) < 2:
        return False
    cmd = argv[1].lower()
    usage = "uso: sussurro [toggle|start|stop|status]"
    if cmd in ("-h", "--help"):
        print(usage)
        raise SystemExit(0)
    if cmd not in ("toggle", "start", "stop", "status"):
        print(usage, file=sys.stderr)
        raise SystemExit(2)
    try:
        sys.stdout.write(ipc_send(cmd))
    except (OSError, socket.timeout):
        print("sussurro nao esta rodando", file=sys.stderr)
        raise SystemExit(1)
    return True
