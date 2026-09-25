"""Cliente do atalho: somente biblioteca padrao, sem carregar Tk, audio ou CUDA."""

import os
import socket
import sys
from pathlib import Path

IPC_SOCK = Path(os.environ.get("XDG_RUNTIME_DIR") or "/tmp") / "sussurro.sock"

# transcricao de arquivo roda o whisper na hora: espera bem mais que um toggle
TRANSCRIBE_TIMEOUT = 900.0

COMANDOS = ("toggle", "start", "stop", "status",
            "toggle-enter", "start-enter", "stop-enter",
            "meeting-start", "meeting-stop", "meeting-pause")


def ipc_send(cmd: str, timeout: float = 1.5) -> str:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect(str(IPC_SOCK))
        sock.sendall((cmd.strip() + "\n").encode("utf-8"))
        partes = []
        while True:  # a resposta do transcribe passa de um recv
            chunk = sock.recv(65536)
            if not chunk:
                break
            partes.append(chunk)
            if partes[-1].endswith(b"\n"):
                break
        return b"".join(partes).decode("utf-8", "replace")


def cli(argv: list[str]) -> bool:
    if len(argv) < 2:
        return False
    cmd = argv[1].lower()
    usage = ("uso: sussurro [toggle|start|stop|status|toggle-enter|start-enter|stop-enter]\n"
             "     sussurro [meeting-start|meeting-stop|meeting-pause]\n"
             "     sussurro transcribe <arquivo de audio>\n"
             "  *-enter: ao terminar, aperta Enter (uso pelo fone)\n"
             "  meeting-*: grava/para/pausa uma reuniao (aba REUNIAO)\n"
             "  transcribe: transcreve o arquivo e grava no historico (JSON na saida)")
    if cmd in ("-h", "--help"):
        print(usage)
        raise SystemExit(0)
    if cmd == "transcribe":
        if len(argv) < 3:
            print(usage, file=sys.stderr)
            raise SystemExit(2)
        # caminho absoluto: quem responde e o processo do Sussurro, com outro cwd
        pedido = f"transcribe {Path(argv[2]).expanduser().resolve()}"
        timeout = TRANSCRIBE_TIMEOUT
    elif cmd in COMANDOS:
        pedido, timeout = cmd, 1.5
    else:
        print(usage, file=sys.stderr)
        raise SystemExit(2)
    try:
        resposta = ipc_send(pedido, timeout)
    except (OSError, socket.timeout):
        print("sussurro nao esta rodando", file=sys.stderr)
        raise SystemExit(1)
    sys.stdout.write(resposta)
    # transcribe sinaliza falha no proprio JSON: o chamador (Hermes) le o codigo de saida
    if cmd == "transcribe" and '"ok": true' not in resposta.lower():
        raise SystemExit(1)
    return True
