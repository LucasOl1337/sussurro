# Porte Linux do Sussurro

Data: 2026-09-04  
Repo: `C:\Projetos\sussurro`  
Máquina deste turno: Windows. Linux do OMART **não está neste PC**.

## O que foi feito no código

Porte em `app.py` (e notas em `README.md` / `requirements.txt`). CUDA continua obrigatório nos dois sistemas — não há caminho CPU.

| Área | Windows | Linux |
|---|---|---|
| CUDA runtime dos wheels NVIDIA | `os.add_dll_directory` + `PATH` em `nvidia/*/bin` | pré-carga `ctypes.CDLL` dos `.so` em `nvidia/*/lib` e `lib64` |
| Dispositivos | host WASAPI | PulseAudio / ALSA / JACK |
| Audio do PC | loopback WASAPI (`soundcard`) | monitor Pulse/PipeWire (`soundcard` + nome monitor) |
| Clipboard (enviar `colar`) | Win32, todos os formatos HGLOBAL | texto via `wl-copy`/`xclip`/`xsel`; sem binário cai em `digitar` |
| Overlay | `-transparentcolor` + `WS_EX_NOACTIVATE\|TOOLWINDOW` | janela dock, opaca (sem chroma-key) |
| Atalho de mouse | `win32_event_filter` + clique suprimido | `pynput.on_click`; o clique **vaza** para o app debaixo |
| Ícone | `.ico` + AppUserModelID | `iconphoto` PNG |
| Fontes | Segoe / Bahnschrift / Cascadia | Noto / DejaVu / Ubuntu / Liberation se as do Windows não existirem |
| Playback do histórico | `sounddevice` (antes era `winsound`) | o mesmo `sounddevice` |
| Scroll do histórico | MouseWheel | MouseWheel + Button-4/5 |

Import de `ctypes.windll`, `ctypes.wintypes`, `winsound` e `os.add_dll_directory` ficou atrás de `IS_WIN`. Sem isso o `app.py` antigo morria no `import` no Linux.

## O que roda (verificado neste PC)

- `python -m py_compile C:\Projetos\sussurro\app.py` → **OK** neste Windows.
- O arquivo passa a ser importável no Linux do ponto de vista das APIs Win32 (não chama `windll`/`winsound`/`add_dll_directory` fora de `IS_WIN`).

## O que não foi verificado

Nada disto foi exercido de verdade neste turno:

- Abrir a janela Tk no Windows depois das mudanças (gravação, atalho, colar, overlay, histórico, CUDA `large-v3`).
- Qualquer execução em Linux. A máquina OMART não está neste PC — **não houve teste lá**.
- Wayland vs X11, Pulse vs PipeWire, presença de GPU NVIDIA no OMART.
- Se os wheels `nvidia-cublas-cu12==12.9.2.10` / `nvidia-cudnn-cu12==9.24.0.43` instalam e carregam no Linux do OMART.
- Playback novo via `sounddevice` no histórico (substituiu `winsound` também no Windows).
- Descoberta CUDA no Windows pelo `import nvidia.cublas` (o caminho antigo `Lib\site-packages\nvidia\*\bin` ficou como reserva se o import não achar `bin`/`lib`).

## O que deve quebrar ou ficar limitado no Linux

Isto é leitura de código e de APIs, não evidência de corrida no OMART.

1. **Sem GPU NVIDIA / sem CUDA usável** — `WhisperModel(..., device="cuda", compute_type="float16")` continua igual. O modelo não sobe. Não inventei fallback CPU.
2. **Wayland** — pynput (atalho global de mouse e `Ctrl+V` / `digitar`) em geral não funciona. Alvo realista: sessão X11.
3. **Clique do atalho não é suprimido** — no Windows o botão lateral não vira “voltar” no browser; no Linux vira. O atalho dispara, mas o evento também chega ao aplicativo debaixo.
4. **Colar** — precisa de `wl-clipboard` ou `xclip`/`xsel` no PATH. Sem isso `set_clipboard_text` falha e o código cai em `keyboard.type`. Backup do clipboard no Linux é só texto, não imagem/arquivo.
5. **Audio do PC** — depende de fonte *monitor* Pulse/PipeWire. Sem monitor, o combo pode ficar vazio ou o loopback levanta `RuntimeError` explícito. PipeWire “puro” sem Pulse compat não foi mapeado.
6. **Overlay** — cantos da barra opacos; `-topmost` e tipo `dock` dependem do window manager. Multi-monitor: `monitor_work_area` usa a tela Tk inteira, sem recortar painel.
7. **Botões laterais do mouse no X11** — o mapeamento `x1`/`x2` assume valores pynput 8/9 (e 6/7). Se o OMART numerar diferente, `Setar` pode não gravar o botão certo.
8. **Pré-carga CUDA** — `LD_LIBRARY_PATH` depois do `exec` não vale para o linker; a pré-carga `CDLL` tenta compensar. Se o layout dos `.so` no wheel Linux for outro, a transcrição ainda cai com biblioteca cublas/cudnn ausente.
9. **Tk / customtkinter** — precisam de display (`DISPLAY`). Headless SSH sem X some a HUD.

## Commit

SHA do porte em `main`: `b4f5341a7c33b4b42c3d30a4f4aa87da0608a9c6`  
Mensagem: `feat: porte do ditado para Windows e Linux`  
Remoto: https://github.com/LucasOl1337/sussurro.git

## Arquivos tocados

- `C:\Projetos\sussurro\app.py`
- `C:\Projetos\sussurro\README.md`
- `C:\Projetos\sussurro\requirements.txt`
- `C:\Projetos\sussurro\relatorio-porte-linux.md` (este arquivo)
