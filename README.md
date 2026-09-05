# Sussurro

Ditado local para Windows e Linux: fale no microfone e o texto aparece digitado/colado onde o cursor estiver. Tudo roda na sua máquina — nenhum áudio sai do computador.

[Release v0.2.0](https://github.com/LucasOl1337/sussurro/releases/tag/v0.2.0) · [Changelog](CHANGELOG.md)

O caminho do áudio é: **microfone → Silero VAD (segmentação de fala) → faster-whisper `large-v3` em CUDA (float16)**, com uma HUD Tkinter discreta e uma barra de overlay que indica gravação/transcrição.

## Requisitos

- Windows 10/11 **ou** Linux (X11 + PulseAudio ou PipeWire; GPU NVIDIA)
- Python 3.11
- GPU NVIDIA (o modelo carrega com `device="cuda", compute_type="float16"`; sem GPU o modelo não sobe neste código)
- Microfone qualquer (a captura tenta 16 kHz e, se o dispositivo não aceitar, reamostra da taxa nativa)
- Linux, para colar via clipboard: `wl-clipboard` e `wtype` (Wayland) ou `xclip`/`xsel` (X11). Em Wayland, configure um atalho no compositor para encaminhar os comandos de gravação. Sem ferramenta de clipboard o envio `colar` cai no modo `digitar`.

## Instalação

Com [uv](https://docs.astral.sh/uv/) (recomendado):

Windows:

```bat
uv venv --python 3.11
uv pip install -r requirements.txt
```

Linux:

```bash
uv venv --python 3.11
uv pip install -r requirements.txt
```

Ou com venv + pip:

Windows:

```bat
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

Linux:

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

As bibliotecas de CUDA (`cublas`, `cudnn`) vêm dos wheels da NVIDIA listados no `requirements.txt` (`nvidia-cublas-cu12` / `nvidia-cudnn-cu12`). No Windows o `app.py` registra `site-packages\nvidia\*\bin` via `os.add_dll_directory()` e prefixa o `PATH`. No Linux pré-carrega os `.so` em `nvidia/*/lib` (e `lib64`) — sem isso a transcrição falha com DLL/`.so` de cublas ausente. Se você instalar as dependências fora de um venv na raiz do projeto, garanta que esses pacotes estejam visíveis no ambiente usado para rodar.

## Omarchy / Hyprland

No Omarchy a barra de gravação segue o monitor do cursor e ganha cantos recortados pelo compositor, e a aba **OMARCHY** aciona o ditado pelo headset MCHOSE X9 sem tocar no PC (roda de volume invertida rápido ou toque duplo no mute, com Enter automático ao terminar). Regras de janela, regra udev e instruções em [`contrib/omarchy/`](contrib/omarchy/README.md).

## Como rodar

Windows:

```bat
.venv\Scripts\activate
python app.py
```

Sem terminal (janela própria, sem console):

```bat
.venv\Scripts\pythonw.exe app.py
```

Linux:

```bash
source .venv/bin/activate
python app.py
```

Na primeira execução o modelo `large-v3` é baixado pelo faster-whisper e depois carregado na GPU (há um aquecimento de uma inferência vazia antes de liberar o botão GRAVAR).

## Como usar

- **Atalho global de mouse** — o botão configurado (padrão: lateral 2 / "frente") liga e desliga a gravação em qualquer aplicativo. No Windows o clique é suprimido, então não vira "voltar/avançar" no browser. No Linux o clique também chega ao aplicativo debaixo (pynput não suprime o evento). Em `Setar` você clica o botão desejado (meio, lateral 1 ou lateral 2) para redefinir.
- **Ação** — `alternar` (clique liga/desliga) ou `segurar` (push-to-talk).
- **Microfone** — seletor com as entradas do host nativo (WASAPI no Windows; Pulse/ALSA no Linux).
- **Fonte** — `microfone` (entrada), `audio do PC` (o que está saindo nas caixas/fones: loopback WASAPI no Windows, monitor Pulse/PipeWire no Linux) ou `os dois` (mistura mic + PC antes do VAD/whisper).
- **Canal do PC** — qual saída/monitor capturar no modo `audio do PC` / `os dois`. `padrao do sistema` usa o dispositivo de reprodução atual. O combo fica desabilitado quando a fonte é só microfone.
- **Transcrição** — `simultaneo`: trechos vão aparecendo conforme você pausa entre frases (corte por VAD após ~0,7 s de silêncio); `final`: acumula tudo e transcreve de uma vez ao parar.
- **Formatação** — depois do whisper, o texto ganha ponto em cláusula nova e quebra de parágrafo em pausa longa (~1,5 s, não na respiração de 0,7 s) e uma linha nova antes de âncoras faladas (`Pergunta 7`, `Questão 12`, `Primeiro`/`Segundo`/`Terceiro`). Não reescreve nem tira “né/sabe”. Vale no colar, no histórico e no copiar. Ditados antigos ficam como estão.
- **Envio** — `colar`: cola via Ctrl+V no campo onde o cursor estiver. No Windows o clipboard original é preservado em todos os formatos; no Linux o backup é só texto (`wl-copy`/`xclip`/`xsel`). `digitar`: simula teclado.
- **Idioma** — `pt`, `en` ou `auto`.
- **Bolinha** — overlay sempre no topo, sem roubar foco e fora do Alt-Tab; laranja = gravando, invertida = transcrevendo. Arraste para reposicionar (a posição é salva como fração da área útil do monitor).
- **Histórico / Ao vivo** — aba com as sessões passadas (tocar o WAV ou copiar o texto) e aba com o texto da sessão atual ("Copiar tudo" leva tudo pra área de transferência).
- **Biblioteca** — aba onde você cadastra as palavras que o whisper escreve errado. Em "sai assim" liste as variantes separadas por vírgula (`grock, groque, nine houter`), em "deve virar" o termo certo (`Grok`), e clique em Adicionar. A troca é aplicada em toda transcrição antes de ela aparecer na tela, ser colada/digitada e ir para o histórico — sem diferença de maiúscula, tolerando espaçamento diferente em termos de duas palavras, e só em palavra inteira (`grok` não mexe em `grokking`). O ✕ remove a entrada; tudo vale na hora, sem reiniciar.
- **Estatísticas** — aba com os números do seu ditado, todos derivados do `history/history.jsonl` (não há contador paralelo): total de palavras, palavras por minuto (palavras ÷ tempo de fala, com o melhor ditado à parte), sequência de dias seguidos e recorde, tempo falado e média por ditado, quanto tempo o mesmo texto levaria digitado a 40 ppm, correções aplicadas pela Biblioteca, mapa de atividade das últimas 26 semanas, distribuição por hora do dia e as palavras que você mais fala (boas candidatas à Biblioteca). Ditados antigos entram na conta — a duração é medida pelo tamanho do WAV; só a contagem de correções começa nesta versão. A janela se alarga sozinha ao abrir a aba (limitada à área útil do monitor) e volta ao tamanho anterior ao sair: nada de número escondido atrás de barra de rolagem.

## Desempenho e diagnóstico no Linux

O comando `sussurro toggle` usa um cliente leve de socket, sem importar Tk, áudio ou CUDA. A interface consulta os comandos a cada 20 ms. No Linux, o microfone abre primeiro na taxa nativa do dispositivo, com blocos de 20 ms; os blocos que ainda estão no mixer são enviados antes de encerrar o ditado.

A cópia via `wl-copy`/`xclip` não captura os pipes dos processos que ficam servindo o clipboard. Capturá-los provocava um timeout de 2 segundos e acionava a digitação de reserva. A restauração do clipboard aguarda 400 ms em segundo plano, respeita cópias feitas pelo usuário durante essa espera e mantém a ordem entre colagens consecutivas.

O modelo continua sendo `large-v3`, CUDA, `float16`, `beam_size=5`. O aquecimento consome o gerador de transcrição e inicializa o VAD antes de liberar a gravação. No modo `final`, os blocos são concatenados uma única vez e o VAD é executado pelo faster-whisper, evitando uma segunda varredura do mesmo áudio.

- `sussurro status`: informa se o modelo está pronto, se está gravando e se ainda há trabalho pendente.
- `sussurro-performance.log`: registra duração do áudio, tempo de transcrição, tempo de entrega e tempo desde o comando de parada. Não contém áudio nem texto ditado. Rotação de 1 MB, com duas cópias anteriores.
- Testes de regressão: `.venv/bin/python -m unittest discover -s tests -v`.

Os exemplos com `sussurro` pressupõem um launcher local com esse nome; ele não é instalado automaticamente. Com o ambiente virtual ativado, também é possível chamar os comandos diretamente no diretório do projeto:

```bash
python app.py toggle
python app.py status
```

Em Wayland, a injeção usa `wtype`; configure o atalho do compositor para executar o Python do ambiente virtual e o `app.py` usando caminhos absolutos, com o argumento `toggle` (ou `start`/`stop`).

## O que fica só na sua máquina

- `history/` — gravações WAV e transcrições das suas sessões (indexadas em `history.jsonl`). É conteúdo seu e privado; não versionamos.
- `settings.json` — preferências locais (atalho, microfone, idioma etc.). Também fica de fora do git.
- `library.json` — sua biblioteca de correções de palavras. Também fica de fora do git.
- `sussurro.log` — log escrito quando rodando sob `pythonw`.

Todos estão no `.gitignore`. Nada é enviado para serviço externo: captura, VAD, modelo e injeção de texto são 100% locais.

## Linux — o que ainda é limitado

- Sem GPU NVIDIA o modelo não sobe (igual ao Windows).
- Wayland: pynput não fornece o atalho global; use o cliente de socket com um atalho do compositor e instale `wtype` para enviar teclas.
- Overlay da barra: sem chroma-key (`-transparentcolor` é Windows); cantos da janela ficam opacos.
- Área útil do monitor: a tela Tk inteira, sem recorte por painel/multi-monitor.
- Clique do atalho não é comido pelo sistema.

## Licença

[MIT](LICENSE)
