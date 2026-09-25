# Sussurro

Ditado local para Windows e Linux: fale no microfone e o texto aparece digitado/colado onde o cursor estiver. Tudo roda na sua máquina — nenhum áudio sai do computador.

[Site](https://lucasol1337.github.io/sussurro/) · [Release v0.5.0](https://github.com/LucasOl1337/sussurro/releases/tag/v0.5.0) · [Changelog](CHANGELOG.md)

O caminho do áudio é: **microfone → Silero VAD (segmentação de fala) → faster-whisper **Turbo na GPU ou Base na CPU**, com modelo selecionável**, com uma HUD Tkinter discreta e uma barra de overlay que indica gravação/transcrição.

## Requisitos

- Windows 10/11 **ou** Linux (X11/Wayland + PulseAudio ou PipeWire)
- Python 3.11
- CPU para transcrição local; GPU NVIDIA com CUDA é opcional e acelera o ditado
- Microfone qualquer (a captura tenta 16 kHz e, se o dispositivo não aceitar, reamostra da taxa nativa)
- Linux, para colar via clipboard: `wl-clipboard` e `wtype` (Wayland) ou `xclip`/`xsel` (X11). Em Wayland, configure um atalho no compositor para encaminhar os comandos de gravação. Sem ferramenta de clipboard o envio `colar` cai no modo `digitar`.

## Instalação

Os comandos abaixo instalam o suporte a GPU NVIDIA. **Sem GPU dedicada**, troque `requirements-cuda.txt` por `requirements.txt`: a instalação em CPU dispensa os pacotes CUDA.

Com [uv](https://docs.astral.sh/uv/) (recomendado):

Windows:

```bat
uv venv --python 3.11
uv pip install -r requirements-cuda.txt --override overrides-cuda.txt
```

Linux:

```bash
uv venv --python 3.11
uv pip install -r requirements-cuda.txt --override overrides-cuda.txt
```

O `--override` troca o onnxruntime de CPU (que o faster-whisper puxa) pelo onnxruntime-gpu, usado pelo Parakeet e pela separação de vozes das reuniões. Sem ele os dois pacotes brigam pelo mesmo módulo e o Parakeet não acha a GPU.

Ou com venv + pip:

Windows:

```bat
python -m venv .venv
.venv\Scripts\pip install -r requirements-cuda.txt
```

Linux:

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements-cuda.txt
.venv/bin/pip uninstall -y onnxruntime onnxruntime-gpu
.venv/bin/pip install --no-deps onnxruntime-gpu==1.24.4
```

As bibliotecas de CUDA (`cublas`, `cudnn`) vêm dos wheels da NVIDIA listados no `requirements-cuda.txt` (`nvidia-cublas-cu12` / `nvidia-cudnn-cu12`). No Windows o `app.py` registra `site-packages\nvidia\*\bin` via `os.add_dll_directory()` e prefixa o `PATH`. No Linux pré-carrega os `.so` em `nvidia/*/lib` (e `lib64`) — sem isso a transcrição falha com DLL/`.so` de cublas ausente. Se você instalar as dependências fora de um venv na raiz do projeto, garanta que esses pacotes estejam visíveis no ambiente usado para rodar.

## Fontes no Linux: use o Python do sistema

O Tk que vem com o CPython do `uv` (python-build-standalone) é compilado **sem Xft**: a interface sai com fontes bitmap, sem antialias. Crie o venv com o Python do sistema (`/usr/bin/python3`, com o `tk` da distro) e as fontes ficam certas. O `app.py` funciona do 3.11 ao 3.14.

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

Na primeira execução o modelo escolhido é baixado pelo faster-whisper; as próximas usam o cache local. O botão GRAVAR só é liberado após carregar e aquecer o modelo. Em **Automático**, o Sussurro escolhe Turbo se detectar CUDA e Base se usar CPU. O seletor mostra o nome real da GPU NVIDIA e do processador encontrados na máquina.

## Escolha do modelo

No card de configuração, escolha **MODELO** e **EXECUTAR EM** e clique em **Aplicar modelo**. A troca acontece sem reiniciar e só é aceita fora de um ditado ou transcrição em andamento. O cabeçalho e `python app.py status` mostram o modelo e dispositivo ativos. A preferência só é salva depois que o modelo carrega; se a troca falhar, o app tenta recuperar o anterior pelo cache.

| Seu PC / prioridade | Ponto de partida recomendado | Execução |
| --- | --- | --- |
| PC fraco, CPU antigo | **Base**; Tiny se Base ainda ficar lento | CPU, INT8 |
| Sem GPU dedicada, CPU recente | **Small**; Base para menor espera | CPU, INT8 |
| Intermediário, como RTX 3060 | **Turbo** | CUDA, INT8/FP16 |
| Forte, como RTX 4070 Ti Super | **Turbo** | CUDA, INT8/FP16 |
| Fortíssimo, como RTX 4090 | **Turbo** para ditado; Large-v3 para priorizar precisão | CUDA, INT8/FP16 |

O seletor oferece **Tiny, Base, Small, Medium, Turbo e Large-v3**, todos multilíngues. Medium é uma opção intermediária para comparação; Turbo é a recomendação geral em GPU. Uma placa mais forte não torna necessário escolher um modelo maior. A qualidade depende do idioma, ruído, sotaque e vocabulário: compare com suas gravações. Os perfis acima são recomendações, não benchmarks dessas placas, e não garantem ditado em tempo real em CPU.

Se houver uma GPU AMD junto da NVIDIA, o Sussurro a identifica e explica seu estado abaixo do seletor, mas não oferece um botão AMD que falharia. O pacote Python normal do CTranslate2 usa CUDA; AMD requer uma compilação ROCm/HIP separada. Os dois backends não se somam para acelerar um único ditado, e manter a AMD livre evita disputar memória compartilhada quando ela é integrada ao processador.

O Turbo reduz o decodificador do large-v3 de 32 para 4 camadas, com uma pequena perda de qualidade reportada pelos autores. INT8 reduz a precisão numérica para economizar memória. Fontes: [modelo oficial Turbo](https://huggingface.co/openai/whisper-large-v3-turbo) e [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Não incluímos modelos `.en` ou `distil-large-v3`, que são voltados ao inglês, na seleção para ditado em português.

### Parakeet (NVIDIA)

O **Parakeet TDT 0.6B v3** é o modelo de fala da NVIDIA que o voxtype do Omarchy também oferece. No Sussurro ele roda só na GPU, em FP16: no primeiro uso baixa 2,5 GB e converte uma vez para 1,3 GB, e ocupa uns 2,7 GB de VRAM (Turbo: 1,3 GB; Large-v3: 2,1 GB). É o mais rápido de todos: 25 s de fala em 0,1 s e 90 s em 0,4 s numa RTX 4070 Ti Super, contra 0,3 s e 0,7 s do Turbo.

O custo é o idioma. O Parakeet não aceita idioma fixo: ele decide sozinho a cada trecho, e em português com termos técnicos em inglês às vezes escorrega pro inglês. Em 25 ditados reais em português, 7 saíram com pedaço em inglês. Também não usa as `hotwords` da Biblioteca (as trocas depois do texto continuam valendo). Para ditado em português, Turbo ou Large-v3 continuam sendo a escolha; o Parakeet está no seletor e no COMPARAR para você medir na sua voz.

## Comparar a mesma frase entre modelos

Abra a aba **COMPARAR**, selecione os modelos e clique em **Gravar frase**. Fale uma vez e clique em **Parar e comparar**. O áudio do microfone selecionado é limitado a 30 segundos e reutilizado em todos os modelos. Você também pode carregar um WAV PCM de 16 bits, mono ou estéreo, de até 30 segundos.

- **Paralelo (2 por vez):** até dois modelos executam ao mesmo tempo. Eles dividem os recursos; os tempos refletem essa disputa.
- **Individual (1 por vez):** um modelo termina antes de começar o seguinte, facilitando a comparação de velocidade.
- **Por modelo:** texto reconhecido, dispositivo/precisão, tempo de transcrição após aquecimento, carga/aquecimento, preparo/download e espera total desde o início da comparação (incluindo fila).
- **Comparar novamente:** reutiliza a mesma frase com outra seleção de modelos, dispositivo ou modo, sem gravar de novo.
- **Cancelar:** encerra os processos da comparação e libera seus modelos. A falha de um modelo aparece na própria linha e não impede os demais de terminar.

Todos recebem exatamente o mesmo WAV, idioma e `beam_size=5`, sem substituições da Biblioteca, para expor as diferenças de reconhecimento. O primeiro uso baixa os pesos. A comparação usa processos separados e mantém o modelo normal de ditado intacto; enquanto grava/compara, o ditado, a troca de modelo e a transcrição de arquivos do app ficam bloqueados para evitar interferência. Outros aplicativos ainda podem disputar CPU/GPU. Se faltar VRAM, tente Individual, menos modelos ou CPU.

O áudio de comparação fica apenas na memória e em um WAV temporário sob `~/.cache/sussurro/compare`, removido ao terminar/cancelar; ele não depende da cota do `/tmp` compartilhado. Os testes não entram no histórico de ditados. Resultados ficam na aba enquanto o app estiver aberto. O comparador não calcula uma nota de precisão: leia as transcrições e compare com a frase que você falou.

### Atualizar de uma versão antiga

Feche o Sussurro, atualize o código (`git pull --ff-only`, em um checkout sem alterações pendentes), atualize as dependências e abra novamente. Vindo de antes do Parakeet e das reuniões, na instalação CUDA troque o onnxruntime de CPU pelo de GPU: `uv pip uninstall onnxruntime` e depois `uv pip install -r requirements-cuda.txt --override overrides-cuda.txt`. Se baixou um ZIP, extraia a nova versão e preserve `settings.json`, `library.json` e `history/` da instalação anterior.

Instalações anteriores à v0.4.0 não tinham escolha de modelo: ao atualizar, recebem **Automático → Turbo na GPU / Base na CPU**. Histórico, biblioteca e demais preferências são preservados. Escolhas de modelo feitas a partir desta versão persistem nas próximas atualizações. Não há atualizador automático: é necessário instalar a nova versão.

## Como usar

- **Atalho global de mouse** — o botão configurado (padrão: lateral 2 / "frente") liga e desliga a gravação em qualquer aplicativo. No Windows o clique é suprimido, então não vira "voltar/avançar" no browser. No Linux o clique também chega ao aplicativo debaixo (pynput não suprime o evento). Em `Setar` você clica o botão desejado (meio, lateral 1 ou lateral 2) para redefinir.
- **Ação** — `alternar` (clique liga/desliga) ou `segurar` (push-to-talk).
- **Microfone** — seletor com nomes legíveis e uma explicação da entrada selecionada. `… — padrão do sistema (PipeWire)` acompanha automaticamente o microfone escolhido no sistema; `USB direto` abre aquele aparelho sem acompanhar trocas; PulseAudio, ALSA e JACK aparecem identificados como rotas de compatibilidade ou de áudio profissional. O identificador técnico continua salvo internamente para não quebrar preferências antigas.
- **Fonte** — `microfone` (entrada), `audio do PC` (o que está saindo nas caixas/fones: loopback WASAPI no Windows, monitor Pulse/PipeWire no Linux) ou `os dois` (mistura mic + PC antes do VAD/whisper).
- **Canal do PC** — qual saída/monitor capturar no modo `audio do PC` / `os dois`. `padrao do sistema` usa o dispositivo de reprodução atual. O combo fica desabilitado quando a fonte é só microfone.
- **Transcrição** — `simultaneo`: trechos vão aparecendo conforme você pausa entre frases (corte por VAD após ~0,7 s de silêncio); `final`: acumula tudo e transcreve de uma vez ao parar.
- **Formatação** — depois do whisper, o texto ganha ponto em cláusula nova e quebra de parágrafo em pausa longa (~1,5 s, não na respiração de 0,7 s) e uma linha nova antes de âncoras faladas (`Pergunta 7`, `Questão 12`, `Primeiro`/`Segundo`/`Terceiro`). Não reescreve nem tira “né/sabe”. Vale no colar, no histórico e no copiar. Ditados antigos ficam como estão.
- **Envio** — `colar`: no Linux, foca a janela (e o campo web) sob o mouse e cola via Ctrl+V. No Windows cola via Ctrl+V no campo onde o cursor estiver. No Windows o clipboard original é preservado em todos os formatos; no Linux o backup é só texto (`wl-copy`/`xclip`/`xsel`). `digitar`: simula teclado.
- **Idioma** — `pt`, `en` ou `auto`.
- **Bolinha** — overlay sempre no topo, sem roubar foco e fora do Alt-Tab; laranja = gravando, invertida = transcrevendo. Arraste para reposicionar (a posição é salva como fração da área útil do monitor).
- **Histórico / Ao vivo** — aba com as sessões passadas (tocar o WAV ou copiar o texto) e aba com o texto da sessão atual ("Copiar tudo" leva tudo pra área de transferência). Se o Whisper falhar, o WAV continua no histórico com um aviso; clique na entrada ou no botão ↻ para tentar transcrevê-lo novamente.
- **Biblioteca** — aba onde você cadastra as palavras que o whisper escreve errado. Em "sai assim" liste as variantes separadas por vírgula (`grock, groque, nine houter`), em "deve virar" o termo certo (`Grok`), e clique em Adicionar. A troca é aplicada em toda transcrição antes de ela aparecer na tela, ser colada/digitada e ir para o histórico — sem diferença de maiúscula, tolerando espaçamento diferente em termos de duas palavras, e só em palavra inteira (`grok` não mexe em `grokking`). O ✕ remove a entrada; tudo vale na hora, sem reiniciar.
- **Estatísticas** — aba com os números do seu ditado, todos derivados do `history/history.jsonl` (não há contador paralelo): total de palavras, palavras por minuto (palavras ÷ tempo de fala, com o melhor ditado à parte), sequência de dias seguidos e recorde, tempo falado e média por ditado, quanto tempo o mesmo texto levaria digitado a 40 ppm, correções aplicadas pela Biblioteca, mapa de atividade das últimas 26 semanas, distribuição por hora do dia e as palavras que você mais fala (boas candidatas à Biblioteca). Ditados antigos entram na conta — a duração é medida pelo tamanho do WAV; só a contagem de correções começa nesta versão. A janela se alarga sozinha ao abrir a aba (limitada à área útil do monitor) e volta ao tamanho anterior ao sair: nada de número escondido atrás de barra de rolagem.

## Reuniões (Linux)

A aba **REUNIÃO** grava o seu microfone e o que o PC toca em duas trilhas separadas, e ao parar transcreve com quem falou o quê. Funciona com qualquer app de chamada, porque só escuta o áudio do sistema: nenhum bot entra na call e nada sai da máquina. Portado do [Meeting Recorder](https://github.com/jankeesvw/omarchy-meeting-recorder) de Jankees van Woezik (MIT), trocando o whisper.cpp na CPU pelo modelo do ditado na GPU.

- **Antes de começar**, os dois medidores já mostram o microfone e o áudio do PC. Nome e idioma podem mudar durante a call; **Pausar** congela as duas trilhas.
- **Cada lado é transcrito separado**, então duas pessoas falando juntas ou música por baixo não apagam a voz mais baixa, e o lado de cada frase é a própria trilha (Você / Remoto). Sem fone, a voz dos outros vaza no seu microfone; o eco é reconhecido e fica de fora.
- **Várias vozes no mesmo lado** (um colega do seu lado, três pessoas do outro) são separadas pelo [Nemotron 3 Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization) da NVIDIA na GPU: Remoto 1, Remoto 2 e assim por diante. Cada falante ganha um campo de nome e uma cor.
- **Importar áudio** transcreve um arquivo qualquer que o ffmpeg lê (memo do celular, call gravada em outro lugar), separando as vozes; o número de falantes pode ser fixado.
- **Na tela da reunião**: clique numa linha para ouvir dali, renomeie falantes, copie a transcrição em Markdown, abra a pasta ou transcreva de novo em outro idioma. A Biblioteca vale aqui também.
- **Se o app cair gravando**, as trilhas já estão no cache e a aba oferece salvar ou descartar na próxima abertura.
- **Atalhos**: `python app.py meeting-start`, `meeting-stop` e `meeting-pause` controlam a gravação sem abrir a janela.

Cada reunião vira uma pasta em `~/Documentos/Reuniões` (a pasta de documentos do `xdg-user-dir`), com o prefixo de data para ordenar: `audio.ogg` (os dois lados nivelados), `transcricao.md`, `reuniao.json` (falas, nomes e modelo) e `.trilhas/` com cada lado separado, que é o que "Refazer" usa. Requer `parec`/`pacat` (libpulse) e `ffmpeg` com Opus. O modelo de vozes (~200 MB) baixa no primeiro uso.

No bench do Meeting Recorder (os casos de teste dele com gabarito, rodado com `sussurro_meeting.py`), com Turbo na GPU: 97,2% das palavras com a pessoa certa na call (o app original: 94,1%), 93,5% no arquivo importado (90,7%) e 99,4% numa reunião real do corpus AMI (97,6%), em 2 a 6 s por caso contra 20 a 160 s do original na CPU.

## Desempenho e diagnóstico no Linux

O comando `sussurro toggle` usa um cliente leve de socket, sem importar Tk, áudio ou CUDA. A interface consulta os comandos a cada 20 ms. No Linux, o microfone abre primeiro na taxa nativa do dispositivo, com blocos de 20 ms; os blocos que ainda estão no mixer são enviados antes de encerrar o ditado.

A cópia via `wl-copy`/`xclip` não captura os pipes dos processos que ficam servindo o clipboard. Capturá-los provocava um timeout de 2 segundos e acionava a digitação de reserva. A restauração do clipboard aguarda 400 ms em segundo plano, respeita cópias feitas pelo usuário durante essa espera e mantém a ordem entre colagens consecutivas.

O padrão em GPU NVIDIA é `large-v3-turbo`, CUDA, `int8_float16`; em CPU é `base`, `int8`. Se o hardware não suportar essa precisão, é usada uma alternativa suportada. A decodificação usa `beam_size=5`. O aquecimento consome o gerador de transcrição e inicializa o VAD antes de liberar a gravação. No modo `final`, os blocos são concatenados uma única vez e o VAD é executado pelo faster-whisper, evitando uma segunda varredura do mesmo áudio.

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

- A instalação padrão acelera na GPU NVIDIA/CUDA. CTranslate2 também pode ser compilado separadamente com ROCm/HIP, mas essa build não convive com a build CUDA no processo atual; por isso uma AMD detectada é informada, não oferecida como opção falsa. GPUs AMD/Intel usam CPU nesta instalação.
- Wayland: pynput não fornece o atalho global; use o cliente de socket com um atalho do compositor e instale `wtype` para enviar teclas.
- Overlay da barra: sem chroma-key (`-transparentcolor` é Windows); cantos da janela ficam opacos.
- Área útil do monitor: a tela Tk inteira, sem recorte por painel/multi-monitor.
- Clique do atalho não é comido pelo sistema.

## Licença

[MIT](LICENSE)
