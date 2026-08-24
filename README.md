# Sussurro

Ditado local para Windows: fale no microfone e o texto aparece digitado/colado onde o cursor estiver. Tudo roda na sua máquina — nenhum áudio sai do computador.

O caminho do áudio é: **microfone → Silero VAD (segmentação de fala) → faster-whisper `large-v3` em CUDA (float16)**, com uma HUD Tkinter discreta e uma bolinha de overlay que indica gravação/transcrição.

## Requisitos

- Windows 10/11
- Python 3.11
- GPU NVIDIA recomendada (o modelo carrega com `device="cuda", compute_type="float16"`; sem GPU o modelo não sobe neste código)
- Microfone qualquer (a captura tenta 16 kHz e, se o dispositivo WASAPI não aceitar, reamostra da taxa nativa)

## Instalação

Com [uv](https://docs.astral.sh/uv/) (recomendado):

```bat
uv venv --python 3.11
uv pip install -r requirements.txt
```

Ou com venv + pip:

```bat
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

As DLLs de CUDA (`cublas`, `cudnn`) vêm dos wheels da NVIDIA listados no `requirements.txt` (`nvidia-cublas-cu12` / `nvidia-cudnn-cu12`). O `app.py` registra os diretórios `site-packages\nvidia\*\bin` via `os.add_dll_directory()` e prefixa o `PATH` — sem isso a transcrição falha com `Library cublas64_12.dll is not found`. Se você instalar as dependências fora de um venv na raiz do projeto, garanta que esses pacotes estejam visíveis no ambiente usado para rodar.

## Como rodar

```bat
.venv\Scripts\activate
python app.py
```

Sem terminal (janela própria, sem console):

```bat
.venv\Scripts\pythonw.exe app.py
```

Na primeira execução o modelo `large-v3` é baixado pelo faster-whisper e depois carregado na GPU (há um aquecimento de uma inferência vazia antes de liberar o botão GRAVAR).

## Como usar

- **Atalho global de mouse** — o botão configurado (padrão: lateral 2 / "frente") liga e desliga a gravação em qualquer aplicativo; o clique é suprimido, então não vira "voltar/avançar" no browser. Em `Setar` você clica o botão desejado (meio, lateral 1 ou lateral 2) para redefinir.
- **Ação** — `alternar` (clique liga/desliga) ou `segurar` (push-to-talk).
- **Microfone** — seletor com as entradas WASAPI disponíveis.
- **Fonte** — `microfone` (entrada), `audio do PC` (o que está saindo nas caixas/fones, via loopback WASAPI) ou `os dois` (mistura mic + PC antes do VAD/whisper).
- **Canal do PC** — qual saída capturar no modo `audio do PC` / `os dois`. `padrao do sistema` usa o dispositivo de reprodução atual do Windows. O combo fica desabilitado quando a fonte é só microfone.
- **Transcrição** — `simultaneo`: trechos vão aparecendo conforme você pausa entre frases (corte por VAD após ~0,7 s de silêncio); `final`: acumula tudo e transcreve de uma vez ao parar.
- **Envio** — `colar`: cola via Ctrl+V no campo onde o cursor estiver (o clipboard original é preservado — backup/restauração de todos os formatos); `digitar`: simula teclado.
- **Idioma** — `pt`, `en` ou `auto`.
- **Bolinha** — overlay sempre no topo, sem roubar foco e fora do Alt-Tab; laranja = gravando, invertida = transcrevendo. Arraste para reposicionar (a posição é salva como fração da área útil do monitor).
- **Histórico / Ao vivo** — aba com as sessões passadas (tocar o WAV ou copiar o texto) e aba com o texto da sessão atual ("Copiar tudo" leva tudo pra área de transferência).
- **Biblioteca** — aba onde você cadastra as palavras que o whisper escreve errado. Em "sai assim" liste as variantes separadas por vírgula (`grock, groque, nine houter`), em "deve virar" o termo certo (`Grok`), e clique em Adicionar. A troca é aplicada em toda transcrição antes de ela aparecer na tela, ser colada/digitada e ir para o histórico — sem diferença de maiúscula, tolerando espaçamento diferente em termos de duas palavras, e só em palavra inteira (`grok` não mexe em `grokking`). O ✕ remove a entrada; tudo vale na hora, sem reiniciar.
- **Estatísticas** — aba com os números do seu ditado, todos derivados do `history/history.jsonl` (não há contador paralelo): total de palavras, palavras por minuto (palavras ÷ tempo de fala, com o melhor ditado à parte), sequência de dias seguidos e recorde, tempo falado e média por ditado, quanto tempo o mesmo texto levaria digitado a 40 ppm, correções aplicadas pela Biblioteca, mapa de atividade das últimas 26 semanas, distribuição por hora do dia e as palavras que você mais fala (boas candidatas à Biblioteca). Ditados antigos entram na conta — a duração é medida pelo tamanho do WAV; só a contagem de correções começa nesta versão. A janela se alarga sozinha ao abrir a aba (limitada à área útil do monitor) e volta ao tamanho anterior ao sair: nada de número escondido atrás de barra de rolagem.

## O que fica só na sua máquina

- `history/` — gravações WAV e transcrições das suas sessões (indexadas em `history.jsonl`). É conteúdo seu e privado; não versionamos.
- `settings.json` — preferências locais (atalho, microfone, idioma etc.). Também fica de fora do git.
- `library.json` — sua biblioteca de correções de palavras. Também fica de fora do git.
- `sussurro.log` — log escrito quando rodando sob `pythonw`.

Todos estão no `.gitignore`. Nada é enviado para serviço externo: captura, VAD, modelo e injeção de texto são 100% locais.

## Licença

[MIT](LICENSE)
