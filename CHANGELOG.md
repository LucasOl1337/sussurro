# Changelog

As mudanças de cada versão do Sussurro são registradas aqui. As versões seguem o formato `MAJOR.MINOR.PATCH`.

## [Não lançado]

### Adicionado

- Aba **OMARCHY** (Linux): gestos do headset MCHOSE X9 acionam o ditado sem tocar no PC — roda de volume invertida rápido ou toque duplo no mute (detectado pelo silêncio digital do mic). Liga/desliga, Enter automático, ajuste dos tempos, mic reserva, instalação da regra udev e feed de eventos, tudo na aba. Módulo `sussurro_devices.py`, pronto para outros perfis de fone.
- Comandos `toggle-enter`, `start-enter` e `stop-enter` por socket local: a sessão termina apertando Enter depois da última colagem, confirmando o envio da frase.
- `contrib/omarchy/`: regras de janela do Hyprland (`sussurro.lua`), regra udev do fone e passo a passo.

### Corrigido

- Barra de gravação no Hyprland: aparece no monitor onde o cursor está (antes ficava sempre no monitor do meio, porque o Tk em XWayland vê uma tela única e o ponteiro dele congela fora de janelas X). Agora pergunta ao socket do Hyprland. Módulo `sussurro_hypr.py`.
- Barra de gravação sem o retângulo escuro atrás da cápsula: no Hyprland ela vira uma janela gerenciada (classe `SussurroBar`) e a regra `rounding` recorta os cantos.

## [0.2.0] — 2026-09-05

Suporte ao Linux e correções de latência na captura, no atalho e na colagem. O modelo permanece `large-v3`, CUDA, `float16`, com `beam_size=5`.

### Adicionado

- Porte para Linux: carregamento das bibliotecas CUDA, captura por ALSA/PulseAudio/PipeWire, monitor de áudio do PC, clipboard e adaptações da interface.
- Comandos `toggle`, `start`, `stop` e `status` por socket local, permitindo integrar o ditado aos atalhos do compositor Wayland. Disponíveis com `python app.py <comando>`.
- Injeção de texto em Wayland pelo `wtype`.
- Log local de desempenho com tempo de transcrição, entrega e parada até entrega, sem texto ditado ou áudio e com rotação de arquivos.
- Nove testes de regressão do caminho de ditado e clipboard.

### Melhorado

- Cliente do atalho usa somente a biblioteca padrão, sem carregar a interface ou as bibliotecas de IA a cada comando.
- Captura de microfone no Linux usa blocos de 20 ms e tenta primeiro a taxa nativa do dispositivo, com reamostragem para 16 kHz.
- Interface consulta os eventos a cada 20 ms.
- No modo final, os blocos de áudio são concatenados uma única vez e o filtro de voz é executado pelo faster-whisper, evitando trabalho duplicado.
- Histórico redesenhado com lista própria, reprodução do áudio e cópia do texto.
- Formatação da transcrição com pontuação entre cláusulas, parágrafos em pausas longas e quebras antes de âncoras de listas faladas.

### Corrigido

- Timeout de 2 segundos na colagem Linux: os processos de clipboard que permanecem em segundo plano não mantêm mais a chamada esperando pelo fechamento de stdout/stderr.
- Restauração do clipboard ocorre em segundo plano, respeita conteúdo copiado posteriormente pelo usuário e preserva o intervalo de leitura entre colagens consecutivas.
- Aquecimento do Whisper consome o gerador de transcrição e inicializa o VAD antes de liberar a gravação.
- Últimos blocos do mixer são enviados antes do marcador de parada.
- Nova sessão não sobrescreve uma sessão ainda em entrega ou arquivamento; cancelamentos não baixam pendências de outra sessão.
- Atualizações de widgets passam pela thread principal do Tk, evitando acessos de outra thread no Linux.
- Wayland não inicia o listener global de mouse do pynput; os comandos são encaminhados pelo compositor.
- Ctrl+V no Wayland envia explicitamente a tecla V pelo `wtype`.

### Validação

Medições locais em Linux/Hyprland com RTX 4070 Ti SUPER e microfone FIFINE:

| Medição | Antes | Depois |
| --- | ---: | ---: |
| Comando do atalho com caches aquecidos | 272 ms | 23,4 ms, mediana de 12 execuções |
| Chamada isolada ao `wl-copy` | 2.001,7 ms, timeout | 15,7 ms, sucesso |
| Tamanho dos blocos de captura no Linux | 100 ms | 20 ms |

- Colagem completa em um campo GTK nativo de Wayland: 86,8 ms desde o início da operação, com acentos e quebra de linha preservados. Esse tempo não inclui captura, transcrição ou acionamento do atalho.
- Nove testes de regressão aprovados; oito gravações locais produziram o mesmo texto de referência após as otimizações. Um teste com silêncio não produziu texto.
- A precisão do modelo e o tamanho da busca de decodificação foram mantidos. Os resultados são medições desta máquina, não uma promessa de desempenho em todo sistema.

### Compatibilidade

- GPU NVIDIA/CUDA continua obrigatória; não há fallback para CPU.
- Wayland requer `wl-clipboard`, `wtype` e um atalho configurado no compositor. O launcher local chamado `sussurro` e as configurações do compositor não são instalados automaticamente pelo repositório.
- A colagem foi validada em GTK nativo de Wayland. A janela de teste Tk/Xwayland não confirmou inserção; a integração com outros aplicativos precisa ser verificada conforme o ambiente.
- Não houve nova validação de execução no Windows para esta release.

## [0.1.0] — 2026-08-24

Primeira release, voltada ao Windows.

### Adicionado

- Ditado local com Silero VAD e faster-whisper `large-v3` em CUDA/float16.
- Atalho global de mouse com modos alternar e segurar.
- Transcrição simultânea ou ao final; captura de microfone, áudio do PC ou ambos.
- Barra flutuante com onda ao vivo, confirmação e cancelamento.
- Biblioteca de correções exatas e fonéticas, com termos usados como hotwords.
- Histórico com reprodução de áudio e cópia, visualização ao vivo e estatísticas derivadas das sessões.
- Colagem com preservação dos formatos do clipboard no Windows.

[0.2.0]: https://github.com/LucasOl1337/sussurro/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/LucasOl1337/sussurro/releases/tag/v0.1.0
