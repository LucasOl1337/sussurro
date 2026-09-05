# Sussurro no Omarchy (Hyprland)

Dois arquivos para copiar e uma aba nova no app.

## Regras de janela: `sussurro.lua`

```sh
cp contrib/omarchy/sussurro.lua ~/.config/hypr/sussurro.lua
```

E em `~/.config/hypr/hyprland.lua`, depois do `require("default.hypr.omarchy")`:

```lua
require("hypr.sussurro")
```

O que as regras fazem:

- Janela principal (`Sussurro`): opaca e renderizada mesmo sem foco, para o CUDA/Tk nao travarem em outro workspace.
- Barra de gravacao (`SussurroBar`, 152x40): flutuante, fixa em todos os workspaces, sem foco (o Enter e o Ctrl+V continuam indo pro app de baixo), sem sombra/animacao e com `rounding = 20`. O Tk em X11 nao tem transparencia por pixel, entao e o Hyprland que recorta os cantos da capsula.

Valide com `hyprctl reload && hyprctl configerrors`.

A barra pergunta ao Hyprland (socket `.socket.sock`) onde o cursor esta e qual a area util do monitor, porque o Tk em XWayland enxerga uma tela unica com todos os monitores e o ponteiro dele congela fora de janelas X. Coordenadas coincidem com as do X com `xwayland:force_zero_scaling` (padrao do Omarchy) e monitores em escala 1.

## Gestos do headset: `70-mchose-x9.rules`

A aba **OMARCHY** do app aciona o ditado pelo headset MCHOSE X9 sem tocar no PC. O fone so entrega ao PC volume +/-, play/pause e o audio; o mute do mic nao gera evento (so zera o audio e toca um aviso). Por isso os gestos sao:

| Gesto | Como e detectado |
|---|---|
| Roda de volume invertida rapido (vol+ e vol- em < 0,5 s) | evdev do *Consumer Control* do fone |
| Toque duplo no mute (mic zerado por ate 7 s) | silencio digital no fluxo do mic, ignorando o aviso de voz (~1,5 s) |

Girar a roda em uma direcao so continua sendo volume. Com "Enter automatico" ligado, ao terminar de colar o Sussurro aperta Enter. O mic padrao do sistema segue o fone (X9 com som -> X9; mudo/desligado por 10 s -> mic reserva); para o Sussurro seguir o padrao, escolha o microfone `pipewire` no card de configuracao.

Ler os botoes do fone sem root exige a regra udev. O botao **Instalar regra udev** da aba faz isso via `pkexec`; a mao:

```sh
sudo install -Dm644 contrib/omarchy/70-mchose-x9.rules /etc/udev/rules.d/70-mchose-x9.rules
sudo udevadm control --reload && sudo udevadm trigger --subsystem-match=input --action=change
```

Outros headsets entram como perfis em `sussurro_devices.py` (`PROFILES`): nome do dispositivo evdev e trecho do nome da fonte de audio.
