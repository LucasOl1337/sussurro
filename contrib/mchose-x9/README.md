# Sussurro pelo headset MCHOSE X9

Aciona o ditado pelo próprio fone, sem tocar no PC, e confirma o envio da frase com Enter.

## Como funciona

O X9 (dongle 2.4 GHz, chip C-Media, USB `3837:6045`) só entrega ao PC três botões: volume +, volume − e play/pause. O botão de mute do microfone não gera evento algum; ele apenas zera o áudio do mic (silêncio digital exato) e toca um aviso de voz. Clique duplo no play troca o modo 2.4G/Bluetooth. Segurar o play chega como um toque comum.

O daemon `x9-sussurro` observa duas coisas:

| Gesto | Detecção | Ação |
|---|---|---|
| Inverter a roda de volume rápido (vol+ e vol− em menos de 0,5 s) | evdev do dispositivo *Consumer Control* do X9 | `toggle-enter` no socket do Sussurro |
| Toque duplo no mute (mic zerado por até 7 s) | fluxo do mic via `parec` | `toggle-enter` (o aviso de voz do fone gera um segundo zero de ~1,5 s, ignorado por assinatura) |

Girar a roda em uma direção só continua sendo volume. O volume sobe e desce um passo no gesto e fica onde estava.

Além disso, o daemon escolhe o microfone padrão do sistema: X9 quando ele está com áudio ativo, outro mic (FIFINE) quando o X9 está mutado ou desligado por 10 s. O Sussurro fica com `device_name: "pipewire"` no `settings.json` para seguir o mic padrão.

Os comandos `toggle-enter`, `start-enter` e `stop-enter` fazem o Sussurro apertar Enter (`wtype -k Return`) depois da última colagem da sessão, só se algum texto foi colado. O mouse 4 continua usando `toggle`, sem Enter.

## Instalação

```sh
install -Dm755 contrib/mchose-x9/x9-sussurro ~/.local/bin/x9-sussurro
install -Dm644 contrib/mchose-x9/x9-sussurro.service ~/.config/systemd/user/x9-sussurro.service
sudo install -Dm644 contrib/mchose-x9/70-mchose-x9.rules /etc/udev/rules.d/70-mchose-x9.rules
sudo udevadm control --reload && sudo udevadm trigger --subsystem-match=input --action=change
systemctl --user daemon-reload && systemctl --user enable --now x9-sussurro.service
```

Ajuste `SRC_X9` e `SRC_FALLBACK` no topo do script para os nomes das suas fontes (`pactl list short sources`). Log ao vivo: `journalctl --user -u x9-sussurro -f`.

## Parâmetros

- `REVERSAL_WINDOW` (0,5 s): janela da inversão da roda. Diminua se disparar ao ajustar volume; aumente se falhar.
- `TAP_MAX` (7 s), `BLIP_MAX_GAP` (1,3 s), `BLIP_MAX_DUR` (2 s): assinatura do toque duplo no mute e do aviso de voz.
- `IDLE_SWITCH` (10 s): tempo de X9 mutado/desligado até o mic padrão voltar ao FIFINE.
