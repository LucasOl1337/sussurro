"""Friendly audio input names while keeping PortAudio identifiers stable."""

from dataclasses import dataclass, replace
import re


@dataclass(frozen=True)
class AudioInput:
    """An input shown to the user but persisted by its native PortAudio name."""

    name: str
    index: int
    label: str
    description: str
    priority: int = 10
    source: str | None = None  # fonte PipeWire/Pulse gravada pela rota "pulse" (PULSE_SOURCE)


def _clean_hardware_name(name: str) -> str:
    text = re.sub(r"\s*\(hw:[^)]+\)\s*$", "", name).strip()
    if ":" in text:
        controller, endpoint = (part.strip() for part in text.split(":", 1))
        if endpoint and endpoint.lower() not in {"usb audio", "audio"}:
            text = endpoint
        elif controller:
            text = controller
    text = re.sub(r"\s+USB Audio$", "", text, flags=re.I).strip()
    return text or name


def describe_input(name: str, index: int, system_microphone: str = "") -> AudioInput:
    """Turn PortAudio implementation names into a useful label and explanation."""
    lowered = name.strip().lower()
    current = system_microphone or "o microfone escolhido no sistema"

    if lowered == "pipewire":
        return AudioInput(
            name, index,
            f"{current} — padrão do sistema (PipeWire)",
            f"Segue automaticamente o microfone padrão do sistema. Agora: {current}.",
            0,
        )
    if lowered == "pulse":
        return AudioInput(
            name, index,
            "Compatibilidade PulseAudio — segue o sistema",
            "Rota de compatibilidade para aplicativos antigos; também segue o áudio do sistema.",
            2,
        )
    if lowered == "default":
        return AudioInput(
            name, index,
            "Compatibilidade ALSA — segue o sistema",
            "Rota genérica do Linux. Prefira PipeWire quando ele estiver disponível.",
            3,
        )
    if lowered == "jack":
        return AudioInput(
            name, index,
            "JACK — roteamento de áudio profissional",
            "Captura a entrada conectada no servidor JACK, usada em fluxos de áudio profissional.",
            4,
        )

    clean = _clean_hardware_name(name)
    if "usb" in lowered or "hw:" in lowered:
        connection = "USB direto" if "usb" in lowered else "hardware direto"
        return AudioInput(
            name, index, f"{clean} — {connection}",
            f"Captura {clean} diretamente, sem acompanhar mudanças do microfone padrão.",
            10,
        )
    return AudioInput(
        name, index, clean,
        f"Captura diretamente a entrada {clean}.",
        10,
    )


def prepare_input_devices(devices, system_microphone: str = "") -> dict[str, AudioInput]:
    """Build a native-name map with stable, unique labels in presentation order."""
    inputs = [
        describe_input(str(device["name"]), int(device["index"]), system_microphone)
        for device in devices
        if int(device.get("max_input_channels", 0)) > 0
    ]
    inputs.sort(key=lambda device: (device.priority, device.label.casefold(), device.index))

    counts: dict[str, int] = {}
    result: dict[str, AudioInput] = {}
    for device in inputs:
        count = counts.get(device.label, 0) + 1
        counts[device.label] = count
        if count > 1:
            device = replace(device, label=f"{device.label} ({count})")
        result[device.name] = device
    return result


SYSTEM_DEFAULT = "pipewire"
_CHANNEL_SUFFIX = re.compile(r"\s+(mono|analog stereo|digital stereo.*|stereo|mono fallback)$", re.I)


def source_label(description: str) -> str:
    """'MCHOSE X9 Mono' -> 'MCHOSE X9'; o canal nao diz nada pra quem escolhe o mic."""
    text = _CHANNEL_SUFFIX.sub("", description.strip()).strip()
    return text or description.strip()


def pipewire_inputs(devices, sources, current: str = "") -> dict[str, AudioInput]:
    """Padrao do PC + so os microfones fisicos do PipeWire, sem os plugins do ALSA.

    `devices` vem do PortAudio (precisa das rotas "pipewire" e "pulse"); `sources` e
    o `pactl -f json list sources`. Monitores (som que o PC toca), sinks virtuais e
    filtros ficam de fora: so entra o que tem `device.class` igual a "sound".
    Devolve {} se faltar uma das rotas, e o chamador cai na lista do PortAudio.
    """
    by_name = {str(d["name"]): int(d["index"]) for d in devices
               if int(d.get("max_input_channels", 0)) > 0}
    if SYSTEM_DEFAULT not in by_name:
        return {}
    labels = {s.get("name"): source_label(s.get("description") or s.get("name", ""))
              for s in sources}
    now = labels.get(current) or "o microfone escolhido no sistema"
    result = {SYSTEM_DEFAULT: AudioInput(
        SYSTEM_DEFAULT, by_name[SYSTEM_DEFAULT], "Padrão do PC",
        f"Segue o microfone padrão do sistema. Agora: {now}.", 0)}
    if "pulse" not in by_name:
        return result
    seen: dict[str, int] = {}
    for source in sources:
        props = source.get("properties") or {}
        name = source.get("name")
        if not name or props.get("device.class") != "sound" or name.endswith(".monitor"):
            continue
        label = labels[name]
        seen[label] = seen.get(label, 0) + 1
        if seen[label] > 1:
            label = f"{label} ({seen[label]})"
        result["pulse:" + name] = AudioInput(
            "pulse:" + name, by_name["pulse"], label,
            f"Grava sempre {label}, mesmo que o padrão do sistema mude.", 10, source=name)
    return result
