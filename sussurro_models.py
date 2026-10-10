"""Model choices and hardware resolution, independent of audio and desktop input."""
import os
from dataclasses import dataclass
from pathlib import Path

from sussurro_cloud import ELEVENLABS, PROVIDER_NAMES, XAI, is_cloud

LOCAL_DEFAULT = "large-v3-turbo"

MODEL_LABELS = {
    "large-v3-turbo": "Turbo · local, rápido",
    "large-v3": "Large-v3 · local, precisão",
    ELEVENLABS: "ElevenLabs Scribe · nuvem",
    XAI: "Grok (xAI) · nuvem",
}
DEVICE_LABELS = {"auto": "Automatico", "cuda": "GPU NVIDIA", "cpu": "CPU"}
DEFAULT_MODEL_SETTINGS = {"whisper_model": LOCAL_DEFAULT, "whisper_device": "auto"}


@dataclass(frozen=True)
class ModelConfig:
    model: str
    device: str
    compute_type: str

    @property
    def label(self):
        if self.engine == "cloud":
            return f"{PROVIDER_NAMES[self.model]} · NUVEM"
        return f"{self.model} · {self.device.upper()} · {self.compute_type}"

    @property
    def engine(self):
        return engine_of(self.model)


def engine_of(model):
    """Nuvem vai por HTTP (sussurro_cloud); o resto e faster-whisper/CTranslate2."""
    return "cloud" if is_cloud(model) else "whisper"


def cache_dir(*parts) -> Path:
    """Cache privado do Sussurro (comparacao, reunioes em andamento, modelo de vozes)."""
    override = os.environ.get('SUSSURRO_CACHE_DIR')
    if override:
        root = Path(override).expanduser()
    elif os.name == 'nt':
        local = os.environ.get('LOCALAPPDATA')
        root = Path(local) / 'Sussurro' if local else Path.home() / 'AppData' / 'Local' / 'Sussurro'
    else:
        xdg = os.environ.get('XDG_CACHE_HOME')
        root = Path(xdg).expanduser() / 'sussurro' if xdg else Path.home() / '.cache' / 'sussurro'
    path = root.joinpath(*parts)
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def normalize_model_settings(settings):
    """Escolha valida sobrevive a atualizacao; Automatico, Tiny, Base, Small, Medium e
    Parakeet (fora do seletor desde a v0.8) e qualquer valor invalido viram Turbo."""
    model = settings.get("whisper_model")
    device = settings.get("whisper_device")
    return {
        "whisper_model": model if isinstance(model, str) and model in MODEL_LABELS else LOCAL_DEFAULT,
        "whisper_device": device if isinstance(device, str) and device in DEVICE_LABELS else "auto",
    }


def resolve_model_config(settings, backend=None):
    settings = normalize_model_settings(settings)
    model = settings["whisper_model"]
    if is_cloud(model):
        return ModelConfig(model, "cloud", "api")  # nao toca em CUDA nem ocupa VRAM
    if backend is None:
        import ctranslate2 as backend
    device = settings["whisper_device"]
    if device != "cpu":
        try:
            cuda = backend.get_cuda_device_count() > 0
        except (RuntimeError, ValueError):
            cuda = False
        if device == "cuda" and not cuda:
            raise RuntimeError("GPU NVIDIA indisponivel. Escolha CPU ou Automatico.")
        device = "cuda" if cuda else "cpu"
    supported = backend.get_supported_compute_types(device)
    candidates = ("int8_float16", "float16", "int8_float32", "float32") if device == "cuda" else ("int8", "int8_float32", "float32")
    compute = next((kind for kind in candidates if kind in supported), None)
    if compute is None:
        raise RuntimeError(f"Nenhuma precisao suportada em {device}. Escolha outro dispositivo.")
    return ModelConfig(model, device, compute)


def fallback_config(settings, backend=None):
    """Whisper local usado quando a nuvem cai: Turbo no dispositivo escolhido."""
    return resolve_model_config({"whisper_model": LOCAL_DEFAULT,
                                 "whisper_device": normalize_model_settings(settings)["whisper_device"]},
                                backend)


def model_path(model, download):
    """Use complete cached weights offline; finish partial/first downloads on demand."""
    from huggingface_hub.errors import LocalEntryNotFoundError
    try:
        path = download(model, local_files_only=True)
        if all((Path(path) / name).is_file() for name in ('model.bin', 'config.json', 'tokenizer.json')):
            return path
    except LocalEntryNotFoundError:
        pass
    return download(model)
