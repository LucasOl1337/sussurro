"""Model choices and hardware resolution, independent of audio and desktop input."""
from dataclasses import dataclass

MODEL_LABELS = {
    "auto": "Automatico (recomendado)",
    "tiny": "Tiny — minimo consumo",
    "base": "Base — CPU basico",
    "small": "Small — CPU moderno",
    "medium": "Medium — intermediario",
    "large-v3-turbo": "Turbo — GPU recomendado",
    "large-v3": "Large-v3 — precisao",
}
DEVICE_LABELS = {"auto": "Automatico", "cuda": "GPU NVIDIA", "cpu": "CPU"}
DEFAULT_MODEL_SETTINGS = {"whisper_model": "auto", "whisper_device": "auto"}


@dataclass(frozen=True)
class ModelConfig:
    model: str
    device: str
    compute_type: str

    @property
    def label(self):
        return f"{self.model} · {self.device.upper()} · {self.compute_type}"


def normalize_model_settings(settings):
    """Missing/invalid choices adopt the new default; explicit choices survive updates."""
    result = {}
    for key, choices in (("whisper_model", MODEL_LABELS), ("whisper_device", DEVICE_LABELS)):
        value = settings.get(key)
        result[key] = value if isinstance(value, str) and value in choices else "auto"
    return result



def resolve_model_config(settings, backend=None):
    if backend is None:
        import ctranslate2 as backend
    settings = normalize_model_settings(settings)
    device = settings["whisper_device"]
    if device != "cpu":
        try:
            cuda = backend.get_cuda_device_count() > 0
        except (RuntimeError, ValueError):
            cuda = False
        if device == "cuda" and not cuda:
            raise RuntimeError("GPU NVIDIA indisponivel. Escolha CPU ou Automatico.")
        device = "cuda" if cuda else "cpu"
    model = settings["whisper_model"]
    if model == "auto":
        model = "large-v3-turbo" if device == "cuda" else "base"
    supported = backend.get_supported_compute_types(device)
    candidates = ("int8_float16", "float16", "int8_float32", "float32") if device == "cuda" else ("int8", "int8_float32", "float32")
    compute = next((kind for kind in candidates if kind in supported), None)
    if compute is None:
        raise RuntimeError(f"Nenhuma precisao suportada em {device}. Escolha outro dispositivo.")
    return ModelConfig(model, device, compute)


def model_path(model, download):
    """Use complete cached weights offline; finish partial/first downloads on demand."""
    from pathlib import Path
    from huggingface_hub.errors import LocalEntryNotFoundError
    try:
        path = download(model, local_files_only=True)
        if all((Path(path) / name).is_file() for name in ('model.bin', 'config.json', 'tokenizer.json')):
            return path
    except LocalEntryNotFoundError:
        pass
    return download(model)
