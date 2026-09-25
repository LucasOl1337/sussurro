"""Parakeet TDT 0.6B v3 (NVIDIA) na GPU, com a mesma cara do WhisperModel do faster-whisper.

O ditado, o historico e a comparacao chamam `model.transcribe(audio, ...)` e leem
`seg.start`, `seg.end`, `seg.text` (e `seg.words` na reuniao): este adaptador devolve
isso a partir do onnx-asr. O Parakeet so aceita trechos curtos (atencao cheia), entao
audio longo e cortado pelo Silero VAD e os trechos vao em lotes pequenos para a GPU.

Diferencas que o usuario sente: nao ha idioma forcado (o modelo decide sozinho e, em
portugues com termos em ingles, as vezes escorrega pro ingles) nem `hotwords` (a
Biblioteca continua corrigindo depois do texto).
"""
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from sussurro_models import PARAKEET, cache_dir

SAMPLE_RATE = 16000
REPO = "istupakov/parakeet-tdt-0.6b-v3-onnx"
# Revisao fixa: uma troca no repositorio nunca chega ao app sem ser vista.
REVISION = "8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce"
FP32_FILES = ("config.json", "vocab.txt", "nemo128.onnx", "decoder_joint-model.onnx",
              "encoder-model.onnx", "encoder-model.onnx.data")
DONE = ".fp16-pronto"
# Trecho maximo por inferencia e tamanho do lote: o encoder e atencao cheia, e o lote
# e preenchido ate o trecho mais longo.
MAX_CHUNK_S = 30
BATCH_SECONDS = 90
FRAME_S = 0.08  # passo do encoder (subsampling 8 x hop de 10 ms)
# Trecho de comprimento no pre_encode: fica em fp32, o conversor quebra os tipos ali.
_LENGTH_PATH = re.compile(r"^/pre_encode/(Cast|Constant|Add|Div|Floor)(_\d+)?$")


def model_dir(download=True, status=None) -> Path:
    """Pasta com o Parakeet em FP16. No primeiro uso baixa o FP32 (2,5 GB) e converte uma vez."""
    target = cache_dir("models") / f"{PARAKEET}-fp16"
    if (target / DONE).is_file():
        return target
    if not download:
        raise FileNotFoundError(f"Parakeet ainda nao foi preparado em {target}")
    from huggingface_hub import hf_hub_download
    work = cache_dir("models") / f".{PARAKEET}-download"
    for name in FP32_FILES:
        if status:
            status(f"Baixando Parakeet ({name})...")
        hf_hub_download(REPO, name, revision=REVISION, local_dir=work)
    if status:
        status("Convertendo Parakeet para FP16 (so no primeiro uso)...")
    partial = target.with_name(target.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    _encoder_to_fp16(work, partial)
    for name in ("config.json", "vocab.txt", "nemo128.onnx", "decoder_joint-model.onnx"):
        shutil.copyfile(work / name, partial / name)
    (partial / DONE).write_text(REVISION + "\n")
    shutil.rmtree(target, ignore_errors=True)
    partial.rename(target)
    shutil.rmtree(work, ignore_errors=True)
    return target


def _encoder_to_fp16(src: Path, dst: Path):
    """Metade da VRAM (4,9 -> 2,7 GB) com a mesma transcricao do FP32."""
    import onnx
    from onnx.external_data_helper import load_external_data_for_model
    from onnxconverter_common import float16
    shapes = dst / "encoder-shapes.onnx"
    # Tipos inferidos antes: sem eles o conversor erra os Casts do grafo (>2 GB, via arquivo).
    onnx.shape_inference.infer_shapes_path(str(src / "encoder-model.onnx"), str(shapes))
    model = onnx.load(str(shapes), load_external_data=False)
    load_external_data_for_model(model, str(src))
    shapes.unlink()
    block = [n.name for n in model.graph.node if _LENGTH_PATH.match(n.name)]
    model = float16.convert_float_to_float16(model, keep_io_types=True, disable_shape_infer=True,
                                             node_block_list=block)
    # O value_info antigo ainda diz float onde agora e float16; o onnxruntime reinfere.
    del model.graph.value_info[:]
    onnx.save(model, str(dst / "encoder-model.onnx"), save_as_external_data=True,
              all_tensors_to_one_file=True, location="encoder-model.onnx.data")


@dataclass
class Word:
    start: float
    end: float
    word: str
    probability: float


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list = field(default_factory=list)
    no_speech_prob: float = 0.0
    avg_logprob: float = 0.0


def words_from_tokens(tokens, times, logprobs, offset, end):
    """Tokens BPE com tempo de inicio -> palavras. Token com espaco na frente abre palavra."""
    words = []
    for i, token in enumerate(tokens):
        start = offset + times[i]
        stop = offset + (times[i + 1] if i + 1 < len(times) else times[i] + FRAME_S)
        prob = float(np.exp(logprobs[i])) if logprobs else 1.0
        if words and not token.startswith(" "):
            last = words[-1]
            last.word += token
            last.end = min(stop, end)
            last.probability = min(last.probability, prob)
        else:
            words.append(Word(start, min(stop, end), token, prob))
    for w in words:
        w.end = max(w.end, w.start)
    return words


def speech_chunks(audio, vad_filter):
    """Trechos (inicio, fim) em amostras. Sem VAD, audio curto vai inteiro."""
    if not vad_filter and audio.size <= MAX_CHUNK_S * SAMPLE_RATE:
        return [(0, audio.size)] if audio.size else []
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    options = VadOptions(max_speech_duration_s=MAX_CHUNK_S, min_silence_duration_ms=500,
                         speech_pad_ms=200)
    spans = [(s["start"], s["end"]) for s in get_speech_timestamps(audio, options)]
    return merge_spans(spans, MAX_CHUNK_S * SAMPLE_RATE)


def merge_spans(spans, limit):
    """Junta falas vizinhas em janelas de ate `limit`: frase curta e isolada e onde o
    Parakeet, sem idioma fixo, mais escorrega pra outra lingua."""
    merged = []
    for start, end in spans:
        if merged and end - merged[-1][0] <= limit:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def batches(chunks):
    """Indices dos trechos em lotes de ate BATCH_SECONDS, parecidos em tamanho."""
    order = sorted(range(len(chunks)), key=lambda i: chunks[i][1] - chunks[i][0])
    group, longest = [], 0
    for i in order:
        size = chunks[i][1] - chunks[i][0]
        if group and max(longest, size) * (len(group) + 1) > BATCH_SECONDS * SAMPLE_RATE:
            yield group
            group, longest = [], 0
        group.append(i)
        longest = max(longest, size)
    if group:
        yield group


class ParakeetModel:
    """`transcribe()` compativel com o faster-whisper; `language`/`beam_size`/`hotwords` sao ignorados."""

    def __init__(self, path):
        import onnx_asr
        import onnxruntime as ort
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("onnxruntime sem CUDA: instale onnxruntime-gpu (requirements-cuda.txt).")
        options = ort.SessionOptions()
        options.log_severity_level = 3  # os avisos de Memcpy do pre-processador sao esperados
        self._asr = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", str(path),
                                        sess_options=options,
                                        providers=["CUDAExecutionProvider"]).with_timestamps()

    def transcribe(self, audio, language=None, beam_size=5, vad_filter=False, hotwords=None,
                   word_timestamps=False, **_):
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        chunks = speech_chunks(audio, vad_filter)
        results = [None] * len(chunks)
        for group in batches(chunks):
            out = self._asr.recognize([audio[chunks[i][0]:chunks[i][1]] for i in group])
            for i, result in zip(group, out):
                results[i] = result
        segments = []
        for (start, end), result in zip(chunks, results):
            text = result.text.strip()
            if not text:
                continue
            offset, stop = start / SAMPLE_RATE, end / SAMPLE_RATE
            words = words_from_tokens(result.tokens or [], result.timestamps or [],
                                      result.logprobs, offset, stop)
            logprob = float(np.mean(result.logprobs)) if result.logprobs else 0.0
            segments.append(Segment(words[0].start if words else offset, stop, " " + text,
                                    words if word_timestamps else [], avg_logprob=logprob))
        info = SimpleNamespace(language=language or "auto", language_probability=0.0,
                               duration=audio.size / SAMPLE_RATE)
        return segments, info

    def close(self):
        """Solta as sessoes (e a VRAM) antes de carregar outro modelo."""
        self._asr = None
        import gc
        gc.collect()

