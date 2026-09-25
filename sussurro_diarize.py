# Portado de jankeesvw/omarchy-meeting-recorder, MIT, (c) 2026 Jankees van Woezik.
"""Who speaks when: NVIDIA Nemotron 3 Diarization (Streaming Sortformer) via ONNX Runtime.

The model gives every 10 ms frame a probability for up to eight speakers, numbered
in the order they are first heard. The ONNX export holds no state: features, the
chunk loop and the speaker cache/FIFO live here and follow
`Nemotron3DiarizationSpeakerCache` from Hugging Face transformers.
onnxruntime and huggingface_hub are imported lazily: pure post-processing works
without them.
"""
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sussurro_models import cache_dir

REPO = "onnx-community/Nemotron-3-Diarization-ONNX"
# Revisao fixa: mudanca upstream nao chega no app sem revisao
REVISION = "353b6f8ad2cac3580e982d7fbdf0a010786b0406"
GRAPH = "onnx/model_fp16.onnx"

RATE = 16000
HOP = 160
N_FFT = 512
WIN = 400
BINS = N_FFT // 2 + 1
MELS = 128
PREEMPHASIS = 0.97
SPEAKERS = 8

# Config de streaming do modelo, tamanhos offline (iguais ao original)
HIDDEN = 512
FACTOR = 8
CHUNK = 340
RIGHT_CONTEXT = 40
FIFO = 40
UPDATE_PERIOD = 300
CACHE = 264
SILENCE_PER_SPEAKER = 1
SCORE_THRESHOLD = 0.25
LATEST_BOOST = 0.05
MIN_POSITIVE_RATE = 0.5
STRONG_BOOST_RATE = 0.75
WEAK_BOOST_RATE = 1.5


class Cancelled(RuntimeError):
    def __init__(self, message="cancelado"):
        super().__init__(message)


@dataclass(frozen=True)
class Turn:
    start_ms: int
    end_ms: int
    speaker: int  # 0.. na ordem em que a voz aparece


# --- modelo -------------------------------------------------------------------------------

def _side_by_side(graph, data):
    """ORT recusa pesos externos cujo caminho real sai da pasta do grafo, e o cache do HF
    com blobs compartilhados guarda cada arquivo numa pasta. Junta por hardlink (ou copia)."""
    graph, data = Path(graph), Path(data)
    if graph.resolve().parent == data.resolve().parent:
        return graph
    folder = cache_dir('diarize', REVISION[:12])
    for source in (graph, data):
        target = folder / source.name  # nome que o grafo referencia, nao o do blob
        if target.is_file():
            continue
        part = target.with_name(target.name + '.part')
        part.unlink(missing_ok=True)
        try:
            os.link(source.resolve(), part)
        except OSError:
            shutil.copyfile(source.resolve(), part)
        os.replace(part, target)
    return folder / graph.name


def ensure_model(progress=None):
    """Graph path, downloading graph + weights (about 200 MB) on first use.
    `progress(msg: str)` is told before a download starts."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    paths = []
    for name in (GRAPH, GRAPH + '_data'):
        try:
            path = hf_hub_download(REPO, name, revision=REVISION, local_files_only=True)
        except LocalEntryNotFoundError:
            if progress:
                progress("Baixando o modelo de falantes (~200 MB)")
            path = hf_hub_download(REPO, name, revision=REVISION)
        paths.append(path)
    return _side_by_side(*paths)


def _sigmoid(x):
    with np.errstate(over='ignore'):
        return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


class Diarizer:
    """One ONNX session; reuse it for several files and `close()` to free the VRAM.
    `providers=None` means GPU only (CUDAExecutionProvider); others are for tests."""

    def __init__(self, providers=None):
        cuda = providers is None
        if cuda:
            from sussurro_cuda import _prepare_cuda_libs
            _prepare_cuda_libs()
        import onnxruntime as ort
        if cuda:
            if 'CUDAExecutionProvider' not in ort.get_available_providers():
                raise RuntimeError("GPU NVIDIA indisponivel para a diarizacao: onnxruntime sem CUDAExecutionProvider.")
            # Arena crescendo so o pedido: pico ~510 MiB em vez de ~750 MiB, mesma velocidade
            providers = [('CUDAExecutionProvider', {'arena_extend_strategy': 'kSameAsRequested'})]
        options = ort.SessionOptions()
        options.log_severity_level = 3  # avisos de Memcpy/nos em CPU a cada sessao: so erro
        self._session = ort.InferenceSession(str(ensure_model()), options, providers=providers)
        # Sem as libs CUDA o ORT cai pra CPU calado: aqui isso vira erro
        if cuda and 'CUDAExecutionProvider' not in self._session.get_providers():
            self.close()
            raise RuntimeError("GPU NVIDIA indisponivel para a diarizacao: falha ao carregar o CUDA no onnxruntime.")
        self._silence = None

    def close(self):
        self._session = None

    def _run(self, features, cached, total):
        """One chunk: (logits per 10 ms, chunk embeds, silence embed). O grafo fp16 ja
        recebe e devolve float32."""
        logits, embeds, silence = self._session.run(['logits', 'chunk_embeds', 'silence_embeds'], {
            'input_features': features[None],
            'cached_embeds': cached[None],
            'attention_mask': np.ones((1, total), np.int64),
        })
        return logits[0], embeds[0], silence

    def probabilities(self, samples, progress=None, abort=None):
        """Speaker probabilities per 10 ms of `samples` (float32 mono 16 kHz): (frames, 8).
        `progress(frac: float)` goes from 0 to 1 over the chunks; a set `abort` Event raises Cancelled."""
        if self._session is None:
            raise RuntimeError("Diarizer fechado")
        samples = np.asarray(samples, np.float32)
        mel = mel_filters()
        frames = 1 + len(samples) // HOP
        valid = len(samples) // HOP
        steps = -(-frames // FACTOR)
        cache = _Cache()
        out = []
        start = 0
        while start < steps:
            if abort is not None and abort.is_set():
                raise Cancelled()
            end = min(start + CHUNK, steps)
            lookahead = min(end + RIGHT_CONTEXT, steps)
            rows = (lookahead - start) * FACTOR
            first = start * FACTOR
            features = log_mel(samples, mel, first, min(first + rows, frames), valid, rows)
            cached = cache.embeds()
            logits, embeds, silence = self._run(features, cached, len(cached) + lookahead - start)
            if self._silence is None:
                self._silence = silence.reshape(HIDDEN)
            out.append(logits[len(cached) * FACTOR:(len(cached) + end - start) * FACTOR])
            cache.update(np.concatenate([cached, embeds]), logits, end - start, self._silence)
            if progress:
                progress(end / steps)
            start = end
        return _sigmoid(np.concatenate(out)[:frames])


# --- features ------------------------------------------------------------------------------

# Escala mel Slaney: linear ate 1 kHz, log depois
F_SP, MIN_LOG_HZ, LOGSTEP = 200.0 / 3.0, 1000.0, math.log(6.4) / 27.0


def _hz_to_mel(hz):
    return MIN_LOG_HZ / F_SP + math.log(hz / MIN_LOG_HZ) / LOGSTEP if hz >= MIN_LOG_HZ else hz / F_SP


def _mel_to_hz(mel):
    min_log_mel = MIN_LOG_HZ / F_SP
    return np.where(mel >= min_log_mel, MIN_LOG_HZ * np.exp(LOGSTEP * (mel - min_log_mel)), F_SP * mel)


def mel_filters():
    """librosa's Slaney filterbank (norm="slaney"), 257 FFT bins -> 128 bands, 0-8 kHz: (128, 257)."""
    top = _hz_to_mel(RATE / 2)
    points = _mel_to_hz(top * np.arange(MELS + 2) / (MELS + 1))
    fft_hz = np.arange(BINS) * (RATE / 2) / (BINS - 1)
    lower, center, upper = points[:-2, None], points[1:-1, None], points[2:, None]
    rising = (fft_hz - lower) / (center - lower)
    falling = (upper - fft_hz) / (upper - center)
    return (np.maximum(np.minimum(rising, falling), 0.0) * (2.0 / (upper - lower))).astype(np.float32)


def _window():
    """Hann simetrica de 400 centrada em 512."""
    offset = (N_FFT - WIN) // 2
    window = np.zeros(N_FFT, np.float32)
    window[offset:offset + WIN] = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(WIN) / (WIN - 1))
    return window


def _padded(samples, lo, hi):
    """Posicoes lo:hi do sinal com pre-enfase (amostra 0 intacta) e N_FFT/2 zeros dos dois lados."""
    out = np.zeros(hi - lo, np.float32)
    pad = N_FFT // 2
    a, b = max(lo - pad, 0), min(hi - pad, len(samples))
    if a < b:
        prev = samples[a - 1:b - 1] if a > 0 else np.concatenate(([0.0], samples[:b - 1])).astype(np.float32)
        out[a + pad - lo:b + pad - lo] = samples[a:b] - np.float32(PREEMPHASIS) * prev
    return out


def power(samples, start, end):
    """|STFT|^2 (torch.stft center=True) of frames start:end: (end - start, 257)."""
    if end <= start:
        return np.zeros((0, BINS), np.float32)
    signal = _padded(samples, start * HOP, (end - 1) * HOP + N_FFT)
    frames = np.lib.stride_tricks.sliding_window_view(signal, N_FFT)[::HOP]
    spectrum = np.fft.rfft(frames * _window(), axis=1)
    return (spectrum.real ** 2 + spectrum.imag ** 2).astype(np.float32)


def log_mel(samples, mel, start, end, valid, rows):
    """log(mel + 2^-24) of frames start:end, zero from `valid` on, padded to `rows`: (rows, 128)."""
    out = np.zeros((rows, MELS), np.float32)
    stop = min(end, valid)
    if stop > start:
        out[:stop - start] = np.log(power(samples, start, stop) @ mel.T + np.float32(2.0 ** -24))
    return out


# --- speaker cache -------------------------------------------------------------------------

def _pool_probs(logits):
    """Sigmoid dos logits de 10 ms, media por passo do encoder: (steps, 8)."""
    steps = len(logits) // FACTOR
    return _sigmoid(logits[:steps * FACTOR]).reshape(steps, FACTOR, -1).mean(axis=1, dtype=np.float32)


def _frame_scores(probs):
    """High for frames that clearly belong to one speaker; -inf where that speaker is silent."""
    budget = CACHE // SPEAKERS - SILENCE_PER_SPEAKER
    min_positive = int(np.floor(budget * MIN_POSITIVE_RATE))
    t = np.float32(SCORE_THRESHOLD)
    complements = np.log(np.maximum(1 - probs, t))
    scores = (np.log(np.maximum(probs, t)) - complements + complements.sum(axis=1, keepdims=True)
              - np.log(np.float32(0.5)))
    scores = np.where(probs > 0.5, scores, -np.inf).astype(np.float32)
    enough = (scores > 0).sum(axis=0) >= min_positive
    weak = np.isfinite(scores) & (scores <= 0) & enough
    scores[weak] = -np.inf
    return scores


def _boost(scores, count, amount):
    """Soma `amount` aos `count` maiores scores de cada falante (ordem estavel, empate = frame antes)."""
    count = min(count, len(scores))
    for s in range(scores.shape[1]):
        top = np.argsort(-scores[:, s], kind='stable')[:count]
        scores[top, s] += np.float32(amount)


def _compress(embeds, probs, silence):
    """Keeps the CACHE most telling frames, grouped by speaker in their original order,
    with one slot of learned silence per speaker."""
    frames = len(embeds)
    scores = _frame_scores(probs)
    scores[CACHE:] += np.float32(LATEST_BOOST)
    budget = CACHE // SPEAKERS - SILENCE_PER_SPEAKER
    _boost(scores, int(np.floor(budget * STRONG_BOOST_RATE)), -2.0 * np.log(0.5))
    _boost(scores, int(np.floor(budget * WEAK_BOOST_RATE)), -np.log(0.5))
    scored = frames + SILENCE_PER_SPEAKER
    # Indice plano falante-major sobre frames + slots de silencio (+inf)
    flat = np.full((SPEAKERS, scored), np.inf, np.float32)
    flat[:, :frames] = scores.T
    flat = flat.reshape(-1)
    order = np.argsort(-flat, kind='stable')[:CACHE]
    sentinel = scored * SPEAKERS
    picked = np.sort(np.where(flat[order] == -np.inf, sentinel, order))
    frame = np.where(picked == sentinel, frames, np.minimum(picked % scored, frames))
    all_embeds = np.concatenate([embeds, silence[None]])
    all_probs = np.concatenate([probs, np.zeros((1, probs.shape[1]), np.float32)])
    return all_embeds[frame], all_probs[frame]


class _Cache:
    """Arrival-Order Speaker Cache and FIFO, rows of HIDDEN."""

    def __init__(self):
        self.cache_embeds = np.zeros((0, HIDDEN), np.float32)
        self.cache_probs = np.zeros((0, SPEAKERS), np.float32)
        self.fifo = np.zeros((0, HIDDEN), np.float32)
        self.compressed = False

    def embeds(self):
        return np.concatenate([self.cache_embeds, self.fifo])

    def update(self, embeds, logits, chunk, silence):
        """Chunk vai pra FIFO; o que transborda desce pro cache, comprimido quando passa de CACHE."""
        cache_len, fifo_len = len(self.cache_embeds), len(self.fifo)
        probs = _pool_probs(logits)
        begin = cache_len + fifo_len
        fifo = np.concatenate([self.fifo, embeds[begin:begin + chunk]])
        popped = 0 if len(fifo) <= FIFO else min(max(UPDATE_PERIOD, len(fifo) - FIFO), len(fifo))
        if popped:
            fifo_probs = probs[cache_len:cache_len + len(fifo)]
            cache_probs = self.cache_probs[:cache_len] if self.compressed else probs[:cache_len]
            cache_embeds = np.concatenate([self.cache_embeds, fifo[:popped]])
            cache_probs = np.concatenate([cache_probs, fifo_probs[:popped]])
            fifo = fifo[popped:]
            if len(cache_embeds) > CACHE:
                cache_embeds, cache_probs = _compress(cache_embeds, cache_probs, silence)
                self.compressed = True
            self.cache_embeds, self.cache_probs = cache_embeds, cache_probs
        self.fifo = fifo


# --- turnos --------------------------------------------------------------------------------

def turns(samples, speakers=None, diarizer=None, progress=None, abort=None):
    """Speaker turns of `samples` (float32 mono 16 kHz). `speakers` fixes how many there are;
    None lets the model decide. Without `diarizer` a GPU one is created and closed here.
    `progress(frac: float)` only gets the chunk loop; call ensure_model(progress) first
    for download messages."""
    own = diarizer is None
    if own:
        diarizer = Diarizer()
    try:
        probs = diarizer.probabilities(samples, progress, abort)
    finally:
        if own:
            diarizer.close()
    raw = _segments(probs)
    # Voz ouvida so por poucos segundos quase sempre e outra pessoa num momento ruim
    raw = _keep_largest(raw, speakers) if speakers is not None else _absorb_small_clusters(raw)
    return _renumber(raw)


def _segments(probs):
    """Trechos com prob > 0.5 em ms (frame = 10 ms); pausa < 500 ms emendada, trecho < 300 ms fora."""
    raw = []
    for s in range(probs.shape[1]):
        on = np.concatenate(([False], probs[:, s] > 0.5, [False]))
        edges = np.flatnonzero(np.diff(on.astype(np.int8)))
        runs = []
        for a, b in zip(edges[::2] * 10, edges[1::2] * 10):
            if runs and a - runs[-1][1] < 500:
                runs[-1][1] = int(b)
            else:
                runs.append([int(a), int(b)])
        raw += [(a, b, s) for a, b in runs if b - a >= 300]
    return raw


def _spoken(raw):
    total = {}
    for start, end, who in raw:
        total[who] = total.get(who, 0) + end - start
    return total


def _keep_largest(raw, n):
    """Keeps the `n` speakers with the most speech; the others go to the nearest kept turn."""
    ranked = sorted(_spoken(raw).items(), key=lambda item: (-item[1], item[0]))
    kept = {who for who, _ in ranked[:max(n, 1)]}
    return _reassign(raw, lambda who: who in kept)


def _absorb_small_clusters(raw):
    """Cluster com pouca fala (< 4 s ou < 4% do total) vai pro falante do turno mantido mais perto."""
    spoken = _spoken(raw)
    floor = max(sum(spoken.values()) * 4 // 100, 4000)
    return _reassign(raw, lambda who: spoken.get(who, 0) >= floor)


def _distance(middle, start, end):
    return start - middle if middle < start else max(middle - end, 0)


def _reassign(raw, keeps):
    anchors = [turn for turn in raw if keeps(turn[2])]
    if not anchors:
        return raw
    result = []
    for start, end, who in raw:
        if not keeps(who):
            middle = (start + end) // 2
            who = min(anchors, key=lambda a: _distance(middle, a[0], a[1]))[2]
        result.append((start, end, who))
    return result


def _renumber(raw):
    """Ordena por inicio e numera os falantes na ordem em que aparecem."""
    order = []
    result = []
    for start, end, who in sorted(raw, key=lambda turn: turn[0]):
        if who not in order:
            order.append(who)
        result.append(Turn(int(start), int(end), order.index(who)))
    return result


def single(samples):
    """The whole file as one speaker."""
    return [Turn(0, len(samples) * 1000 // RATE, 0)]


def speaker_at(turns, start_ms, end_ms):
    """Speaker overlapping start_ms:end_ms the most, or of the nearest turn (words in a gap)."""
    end_ms = max(end_ms, start_ms + 1)
    overlap = {}
    for turn in turns:
        shared = min(turn.end_ms, end_ms) - max(turn.start_ms, start_ms)
        if shared > 0:
            overlap[turn.speaker] = overlap.get(turn.speaker, 0) + shared
    if overlap:
        return max(overlap.items(), key=lambda item: (item[1], -item[0]))[0]
    if not turns:
        return 0
    middle = (start_ms + end_ms) // 2
    return min(turns, key=lambda t: _distance(middle, t.start_ms, t.end_ms)).speaker


def turn_start_near(turns, speaker, around_ms):
    """Start of `speaker`'s turn within 1.5 s of `around_ms`, or None."""
    near = [t for t in turns if t.speaker == speaker and abs(t.start_ms - around_ms) <= 1500]
    return min(near, key=lambda t: abs(t.start_ms - around_ms)).start_ms if near else None
