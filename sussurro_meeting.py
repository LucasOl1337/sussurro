"""Reuniao: duas trilhas (voce no microfone, os outros no audio do PC) viram uma
transcricao com quem falou o que.

Portado de jankeesvw/omarchy-meeting-recorder (MIT, (c) 2026 Jankees van Woezik):
cada lado passa pelo reconhecedor separado, entao duas pessoas falando juntas ou
uma musica por baixo nao apagam a voz mais baixa, e o lado de cada frase e a propria
trilha. O eco dos outros vazando no seu microfone (sem fone) fica de fora. Varias
vozes no mesmo lado sao separadas pelo Nemotron (sussurro_diarize).

Diferenca daqui: o faster-whisper roda na GPU sobre a trilha com o que nao e fala
zerado; o Silero pula os silencios e devolve os tempos ja na linha do tempo real,
sem a colagem de trechos que o original faz para o whisper.cpp.
"""
import bisect
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass

import numpy as np

RATE = 16000
FRAME = RATE * 30 // 1000  # 30 ms
PARAGRAPH_PAUSE_MS = 3000  # pausa maior abre paragrafo, mesmo falante
PARAGRAPH_MAX_MS = 90_000  # turno longo e quebrado: a linha continua util pra pular
NEW_STRETCH_MS = 1400      # silencio que separa dois trechos de som (padding + juncao)

LANGUAGES = {"pt": "Português", "en": "Inglês", "auto": "Automático", "es": "Espanhol",
             "fr": "Francês", "de": "Alemão", "it": "Italiano", "nl": "Holandês"}
# Sem contexto o whisper as vezes cai no modo "tudo minusculo, sem ponto" e fica nele a
# trilha inteira; sem ponto a frase nao fecha e o falante nao troca. Um comeco pontuado
# no idioma certo segura a pontuacao.
PROMPTS = {"pt": "Olá. Tudo bem? Então, vamos começar a reunião.",
           "en": "Hello. Okay, let's begin the meeting.",
           "es": "Hola. ¿Qué tal? Bueno, empecemos la reunión.",
           "fr": "Bonjour. Ça va ? Bon, commençons la réunion.",
           "de": "Hallo. Wie geht's? Also, fangen wir mit dem Meeting an.",
           "it": "Ciao. Come va? Allora, iniziamo la riunione.",
           "nl": "Hallo. Hoe gaat het? Oké, laten we beginnen met de vergadering."}


@dataclass(frozen=True)
class Labels:
    you: str = "Você"
    remote: str = "Remoto"
    speaker: str = "Falante"


ENGLISH = Labels("You", "Remote", "Speaker")


class Cancelled(Exception):
    pass


@dataclass
class Line:
    start_ms: int
    end_ms: int
    speaker: str
    text: str


@dataclass
class Transcript:
    lines: list
    language: str
    duration_secs: int


def _check(abort):
    if abort is not None and abort.is_set():
        raise Cancelled("transcricao cancelada")


# ---------------------------------------------------------------------------
# Nivel e trechos com som

def frame_levels(track):
    """RMS de cada quadro de 30 ms (o ultimo, incompleto, conta com o que tem)."""
    if track.size == 0:
        return np.zeros(0, dtype=np.float32)
    frames = -(-track.size // FRAME)
    padded = np.zeros(frames * FRAME, dtype=np.float32)
    padded[:track.size] = track
    sums = np.einsum("ij,ij->i", padded.reshape(frames, FRAME), padded.reshape(frames, FRAME))
    counts = np.full(frames, FRAME, dtype=np.float32)
    counts[-1] = track.size - (frames - 1) * FRAME
    return np.sqrt(sums / counts)


def is_silent(track):
    """Abaixo de uns -50 dBFS nao ha fala pra achar, so alucinacao."""
    return track.size == 0 or float(np.max(np.abs(track))) < 0.003


def active_level(track):
    """Volume enquanto alguem fala: percentil 95 dos quadros, pausas nao puxam pra baixo."""
    levels = np.sort(frame_levels(track))
    return float(levels[len(levels) * 95 // 100]) if levels.size else 0.0


def soft_clip(x):
    """Deixa ate 0,8 intacto e curva o que passa disso em direcao a 1,0."""
    knee = 0.8
    over = np.abs(x) > knee
    out = x.copy()
    out[over] = np.sign(x[over]) * (knee + (1 - knee) * np.tanh((np.abs(x[over]) - knee) / (1 - knee)))
    return out


def level(track):
    """Leva a trilha a um volume de fala parecido; trilha que e so ruido fica como esta."""
    track = np.asarray(track, dtype=np.float32)
    current = active_level(track)
    if is_silent(track) or current < 0.003:
        return track.copy()
    return soft_clip(track * np.float32(np.clip(0.1 / current, 0.25, 8.0)))


@dataclass(frozen=True)
class Region:
    start: int  # com padding antes do som
    onset: int  # onde o som comeca
    end: int


def active_frames(track, frames):
    """Quadros com som: acima de 4x o chao de ruido da propria trilha."""
    active = np.zeros(frames, dtype=bool)
    energies = frame_levels(track)
    if energies.size == 0:
        return active
    floor = np.sort(energies)[energies.size // 10]
    threshold = max(floor * 4.0, 0.002)
    n = min(frames, energies.size)
    active[:n] = energies[:n] >= threshold
    return active


def regions_from(active, length):
    pad, merge_gap = RATE * 300 // 1000, RATE * 800 // 1000
    regions = []
    for i in np.flatnonzero(active):
        onset = int(i) * FRAME
        start, end = max(0, onset - pad), min(length, (int(i) + 1) * FRAME + pad)
        if regions and start <= regions[-1].end + merge_gap:
            last = regions[-1]
            regions[-1] = Region(last.start, last.onset, max(last.end, end))
        else:
            regions.append(Region(start, onset, end))
    return regions


def speech_regions(tracks, length):
    """Partes com som. Cada trilha tem o proprio limiar: musica de um lado nao esconde o outro."""
    frames = -(-length // FRAME)
    active = np.zeros(frames, dtype=bool)
    for track in tracks:
        active |= active_frames(track, frames)
    return regions_from(active, length)


def own_speech_regions(mic, computer):
    """Partes do microfone (ja nivelado) com a sua voz. Pelo alto-falante os outros vazam
    no mic, um pouco atrasados e sempre mais baixos: um quadro do mic so conta se tiver
    ao menos metade do volume do PC mais alto em volta, e so em sequencias de 3 quadros."""
    around, run = 3, 3
    frames = -(-mic.size // FRAME)
    own = frame_levels(mic)
    other = np.zeros(frames, dtype=np.float32)
    theirs = frame_levels(computer)[:frames]
    other[:theirs.size] = theirs
    active = active_frames(mic, frames)
    if frames:
        padded = np.pad(other, around)
        windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * around + 1)
        active &= own[:frames] * 2.0 >= windows.max(axis=1)
    i = 0
    while i < frames:
        if not active[i]:
            i += 1
            continue
        start = i
        while i < frames and active[i]:
            i += 1
        if i - start < run:
            active[start:i] = False
    return regions_from(active, mic.size)


def only(track, regions):
    """A trilha com tudo fora de `regions` em silencio."""
    out = np.zeros_like(track)
    for r in regions:
        out[r.start:r.end] = track[r.start:r.end]
    return out


# ---------------------------------------------------------------------------
# Quem fala

class Speakers:
    """Um lado da gravacao (a trilha e o falante, dividido quando ha varias vozes nela)
    ou as vozes achadas num arquivo importado."""

    def __init__(self, label, turns, side=True):
        self.label, self.turns, self.side = label, turns, side

    def speaker(self, start_ms, end_ms):
        if self.side and not self.turns:
            return self.label
        import sussurro_diarize as diarize
        return f"{self.label} {diarize.speaker_at(self.turns, start_ms, end_ms) + 1}"

    def takeover_ms(self, speaker, around_ms):
        """Onde `speaker` comeca a falar perto de `around_ms`, se da pra saber melhor que o whisper."""
        prefix = self.label + " "
        if not self.turns or not speaker.startswith(prefix) or not speaker[len(prefix):].isdigit():
            return None
        import sussurro_diarize as diarize
        index = int(speaker[len(prefix):]) - 1
        return diarize.turn_start_near(self.turns, index, around_ms) if index >= 0 else None

    def cuts_at_pauses(self):
        """Com diarizacao um segmento pode ter duas vozes sem ponto entre elas: corta na pausa."""
        return bool(self.turns) or not self.side


# ---------------------------------------------------------------------------
# Reconhecimento

@dataclass
class Word:
    text: str
    start_ms: int
    end_ms: int
    no_speech: float
    segment: int


def recognize(model, audio, language, progress=None, abort=None, hotwords=None):
    """Palavras com tempo real de `audio` (16 kHz, com o que nao e fala zerado) e o idioma."""
    lang = None if language == "auto" else language
    duration = max(audio.size / RATE, 1e-6)
    if hasattr(model, "model") and hasattr(model, "feature_extractor"):  # WhisperModel
        if lang is None:
            lang, _, _ = model.detect_language(audio, vad_filter=True)
        # Sequencial de proposito: em lote cada janela de 30 s vem sem o texto da anterior,
        # e janela que comeca no meio da frase sai sem pontuacao (a frase nao fecha e o
        # falante nao troca). Na GPU uma hora de reuniao ainda sai em menos de um minuto.
        segments, info = model.transcribe(audio, language=lang, beam_size=5, word_timestamps=True,
                                          vad_filter=True, initial_prompt=PROMPTS.get(lang),
                                          hotwords=hotwords)
    else:  # ParakeetModel e qualquer adaptador com a mesma cara
        segments, info = model.transcribe(audio, language=lang, vad_filter=True, word_timestamps=True)
    words = []
    for index, seg in enumerate(segments):
        _check(abort)
        for w in seg.words or []:
            text = w.word.strip()
            if text:
                words.append(Word(text, int(w.start * 1000), int(max(w.end, w.start) * 1000),
                                  float(seg.no_speech_prob), index))
        if progress:
            progress(min(1.0, seg.end / duration))
    detected = getattr(info, "language", None)
    return words, (detected if detected and detected != "auto" else None)


# ---------------------------------------------------------------------------
# Linhas

STOCK = {
    # o que o whisper diz quando so ouve silencio ou ruido, aprendido em legenda
    "thank you", "thank you very much", "thanks", "thanks for watching", "thank you for watching",
    "bye", "bye bye", "you", "okay", "so", "subtitles by the amaraorg community", "please subscribe",
    "obrigado", "obrigada", "muito obrigado", "obrigado por assistir", "tchau", "tchau tchau",
    "tchau tchau tchau", "até a próxima", "legendas pela comunidade amaraorg",
    "inscrevase no canal", "se inscreva no canal", "legenda adriana zanotto", "e aí", "é isso",
    "merci", "vielen dank", "gracias",
}


def _normalize(text):
    kept = "".join(c for c in text.lower() if c.isalnum() or c.isspace())
    return " ".join(kept.split())


def is_stock_phrase(text):
    norm = _normalize(text)
    return norm in STOCK or norm.startswith(("subtitles by", "legendas pela", "legenda por"))


def drop_hallucinations(segments, audio, rate=RATE):
    """Ditado: tira o "Tchau." / "Obrigado." que o whisper inventa no silencio do fim.
    So cai o segmento que e inteiro uma frase-padrao e esta bem abaixo da fala do
    proprio ditado (ou que o whisper achou que nem era fala); um "obrigado" dito de
    verdade tem o volume da sua voz e fica."""
    speech = active_level(np.asarray(audio, dtype=np.float32))
    kept = []
    for seg in segments:
        if is_stock_phrase(seg.text):
            part = audio[int(seg.start * rate):max(int(seg.end * rate), int(seg.start * rate) + 1)]
            loudness = float(np.sqrt(np.mean(part * part))) if part.size else 0.0
            if getattr(seg, "no_speech_prob", 0.0) > 0.3 or loudness < speech * 0.25:
                continue
        kept.append(seg)
    return kept


def is_noise_marker(text):
    """"[BLANK_AUDIO]", "(música)", "*aplausos*" e parecidos."""
    t = text.strip()
    wrapped = any(t.startswith(a) and t.endswith(b) for a, b in ("[]", "()", "**"))
    return wrapped or not any(c.isalnum() for c in t)


def ends_sentence(text):
    return text.rstrip().rstrip("\"')”’").endswith((".", "?", "!", "…"))


def _span_rms(track, start_ms, end_ms):
    part = track[start_ms * RATE // 1000:max(end_ms, start_ms + 1) * RATE // 1000]
    return float(np.sqrt(np.mean(part * part))) if part.size else 0.0


def _region_of(regions, starts, ms):
    """Indice do trecho de som que contem (ou fica mais perto de) `ms`; `starts` e o
    inicio de cada trecho, em ordem (busca binaria: reuniao longa tem milhares)."""
    sample = ms * RATE // 1000
    i = bisect.bisect_right(starts, sample) - 1
    candidates = [j for j in (i, i + 1) if 0 <= j < len(regions)]

    def distance(j):
        r = regions[j]
        return 0 if r.start <= sample < r.end else min(abs(sample - r.start), abs(sample - r.end))
    return min(candidates, key=distance)


def phrases(words, regions, speakers, track, paragraphs):
    """Palavras -> linhas: corta onde o falante muda, onde a frase termina e em cada trecho
    de silencio. Frase nunca e dividida entre falantes."""
    per_segment = {}
    for w in words:
        per_segment[w.segment] = per_segment.get(w.segment, 0) + 1

    pieces = []  # dicts: words, start, end, region, no_speech, segment
    starts = [r.start for r in regions]
    for w in words:
        region = _region_of(regions, starts, w.start_ms) if regions else 0
        start, end = w.start_ms, max(w.end_ms, w.start_ms)
        last = pieces[-1] if pieces else None
        sentence_ended = last is not None and ends_sentence(last["words"][-1])
        turn_at_pause = (speakers.cuts_at_pauses() and last is not None
                         and start - last["end"] >= 250
                         and speakers.speaker(last["start"], last["end"]) != speakers.speaker(start, end))
        gap = last is not None and start - last["end"] >= NEW_STRETCH_MS
        if (last is not None and last["region"] == region and last["segment"] == w.segment
                and not sentence_ended and not turn_at_pause and not gap):
            last["words"].append(w.text)
            last["end"] = max(end, last["end"])
            last["no_speech"] = max(last["no_speech"], w.no_speech)
        else:
            pieces.append({"words": [w.text], "start": start, "end": end, "region": region,
                           "no_speech": w.no_speech, "segment": w.segment})

    # As primeiras palavras de um trecho comecam onde o som comeca; o whisper tende a
    # po-las no inicio do padding.
    previous_region = None
    for p in pieces:
        if regions and p["region"] != previous_region:
            onset = regions[p["region"]].onset * 1000 // RATE
            if abs(p["start"] - onset) < 1500:
                shift = onset - p["start"]
                p["start"] = onset
                p["end"] += max(shift, 0)
        previous_region = p["region"]
        p["end"] = max(p["end"], p["start"] + 250 * len(p["words"]))  # ~250 ms por palavra

    # Pedaco que para sem fim de frase junta com o seguinte se vier logo depois.
    sentences = []
    for p in pieces:
        if sentences and not ends_sentence(sentences[-1]["words"][-1]) and p["start"] - sentences[-1]["end"] < 3000:
            prev = sentences[-1]
            prev["words"].extend(p["words"])
            prev["end"] = max(prev["end"], p["end"])
            prev["no_speech"] = max(prev["no_speech"], p["no_speech"])
        else:
            sentences.append(p)

    lines = []
    for p in sentences:
        text = " ".join(p["words"])
        whole = per_segment.get(p["segment"]) == len(p["words"])
        if is_noise_marker(text) or _hallucination(text, p, whole, track):
            continue
        speaker = speakers.speaker(p["start"], p["end"])
        changed = not lines or lines[-1].speaker != speaker
        if changed:
            takeover = speakers.takeover_ms(speaker, p["start"])
            if takeover is not None:
                floor = lines[-1].start_ms + 1 if lines else 0
                p["start"] = max(takeover, floor)
                p["end"] = max(p["end"], p["start"] + 1)
        last = lines[-1] if lines else None
        if (paragraphs and last is not None and last.speaker == speaker
                and (p["start"] - last.end_ms < PARAGRAPH_PAUSE_MS or not ends_sentence(last.text))
                and p["end"] - last.start_ms < PARAGRAPH_MAX_MS):
            last.text += " " + text
            last.end_ms = p["end"]
        else:
            lines.append(Line(p["start"], p["end"], speaker, text))
    return lines


def _hallucination(text, piece, whole, track):
    loudness = _span_rms(track, piece["start"], piece["end"])
    if loudness < 0.004 or piece["no_speech"] > 0.85:
        return True  # whisper falando por cima de silencio
    return whole and is_stock_phrase(text) and (piece["no_speech"] > 0.3 or loudness < 0.02)


def interleave(sentences, labels):
    """As frases dos dois lados na ordem em que foram ditas, em paragrafos por falante.
    Frase sua que repete o que o outro lado disse no mesmo momento e eco e sai."""
    def is_local(speaker):
        return speaker == labels.you or re.fullmatch(re.escape(labels.you) + r" \d+", speaker)

    def words(text):
        return [w for w in (re.sub(r"^\W+|\W+$", "", t).lower() for t in text.split()) if w]

    def trigrams(ws):
        return [" ".join(ws[i:i + 3]) for i in range(len(ws) - 2)]

    sentences = sorted(sentences, key=lambda s: s.start_ms)
    remote = [(s.start_ms, s.end_ms, words(s.text)) for s in sentences if not is_local(s.speaker)]
    remote_starts = [r[0] for r in remote]
    longest = max((end - start for start, end, _ in remote), default=0)

    def is_echo(mine):
        own_words = words(mine.text)
        # so as falas do outro lado que encostam nesta (+-2 s): busca binaria pelo inicio
        lo = bisect.bisect_left(remote_starts, mine.start_ms - 2000 - longest)
        hi = bisect.bisect_right(remote_starts, mine.end_ms + 2000)
        near = [ws for start, end, ws in remote[lo:hi] if mine.start_ms < end + 2000]
        own = trigrams(own_words)
        if not own:
            n = len(own_words)
            return bool(n) and any(theirs[i:i + n] == own_words for theirs in near
                                   for i in range(len(theirs) - n + 1))
        theirs = {t for ws in near for t in trigrams(ws)}
        return sum(t in theirs for t in own) * 2 >= len(own)

    keep = [not is_local(s.speaker) or not is_echo(s) for s in sentences]
    out = []
    for s, k in zip(sentences, keep):
        if not k:
            continue
        last = out[-1] if out else None
        if (last is not None and last.speaker == s.speaker
                and (s.start_ms - last.end_ms < PARAGRAPH_PAUSE_MS or not ends_sentence(last.text))
                and s.end_ms - last.start_ms < PARAGRAPH_MAX_MS):
            last.text += " " + s.text
            last.end_ms = max(last.end_ms, s.end_ms)
        else:
            out.append(Line(s.start_ms, s.end_ms, s.speaker, s.text))
    return out


# ---------------------------------------------------------------------------
# Pipelines

class Progress:
    """Junta etapa, fracao e linhas ao vivo num callback so: report(kind, value)."""

    def __init__(self, report=None):
        self.report = report or (lambda kind, value: None)

    def stage(self, text):
        self.report("stage", text)

    def fraction(self, value):
        self.report("progress", float(value))

    def line(self, text):
        self.report("line", text)


def _voices(track, diarizer, progress, abort):
    """Turnos quando ha mais de uma voz no lado; nada quando e uma pessoa so."""
    import sussurro_diarize as diarize
    if is_silent(track):
        return []
    turns = diarize.turns(track, None, diarizer=diarizer, abort=abort,
                          progress=lambda f: progress.fraction(f))
    return turns if any(t.speaker > 0 for t in turns) else []


def _open_diarizer(progress):
    import sussurro_diarize as diarize
    progress.stage("Preparando o modelo de vozes...")
    diarize.ensure_model(progress=lambda msg: progress.stage(msg))
    return diarize.Diarizer()


def transcribe_meeting(mic, computer, language, model, *, labels=Labels(), find_voices=True,
                       report=None, abort=None, lock=None, hotwords=None):
    """Transcreve uma reuniao gravada (duas trilhas 16 kHz). `lock` e o lock do modelo
    compartilhado com o ditado: so o whisper roda com ele tomado."""
    progress = Progress(report)
    mic, computer = np.asarray(mic, np.float32), np.asarray(computer, np.float32)
    duration = max(mic.size, computer.size) // RATE
    if is_silent(mic) and is_silent(computer):
        return Transcript([], language if language != "auto" else "unknown", duration)
    mic, computer = level(mic), level(computer)
    mic_regions = own_speech_regions(mic, computer)
    computer_regions = speech_regions([computer], computer.size)
    if not mic_regions and not computer_regions:
        return Transcript([], language if language != "auto" else "unknown", duration)

    local, remote = [], []
    if find_voices:
        diarizer = _open_diarizer(progress)
        try:
            progress.stage("Separando as vozes de quem está com você...")
            local = _voices(only(mic, mic_regions), diarizer, progress, abort)
            progress.stage("Separando as vozes do outro lado...")
            remote = _voices(computer, diarizer, progress, abort)
        finally:
            diarizer.close()

    length = lambda regions: sum(r.end - r.start for r in regions)
    total = max(length(mic_regions) + length(computer_regions), 1)
    sides = [(mic, mic_regions, Speakers(labels.you, local)),
             (computer, computer_regions, Speakers(labels.remote, remote))]
    # O lado com mais som primeiro: no automatico, o idioma dele vale pros dois.
    sides.sort(key=lambda side: -length(side[1]))
    lines, detected, done = [], None, 0.0
    for track, regions, speakers in sides:
        if not regions:
            continue
        share = length(regions) / total
        progress.stage("Transcrevendo " + ("você" if speakers.label == labels.you else "o outro lado") + "...")
        words, found = _recognize_locked(model, only(track, regions), language, lock, abort,
                                         lambda f, d=done, s=share: progress.fraction(d + s * f), hotwords)
        if language == "auto" and found:
            language, detected = found, found
        side_lines = phrases(words, regions, speakers, track, paragraphs=False)
        for line in side_lines:
            progress.line(f"{line.speaker}: {line.text}")
        lines.extend(side_lines)
        done += share
    progress.fraction(1.0)
    return Transcript(interleave(lines, labels), language if language != "auto" else (detected or "unknown"),
                      duration)


def transcribe_single(track, language, speakers, model, *, labels=Labels(), report=None,
                      abort=None, lock=None, hotwords=None):
    """Um arquivo so (importado): as vozes sao separadas pelo Nemotron. `speakers` fixa
    quantas pessoas falam; None deixa o modelo decidir, 1 pula a separacao."""
    import sussurro_diarize as diarize
    progress = Progress(report)
    track = np.asarray(track, np.float32)
    duration = track.size // RATE
    if is_silent(track):
        return Transcript([], language if language != "auto" else "unknown", duration)
    levelled = level(track)
    regions = speech_regions([track], track.size)
    if not regions:
        return Transcript([], language if language != "auto" else "unknown", duration)
    if speakers == 1:
        turns = diarize.single(track)
    else:
        diarizer = _open_diarizer(progress)
        try:
            progress.stage("Separando as vozes...")
            turns = diarize.turns(track, speakers, diarizer=diarizer, abort=abort,
                                  progress=lambda f: progress.fraction(f))
        finally:
            diarizer.close()
    who = Speakers(labels.speaker, turns, side=False)
    progress.stage("Transcrevendo...")
    words, found = _recognize_locked(model, only(levelled, regions), language, lock, abort,
                                     progress.fraction, hotwords)
    lines = phrases(words, regions, who, levelled, paragraphs=True)
    for line in lines:
        progress.line(f"{line.speaker}: {line.text}")
    progress.fraction(1.0)
    return Transcript(lines, language if language != "auto" else (found or "unknown"), duration)


def _recognize_locked(model, audio, language, lock, abort, fraction, hotwords=None):
    _check(abort)
    if lock is None:
        return recognize(model, audio, language, fraction, abort, hotwords)
    with lock:
        return recognize(model, audio, language, fraction, abort, hotwords)


# ---------------------------------------------------------------------------
# Saida

def clock(ms):
    secs = int(ms) // 1000
    h, m, s = secs // 3600, secs // 60 % 60, secs % 60
    return f"{h}:{m:02}:{s:02}" if h else f"{m:02}:{s:02}"


def to_markdown(title, date, transcript, names=None, headings=None):
    """`names` troca o rotulo pelo nome dado pelo usuario (Remoto 2 -> Marina)."""
    names = names or {}
    h = headings or {"date": "Data", "duration": "Duração", "language": "Idioma",
                     "transcript": "Transcrição", "empty": "_Nenhuma fala reconhecida._"}
    out = [f"# {title}", "", f"- **{h['date']}:** {date}",
           f"- **{h['duration']}:** {clock(transcript.duration_secs * 1000)}",
           f"- **{h['language']}:** {LANGUAGES.get(transcript.language, transcript.language)}", "",
           f"## {h['transcript']}", ""]
    if not transcript.lines:
        out.append(h["empty"])
    for line in transcript.lines:
        out += [f"**[{clock(line.start_ms)}] {names.get(line.speaker, line.speaker)}:** {line.text}", ""]
    return "\n".join(out).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Audio e linha de comando

def load_track(path):
    """Qualquer coisa que o ffmpeg le, em 16 kHz mono float32."""
    result = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path),
                             "-f", "f32le", "-ac", "1", "-ar", str(RATE), "-"],
                            capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg nao leu {path}: {result.stderr.decode(errors='replace').strip()}")
    return np.frombuffer(result.stdout, dtype="<f4").copy()


def load_cli_model(name):
    """Modelo para a linha de comando (bench): sempre na GPU."""
    from sussurro_cuda import _prepare_cuda_libs
    _prepare_cuda_libs()
    from sussurro_models import resolve_model_config, model_path
    config = resolve_model_config({"whisper_model": name, "whisper_device": "cuda"})
    if config.engine == "parakeet":
        import sussurro_parakeet
        return sussurro_parakeet.ParakeetModel(sussurro_parakeet.model_dir())
    from faster_whisper import WhisperModel
    from faster_whisper.utils import download_model
    return WhisperModel(model_path(config.model, download_model), device=config.device,
                        compute_type=config.compute_type)


def cli(argv):
    """sussurro_meeting.py transcribe <mic> <pc> | transcribe-file <audio> [--speakers N] |
    diarize <audio>, com [--language xx] [--model nome] [--labels en]. Mesmo formato do
    bench do omarchy-meeting-recorder, pra comparar os dois nos mesmos casos."""
    usage = ("uso: sussurro_meeting.py transcribe <mic> <pc> | transcribe-file <audio> [--speakers N]"
             " | diarize <audio>  [--language auto|pt|en] [--model nome] [--labels en]")
    if len(argv) < 2:
        print(usage, file=sys.stderr)
        return 2
    command, rest = argv[1], argv[2:]
    files, language, model_name, speakers, labels = [], "auto", "large-v3-turbo", None, Labels()
    it = iter(rest)
    for arg in it:
        if arg in ("--language", "-l"):
            language = next(it, "auto")
        elif arg in ("--model", "-m"):
            model_name = next(it, model_name)
        elif arg in ("--speakers", "-s"):
            speakers = int(next(it, "0")) or None
        elif arg == "--labels":
            labels = ENGLISH if next(it, "") == "en" else Labels()
        else:
            files.append(arg)
    started = time.perf_counter()

    def report(kind, value):
        if kind == "stage":
            print(f"[{time.perf_counter() - started:6.1f}s] {value}", file=sys.stderr)
        elif kind == "line":
            print(f"  {value}", file=sys.stderr)

    if command == "diarize" and len(files) == 1:
        import sussurro_diarize as diarize
        turns = diarize.turns(load_track(files[0]), speakers)
        print(json.dumps([{"speaker": t.speaker, "start": t.start_ms / 1000, "end": t.end_ms / 1000}
                          for t in turns]))
        return 0
    if command == "transcribe" and len(files) == 2:
        mic, computer = load_track(files[0]), load_track(files[1])
        transcript = transcribe_meeting(mic, computer, language, load_cli_model(model_name),
                                        labels=labels, report=report)
    elif command == "transcribe-file" and len(files) == 1:
        transcript = transcribe_single(load_track(files[0]), language, speakers,
                                       load_cli_model(model_name), labels=labels, report=report)
    else:
        print(usage, file=sys.stderr)
        return 2
    print(to_markdown("Transcript", time.strftime("%Y-%m-%d %H:%M"), transcript), end="")
    print(f"Pronto em {time.perf_counter() - started:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli(sys.argv))
