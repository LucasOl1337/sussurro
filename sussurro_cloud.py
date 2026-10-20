"""Transcricao na nuvem (opcional): xAI Grok e ElevenLabs Scribe.

`CloudModel.transcribe()` tem a mesma cara do faster-whisper (segmentos com palavras e
tempos), entao ditado, Refazer, IPC e COMPARAR usam a nuvem sem caminho paralelo.

So entra em uso quando o usuario escolhe um modelo de nuvem no seletor: com Turbo ou
Large-v3 nenhum audio sai do computador. Se o provedor escolhido falhar, tenta o outro
(quando ha credencial) e so entao levanta `CloudError`, que o app usa pra cair no
Whisper local.

Credenciais, sem nada salvo em settings.json:
- xAI: `XAI_API_KEY` ou o login do Grok CLI (`~/.grok/auth.json`, o mesmo do SuperGrok).
  Token perto de vencer e renovado pelo proprio CLI, dono das travas do arquivo.
- ElevenLabs: `ELEVENLABS_API_KEY` ou o chaveiro (`secret-tool lookup servico elevenlabs
  tipo api-key`).

Reuniao: `transcribe(..., diarize=True, compress=True)` manda o audio em Ogg/Opus (uma
hora de fala vira ~15 MB) e pede ao provedor quem fala cada palavra (`Word.speaker`).
"""
import json
import os
import re
import select
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SAMPLE_RATE = 16000
SPLIT_GAP_S = 1.5  # pausa que vira segmento separado (o app transforma em paragrafo)

XAI = "cloud-xai"
ELEVENLABS = "cloud-elevenlabs"
PROVIDERS = (ELEVENLABS, XAI)
PROVIDER_NAMES = {XAI: "Grok (xAI)", ELEVENLABS: "ElevenLabs Scribe"}

XAI_URL = "https://api.x.ai/v1/stt"
XAI_MODEL = "grok-voice-transcribe-2.0"
ELEVENLABS_URL = "https://api.elevenlabs.io/v1/speech-to-text"
ELEVENLABS_MODEL = "scribe_v2"
GROK_AUTH = Path.home() / ".grok" / "auth.json"
GROK_ISSUER = "https://auth.x.ai"


class CloudError(RuntimeError):
    """Nuvem indisponivel (rede, credencial, cota): o app cai no Whisper local."""


@dataclass
class Word:
    start: float
    end: float
    word: str
    probability: float = 1.0
    speaker: object = None  # so com diarize: id do falante como o provedor manda


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list = field(default_factory=list)
    no_speech_prob: float = 0.0
    avg_logprob: float = 0.0


def is_cloud(model: str) -> bool:
    return model in PROVIDERS


ISO3 = {"por": "pt", "eng": "en", "spa": "es", "fra": "fr", "deu": "de", "ita": "it", "nld": "nl"}


def short_language(code):
    """'por' (ElevenLabs) e 'pt-BR'/'es-mx' (xAI) viram 'pt'/'es'; vazio vira None."""
    if not code or code == "auto":
        return None
    base = str(code).split("-")[0].split("_")[0].lower()
    return ISO3.get(base, base)


def keyterms(hotwords) -> list:
    """Termos da Biblioteca no formato aceito pelos dois provedores (<50 chars, ate 5 palavras)."""
    if not hotwords:
        return []
    terms = hotwords if isinstance(hotwords, (list, tuple)) else hotwords.split(", ")
    out = []
    for term in terms:
        term = " ".join(str(term).split())
        if (term and len(term) < 50 and len(term.split()) <= 5
                and not any(c in term for c in "<>{}[]\\") and term not in out):
            out.append(term)
    return out[:100]  # ElevenLabs cobra no minimo 20 s por pedido acima de 100 termos


def pcm16(audio) -> bytes:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    return (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()


def wav_bytes(audio) -> bytes:
    import io
    import wave
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm16(audio))
    return buffer.getvalue()


def opus_bytes(audio):
    """Ogg/Opus de voz (32 kbps) pelo ffmpeg; None sem ffmpeg ou sem libopus."""
    if not shutil.which("ffmpeg"):
        return None
    raw = np.asarray(audio, dtype="<f4").reshape(-1).tobytes()
    try:
        result = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "f32le",
                                 "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", "pipe:0",
                                 "-c:a", "libopus", "-b:a", "32k", "-application", "voip",
                                 "-f", "ogg", "pipe:1"], input=raw, capture_output=True, timeout=600)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 and result.stdout else None


def segments_from_words(items, text, duration, gap=SPLIT_GAP_S):
    """Agrupa as palavras em segmentos, cortando nas pausas longas e na troca de falante.

    `items` sao (texto, inicio, fim, e_palavra[, falante]): pecas sem tempo ou de espaco
    (ElevenLabs manda o espaco como item proprio) so entram no texto do segmento.
    """
    segments, current, pieces, last_end, last_speaker = [], [], [], None, None

    def close():
        body = "".join(pieces).strip()
        if body and current:
            segments.append(Segment(current[0].start, current[-1].end, " " + body, list(current)))

    for item in items:
        piece, start, end, is_word = item[:4]
        speaker = item[4] if len(item) > 4 else None
        if (is_word and start is not None and last_end is not None
                and (start - last_end >= gap or speaker != last_speaker)):
            close()
            current, pieces = [], []
        pieces.append(piece)
        if is_word and start is not None:
            current.append(Word(float(start), float(max(end, start)), piece.strip(), speaker=speaker))
            last_end = float(max(end, start))
            last_speaker = speaker
    close()
    if not segments and text and text.strip():
        segments = [Segment(0.0, duration, " " + text.strip())]
    return segments


class CloudModel:
    """Adaptador com `transcribe()` compativel com o faster-whisper."""

    supports_diarization = True  # a reuniao usa os falantes do provedor no lugar do Nemotron

    def __init__(self, provider, *, http=None, secrets=None, fallback=True, usage=None):
        if provider not in PROVIDERS:
            raise ValueError(f"provedor de nuvem desconhecido: {provider}")
        self.provider = provider
        self.fallback = fallback  # COMPARAR desliga: cada linha tem que ser do provedor dela
        self.last_provider = None
        self._secrets = secrets or Credentials()
        if http is None:
            import httpx
            http = httpx.Client(http2=False, headers={"User-Agent": "Sussurro"})
            if usage is None:  # so pedido de verdade entra no livro de gasto
                from sussurro_usage import shared_ledger
                usage = shared_ledger()
        self._http = http
        self._usage = usage
        self._last_headers = None  # cabecalhos da ultima resposta: a ElevenLabs manda o custo neles

    def available(self):
        """Provedores com credencial, o escolhido primeiro."""
        order = [self.provider]
        if self.fallback:
            order += [p for p in PROVIDERS if p != self.provider]
        return [p for p in order if self._secrets.has(p)]

    def check(self):
        if not self._secrets.has(self.provider):
            raise CloudError(self._secrets.missing_message(self.provider))

    def transcribe(self, audio, language=None, beam_size=5, vad_filter=False, hotwords=None,
                   word_timestamps=False, diarize=False, speakers=None, compress=False, **_):
        """`diarize` pede o falante de cada palavra (`speakers` fixa quantos, so ElevenLabs);
        `compress` manda Opus em vez de PCM, para audio longo de reuniao."""
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        duration = audio.size / SAMPLE_RATE
        info = SimpleNamespace(language=language or "auto", language_probability=0.0, duration=duration)
        if audio.size < SAMPLE_RATE // 10:  # ElevenLabs recusa < 100 ms; toque acidental
            return [], info
        terms = keyterms(hotwords)
        packed = opus_bytes(audio) if compress else None
        options = {"diarize": diarize, "speakers": speakers, "packed": packed, "long": compress}
        errors = []
        for provider in self.available() or [self.provider]:
            try:
                call = self._xai if provider == XAI else self._elevenlabs
                segments, detected = call(audio, duration, language, terms, **options)
            except CloudError as error:
                errors.append(f"{PROVIDER_NAMES[provider]}: {error}")
                continue
            except Exception as error:  # resposta fora do formato: tambem cai no proximo / local
                errors.append(f"{PROVIDER_NAMES[provider]}: resposta invalida ({type(error).__name__})")
                continue
            self.last_provider = provider
            if self._usage is not None:
                self._usage.record(provider, duration, self._last_headers)
            info.language = detected or info.language
            return segments, info
        raise CloudError("; ".join(errors) or self._secrets.missing_message(self.provider))

    # -- provedores ---------------------------------------------------------
    def _timeout(self, duration, long=False):
        import httpx
        if long:  # reuniao: upload maior e o provedor ainda separa as vozes
            return httpx.Timeout(60.0 + duration * 0.5, connect=10.0)
        # medido em 20 ditados reais: 60 s de fala voltam em ate ~5 s
        return httpx.Timeout(6.0 + duration * 0.2, connect=4.0)

    def _post(self, url, duration, long=False, **kwargs):
        import httpx
        try:
            response = self._http.post(url, timeout=self._timeout(duration, long), **kwargs)
            self._last_headers = getattr(response, "headers", None)
        except httpx.TimeoutException as error:
            raise CloudError("sem resposta a tempo") from error
        except httpx.HTTPError as error:
            raise CloudError(f"falha de rede ({type(error).__name__})") from error
        return response

    @staticmethod
    def _fail(response):
        try:
            detail = response.json()
            detail = detail.get("detail", detail) if isinstance(detail, dict) else detail
            if isinstance(detail, dict):
                detail = detail.get("message") or detail.get("status") or detail
            detail = detail.get("error", detail) if isinstance(detail, dict) else detail
        except ValueError:
            detail = response.text
        # a mensagem vai pra barra de status: nada que pareca chave ou token sai daqui
        detail = re.sub(r"[A-Za-z0-9_\-.]{20,}", "***", str(detail))
        raise CloudError(f"HTTP {response.status_code}: {detail[:160]}")

    def _xai(self, audio, duration, language, terms, diarize=False, speakers=None, packed=None, long=False):
        data = {"model": XAI_MODEL}
        if language:  # format=true sem idioma volta HTTP 400
            data.update(language=language, format="true")
        if terms:
            data["keyterm"] = terms  # httpx repete o campo pra cada item
        if diarize:
            data["diarize"] = "true"
        upload = ("reuniao.ogg", packed, "audio/ogg") if packed else ("ditado.wav", wav_bytes(audio), "audio/wav")
        files = {"file": upload}

        def post(token):
            return self._post(XAI_URL, duration, long, data=data, files=files,
                              headers={"Authorization": f"Bearer {token}"})

        token = self._secrets.xai_token()
        response = post(token)
        if _auth_refused(response) and self._secrets.xai_is_login():
            # relogio adiantado ou token revogado perto do fim: renova se estiver nos ultimos
            # 10 min e tenta uma vez; token novo longe de vencer nao adianta repetir
            fresh = self._secrets.xai_token(margin=600)
            if fresh != token:
                response = post(fresh)
        if response.status_code != 200:
            self._fail(response)
        body = response.json()
        items = [(w.get("text", "") + " ", w.get("start"), w.get("end"), True,
                  w.get("speaker") if diarize else None)
                 for w in body.get("words") or []]
        return segments_from_words(items, body.get("text", ""), duration), body.get("language")

    def _elevenlabs(self, audio, duration, language, terms, diarize=False, speakers=None, packed=None,
                    long=False):
        data = {"model_id": ELEVENLABS_MODEL, "tag_audio_events": "false",
                "timestamps_granularity": "word"}
        if language:
            data["language_code"] = {"pt": "por", "en": "eng"}.get(language, language)
        if terms:
            data["keyterms"] = terms
        if diarize:
            data["diarize"] = "true"
            if speakers:
                data["num_speakers"] = str(int(speakers))
        if packed:
            upload = ("reuniao.ogg", packed, "audio/ogg")
        else:
            data["file_format"] = "pcm_s16le_16"
            upload = ("ditado.pcm", pcm16(audio), "application/octet-stream")
        response = self._post(ELEVENLABS_URL, duration, long, data=data, files={"file": upload},
                              headers={"xi-api-key": self._secrets.elevenlabs_key()})
        if response.status_code != 200:
            self._fail(response)
        body = response.json()
        items = [(w.get("text", ""), w.get("start"), w.get("end"), w.get("type") == "word",
                  w.get("speaker_id") if diarize and w.get("type") == "word" else None)
                 for w in body.get("words") or [] if w.get("type") != "audio_event"]
        return segments_from_words(items, body.get("text", ""), duration), body.get("language_code")

    def close(self):
        try:
            self._http.close()
        except Exception:
            pass


def require_credentials(provider, secrets=None):
    """Falha cedo (antes de soltar o modelo local) quando o provedor nao tem credencial."""
    secrets = secrets or Credentials()
    if not secrets.has(provider):
        raise CloudError(secrets.missing_message(provider))


def _auth_refused(response):
    # xAI responde chave invalida/vencida com 400 "Incorrect API key"
    if response.status_code in (401, 403):
        return True
    return response.status_code == 400 and "key" in response.text.lower()


class Credentials:
    """Le as credenciais na hora do uso; nunca grava nem imprime o valor."""

    def __init__(self, env=None, auth_path=GROK_AUTH, grok_bin=None):
        self._env = os.environ if env is None else env
        self._auth_path = Path(auth_path)
        self._grok_bin = grok_bin
        self._lock = threading.Lock()
        self._elevenlabs = None

    # -- disponibilidade -----------------------------------------------------
    def has(self, provider):
        try:
            if provider == XAI:
                return bool(self._env.get("XAI_API_KEY") or self._grok_login())
            return bool(self.elevenlabs_key(quiet=True))
        except Exception:
            return False

    @staticmethod
    def missing_message(provider):
        if provider == XAI:
            return "sem login do Grok: rode `grok login` ou defina XAI_API_KEY"
        return ("sem chave da ElevenLabs: defina ELEVENLABS_API_KEY ou guarde com "
                "`secret-tool store --label='ElevenLabs (Sussurro)' servico elevenlabs tipo api-key`")

    # -- ElevenLabs ----------------------------------------------------------
    def elevenlabs_key(self, quiet=False):
        key = self._env.get("ELEVENLABS_API_KEY") or self._elevenlabs
        if not key and shutil.which("secret-tool"):
            try:
                key = subprocess.run(["secret-tool", "lookup", "servico", "elevenlabs", "tipo", "api-key"],
                                     capture_output=True, text=True, timeout=5).stdout.strip()
            except (OSError, subprocess.SubprocessError):
                key = ""
            self._elevenlabs = key or None
        if not key and not quiet:
            raise CloudError(self.missing_message(ELEVENLABS))
        return key

    # -- xAI -----------------------------------------------------------------
    def xai_is_login(self):
        return not self._env.get("XAI_API_KEY")

    def _grok_login(self):
        try:
            entries = json.loads(self._auth_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return next((v for v in entries.values()
                     if isinstance(v, dict) and v.get("oidc_issuer") == GROK_ISSUER and v.get("key")), None)

    @staticmethod
    def _expires(login):
        # "2026-10-10T16:07:24.250057481Z": sempre UTC, nanossegundos nao importam
        try:
            stamp = datetime.strptime(str(login["expires_at"])[:19], "%Y-%m-%dT%H:%M:%S")
        except (KeyError, ValueError):
            return 0.0
        return stamp.replace(tzinfo=timezone.utc).timestamp()

    def xai_token(self, margin=120):
        key = self._env.get("XAI_API_KEY")
        if key:
            return key
        with self._lock:
            login = self._grok_login()
            if login is None:
                raise CloudError(self.missing_message(XAI))
            if self._expires(login) < time.time() + margin:
                self._renew()
                login = self._grok_login()
                if login is None or self._expires(login) < time.time() + 30:
                    raise CloudError("login do Grok vencido e o CLI nao renovou: abra o `grok` uma vez")
            return login["key"]

    def _grok_command(self):
        if self._grok_bin:
            return [str(self._grok_bin)]
        native = Path.home() / ".grok" / "bin" / "grok"
        if native.exists():
            return [str(native)]
        found = shutil.which("grok")
        if not found:
            raise CloudError("login do Grok vencido e o CLI `grok` nao esta instalado")
        return [found]

    def _renew(self):
        """Pede ao CLI pra subir e responder `initialize`: ele renova o token e solta as travas."""
        try:
            process = subprocess.Popen(self._grok_command() + ["agent", "--no-leader", "stdio"],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.DEVNULL, text=True)
        except OSError as error:
            raise CloudError(f"nao consegui abrir o CLI do Grok ({error})") from error
        try:
            process.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                            "params": {"protocolVersion": 1, "clientCapabilities": {}}}) + "\n")
            process.stdin.flush()
            deadline = time.monotonic() + 25
            while time.monotonic() < deadline:
                if select.select([process.stdout], [], [], 1)[0]:
                    line = process.stdout.readline()
                    if not line:
                        break
                    try:
                        if json.loads(line).get("id") == 1:
                            break
                    except ValueError:
                        continue
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
