"""Gasto da nuvem: o que cada provedor cobrou, contado aqui, e o saldo da ElevenLabs.

A ElevenLabs devolve o custo de cada transcricao nos cabecalhos (`character-cost` em
creditos e `fiat-cost-before-overages` em dolar); o Grok nao devolve custo, entao dele
so contam pedidos e minutos de audio. O saldo da conta ElevenLabs vem de
`/v1/user/subscription`, que exige a permissao `user_read` na chave.

O livro fica em `cloud-usage.json` ao lado do app, por mes, sem texto nem audio.
"""
import json
import threading
import time
from datetime import datetime
from pathlib import Path

from sussurro_cloud import ELEVENLABS, PROVIDERS, XAI

USAGE_PATH = Path(__file__).with_name("cloud-usage.json")
SUBSCRIPTION_URL = "https://api.elevenlabs.io/v1/user/subscription"
BALANCE_MAX_AGE_S = 600  # saldo mais velho que isso e buscado de novo; entre buscas, o gasto local desconta
EMPTY = {"calls": 0, "audio_s": 0.0, "credits": 0, "usd": 0.0}


def month_key(now=None):
    return (now or datetime.now()).strftime("%Y-%m")


def charge_from_headers(headers):
    """(creditos, dolares) do pedido; (0, 0.0) quando o provedor nao informa."""
    headers = headers or {}
    try:
        credits = int(float(headers.get("character-cost") or 0))
    except (TypeError, ValueError):
        credits = 0
    try:
        usd = float(headers.get("fiat-cost-before-overages") or 0)
    except (TypeError, ValueError):
        usd = 0.0
    return credits, usd


class UsageLedger:
    """Contador por mes e provedor, gravado a cada pedido (poucos por minuto)."""

    def __init__(self, path=USAGE_PATH):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._months = self._read()
        self.balance = None        # dict da ElevenLabs ou None
        self.balance_error = None  # motivo curto quando o saldo nao vem
        self._balance_at = 0.0
        self._balance_busy = False
        self.version = 0  # sobe a cada gasto ou saldo novo: a UI redesenha quando muda

    def _read(self):
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            months = data.get("months", {})
            return months if isinstance(months, dict) else {}
        except (OSError, ValueError, AttributeError):
            return {}

    def record(self, provider, audio_s, headers=None, now=None):
        if provider not in PROVIDERS:
            return
        credits, usd = charge_from_headers(headers)
        with self._lock:
            row = self._months.setdefault(month_key(now), {}).setdefault(provider, dict(EMPTY))
            row["calls"] += 1
            row["audio_s"] = round(row["audio_s"] + float(audio_s), 2)
            row["credits"] += credits
            row["usd"] = round(row["usd"] + usd, 6)
            if self.balance is not None and credits:
                self.balance["used"] += credits  # saldo na tela anda junto ate a proxima busca
            self.version += 1
            try:
                tmp = self.path.with_suffix(".tmp")
                tmp.write_text(json.dumps({"version": 1, "months": self._months}, indent=1), encoding="utf-8")
                tmp.replace(self.path)
            except OSError:
                pass  # gasto e informativo: disco cheio nao derruba o ditado

    def month(self, provider, now=None):
        with self._lock:
            return dict(self._months.get(month_key(now), {}).get(provider, EMPTY))

    def total(self, provider):
        with self._lock:
            out = dict(EMPTY)
            for rows in self._months.values():
                for key, value in rows.get(provider, {}).items():
                    out[key] = out.get(key, 0) + value
            return out

    # -- saldo ElevenLabs ----------------------------------------------------
    def refresh_balance(self, secrets=None, http=None, force=False):
        """Busca o saldo em segundo plano, no maximo uma vez por minuto."""
        with self._lock:
            if self._balance_busy or (not force and time.monotonic() - self._balance_at < BALANCE_MAX_AGE_S):
                return
            self._balance_busy = True
        threading.Thread(target=self._fetch_balance, args=(secrets, http),
                         name="sussurro-saldo", daemon=True).start()

    def _fetch_balance(self, secrets, http):
        balance, error = None, None
        try:
            balance = elevenlabs_balance(secrets, http)
        except Exception as e:  # rede, chave sem user_read: vira texto na tela
            error = str(e)
        with self._lock:
            self.balance, self.balance_error = balance, error
            self._balance_at = time.monotonic()
            self._balance_busy = False
            self.version += 1


_shared = None
_shared_lock = threading.Lock()


def shared_ledger():
    """Um livro por processo: ditado, Refazer e COMPARAR somam no mesmo arquivo."""
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = UsageLedger()
        return _shared


def elevenlabs_balance(secrets=None, http=None):
    """{'used', 'limit', 'reset' (epoch), 'tier'} da assinatura ElevenLabs."""
    from sussurro_cloud import Credentials
    secrets = secrets or Credentials()
    if not secrets.has(ELEVENLABS):
        raise RuntimeError("sem chave da ElevenLabs")
    key = secrets.elevenlabs_key()
    if http is None:
        import httpx
        http = httpx
    response = http.get(SUBSCRIPTION_URL, headers={"xi-api-key": key}, timeout=10)
    if response.status_code != 200:
        try:
            detail = response.json().get("detail", {})
        except (ValueError, AttributeError):
            detail = {}
        if isinstance(detail, dict) and detail.get("status") == "missing_permissions":
            raise RuntimeError("saldo: falta a permissão user_read na chave")
        raise RuntimeError(f"saldo indisponível (HTTP {response.status_code})")
    body = response.json()
    return {"used": int(body.get("character_count") or 0), "limit": int(body.get("character_limit") or 0),
            "reset": body.get("next_character_count_reset_unix"), "tier": body.get("tier") or ""}


def fmt_minutes(seconds):
    minutes = seconds / 60
    return f"{minutes:.1f} min" if minutes < 10 else f"{minutes:.0f} min"


def fmt_credits(n):
    return f"{int(n):,}".replace(",", ".")


def fmt_usd(value):
    return f"US$ {value:.2f}".replace(".", ",") if value >= 0.01 or not value else "< US$ 0,01"


def spend_line(ledger, provider):
    """Uma linha pro card de configuracao: o que esse provedor gastou no mes."""
    m = ledger.month(provider)
    if provider == ELEVENLABS:
        line = f"Este mês: {fmt_credits(m['credits'])} créditos · {fmt_usd(m['usd'])}"
        if ledger.balance and ledger.balance["limit"]:
            left = max(0, ledger.balance["limit"] - ledger.balance["used"])
            line += f" · restam {fmt_credits(left)} de {fmt_credits(ledger.balance['limit'])}"
        elif ledger.balance_error:
            line += f"\n{ledger.balance_error}"
        return line
    if provider == XAI:
        return (f"Este mês: {fmt_minutes(m['audio_s'])} em {m['calls']} pedidos"
                " · a xAI não informa custo nem saldo pela API")
    return ""
