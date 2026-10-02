"""Tons curtos opcionais, fora da thread de captura e sem interromper playback."""
import logging
import threading

import numpy as np
import sounddevice as sd

_RATE = 24000
_LOCK = threading.Lock()
_FREQUENCIES = {'start': (660, 880), 'stop': (880, 660), 'error': (330, 220)}


def _tone(event):
    frequencies = _FREQUENCIES[event]
    samples = int(.07 * _RATE)
    time = np.arange(samples, dtype=np.float32) / _RATE
    envelope = np.sin(np.linspace(0, np.pi, samples, dtype=np.float32)) ** 2
    parts = [.06 * np.sin(2 * np.pi * frequency * time) * envelope
             for frequency in frequencies]
    return np.concatenate((parts[0], np.zeros(int(.02 * _RATE), dtype=np.float32), parts[1]))


def _play(event):
    if not _LOCK.acquire(blocking=False):
        return
    try:
        try:
            stream = sd.get_stream()
        except RuntimeError:  # nenhum play/rec de conveniencia desde a abertura
            stream = None
        if stream is not None and stream.active:
            return  # sd.play para o playback anterior, entao nunca o interrompemos
        sd.play(_tone(event), _RATE, blocking=False)
    except Exception:
        logging.getLogger(__name__).debug('Som de feedback indisponivel', exc_info=True)
    finally:
        _LOCK.release()


def play(event, *, enabled=False):
    if not enabled or event not in _FREQUENCIES:
        return
    threading.Thread(target=_play, args=(event,), daemon=True).start()
