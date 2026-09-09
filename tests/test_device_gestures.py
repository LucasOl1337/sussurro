"""Replay de estados do microfone, sem audio real ou ativacao do Sussurro."""
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sussurro_devices as devices


class MicrophoneGestureTests(unittest.TestCase):
    def replay(self, frames):
        callback = Mock()
        detector = devices.DeviceGestures({**devices.DEFAULTS, 'mic_follow': False,
                                          'mute_gesture': True}, callback)
        now = [100.0]
        data = [b'']
        frames = iter(frames)
        def select_frame(*args):
            try:
                now[0], data[0] = next(frames)
            except StopIteration:
                detector._stop.set()
                return [], [], []
            return ([123] if data[0] is not None else []), [], []
        with patch.object(devices.os, 'set_blocking'), \
             patch.object(devices.os, 'read', side_effect=lambda *a: data[0]), \
             patch.object(devices.select, 'select', side_effect=select_frame), \
             patch.object(devices.time, 'time', side_effect=lambda: now[0]):
            detector._mic_session(123, 'fake-source')
        return callback

    def test_audio_dropout_does_not_activate_dictation(self):
        sound = b'\x01' * devices.CHUNK_BYTES
        zero = bytes(devices.CHUNK_BYTES)
        callback = self.replay([(100.0, sound), (101.0, zero),
                                (101.5, None), (102.0, None), (103.0, sound)])
        callback.assert_not_called()

    def test_mute_then_audio_resume_still_activates(self):
        sound = b'\x01' * devices.CHUNK_BYTES
        zero = bytes(devices.CHUNK_BYTES)
        callback = self.replay([(100.0, sound), (101.0, zero),
                                (102.0, sound), (102.4, sound)])
        callback.assert_called_once()

    def test_dropout_discards_pending_mute(self):
        sound = b'\x01' * devices.CHUNK_BYTES
        zero = bytes(devices.CHUNK_BYTES)
        callback = self.replay([(100.0, sound), (101.0, zero),
                                (102.0, sound), (102.5, None), (103.0, sound)])
        callback.assert_not_called()


if __name__ == '__main__':
    unittest.main()
