"""Sons testados sem PortAudio real, captura ou janela do Lucas."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import app
import sussurro_sounds as sounds
import test_bar_feedback


class SoundTests(unittest.TestCase):
    def test_default_and_existing_preferences_are_silent(self):
        self.assertFalse(app.DEFAULT_SETTINGS['feedback_sounds'])
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'settings.json'
            file.write_text('{"language":"en"}')
            with patch.object(app, 'SETTINGS_PATH', file):
                self.assertFalse(app.load_settings()['feedback_sounds'])
        with patch.object(sounds.threading, 'Thread') as thread:
            sounds.play('start')
            sounds.play('stop', enabled=False)
            sounds.play('unknown', enabled=True)
            thread.assert_not_called()

    def test_enabled_tone_is_dispatched_to_daemon_thread(self):
        with patch.object(sounds.threading, 'Thread') as thread:
            sounds.play('start', enabled=True)
            thread.assert_called_once_with(target=sounds._play, args=('start',), daemon=True)
            thread.return_value.start.assert_called_once()

    def test_tones_are_short_distinct_float32_and_low_volume(self):
        tones = [sounds._tone(event) for event in ('start', 'stop', 'error')]
        for tone in tones:
            self.assertEqual(tone.dtype, np.float32)
            self.assertEqual(tone.ndim, 1)
            self.assertLessEqual(len(tone) / sounds._RATE, .2)
            self.assertLessEqual(np.max(np.abs(tone)), .061)
            self.assertEqual(tone[0], 0.)
            self.assertAlmostEqual(tone[-1], 0., places=6)
        self.assertFalse(np.array_equal(tones[0], tones[1]))
        self.assertFalse(np.array_equal(tones[1], tones[2]))

    def test_output_is_nonblocking_and_does_not_stop_capture(self):
        with patch.object(sounds.sd, 'get_stream', side_effect=RuntimeError('sem stream')), \
             patch.object(sounds.sd, 'play') as play, \
             patch.object(sounds.sd, 'stop') as stop:
            sounds._play('start')
            self.assertEqual(play.call_args.args[1], sounds._RATE)
            self.assertEqual(play.call_args.kwargs, {'blocking': False})
            stop.assert_not_called()

    def test_active_playback_or_other_feedback_is_not_interrupted(self):
        with patch.object(sounds.sd, 'get_stream', return_value=SimpleNamespace(active=True)), \
             patch.object(sounds.sd, 'play') as play:
            sounds._play('error')
            play.assert_not_called()
            sounds._LOCK.acquire()
            try:
                sounds._play('start')
                play.assert_not_called()
            finally:
                sounds._LOCK.release()

    def test_device_failure_is_swallowed_and_releases_lock(self):
        with patch.object(sounds.sd, 'get_stream', return_value=None), \
             patch.object(sounds.sd, 'play', side_effect=RuntimeError('saida ausente')):
            sounds._play('error')
        self.assertTrue(sounds._LOCK.acquire(blocking=False))
        sounds._LOCK.release()


class AppSoundTests(unittest.TestCase):
    def ui(self):
        ui = test_bar_feedback.AppFeedbackTests().ui()
        ui.settings['feedback_sounds'] = True
        ui.hotkey = SimpleNamespace(active=True)
        return ui

    def test_switch_persists_without_playing_preview(self):
        ui = self.ui()
        ui.sounds_var = Mock()
        ui.sounds_var.get.return_value = False
        ui._save = Mock()
        with patch.object(sounds, 'play') as play:
            ui._on_sounds()
            self.assertFalse(ui.settings['feedback_sounds'])
            ui._save.assert_called_once()
            play.assert_not_called()

    def test_start_stop_and_error_trigger_tones_only_after_result(self):
        ui = self.ui()
        with patch.object(sounds, 'play') as play:
            self.assertTrue(ui._start(inject=True))
            play.assert_called_once_with('start', enabled=True)
            ui.transcriber.recording.set()
            play.reset_mock()
            ui._stop()
            play.assert_called_once_with('stop', enabled=True)
            ui.transcriber.recording.clear()
            play.reset_mock()
            ui.transcriber.start.side_effect = RuntimeError('mic falhou')
            self.assertFalse(ui._start(inject=True))
            play.assert_called_once_with('error', enabled=True)

    def test_busy_and_repeated_stop_do_not_play_success(self):
        ui = self.ui()
        ui.transcriber.model_loading.set()
        with patch.object(sounds, 'play') as play:
            self.assertFalse(ui._start(inject=True))
            ui._stop()
            play.assert_not_called()

    def test_repeated_start_during_recording_does_not_play_or_reopen_capture(self):
        ui = self.ui()
        ui.transcriber.recording.set()
        with patch.object(sounds, 'play') as play:
            self.assertTrue(ui._start(inject=True))
            ui.transcriber.start.assert_not_called()
            play.assert_not_called()

    def test_async_error_plays_error_but_normal_status_is_silent(self):
        ui = self.ui()
        ui.status_queue.put('ERRO: captura parou')
        ui.status_queue.put('Transcrito.')
        with patch.object(sounds, 'play') as play:
            ui._poll()
            play.assert_called_once_with('error', enabled=True)


if __name__ == '__main__':
    unittest.main()
