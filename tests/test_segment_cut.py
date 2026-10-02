"""Overflow de ditado corta em pausa do VAD, conservando audio e sid."""
import queue
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app


class SegmentCutTests(unittest.TestCase):
    def segment(self, audio, spans, mode='simultaneo'):
        with patch.object(threading.Thread, 'start'), patch.object(app.keyboard, 'Controller'):
            t = app.Transcriber(queue.Queue(), queue.Queue())
        t._session_mode = mode
        events = iter([(t._session_id, audio), (t._session_id, None)])
        t._audio_queue.get = lambda: next(events)
        def vad(buffer, options):
            if buffer.size == audio.size:
                return spans
            return [{'start': spans[-1]['start'] - spans[-2]['end'],
                     'end': buffer.size}] if len(spans) > 1 else [{'start': 0, 'end': buffer.size}]
        with patch.object(app, 'get_speech_timestamps', side_effect=vad), \
             patch.object(app, 'VAD_CHECK_EVERY_S', 0):
            with self.assertRaises(StopIteration):
                t._segmenter_loop()
        segments = list(t._segment_queue.queue)
        self.assertIsNone(segments[-1][0])
        return t, segments[:-1]

    def test_overflow_cuts_after_penultimate_speech_span_not_current_phrase(self):
        rate = app.SAMPLE_RATE
        audio = np.arange(int((app.MAX_SEGMENT_S + 1) * rate), dtype=np.float32)
        spans = [{'start': 0, 'end': 10 * rate},
                 {'start': int(10.3 * rate), 'end': 20 * rate},
                 {'start': int(20.3 * rate), 'end': audio.size}]
        t, segments = self.segment(audio, spans)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0][0].size, 20 * rate)
        self.assertAlmostEqual(segments[1][4], .3)
        np.testing.assert_array_equal(np.concatenate([segment[0] for segment in segments]), audio)
        np.testing.assert_array_equal(np.concatenate(t._session_audio), audio)

    def test_single_continuous_span_keeps_hard_overflow_fallback(self):
        audio = np.arange(int((app.MAX_SEGMENT_S + 1) * app.SAMPLE_RATE), dtype=np.float32)
        t, segments = self.segment(audio, [{'start': 0, 'end': audio.size}])
        self.assertEqual(len(segments), 1)
        np.testing.assert_array_equal(segments[0][0], audio)
        self.assertEqual(t._pending, 1)

    def test_below_limit_does_not_cut_at_micro_pause(self):
        rate = app.SAMPLE_RATE
        audio = np.arange(12 * rate, dtype=np.float32)
        spans = [{'start': 0, 'end': 5 * rate},
                 {'start': int(5.3 * rate), 'end': audio.size}]
        t, segments = self.segment(audio, spans)
        self.assertEqual(len(segments), 1)
        np.testing.assert_array_equal(segments[0][0], audio)

    def test_tail_silence_takes_priority_over_overflow_micro_pause(self):
        rate = app.SAMPLE_RATE
        audio = np.arange(int((app.MAX_SEGMENT_S + 1) * rate), dtype=np.float32)
        spans = [{'start': 0, 'end': 10 * rate},
                 {'start': int(10.3 * rate), 'end': 24 * rate}]
        # O restante e silencio: o VAD real nao produz spans depois da fala final.
        with patch.object(app, 'get_speech_timestamps', side_effect=[spans, []]):
            with patch.object(threading.Thread, 'start'), patch.object(app.keyboard, 'Controller'):
                t = app.Transcriber(queue.Queue(), queue.Queue())
            events = iter([(t._session_id, audio), (t._session_id, None)])
            t._audio_queue.get = lambda: next(events)
            with self.assertRaises(StopIteration):
                t._segmenter_loop()
        segments = list(t._segment_queue.queue)
        self.assertEqual(len(segments), 2)  # fala + marcador, nao penultima fala + resto
        self.assertEqual(segments[0][0].size, 24 * rate)
        self.assertIsNone(segments[1][0])

    def test_final_mode_still_transcribes_whole_recording(self):
        audio = np.arange(int((app.MAX_SEGMENT_S + 1) * app.SAMPLE_RATE), dtype=np.float32)
        spans = [{'start': 0, 'end': 10 * app.SAMPLE_RATE},
                 {'start': 11 * app.SAMPLE_RATE, 'end': audio.size}]
        t, segments = self.segment(audio, spans, mode='final')
        self.assertEqual(len(segments), 1)
        np.testing.assert_array_equal(segments[0][0], audio)


if __name__ == '__main__':
    unittest.main()
