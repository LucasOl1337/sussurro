"""Reuniao: nivel, trechos com som, eco no microfone, linhas por falante e a saida em Markdown."""
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import sussurro_meeting as m
import sussurro_meeting_audio as rec
from sussurro_diarize import Turn

RATE = m.RATE


def tone(seconds, amplitude, freq=220.0):
    t = np.arange(int(seconds * RATE)) / RATE
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(seconds):
    return np.zeros(int(seconds * RATE), dtype=np.float32)


def line(start_ms, speaker, text):
    return m.Line(start_ms, start_ms + 2000, speaker, text)


class LevelTests(unittest.TestCase):
    def test_quiet_speech_is_raised_and_silence_left_alone(self):
        quiet = np.concatenate([tone(1, 0.01), silence(1)])
        self.assertGreater(m.active_level(m.level(quiet)), 0.05)
        noise = (np.random.default_rng(1).standard_normal(RATE) * 0.0005).astype(np.float32)
        np.testing.assert_array_equal(m.level(noise), noise)

    def test_soft_clip_keeps_the_body_and_bends_peaks(self):
        x = np.array([0.5, -0.8, 1.5, -3.0], dtype=np.float32)
        y = m.soft_clip(x)
        self.assertEqual(y[0], 0.5)
        self.assertTrue(np.all(np.abs(y) <= 1.0))
        self.assertLess(y[2], 1.0)


class RegionTests(unittest.TestCase):
    def test_sound_with_padding_and_merge(self):
        track = np.concatenate([silence(1), tone(0.5, 0.3), silence(0.5), tone(0.5, 0.3), silence(2)])
        regions = m.speech_regions([track], track.size)
        self.assertEqual(len(regions), 1)  # a pausa de 0,5 s fica dentro do mesmo trecho
        self.assertAlmostEqual(regions[0].onset / RATE, 1.0, delta=0.05)
        self.assertLess(regions[0].start, regions[0].onset)

    def test_echo_of_the_other_side_is_not_your_voice(self):
        theirs = np.concatenate([silence(1), tone(2, 0.3), silence(3)])
        echo = np.roll(theirs, int(0.04 * RATE)) * 0.3  # alto-falante vazando no mic, 40 ms depois
        own = np.concatenate([silence(4), tone(1.5, 0.2, 180), silence(0.5)])
        mic = m.level(echo + own)
        regions = m.own_speech_regions(mic, m.level(theirs))
        self.assertEqual(len(regions), 1)
        self.assertGreater(regions[0].onset / RATE, 3.5)


class InterleaveTests(unittest.TestCase):
    """Os dois testes do original, com os rotulos em portugues."""

    def test_both_sides_come_back_in_the_order_they_spoke(self):
        out = m.interleave([line(0, "Remoto", "Obrigado, posso comecar pelo release."),
                            line(5000, "Remoto", "O beta saiu na segunda."),
                            line(2500, "Você", "Claro, pode ir.")], m.Labels())
        self.assertEqual([s.text for s in out], ["Obrigado, posso comecar pelo release.",
                                                 "Claro, pode ir.", "O beta saiu na segunda."])

    def test_the_other_side_leaking_into_the_mic_is_dropped(self):
        out = m.interleave([line(0, "Remoto 1", "A revisao ainda esta pendente depois de quatro dias."),
                            line(300, "Você", "revisao ainda esta pendente depois de quatro"),
                            line(9000, "Remoto 1", "Parece bom."),
                            line(9100, "Você", "Parece bom."),
                            line(30_000, "Você", "A revisao ainda esta pendente, estou vendo.")], m.Labels())
        self.assertEqual([s.text for s in out if s.speaker == "Você"],
                         ["A revisao ainda esta pendente, estou vendo."])


class PhraseTests(unittest.TestCase):
    def words(self, spec):
        return [m.Word(text, start, end, 0.0, seg) for text, start, end, seg in spec]

    def test_turns_split_lines_and_sentences_stay_whole(self):
        track = tone(8, 0.2)
        turns = [Turn(0, 3000, 0), Turn(3200, 8000, 1)]
        speakers = m.Speakers("Falante", turns, side=False)
        words = self.words([("Bom", 100, 400, 0), ("dia", 400, 800, 0), ("a", 800, 1000, 0),
                            ("todos.", 1000, 2800, 0), ("Oi,", 3300, 3600, 0), ("tudo", 3600, 3900, 0),
                            ("bem?", 3900, 4400, 0)])
        regions = m.speech_regions([track], track.size)
        lines = m.phrases(words, regions, speakers, track, paragraphs=True)
        self.assertEqual([(l.speaker, l.text) for l in lines],
                         [("Falante 1", "Bom dia a todos."), ("Falante 2", "Oi, tudo bem?")])

    def test_whisper_talking_over_silence_is_dropped(self):
        track = np.concatenate([tone(1, 0.2), silence(3)])
        speakers = m.Speakers("Você", [])
        words = self.words([("Obrigado.", 2000, 2600, 0)])
        self.assertEqual(m.phrases(words, [], speakers, track, paragraphs=False), [])

    def test_stock_phrases_and_markers(self):
        self.assertTrue(m.is_stock_phrase("Legendas pela comunidade Amara.org"))
        self.assertTrue(m.is_stock_phrase("Tchau, tchau!"))
        self.assertFalse(m.is_stock_phrase("Obrigado pela revisao do contrato."))
        self.assertTrue(m.is_noise_marker("[Música]"))
        self.assertFalse(m.is_noise_marker("Música boa."))


class FakeModel:
    """Devolve as palavras de um roteiro so onde a trilha tem som."""

    def __init__(self, script):
        self.script = script  # (texto, inicio_s, fim_s)

    def transcribe(self, audio, language=None, vad_filter=True, word_timestamps=True, **_):
        segments = []
        for i, (text, start, end) in enumerate(self.script):
            part = audio[int(start * RATE):int(end * RATE)]
            if part.size and np.abs(part).max() > 0.01:
                n = len(text.split())
                step = (end - start) / n
                words = [SimpleNamespace(word=" " + w, start=start + k * step, end=start + (k + 1) * step)
                         for k, w in enumerate(text.split())]
                segments.append(SimpleNamespace(words=words, no_speech_prob=0.01, end=end))
        return segments, SimpleNamespace(language=language or "pt")


class MeetingTests(unittest.TestCase):
    def test_each_side_is_its_track_and_order_is_kept(self):
        mic = np.concatenate([silence(3), tone(2, 0.2, 180), silence(3)])
        pc = np.concatenate([tone(2.5, 0.2), silence(3), tone(2, 0.2)])
        script = [("Oi pessoal, vamos comecar.", 0.0, 2.5), ("Pode falar.", 3.0, 5.0),
                  ("Beleza, entao eu comeco.", 5.5, 7.5)]
        result = m.transcribe_meeting(mic, pc, "pt", FakeModel(script), find_voices=False)
        self.assertEqual([(l.speaker, l.text) for l in result.lines],
                         [("Remoto", "Oi pessoal, vamos comecar."), ("Você", "Pode falar."),
                          ("Remoto", "Beleza, entao eu comeco.")])
        self.assertEqual(result.duration_secs, 8)

    def test_silence_gives_an_empty_transcript(self):
        result = m.transcribe_meeting(silence(5), silence(5), "pt", FakeModel([("Tchau.", 0, 1)]),
                                      find_voices=False)
        self.assertEqual(result.lines, [])

    def test_cancel_stops_before_recognizing(self):
        import threading
        abort = threading.Event()
        abort.set()
        with self.assertRaises(m.Cancelled):
            speech = np.concatenate([silence(1), tone(1, 0.2), silence(1)])  # tom continuo e chao de ruido
            m.transcribe_meeting(speech, silence(3), "pt", FakeModel([]), find_voices=False, abort=abort)


class MarkdownTests(unittest.TestCase):
    def test_lines_match_the_bench_format_and_names_apply(self):
        t = m.Transcript([m.Line(65_000, 70_000, "Remoto 2", "Tudo certo.")], "pt", 3700)
        text = m.to_markdown("Sync", "25/09/2026 10:00", t, {"Remoto 2": "Marina"})
        self.assertIn("- **Duração:** 1:01:40", text)
        self.assertRegex(text, r"\*\*\[01:05\] Marina:\*\* Tudo certo\.")
        bench = re.compile(r"\*\*\[(?:(\d+):)?(\d+):(\d+)\] ([^:*]+):\*\* (.*)")
        self.assertEqual(bench.search(text).group(4), "Marina")


class FolderTests(unittest.TestCase):
    def test_folder_names_sort_by_date_and_rename_keeps_prefix(self):
        from datetime import datetime
        with tempfile.TemporaryDirectory() as root, patch.object(rec, "meetings_root", return_value=Path(root)):
            folder = rec.new_meeting_dir("Sync: produto/infra", datetime(2026, 9, 25, 10, 5))
            self.assertEqual(folder.name, "202609251005 Sync produto infra")
            self.assertEqual(rec.safe_name("Reunião 10:20"), "Reunião 10h20")
            again = rec.new_meeting_dir("Sync: produto/infra", datetime(2026, 9, 25, 10, 5))
            self.assertTrue(again.name.endswith("(2)"))
            rec.write_manifest(folder, {"title": "x", "lines": []})
            renamed = rec.rename_meeting(folder, "Weekly")
            self.assertEqual(renamed.name, "202609251005 Weekly")
            self.assertEqual(rec.read_manifest(renamed)["title"], "x")

    def test_speech_gain_levels_quiet_tracks_and_leaves_silence(self):
        with tempfile.TemporaryDirectory() as root:
            quiet, empty = Path(root) / "q.raw", Path(root) / "s.raw"
            (np.sin(np.arange(rec.RATE * 2) / 10) * 800).astype("<i2").tofile(quiet)
            np.zeros(rec.RATE, dtype="<i2").tofile(empty)
            self.assertGreater(rec.speech_gain_db(quiet), 6)
            self.assertEqual(rec.speech_gain_db(empty), 0.0)


if __name__ == "__main__":
    unittest.main()


class DictationHallucinationTests(unittest.TestCase):
    def seg(self, start, end, text, no_speech=0.0):
        return SimpleNamespace(start=start, end=end, text=text, no_speech_prob=no_speech)

    def test_quiet_goodbye_at_the_end_goes_and_a_real_one_stays(self):
        audio = np.concatenate([tone(3, 0.2), tone(1, 0.002)])
        segments = [self.seg(0, 3, " Vamos fechar o contrato hoje."), self.seg(3, 4, " Tchau.")]
        self.assertEqual([s.text for s in m.drop_hallucinations(segments, audio)],
                         [" Vamos fechar o contrato hoje."])
        spoken = [self.seg(0, 2.5, " Vamos fechar."), self.seg(2.5, 3, " Obrigado.")]
        self.assertEqual(len(m.drop_hallucinations(spoken, audio)), 2)
