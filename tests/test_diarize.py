"""Speaker diarization: features, speaker cache, chunk loop and turns without GPU; one GPU check if available."""
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import sussurro_diarize as diarize
from sussurro_diarize import Turn


def gpu_skip_reason():
    try:
        import onnxruntime
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import LocalEntryNotFoundError
    except ImportError:
        return 'sem onnxruntime/huggingface_hub'
    if 'CUDAExecutionProvider' not in onnxruntime.get_available_providers():
        return 'sem CUDAExecutionProvider'
    try:
        for name in (diarize.GRAPH, diarize.GRAPH + '_data'):
            hf_hub_download(diarize.REPO, name, revision=diarize.REVISION, local_files_only=True)
    except LocalEntryNotFoundError:
        return 'modelo de diarizacao fora do cache'
    return None


GPU_SKIP = gpu_skip_reason()


def librosa_mel():
    """librosa.filters.mel(sr=16000, n_fft=512, n_mels=128, norm='slaney'), refeito pela formula dela."""
    def hz_to_mel(f):
        f = np.asarray(f, float)
        return np.where(f >= 1000, 15 + np.log(np.maximum(f, 1e-10) / 1000) / (np.log(6.4) / 27), 3 * f / 200)

    def mel_to_hz(m):
        return np.where(m >= 15, 1000 * np.exp(np.log(6.4) / 27 * (m - 15)), 200 * m / 3)

    fftfreqs = np.fft.rfftfreq(512, 1 / 16000)
    mel_f = mel_to_hz(np.linspace(hz_to_mel(0), hz_to_mel(8000), 130))
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    return np.maximum(0, np.minimum(lower, upper)) * (2.0 / (mel_f[2:] - mel_f[:-2]))[:, None]


def reference_log_mel(samples, start, end, valid, rows):
    """Um frame por vez, como o Rust: pre-enfase, padding central, Hann 400 em 512, float64."""
    x = samples.astype(float)
    emph = np.concatenate(([x[0]], x[1:] - 0.97 * x[:-1]))
    padded = np.pad(emph, (256, 256))
    window = np.zeros(512)
    window[56:456] = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(400) / 399)
    mel = librosa_mel()
    out = np.zeros((rows, 128))
    for f in range(start, min(end, valid)):
        power = np.abs(np.fft.rfft(padded[f * 160:f * 160 + 512] * window)) ** 2
        out[f - start] = np.log(mel @ power + 2.0 ** -24)
    return out


class FeatureTests(unittest.TestCase):
    def test_mel_filters_match_librosa_slaney(self):
        filters = diarize.mel_filters()
        self.assertEqual(filters.shape, (128, 257))
        self.assertEqual(filters.dtype, np.float32)
        np.testing.assert_allclose(filters, librosa_mel(), rtol=1e-5, atol=1e-8)
        self.assertTrue((filters.max(axis=1) > 0).all())

    def test_log_mel_matches_frame_by_frame_reference(self):
        samples = np.random.default_rng(1).standard_normal(16000).astype(np.float32) * 0.1
        mel = diarize.mel_filters()
        frames = 1 + len(samples) // 160
        full = diarize.log_mel(samples, mel, 0, frames, frames - 1, 104)
        np.testing.assert_allclose(full, reference_log_mel(samples, 0, frames, frames - 1, 104), atol=2e-3)
        self.assertFalse(full[frames - 1:].any())  # ultimo frame (>= valid) e padding zerados

    def test_block_in_the_middle_cuts_at_valid_and_pads(self):
        samples = np.random.default_rng(2).standard_normal(16000).astype(np.float32) * 0.1
        block = diarize.log_mel(samples, diarize.mel_filters(), 40, 90, 80, 64)
        np.testing.assert_allclose(block, reference_log_mel(samples, 40, 90, 80, 64), atol=2e-3)
        self.assertFalse(block[40:].any())

    def test_first_sample_keeps_its_value_before_emphasis(self):
        samples = np.array([1.0, 1.0, 0.5], np.float32)
        np.testing.assert_allclose(diarize._padded(samples, 256, 259), [1.0, 0.03, 0.5 - 0.97], rtol=1e-6)
        self.assertFalse(diarize._padded(samples, 0, 256).any())

    def test_sine_peaks_in_its_fft_bin(self):
        t = np.arange(16000) / 16000
        power = diarize.power(np.sin(2 * np.pi * 1000 * t).astype(np.float32), 10, 20)
        self.assertEqual(power.shape, (10, 257))
        self.assertTrue((power.argmax(axis=1) == 32).all())  # 1000 Hz / 31.25 Hz por bin


def speakers_logits(total, active):
    """Logits de 10 ms: +6 pro falante de cada passo em `active`, -6 no resto."""
    logits = np.full((total * 8, 8), -6.0, np.float32)
    for step, who in enumerate(active):
        logits[step * 8:(step + 1) * 8, who] = 6.0
    return logits


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.silence = np.full(512, -1.0, np.float32)

    def test_first_chunk_keeps_fifo_and_compresses_cache_by_speaker(self):
        cache = diarize._Cache()
        embeds = np.repeat(np.arange(380, dtype=np.float32)[:, None], 512, axis=1)
        logits = speakers_logits(380, [0] * 170 + [1] * 210)
        cache.update(embeds, logits, 340, self.silence)
        self.assertEqual(cache.fifo[:, 0].tolist(), list(range(300, 340)))
        self.assertEqual(len(cache.cache_embeds), 264)
        self.assertTrue(cache.compressed)
        rows = cache.cache_embeds[:, 0]
        silent = rows == -1
        self.assertEqual(silent.sum(), 8)  # um slot de silencio por falante
        self.assertFalse(cache.cache_probs[silent].any())
        # Agrupado por falante, cada grupo na ordem original
        first_silence = np.flatnonzero(silent)[0]
        speaker0, speaker1 = rows[:first_silence], rows[first_silence + 1:][~silent[first_silence + 1:]]
        self.assertTrue((speaker0 < 170).all() and (np.diff(speaker0) > 0).all())
        self.assertTrue((speaker1 >= 170).all() and (np.diff(speaker1) > 0).all())

    def test_overflowing_fifo_pops_at_least_update_period(self):
        cache = diarize._Cache()
        cache.update(np.zeros((380, 512), np.float32), speakers_logits(380, [0] * 380), 340, self.silence)
        cached = cache.embeds()
        self.assertEqual(len(cached), 304)
        chunk = np.ones((20, 512), np.float32)
        cache.update(np.concatenate([cached, chunk]), speakers_logits(324, [0] * 324), 20, self.silence)
        self.assertEqual(len(cache.fifo), 0)  # 60 > 40: sai tudo (min(max(300, 20), 60))
        self.assertEqual(len(cache.cache_embeds), 264)

    def test_small_chunk_stays_in_fifo(self):
        cache = diarize._Cache()
        cache.update(np.zeros((30, 512), np.float32), speakers_logits(30, [0] * 30), 30, self.silence)
        self.assertEqual((len(cache.cache_embeds), len(cache.fifo)), (0, 30))
        self.assertFalse(cache.compressed)


class FakeSession:
    def __init__(self):
        self.calls = []

    def run(self, names, feeds):
        rows, cached, total = (feeds['input_features'].shape[1], feeds['cached_embeds'].shape[1],
                               feeds['attention_mask'].shape[1])
        self.calls.append((rows, cached, total))
        assert total == cached + rows // 8
        logits = np.full((1, total * 8, 8), -5.0, np.float32)
        logits[..., 0] = 5.0
        embeds = np.random.default_rng(len(self.calls)).standard_normal((1, rows // 8, 512)).astype(np.float32)
        return [logits, embeds, np.zeros(512, np.float32)]


class FakeDiarizer(diarize.Diarizer):
    def __init__(self, providers=None):
        self._session = FakeSession()
        self._silence = None
        self.closed = False

    def close(self):
        self.closed = True
        super().close()


class ChunkLoopTests(unittest.TestCase):
    def test_chunks_carry_cache_and_report_progress(self):
        diarizer = FakeDiarizer()
        seen = []
        probs = diarizer.probabilities(np.zeros(16000 * 60, np.float32), progress=seen.append)
        self.assertEqual(probs.shape, (6001, 8))
        self.assertEqual(probs.dtype, np.float32)
        # 751 passos: 340 + 340 + 71, lookahead de 40 e cache+FIFO de 264+40 depois do primeiro
        self.assertEqual(diarizer._session.calls, [(3040, 0, 380), (3040, 304, 684), (568, 304, 375)])
        self.assertEqual(seen, [340 / 751, 680 / 751, 1.0])
        self.assertTrue((probs[:, 0] > 0.99).all() and (probs[:, 1:] < 0.01).all())

    def test_abort_raises_cancelled(self):
        abort = threading.Event()
        abort.set()
        diarizer = FakeDiarizer()
        with self.assertRaises(diarize.Cancelled) as caught:
            diarizer.probabilities(np.zeros(16000, np.float32), abort=abort)
        self.assertIsInstance(caught.exception, RuntimeError)
        self.assertEqual(str(caught.exception), 'cancelado')
        self.assertEqual(diarizer._session.calls, [])

    def test_closed_diarizer_refuses_work(self):
        diarizer = FakeDiarizer()
        diarizer.close()
        with self.assertRaises(RuntimeError):
            diarizer.probabilities(np.zeros(1600, np.float32))

    def test_turns_closes_only_its_own_diarizer(self):
        samples = np.zeros(16000 * 3, np.float32)
        given = FakeDiarizer()
        self.assertEqual(diarize.turns(samples, diarizer=given), [Turn(0, 3010, 0)])
        self.assertFalse(given.closed)
        created = []
        with patch.object(diarize, 'Diarizer', lambda: created.append(FakeDiarizer()) or created[-1]):
            diarize.turns(samples)
        self.assertTrue(created[0].closed)


class ModelFileTests(unittest.TestCase):
    @unittest.skipIf(os.name == 'nt', 'symlinks')
    def test_split_blobs_are_linked_side_by_side(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for blob, data in (('aa/graph', b'graph'), ('bb/weights', b'weights')):
                (root / 'blobs' / blob).parent.mkdir(parents=True)
                (root / 'blobs' / blob).write_bytes(data)
            snapshot = root / 'snapshot'
            snapshot.mkdir()
            (snapshot / 'model_fp16.onnx').symlink_to(root / 'blobs/aa/graph')
            (snapshot / 'model_fp16.onnx_data').symlink_to(root / 'blobs/bb/weights')
            with patch.dict(os.environ, {'SUSSURRO_CACHE_DIR': str(root / 'cache')}):
                path = diarize._side_by_side(snapshot / 'model_fp16.onnx', snapshot / 'model_fp16.onnx_data')
                self.assertEqual(diarize._side_by_side(snapshot / 'model_fp16.onnx',
                                                       snapshot / 'model_fp16.onnx_data'), path)
            self.assertTrue(path.is_relative_to(root / 'cache'))
            self.assertFalse(path.is_symlink())
            self.assertEqual((path.parent / 'model_fp16.onnx_data').read_bytes(), b'weights')

    def test_files_in_one_folder_are_used_in_place(self):
        with tempfile.TemporaryDirectory() as folder:
            graph, data = Path(folder) / 'model.onnx', Path(folder) / 'model.onnx_data'
            graph.write_bytes(b'g')
            data.write_bytes(b'w')
            self.assertEqual(diarize._side_by_side(graph, data), graph)


class TurnTests(unittest.TestCase):
    def test_speakers_are_numbered_by_first_appearance(self):
        turns = diarize._renumber([(5000, 6000, 3), (0, 1000, 7), (2000, 3000, 3), (7000, 8000, 7)])
        self.assertEqual([t.speaker for t in turns], [0, 1, 1, 0])
        self.assertEqual([t.start_ms for t in turns], [0, 2000, 5000, 7000])

    def test_words_go_to_the_turn_they_overlap_most(self):
        turns = [Turn(0, 2000, 0), Turn(1800, 5000, 1)]
        self.assertEqual(diarize.speaker_at(turns, 1500, 1900), 0)
        self.assertEqual(diarize.speaker_at(turns, 1900, 3000), 1)
        self.assertEqual(diarize.speaker_at(turns, 6000, 6500), 1)  # no vao depois do ultimo: o mais perto
        self.assertEqual(diarize.speaker_at([], 0, 100), 0)
        self.assertEqual(diarize.speaker_at([Turn(0, 1000, 1), Turn(0, 1000, 0)], 0, 1000), 0)  # empate: menor
        self.assertEqual(diarize.turn_start_near(turns, 1, 2500), 1800)
        self.assertIsNone(diarize.turn_start_near(turns, 1, 9000))

    def test_segments_bridge_short_pauses_and_drop_blips(self):
        probs = np.zeros((300, 8), np.float32)
        probs[0:50, 0] = 0.9      # 0-500 ms
        probs[99:150, 0] = 0.9    # pausa de 490 ms: emenda
        probs[200:229, 1] = 0.9   # 290 ms: fora
        probs[250:280, 2] = 0.9   # 300 ms: fica
        self.assertEqual(diarize._segments(probs), [(0, 1500, 0), (2500, 2800, 2)])
        probs[100:150, 0] = 0.9
        probs[99, 0] = 0.0        # pausa de exatos 500 ms: nao emenda
        self.assertEqual(diarize._segments(probs)[:2], [(0, 500, 0), (1000, 1500, 0)])

    def test_keep_largest_gives_the_rest_to_the_nearest_kept_turn(self):
        raw = [(0, 5000, 0), (5000, 6000, 2), (10000, 14000, 1), (20000, 21000, 2)]
        kept = diarize._keep_largest(raw, 2)
        self.assertEqual([who for _, _, who in kept], [0, 0, 1, 1])
        self.assertEqual({who for _, _, who in diarize._keep_largest(raw, 0)}, {0})

    def test_small_clusters_are_absorbed(self):
        raw = [(0, 60000, 0), (60000, 62000, 5), (62000, 120000, 1)]
        self.assertEqual([who for _, _, who in diarize._absorb_small_clusters(raw)], [0, 0, 1])
        few = [(0, 1000, 0), (2000, 3000, 1)]  # ninguem passa de 4 s: fica como esta
        self.assertEqual(diarize._absorb_small_clusters(few), few)

    def test_single_covers_the_whole_file(self):
        self.assertEqual(diarize.single(np.zeros(24000, np.float32)), [Turn(0, 1500, 0)])


@unittest.skipIf(GPU_SKIP, GPU_SKIP or '')
class GpuTests(unittest.TestCase):
    def test_model_runs_on_cuda_and_releases_session(self):
        samples = (np.random.default_rng(3).standard_normal(16000 * 5) * 0.01).astype(np.float32)
        diarizer = diarize.Diarizer()
        try:
            self.assertIn('CUDAExecutionProvider', diarizer._session.get_providers())
            probs = diarizer.probabilities(samples)
            self.assertEqual(probs.shape, (501, 8))
            self.assertEqual(probs.dtype, np.float32)
            self.assertTrue(((probs >= 0) & (probs <= 1)).all())
            self.assertTrue(all(isinstance(t, Turn) for t in diarize.turns(samples, diarizer=diarizer)))
        finally:
            diarizer.close()
        with self.assertRaises(RuntimeError):
            diarizer.probabilities(samples)


if __name__ == '__main__':
    unittest.main()
