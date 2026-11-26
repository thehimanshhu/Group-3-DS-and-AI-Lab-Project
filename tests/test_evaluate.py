"""Check evaluation against Milestone 3 predictions, audio, and weights.

Set M3_DATA_DIR and M3_CHECKPOINT to enable the real-audio checks.
"""

import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate import (
    compare_source_thresholds,
    evaluate_generators,
    evaluate_languages,
    evaluate_splits,
)
from src.metrics import eer


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "notebooks/Milestone-3/run_outputs/predictions.csv"
SAMPLE_RATE = 16_000


class EvaluationReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictions = pd.read_csv(PREDICTIONS)

    def test_eer_on_perfect_scores_and_saved_validation_predictions(self):
        self.assertEqual(eer([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), (0.0, 0.8))
        valid = self.predictions.loc[self.predictions["split"].eq("valid")]
        error, threshold = eer(valid["label"].eq("fake"), valid["p_fake"])
        self.assertEqual(len(valid), 397)
        self.assertAlmostEqual(error, 0.244327, places=6)
        self.assertAlmostEqual(threshold, 0.31553, places=5)

    def test_asvspoof_thresholds_and_confusion_counts(self):
        summary = evaluate_splits(self.predictions).set_index("split")
        counts = summary.loc["test_asvspoof", [
            "real_as_real", "real_as_fake", "fake_as_real", "fake_as_fake",
        ]]
        self.assertEqual(counts.tolist(), [0, 48, 0, 48])
        self.assertEqual(summary["threshold"].nunique(), 1)

        comparison = compare_source_thresholds(
            self.predictions, "asvspoof2019_la", "test_asvspoof"
        ).set_index("threshold_type")
        source = comparison.loc["source"]
        self.assertAlmostEqual(source["threshold"], 0.37740, places=5)
        self.assertEqual(source[["real_as_fake", "fake_as_real"]].tolist(), [14, 19])

    def test_generator_and_english_reports(self):
        generators = evaluate_generators(self.predictions)
        self.assertEqual(generators["detected"].sum(), 150)
        self.assertEqual(generators["fake_clips"].sum(), 270)

        languages = evaluate_languages(self.predictions)
        english = languages.loc[languages["language"].eq("en")]
        self.assertEqual(set(english["split"]), {"test_asvspoof", "test_itw"})


class Milestone3AudioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data_dir = os.environ.get("M3_DATA_DIR")
        checkpoint = os.environ.get("M3_CHECKPOINT")
        if not data_dir and not checkpoint:
            raise unittest.SkipTest("Set M3_DATA_DIR and M3_CHECKPOINT for audio tests.")
        if not data_dir or not checkpoint:
            raise ValueError("Set both M3_DATA_DIR and M3_CHECKPOINT.")

        cls.data_dir = Path(data_dir)
        manifest_path = cls.data_dir / "manifests" / "manifest_subset.csv"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Milestone 3 manifest not found: {manifest_path}")
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(f"Milestone 3 checkpoint not found: {checkpoint}")

        import soundfile as sf
        import torch
        from torch import nn
        from transformers import Wav2Vec2Config, Wav2Vec2Model

        manifest = pd.read_csv(manifest_path, keep_default_na=False)
        valid = manifest.loc[manifest["split"].eq("valid")]
        for index, row in valid.iterrows():
            wav_path = cls.data_dir / row["path"]
            info = sf.info(wav_path)
            if info.channels == 1 and info.samplerate == SAMPLE_RATE and 4 < info.duration <= 8:
                cls.rows = valid.loc[[index]]
                cls.audio, _ = sf.read(wav_path, dtype="float32")
                break
        else:
            raise ValueError("No 16 kHz mono validation clip longer than 4 s and at most 8 s.")

        class Milestone3Detector(nn.Module):
            """Match the encoder and head in 02_pipeline_check.ipynb."""

            def __init__(self):
                super().__init__()
                config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-xls-r-300m")
                self.encoder = Wav2Vec2Model(config)
                self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(config.hidden_size, 2))

            def forward(self, wave, mask):
                frames = self.encoder(wave, attention_mask=mask).last_hidden_state
                frame_mask = self.encoder._get_feature_vector_attention_mask(
                    frames.shape[1], mask
                )
                pooled = (frames * frame_mask[..., None]).sum(1)
                pooled = pooled / frame_mask.sum(1, keepdim=True)
                return self.head(pooled)

        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.model = Milestone3Detector()
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
        cls.model.load_state_dict(weights)
        cls.model.to(cls.device).eval()

    def test_score_returns_both_probabilities_on_m3_audio(self):
        from src.evaluate import score

        result = score(
            self.model, self.rows, self.data_dir,
            {"sample_rate": SAMPLE_RATE, "crop_seconds": 4.0}, self.device,
        )
        pd.testing.assert_frame_equal(result[self.rows.columns], self.rows)
        probabilities = result[["p_fake", "p_fake_middle"]].to_numpy()
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())

    def test_saliency_is_finite_and_aligned_with_audio(self):
        from src.explain import saliency

        values = saliency(self.model, self.audio, self.device)
        self.assertEqual(values.shape, self.audio.shape)
        self.assertTrue(np.isfinite(values).all())
        self.assertGreater(values.max(), 0)


class BatchedScoringTests(unittest.TestCase):
    """score() batches windows from many clips; it must give the same scores as one clip at a time."""

    def test_batched_scores_match_clip_by_clip(self):
        import tempfile

        import soundfile as sf
        import torch

        from src.evaluate import _score_clip, score
        from src.model import build_model

        torch.manual_seed(0)
        tiny = {"name": "tiny", "freeze_feature_encoder": True, "freeze_transformer_layers": 0,
                "dropout": 0.1}
        rng = np.random.default_rng(0)
        # 1 s to 13 s: one window, padded windows, a tail window, a separate middle window
        seconds = [1.0, 2.5, 4.0, 5.3, 7.9, 13.1, 3.2, 9.0]
        for pooling in ("mean", "attentive"):
            model = build_model({**tiny, "pooling": pooling}).eval()
            with tempfile.TemporaryDirectory() as folder:
                paths = []
                for i, s in enumerate(seconds):
                    sf.write(Path(folder) / f"{i}.wav", 0.1 * rng.standard_normal(int(SAMPLE_RATE * s)),
                             SAMPLE_RATE, subtype="PCM_16")
                    paths.append(f"{i}.wav")
                rows = pd.DataFrame({"path": paths})
                expected = [_score_clip(model, sf.read(Path(folder) / p, dtype="float32")[0], SAMPLE_RATE, "cpu")
                            for p in paths]
                for batch_size in (1, 3, 64):  # windows of one clip split over batches, and all in one
                    got = score(model, rows, folder, {"sample_rate": SAMPLE_RATE, "crop_seconds": 4.0}, "cpu",
                                batch_size=batch_size, workers=2)
                    np.testing.assert_allclose(got[["p_fake", "p_fake_middle"]].to_numpy(), expected, atol=1e-5,
                                               err_msg=f"{pooling}, batch {batch_size}")


class TrimStartTests(unittest.TestCase):
    def test_trim_start_scores_the_clip_without_its_first_second(self):
        import tempfile

        import soundfile as sf
        import torch

        from src.evaluate import _score_clip, score
        from src.model import build_model

        torch.manual_seed(0)
        model = build_model({"name": "tiny", "freeze_feature_encoder": True, "freeze_transformer_layers": 0,
                             "dropout": 0.1, "pooling": "mean"}).eval()
        rng = np.random.default_rng(1)
        clips = {"long.wav": 0.1 * rng.standard_normal(SAMPLE_RATE * 5),     # trimmed to 4 s
                 "short.wav": 0.1 * rng.standard_normal(SAMPLE_RATE * 3 // 2)}  # 1.5 s: scored whole
        with tempfile.TemporaryDirectory() as folder:
            for name, audio in clips.items():
                sf.write(Path(folder) / name, audio, SAMPLE_RATE, subtype="FLOAT")
            got = score(model, pd.DataFrame({"path": list(clips)}), folder,
                        {"sample_rate": SAMPLE_RATE, "crop_seconds": 4.0}, "cpu", workers=0, trim_start=1.0)
        audio = {k: v.astype(np.float32) for k, v in clips.items()}
        expected = [_score_clip(model, audio["long.wav"][SAMPLE_RATE:], SAMPLE_RATE, "cpu"),
                    _score_clip(model, audio["short.wav"], SAMPLE_RATE, "cpu")]
        np.testing.assert_allclose(got[["p_fake", "p_fake_middle"]].to_numpy(), expected, atol=1e-5)
