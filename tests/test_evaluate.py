"""Check evaluation against Milestone 3 predictions, audio, and weights.

Set M3_DATA_DIR and M3_CHECKPOINT to enable the real-audio checks.
"""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate import (
    _prepare_window,
    _window_starts,
    compare_source_thresholds,
    evaluate_generators,
    evaluate_languages,
    evaluate_splits,
    score,
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


class BatchedScoringTests(unittest.TestCase):
    def test_batched_scores_match_original_clip_by_clip_fp32(self):
        import soundfile as sf
        import torch

        class WindowModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Linear(2, 2, bias=False)
                with torch.no_grad():
                    self.projection.weight.copy_(torch.tensor([[0.0, 0.0], [0.4, 0.3]]))
                self.batch_sizes = []

            def forward(self, wave, mask):
                self.batch_sizes.append(len(wave))
                features = torch.stack((wave[:, 0], mask.float().mean(dim=1)), dim=1)
                return self.projection(features)

        sample_rate = 100
        window_size = 4 * sample_rate
        generator = np.random.default_rng(42)
        # Short, exact window, off-grid tail with a distinct center, and long enough
        # for one clip to span several batches.
        clips = [generator.normal(size=length) for length in (210, 400, 650, 1450)]
        rows = pd.DataFrame(
            {"path": [f"{index}.wav" for index in range(len(clips))]},
            index=[5, 2, 5, 8],
        )
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")

        with tempfile.TemporaryDirectory() as folder:
            for path, audio in zip(rows["path"], clips):
                sf.write(Path(folder) / path, audio, sample_rate, subtype="FLOAT")

            for device in devices:
                with self.subTest(device=device):
                    model = WindowModel().to(device).eval()
                    reference = []
                    with torch.inference_mode():
                        for path in rows["path"]:
                            audio, _ = sf.read(Path(folder) / path, dtype="float32")
                            starts = _window_starts(len(audio), window_size, 2 * sample_rate)
                            sliding_count = len(starts)
                            middle = max(0, (len(audio) - window_size) // 2)
                            if middle not in starts:
                                starts.append(middle)
                            waves, masks = zip(*(
                                _prepare_window(audio, start, window_size) for start in starts
                            ))
                            logits = model(
                                torch.tensor(np.stack(waves), device=device),
                                torch.tensor(np.stack(masks), device=device),
                            )
                            probabilities = logits.softmax(dim=1)[:, 1]
                            reference.append([
                                probabilities[:sliding_count].mean().item(),
                                probabilities[starts.index(middle)].item(),
                            ])

                    model.batch_sizes.clear()
                    result = score(
                        model, rows, folder, {"sample_rate": sample_rate}, device,
                        batch_size=3, fp16=device == "cuda",
                    )
                    pd.testing.assert_frame_equal(result[rows.columns], rows)
                    np.testing.assert_allclose(
                        result[["p_fake", "p_fake_middle"]], reference,
                        atol=1e-3, rtol=0,
                    )
                    self.assertEqual(model.batch_sizes, [3, 3, 3, 3, 2])


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
