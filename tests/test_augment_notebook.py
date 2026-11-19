"""Notebook-facing augmentation API and default-config checks.

Run from the repository root:
    python -m unittest discover -s tests -v
FFmpeg integration tests skip when the executable is unavailable.
"""

import shutil
import sys
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import augment as aug

ROOT = Path(__file__).resolve().parents[1]


def sine(seconds=1.0, frequency=440.0, amplitude=0.2):
    times = np.arange(round(aug.TARGET_SR * seconds)) / aug.TARGET_SR
    return (amplitude * np.sin(2 * np.pi * frequency * times)).astype(np.float32)


def peak_frequency(wave):
    return np.argmax(np.abs(np.fft.rfft(wave))) * aug.TARGET_SR / len(wave)


class NotebookSignatureTests(unittest.TestCase):
    def test_notebook_waveform_effects_accept_sr_and_rng(self):
        clean = sine()
        rng = np.random.default_rng(42)
        versions = {
            "noise": aug.add_noise(clean, rng),
            "gain": aug.random_gain(clean, rng),
            "speed": aug.speed(clean, 16000, rng),
            "pitch": aug.pitch(clean, 16000, rng),
            "reverberation": aug.reverb(clean, 16000, rng),
            "RawBoost": aug.rawboost(clean, rng),
        }
        for name, result in versions.items():
            with self.subTest(effect=name):
                self.assertEqual(result.dtype, np.float32)
                self.assertTrue(np.isfinite(result).all())
                self.assertFalse(np.shares_memory(result, clean))

    def test_speed_and_pitch_default_ranges_are_seeded(self):
        clean = sine()
        for transform in (aug.speed, aug.pitch):
            with self.subTest(effect=transform.__name__):
                first = transform(clean, 16000, np.random.default_rng(42))
                second = transform(clean, 16000, np.random.default_rng(42))
                third = transform(clean, 16000, np.random.default_rng(43))
                np.testing.assert_array_equal(first, second)
                self.assertFalse(np.array_equal(first, third))

    def test_speed_and_pitch_do_not_emit_runtime_warnings(self):
        clean = sine()
        for transform in (aug.speed, aug.pitch):
            with self.subTest(effect=transform.__name__):
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)
                    transform(clean, 16000, np.random.default_rng(42))

    def test_fixed_speed_and_pitch_values_remain_available(self):
        clean = sine()
        stretched = aug.speed(clean, 16000, np.random.default_rng(42), rate=1.1)
        self.assertEqual(len(stretched), round(len(clean) / 1.1))
        self.assertAlmostEqual(peak_frequency(stretched), 440, delta=3)
        shifted = aug.pitch(clean, 16000, np.random.default_rng(42), semitones=2)
        self.assertEqual(len(shifted), len(clean))
        self.assertAlmostEqual(peak_frequency(shifted), 440 * 2 ** (2 / 12), delta=4)

    def test_rawboost_notebook_call_uses_the_default_combination(self):
        clean = sine(0.25)
        default = aug.rawboost(clean, np.random.default_rng(42))
        explicit = aug.rawboost(
            clean,
            np.random.default_rng(42),
            ("convolutive", "impulsive"),
        )
        np.testing.assert_array_equal(default, explicit)

    def test_effect_rate_and_rng_validation(self):
        clean = sine()
        for transform in (aug.speed, aug.pitch):
            with self.subTest(effect=transform.__name__):
                with self.assertRaisesRegex(ValueError, "16 kHz"):
                    transform(clean, 8000, np.random.default_rng(42))
                with self.assertRaisesRegex(ValueError, "Generator"):
                    transform(clean, 16000, None)

    def test_codec_defaults_to_phone_channel_and_random_g711(self):
        clean = sine()
        names = []

        def record(wave, sr, name, ffmpeg_path):
            names.append(name)
            self.assertEqual(sr, 16000)
            self.assertEqual(ffmpeg_path, "ffmpeg")
            return wave.copy()

        with patch.object(aug, "phone_channel", side_effect=record) as phone:
            for seed in range(8):
                aug.codec(clean, 16000, np.random.default_rng(seed))
        self.assertEqual(phone.call_count, 8)
        self.assertEqual(set(names), {"mulaw", "alaw"})

    def test_codec_mode_can_skip_the_phone_band_limit(self):
        clean = sine()
        with (
            patch.object(
                aug, "_codec_roundtrip", return_value=clean.copy()
            ) as roundtrip,
            patch.object(
                aug,
                "phone_channel",
                side_effect=AssertionError("phone channel accessed"),
            ),
        ):
            result = aug.codec(
                clean,
                16000,
                np.random.default_rng(42),
                mode="codec",
                name="alaw",
            )
        args = roundtrip.call_args.args
        np.testing.assert_array_equal(args[0], clean)
        self.assertEqual(args[1:], (16000, "alaw", "ffmpeg"))
        np.testing.assert_array_equal(result, clean)

    def test_codec_rejects_unknown_mode_and_name(self):
        clean = sine()
        with self.assertRaisesRegex(ValueError, "mode"):
            aug.codec(clean, 16000, np.random.default_rng(42), mode="amr")
        with self.assertRaisesRegex(ValueError, "name"):
            aug.codec(clean, 16000, np.random.default_rng(42), name="amr")

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg executable unavailable")
    def test_notebook_codec_call_end_to_end(self):
        clean = sine()
        first = aug.codec(clean, 16000, np.random.default_rng(42))
        second = aug.codec(clean, 16000, np.random.default_rng(42))
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), len(clean))
        self.assertEqual(first.dtype, np.float32)
        self.assertTrue(np.isfinite(first).all())
        self.assertFalse(np.array_equal(first, clean))
        self.assertAlmostEqual(peak_frequency(first), 440, delta=2)
        high = aug.codec(sine(frequency=6000), 16000, np.random.default_rng(42))
        low = aug.codec(sine(frequency=1000), 16000, np.random.default_rng(42))
        self.assertLess(np.sqrt(np.mean(high**2)), 0.02 * np.sqrt(np.mean(low**2)))


class DefaultConfigTests(unittest.TestCase):
    def test_module_defaults_match_the_notebook_contract(self):
        config = aug.AugmentationConfig()
        self.assertEqual(config.speed["rate"], (0.9, 1.1))
        self.assertEqual(config.pitch["semitones"], (-2.0, 2.0))
        self.assertEqual(config.channel["mode"], "phone")
        self.assertEqual(config.channel["codec"], "random")
        self.assertEqual(config.channel["ffmpeg_path"], "ffmpeg")

    def test_default_yaml_is_valid_and_leaves_audio_unchanged(self):
        import yaml

        document = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
        augment_cfg = document["augment"]
        self.assertEqual(set(augment_cfg), set(aug.EFFECT_ORDER))
        config = aug.AugmentationConfig.from_dict(augment_cfg)
        for name in aug.EFFECT_ORDER:
            self.assertFalse(getattr(config, name)["enabled"])
        clean = sine()
        rng = np.random.default_rng(42)
        before = rng.bit_generator.state
        result = aug.apply(clean, 16000, augment_cfg, rng)
        np.testing.assert_array_equal(result, clean)
        self.assertFalse(np.shares_memory(result, clean))
        self.assertEqual(before, rng.bit_generator.state)

    def test_apply_routes_channel_to_codec(self):
        clean = sine()
        config = {
            "channel": {
                "enabled": True,
                "p": 1,
                "mode": "phone",
                "codec": "alaw",
            }
        }
        with patch.object(aug, "phone_channel", return_value=clean.copy()) as phone:
            result = aug.apply(clean, 16000, config, np.random.default_rng(42))
        args = phone.call_args.args
        np.testing.assert_array_equal(args[0], clean)
        self.assertEqual(args[1:], (16000, "alaw", "ffmpeg"))
        np.testing.assert_array_equal(result, clean)


if __name__ == "__main__":
    unittest.main()
