"""Synthetic augmentation checks; no corpus, model, or network access.

Run from the repository root:
    python -m unittest discover -s scratchpad/vikram/milestone-4/tests -v
Or from scratchpad/vikram/milestone-4:
    python -m unittest discover -s tests -v
FFmpeg integration tests skip when the executable is unavailable.
"""

from contextlib import ExitStack
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

# Support repository-root discovery and direct execution after relocation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import augment as aug  # noqa: E402


def sine(seconds=1.0, frequency=440.0, amplitude=0.2):
    times = np.arange(round(aug.TARGET_SR * seconds)) / aug.TARGET_SR
    return (amplitude * np.sin(2 * np.pi * frequency * times)).astype(np.float32)


def measured_snr(clean, noisy):
    return 10 * np.log10(
        np.mean(clean.astype(np.float64) ** 2)
        / np.mean((noisy.astype(np.float64) - clean) ** 2)
    )


def peak_frequency(wave):
    return np.argmax(np.abs(np.fft.rfft(wave))) * aug.TARGET_SR / len(wave)


class ConfigTests(unittest.TestCase):
    def test_defaults_and_dictionary_roundtrip(self):
        config = aug.AugmentationConfig()
        for name in aug.EFFECT_ORDER:
            self.assertFalse(getattr(config, name)["enabled"])
            self.assertEqual(getattr(config, name)["p"], 0.5)
        self.assertEqual(config.rawboost["components"], ("convolutive", "impulsive"))
        restored = aug.AugmentationConfig.from_dict(
            json.loads(json.dumps(asdict(config)))
        )
        self.assertEqual(asdict(config), asdict(restored))

    def test_configuration_does_not_mutate_or_share_dictionaries(self):
        supplied = {"speed": {"enabled": True, "rate": [0.9, 1.1]}}
        before = json.dumps(supplied)
        config = aug.AugmentationConfig.from_dict(supplied)
        self.assertEqual(json.dumps(supplied), before)
        supplied["speed"]["rate"][0] = 0
        self.assertEqual(config.speed["rate"], (0.9, 1.1))
        config.noise["enabled"] = True
        self.assertFalse(aug.AugmentationConfig().noise["enabled"])

    def test_invalid_settings_fail_clearly(self):
        invalid = [
            {"unknown": {}},
            {"noise": {"unknown": 1}},
            {"gain": {"enabled": "yes"}},
            {"gain": {"p": -0.1}},
            {"gain": {"p": 1.1}},
            {"gain": {"p": True}},
            {"gain": {"gain_db": [6, -6]}},
            {"speed": {"rate": 0}},
            {"speed": {"rate": [0.9, float("nan")]}},
            {"shift": {"seconds": [0, 1, 2]}},
            {"pitch": []},
            {"reverb": {"rt60_seconds": 0}},
            {"noise": {"kind": "unknown"}},
            {"noise": {"enabled": True, "kind": "musan"}},
            {"channel": {"codec": "amr"}},
            {"channel": {"mode": "unknown"}},
            {"channel": {"ffmpeg_path": ""}},
            {"rawboost": {"components": []}},
            {"rawboost": {"components": ["impulsive", "impulsive"]}},
            {"rawboost": {"components": ["unknown"]}},
            {"rawboost": {"n_bands": 0}},
            {"rawboost": {"nonlinear_order": True}},
            {"rawboost": {"frequency_hz": [20, 8001]}},
            {"rawboost": {"filter_taps": [10.5, 20]}},
            {"rawboost": {"max_impulsive_percent": 101}},
            {"rawboost": {"impulsive_gain": -1}},
            {"peak_ceiling": 1},
        ]
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(ValueError):
                aug.AugmentationConfig.from_dict(config)

    def test_fixed_values_and_json_lists_are_accepted(self):
        config = aug.AugmentationConfig.from_dict(
            {
                "noise": {"snr_db": [10, 20]},
                "speed": {"rate": 1.1},
                "gain": {"gain_db": 3},
            }
        )
        self.assertEqual(config.noise["snr_db"], (10, 20))
        self.assertEqual(config.speed["rate"], (1.1, 1.1))
        self.assertEqual(config.gain["gain_db"], (3, 3))


class WaveformTests(unittest.TestCase):
    def setUp(self):
        self.clean = sine()

    def test_invalid_waveforms_rate_and_rng_are_rejected(self):
        for wave in ([], [float("nan")], [float("inf")], [[0.1, 0.2]], [1 + 2j]):
            with self.subTest(wave=wave), self.assertRaises(ValueError):
                aug.apply(wave, 16000, {}, np.random.default_rng(42))
        with self.assertRaisesRegex(ValueError, "16 kHz"):
            aug.apply(self.clean, 8000, {}, np.random.default_rng(42))
        with self.assertRaisesRegex(ValueError, "Generator"):
            aug.apply(self.clean, 16000, {}, None)

    def test_disabled_pipeline_is_an_independent_exact_copy_without_rng_draws(self):
        clean = np.array([1.0, -1.0, 0.5], dtype=np.float32)
        rng = np.random.default_rng(42)
        before = json.dumps(rng.bit_generator.state)
        result = aug.apply(clean, 16000, {}, rng)
        np.testing.assert_array_equal(result, clean)
        self.assertFalse(np.shares_memory(result, clean))
        self.assertEqual(before, json.dumps(rng.bit_generator.state))

    def test_gaussian_noise_has_requested_snr_and_is_reproducible(self):
        first = aug.add_noise(self.clean, np.random.default_rng(42), 15)
        second = aug.add_noise(self.clean, np.random.default_rng(42), 15)
        np.testing.assert_array_equal(first, second)
        self.assertAlmostEqual(measured_snr(self.clean, first), 15, places=3)
        self.assertEqual(first.dtype, np.float32)
        self.assertEqual(first.shape, self.clean.shape)
        zeros = np.zeros(128, dtype=np.float32)
        np.testing.assert_array_equal(
            aug.add_noise(zeros, np.random.default_rng(42)), zeros
        )

    def test_gain_is_in_db_without_rms_renormalization(self):
        result = aug.random_gain(self.clean, np.random.default_rng(42), (6, 6))
        np.testing.assert_allclose(result, self.clean * 10 ** (6 / 20), rtol=1e-6)

    def test_speed_changes_duration_but_not_pitch(self):
        result = aug.speed(self.clean, 1.1)
        self.assertEqual(len(result), round(len(self.clean) / 1.1))
        self.assertAlmostEqual(peak_frequency(result), 440, delta=3)
        neutral = aug.speed(self.clean, 1)
        np.testing.assert_array_equal(neutral, self.clean)
        self.assertFalse(np.shares_memory(neutral, self.clean))

    def test_pitch_preserves_duration(self):
        result = aug.pitch(self.clean, 16000, 2)
        self.assertEqual(len(result), len(self.clean))
        self.assertAlmostEqual(peak_frequency(result), 440 * 2 ** (2 / 12), delta=4)
        np.testing.assert_array_equal(aug.pitch(self.clean, 16000, 0), self.clean)

    def test_shift_zero_fills_without_wrapping(self):
        wave = np.arange(1, 6, dtype=np.float32)
        np.testing.assert_array_equal(
            aug.time_shift(wave, 16000, 2 / 16000), [0, 0, 1, 2, 3]
        )
        np.testing.assert_array_equal(
            aug.time_shift(wave, 16000, -2 / 16000), [3, 4, 5, 0, 0]
        )
        np.testing.assert_array_equal(aug.time_shift(wave, 16000, 0), wave)
        np.testing.assert_array_equal(
            aug.time_shift(wave, 16000, 1), np.zeros_like(wave)
        )

    def test_reverb_is_causal_with_controlled_tail_energy_and_decay(self):
        impulse = np.zeros(16000, dtype=np.float32)
        impulse[0] = 1
        result = aug.reverb(impulse, 16000, np.random.default_rng(42), 0.1, 6)
        self.assertEqual(result[0], 1)
        self.assertEqual(len(result), len(impulse))
        self.assertAlmostEqual(-10 * np.log10(np.sum(result[1:] ** 2)), 6, places=4)
        energy = np.cumsum(result[1:].astype(np.float64)[::-1] ** 2)[::-1]
        decay_db = 10 * np.log10(np.maximum(energy / energy[0], np.finfo(float).tiny))
        window = (decay_db <= -5) & (decay_db >= -25)
        slope, _ = np.polyfit(
            np.arange(len(energy))[window] / 16000, decay_db[window], 1
        )
        self.assertAlmostEqual(-60 / slope, 0.1, delta=0.02)
        np.testing.assert_array_equal(
            result, aug.reverb(impulse, 16000, np.random.default_rng(42), 0.1, 6)
        )

    def test_individual_effects_do_not_mutate_input(self):
        original = self.clean.copy()
        for transform in (
            lambda: aug.add_noise(self.clean, np.random.default_rng(42)),
            lambda: aug.random_gain(self.clean, np.random.default_rng(42)),
            lambda: aug.speed(self.clean, 1.1),
            lambda: aug.pitch(self.clean, 16000, 2),
            lambda: aug.time_shift(self.clean, 16000, 0.1),
            lambda: aug.reverb(self.clean, 16000, np.random.default_rng(42)),
            lambda: aug.rawboost(self.clean, 16000, np.random.default_rng(42)),
        ):
            result = transform()
            np.testing.assert_array_equal(self.clean, original)
            self.assertFalse(np.shares_memory(result, self.clean))
            self.assertTrue(np.isfinite(result).all())
            self.assertEqual(result.dtype, np.float32)


class MusanTests(unittest.TestCase):
    def test_local_stereo_lower_rate_noise_is_tiled_and_calibrated(self):
        import soundfile as sf

        clean = sine()
        noise = np.random.default_rng(1).normal(0, 0.05, (1000, 2)).astype(np.float32)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "noise.wav"
            sf.write(path, noise, 8000, subtype="FLOAT")
            original_bytes = path.read_bytes()
            first = aug.add_musan_noise(
                clean, 16000, np.random.default_rng(42), folder, 20
            )
            second = aug.add_musan_noise(
                clean, 16000, np.random.default_rng(42), folder, 20
            )
            self.assertEqual(original_bytes, path.read_bytes())
            self.assertEqual(
                [item.name for item in Path(folder).iterdir()], ["noise.wav"]
            )
        self.assertAlmostEqual(measured_snr(clean, first), 20, places=3)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, clean.shape)

    def test_long_noise_uses_random_segments(self):
        import soundfile as sf

        clean = sine(0.25)
        noise = np.random.default_rng(1).normal(0, 0.05, 64000).astype(np.float32)
        with tempfile.TemporaryDirectory() as folder:
            nested = Path(folder) / "noise"
            nested.mkdir()
            sf.write(nested / "long.FLAC", noise, 16000)
            first = aug.add_musan_noise(clean, 16000, np.random.default_rng(42), folder)
            second = aug.add_musan_noise(
                clean, 16000, np.random.default_rng(43), folder
            )
        self.assertFalse(np.array_equal(first, second))
        self.assertAlmostEqual(measured_snr(clean, first), 15, places=3)

    def test_missing_empty_silent_and_corrupt_assets_raise(self):
        import soundfile as sf

        with tempfile.TemporaryDirectory() as root:
            missing = Path(root) / "missing"
            with self.assertRaisesRegex(ValueError, "does not exist"):
                aug.add_musan_noise(sine(), 16000, np.random.default_rng(42), missing)
            with self.assertRaisesRegex(ValueError, "No local"):
                aug.add_musan_noise(sine(), 16000, np.random.default_rng(42), root)
            sf.write(Path(root) / "silent.wav", np.zeros(16000), 16000)
            with self.assertRaisesRegex(ValueError, "silent"):
                aug.add_musan_noise(sine(), 16000, np.random.default_rng(42), root)
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "corrupt.wav").write_bytes(b"not a wav")
            with self.assertRaisesRegex(ValueError, "MUSAN noise file"):
                aug.add_musan_noise(sine(), 16000, np.random.default_rng(42), root)


class RawBoostTests(unittest.TestCase):
    def test_notch_filter_is_float64_and_attenuates_the_stopband(self):
        from scipy.signal import freqz

        config = aug.AugmentationConfig(
            rawboost={
                "n_bands": 1,
                "frequency_hz": 1000,
                "bandwidth_hz": 400,
                "filter_taps": 101,
            }
        )
        coefficients = aug._rawboost_notches(
            16000, np.random.default_rng(42), config.rawboost, (0, 0)
        )
        self.assertEqual(coefficients.dtype, np.float64)
        self.assertEqual(coefficients.shape, (101,))
        self.assertTrue(np.isfinite(coefficients).all())
        _, response = freqz(coefficients, worN=[0, 1000, 3000], fs=16000)
        self.assertLess(abs(response[1]), 0.1 * abs(response[0]))
        self.assertGreater(abs(response[2]), 0.9 * abs(response[0]))

    def test_nonlinear_gain_bounds_are_ordered_for_generator(self):
        with patch.object(
            aug, "_rawboost_notches", return_value=np.array([1.0])
        ) as filters:
            aug.rawboost(
                sine(0.25),
                16000,
                np.random.default_rng(42),
                ("convolutive",),
                nonlinear_order=2,
            )
        np.testing.assert_array_equal(filters.call_args_list[1].args[3], [-20, -5])

    def test_components_and_combinations_are_reproducible_and_length_preserving(self):
        clean = sine(0.25)
        for components in (
            ("convolutive",),
            ("impulsive",),
            ("stationary",),
            ("convolutive", "impulsive"),
            ("convolutive", "impulsive", "stationary"),
        ):
            with self.subTest(components=components):
                first = aug.rawboost(
                    clean, 16000, np.random.default_rng(42), components
                )
                second = aug.rawboost(
                    clean, 16000, np.random.default_rng(42), components
                )
                np.testing.assert_array_equal(first, second)
                self.assertEqual(first.shape, clean.shape)
                self.assertEqual(first.dtype, np.float32)
                self.assertTrue(np.isfinite(first).all())
                self.assertFalse(np.array_equal(clean, first))

    def test_stationary_component_calibrates_snr(self):
        clean = sine()
        result = aug.rawboost(
            clean,
            16000,
            np.random.default_rng(42),
            ("stationary",),
            stationary_snr_db=20,
        )
        self.assertAlmostEqual(measured_snr(clean, result), 20, places=3)

    def test_silence_and_short_inputs_remain_finite(self):
        for clean in (
            np.zeros(100, dtype=np.float32),
            np.array([0.1], dtype=np.float32),
        ):
            result = aug.rawboost(
                clean,
                16000,
                np.random.default_rng(42),
                ("convolutive", "impulsive", "stationary"),
            )
            self.assertEqual(result.shape, clean.shape)
            self.assertTrue(np.isfinite(result).all())
            if not np.any(clean):
                np.testing.assert_array_equal(result, clean)

    def test_unknown_direct_parameters_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown RawBoost"):
            aug.rawboost(sine(), 16000, np.random.default_rng(42), misspelled=1)


class PipelineTests(unittest.TestCase):
    def test_documented_order(self):
        names = {
            "speed": "speed",
            "pitch": "pitch",
            "time_shift": "shift",
            "reverb": "reverb",
            "rawboost": "rawboost",
            "add_noise": "noise",
            "random_gain": "gain",
            "phone_channel": "channel",
        }
        calls = []
        config = {name: {"enabled": True, "p": 1} for name in aug.EFFECT_ORDER}
        with ExitStack() as stack:
            for function, name in names.items():

                def record(wave, *args, effect=name, **kwargs):
                    calls.append(effect)
                    return wave.copy()

                stack.enter_context(patch.object(aug, function, side_effect=record))
            aug.apply(sine(), 16000, config, np.random.default_rng(42))
        self.assertEqual(calls, list(aug.EFFECT_ORDER))

    def test_default_probabilities_are_independent(self):
        config = aug.AugmentationConfig.from_dict(
            {
                "noise": {"enabled": True},
                "gain": {"enabled": True},
            }
        )
        counts = [0, 0, 0, 0]
        rng = np.random.default_rng(42)
        with (
            patch.object(aug, "add_noise", side_effect=lambda wave, *args: wave + 0.1),
            patch.object(
                aug, "random_gain", side_effect=lambda wave, *args: wave + 0.2
            ),
        ):
            for _ in range(2000):
                result = aug.apply(np.zeros(1, dtype=np.float32), 16000, config, rng)
                counts[round(float(result[0]) * 10)] += 1
        for count in counts:
            self.assertTrue(350 < count < 650, counts)

    def test_seeded_pipeline_does_not_use_or_change_global_rng(self):
        config = {
            name: {"enabled": True, "p": 1}
            for name in (
                "speed",
                "pitch",
                "shift",
                "reverb",
                "rawboost",
                "noise",
                "gain",
            )
        }
        state = np.random.get_state()
        first = aug.apply(sine(), 16000, config, np.random.default_rng(42))
        after = np.random.get_state()
        self.assertEqual(state[0], after[0])
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])
        second = aug.apply(sine(), 16000, config, np.random.default_rng(42))
        np.testing.assert_array_equal(first, second)
        third = aug.apply(sine(), 16000, config, np.random.default_rng(43))
        self.assertFalse(np.array_equal(first, third))
        self.assertLessEqual(float(np.max(np.abs(first))), 0.99)

    def test_peak_safety_does_not_renormalize_quiet_audio(self):
        config = {"gain": {"enabled": True, "p": 1, "gain_db": 20}}
        loud = aug.apply(sine(), 16000, config, np.random.default_rng(42))
        self.assertLessEqual(float(np.max(np.abs(loud))), 0.99)
        quiet = aug.apply(
            sine(),
            16000,
            {"gain": {"enabled": True, "p": 1, "gain_db": -6}},
            np.random.default_rng(42),
        )
        np.testing.assert_allclose(quiet, sine() * 10 ** (-6 / 20), rtol=1e-6)

    def test_disabled_or_zero_probability_resources_are_not_accessed(self):
        for enabled in (False, True):
            config = {
                "noise": {
                    "enabled": enabled,
                    "p": 0,
                    "kind": "musan",
                    "folder": "/missing",
                },
                "channel": {
                    "enabled": enabled,
                    "p": 0,
                    "ffmpeg_path": "/missing/ffmpeg",
                },
            }
            with (
                patch.object(
                    aug, "add_musan_noise", side_effect=AssertionError("noise accessed")
                ),
                patch.object(
                    aug, "_ffmpeg", side_effect=AssertionError("ffmpeg accessed")
                ),
            ):
                np.testing.assert_array_equal(
                    aug.apply(sine(), 16000, config, np.random.default_rng(42)), sine()
                )

    def test_selected_resource_errors_name_the_effect(self):
        config = {
            "channel": {
                "enabled": True,
                "p": 1,
                "mode": "codec",
                "ffmpeg_path": "/missing/ffmpeg",
            }
        }
        with self.assertRaisesRegex(ValueError, "channel augmentation failed.*FFmpeg"):
            aug.apply(sine(), 16000, config, np.random.default_rng(42))

    def test_import_does_not_load_optional_libraries_or_write_outputs(self):
        root = str(Path(__file__).resolve().parents[1])
        script = """
import os
import sys
before = os.listdir('.')
import src.augment
assert not {'librosa', 'scipy', 'soundfile', 'torch', 'datasets', 'transformers'} & sys.modules.keys()
assert os.listdir('.') == before
"""
        with tempfile.TemporaryDirectory() as folder:
            subprocess.run(
                [sys.executable, "-c", script],
                cwd=folder,
                check=True,
                env={**os.environ, "PYTHONPATH": root, "PYTHONDONTWRITEBYTECODE": "1"},
                capture_output=True,
            )


class CodecTests(unittest.TestCase):
    def test_missing_ffmpeg_and_failed_codec_have_clear_errors(self):
        with self.assertRaisesRegex(RuntimeError, "FFmpeg unavailable"):
            aug.codec(sine(), 16000, ffmpeg_path="/missing/ffmpeg")
        failed = subprocess.CompletedProcess(
            [], 1, stdout=b"", stderr=b"Unknown encoder"
        )
        with patch.object(aug.subprocess, "run", return_value=failed):
            with self.assertRaisesRegex(RuntimeError, "Unknown encoder"):
                aug.codec(sine(), 16000)

    def test_codec_timeout_has_clear_error(self):
        with patch.object(
            aug.subprocess, "run", side_effect=subprocess.TimeoutExpired("ffmpeg", 30)
        ):
            with self.assertRaisesRegex(RuntimeError, "FFmpeg unavailable or failed"):
                aug.codec(sine(), 16000)

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg executable unavailable")
    def test_real_g711_roundtrips(self):
        clean = sine()
        original = clean.copy()
        for name in ("mulaw", "alaw"):
            with self.subTest(codec=name):
                result = aug.codec(clean, 16000, name)
                self.assertEqual(result.shape, clean.shape)
                self.assertEqual(result.dtype, np.float32)
                self.assertTrue(np.isfinite(result).all())
                self.assertFalse(np.array_equal(result, clean))
                self.assertGreater(measured_snr(clean, result), 25)
                np.testing.assert_array_equal(result, aug.codec(clean, 16000, name))
                np.testing.assert_array_equal(clean, original)

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg executable unavailable")
    def test_phone_channel_length_and_bandwidth(self):
        for length in (16000, 16001):
            clean = sine(seconds=length / 16000)
            result = aug.phone_channel(clean, 16000)
            self.assertEqual(len(result), length)
            self.assertEqual(result.dtype, np.float32)
            self.assertTrue(np.isfinite(result).all())
            self.assertAlmostEqual(peak_frequency(result), 440, delta=2)
        high = aug.phone_channel(sine(frequency=6000), 16000)
        low = aug.phone_channel(sine(frequency=1000), 16000)
        self.assertLess(np.sqrt(np.mean(high**2)), 0.02 * np.sqrt(np.mean(low**2)))


if __name__ == "__main__":
    unittest.main()
