"""Milestone-4 run-config checks; no corpus, model, GPU or network access.

Run from the repository root:
    python -m unittest discover -s tests -p 'test_configs.py' -v
"""

import copy
import shutil
import sys
import unittest
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import yaml

# Support repository-root discovery and direct execution after relocation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import augment as aug

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
V_RUNS = [f"V-{number:02d}" for number in range(1, 12)]
RUN_IDS = ["T-01", *V_RUNS]

# run_id: (owner, experiment.param, experiment.value)
METADATA = {
    "T-01": ("Tarika", None, None),
    "V-01": ("Vikram", "model.dropout", 0.0),
    "V-02": ("Vikram", "model.dropout", 0.3),
    "V-03": ("Vikram", "model.dropout", 0.5),
    "V-04": ("Vikram", "train.weight_decay", 0.0),
    "V-05": ("Vikram", "train.weight_decay", 0.1),
    "V-06": ("Vikram", "augment", "noise + gain"),
    "V-07": ("Vikram", "augment", "speed + pitch + reverb"),
    "V-08": ("Vikram", "augment", "rawboost"),
    "V-09": ("Vikram", "augment", "codec / phone"),
    "V-10": ("Vikram", "augment", "all"),
    "V-11": ("Vikram", "stability", "no warmup, no grad clipping"),
}

# Every V config must equal default.yaml apart from these keys and the
# common sweep protocol.
CHANGES = {
    "V-01": {"model.dropout": 0.0},
    "V-02": {"model.dropout": 0.3},
    "V-03": {"model.dropout": 0.5},
    "V-04": {"train.weight_decay": 0.0},
    "V-05": {"train.weight_decay": 0.1},
    "V-06": {
        "augment.noise.enabled": True,
        "augment.gain.enabled": True,
    },
    "V-07": {
        "augment.speed.enabled": True,
        "augment.pitch.enabled": True,
        "augment.reverb.enabled": True,
    },
    "V-08": {"augment.rawboost.enabled": True},
    "V-09": {
        "augment.channel.enabled": True,
        "augment.channel.mode": "phone",
    },
    "V-10": {
        "augment.speed.enabled": True,
        "augment.pitch.enabled": True,
        "augment.shift.enabled": True,
        "augment.reverb.enabled": True,
        "augment.rawboost.enabled": True,
        "augment.noise.enabled": True,
        "augment.gain.enabled": True,
        "augment.channel.enabled": True,
        "augment.channel.mode": "phone",
    },
    "V-11": {
        "train.warmup_ratio": 0.0,
        "train.grad_clip_norm": None,
    },
}

IGNORED = {
    "run_id",
    "experiment.owner",
    "experiment.param",
    "experiment.value",
    "data.tune_subset",
    "train.epochs",
}


def sine(seconds=0.5, frequency=440.0, amplitude=0.2):
    times = np.arange(round(aug.TARGET_SR * seconds)) / aug.TARGET_SR
    return (amplitude * np.sin(2 * np.pi * frequency * times)).astype(np.float32)


def load(run_id):
    path = CONFIGS / f"{run_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Missing run config: {path}")
    return yaml.safe_load(path.read_text())


def flatten(mapping, prefix=""):
    flat = {}
    for key, value in mapping.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            flat.update(flatten(value, dotted))
        else:
            flat[dotted] = value
    return flat


def comparable(config):
    return {key: value for key, value in flatten(config).items() if key not in IGNORED}


class ConfigFileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.default = load("default")
        cls.default_comparable = comparable(cls.default)

    def test_run_configs_exist_and_match_their_file_names(self):
        self.assertEqual(
            sorted(path.stem for path in CONFIGS.glob("V-*.yaml")),
            V_RUNS,
        )
        for run_id in RUN_IDS:
            with self.subTest(run_id=run_id):
                self.assertEqual(load(run_id)["run_id"], run_id)

    def test_common_sweep_protocol_and_metadata(self):
        for run_id in RUN_IDS:
            with self.subTest(run_id=run_id):
                config = load(run_id)
                self.assertEqual(config["seed"], 42)
                self.assertIs(config["data"]["tune_subset"], True)
                self.assertEqual(config["train"]["epochs"], 3)
                owner, param, value = METADATA[run_id]
                self.assertEqual(config["experiment"]["owner"], owner)
                self.assertEqual(config["experiment"]["param"], param)
                self.assertEqual(config["experiment"]["value"], value)

    def test_default_config_is_still_the_full_data_baseline(self):
        self.assertEqual(self.default["train"]["epochs"], 5)
        self.assertFalse(self.default["data"].get("tune_subset", False))
        for name in aug.EFFECT_ORDER:
            with self.subTest(effect=name):
                self.assertFalse(self.default["augment"][name]["enabled"])

    def test_t01_is_default_apart_from_the_sweep_protocol(self):
        self.assertEqual(comparable(load("T-01")), self.default_comparable)

    def test_each_v_run_changes_only_its_setting(self):
        for run_id, changes in CHANGES.items():
            expected = dict(self.default_comparable)
            expected.update(changes)
            with self.subTest(run_id=run_id):
                self.assertEqual(comparable(load(run_id)), expected)

    def test_augmentation_blocks_validate_with_expected_effects(self):
        for run_id in V_RUNS:
            config = load(run_id)
            parsed = aug.AugmentationConfig.from_dict(config["augment"])
            with self.subTest(run_id=run_id):
                for name in aug.EFFECT_ORDER:
                    expected = CHANGES.get(run_id, {}).get(
                        f"augment.{name}.enabled", False
                    )
                    self.assertEqual(getattr(parsed, name)["enabled"], expected)
                    if expected:
                        self.assertEqual(getattr(parsed, name)["p"], 0.5)

    def test_augmentation_defaults_are_not_duplicated_in_yaml(self):
        for run_id in V_RUNS:
            for name in aug.EFFECT_ORDER:
                supplied = load(run_id)["augment"][name]
                with self.subTest(run_id=run_id, effect=name):
                    self.assertTrue(set(supplied).issubset({"enabled", "mode"}))
                    self.assertNotIn("p", supplied)

    def test_expected_module_defaults(self):
        v06 = aug.AugmentationConfig.from_dict(load("V-06")["augment"])
        self.assertEqual(v06.noise["kind"], "gaussian")
        self.assertEqual(v06.noise["snr_db"], (15.0, 15.0))
        self.assertEqual(v06.gain["gain_db"], (-6.0, 6.0))

        v07 = aug.AugmentationConfig.from_dict(load("V-07")["augment"])
        self.assertEqual(v07.speed["rate"], (0.9, 1.1))
        self.assertEqual(v07.pitch["semitones"], (-2.0, 2.0))
        self.assertEqual(v07.reverb["rt60_seconds"], 0.3)
        self.assertEqual(v07.reverb["drr_db"], 6.0)

        v08 = aug.AugmentationConfig.from_dict(load("V-08")["augment"])
        self.assertEqual(v08.rawboost["components"], ("convolutive", "impulsive"))

        for run_id in ("V-09", "V-10"):
            channel = aug.AugmentationConfig.from_dict(load(run_id)["augment"]).channel
            with self.subTest(run_id=run_id):
                self.assertEqual(channel["mode"], "phone")
                self.assertEqual(channel["codec"], "random")

    def test_v11_disables_warmup_and_gradient_clipping(self):
        train = load("V-11")["train"]
        self.assertEqual(train["warmup_ratio"], 0.0)
        self.assertIsNone(train["grad_clip_norm"])

    def test_augmentation_configs_apply_to_a_synthetic_wave(self):
        wave = sine()
        for run_id in ("V-06", "V-07", "V-08", "V-09", "V-10"):
            augment_cfg = copy.deepcopy(load(run_id)["augment"])
            for name in aug.EFFECT_ORDER:
                if augment_cfg[name]["enabled"]:
                    augment_cfg[name]["p"] = 1.0

            if run_id in ("V-09", "V-10") and shutil.which("ffmpeg") is None:
                if run_id == "V-09":
                    continue
                augment_cfg["channel"]["enabled"] = False

            with self.subTest(run_id=run_id):
                result = aug.apply(
                    wave,
                    aug.TARGET_SR,
                    augment_cfg,
                    np.random.default_rng(42),
                )
                self.assertEqual(result.dtype, np.float32)
                self.assertEqual(result.ndim, 1)
                self.assertGreater(result.size, 0)
                self.assertTrue(np.isfinite(result).all())
                if run_id in ("V-07", "V-10"):
                    # Speed changes duration; all other effects preserve it.
                    self.assertLessEqual(
                        abs(result.size - wave.size),
                        int(0.15 * wave.size) + 2,
                    )
                else:
                    self.assertEqual(result.size, wave.size)


if __name__ == "__main__":
    unittest.main()
