# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Frozen XLS-R — reproducible classifier comparison
#
# Convert with `uvx jupytext --to ipynb model_notebook.py`, upload to Colab,
# select a GPU runtime, and run cells in order. **Edit SOURCE_CONFIGS first:**
# paths must point to the full, already-preprocessed `DatasetDict` directories,
# not their parent folders or preprocessing smoke artifacts. No raw dataset is
# downloaded, re-split, augmented, or overwritten here.
#
# **Labels:** 0 = bonafide; 1 = spoof. XLS-R is completely frozen, in evaluation
# mode, with SpecAugment off. Its configurable layer-6 frame features feed nine
# traditional/XGBoost models, three PyTorch heads, and a dummy baseline.
# Traditional models/MLP use temporal mean/std vectors; ASP/CNN learn from the
# corresponding frame sequences. ASP/CNN are research-inspired baselines,
# not reproductions of a published system.
#
# Start in **smoke** mode (32/16/16 clips per sampling group). Comparison uses
# configurable 400/100/100 group quotas. Every group supplies equal numbers of
# both labels; explicitly pair real-only and synthetic-only sources if needed.
# Full recordings are used unless MAX_SECONDS is explicitly set. A long clip
# can still exhaust a free GPU: reduce the batch size or enable that cap yourself.
#
# Rank by **validation EER**, never test results. Test scoring is a separate,
# opt-in final cell. Features, models, and reports persist on Drive; confirm
# actual account capacity before enabling new publication. Completed artifacts
# are immutable. Incomplete/corrupt outputs need a new work/output location;
# this notebook never deletes or silently repairs them.

# %% Setup — preserve Colab's PyTorch/CUDA; install before importing.
# ruff: noqa: E402
# Dependency installation intentionally precedes imports in this notebook.
from __future__ import annotations

import importlib.metadata
import json
import subprocess
import sys

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion

NOTEBOOK_RUN = __name__ == "__main__"


def installed_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def ensure_dependencies():
    """Install bounded APIs without replacing the runtime's PyTorch/CUDA stack."""
    version = installed_version("torch")
    if version is None:
        raise RuntimeError("Select a Colab PyTorch GPU runtime before running setup.")
    pair = tuple(int(part) for part in version.split("+")[0].split(".")[:2])
    codec_versions = {
        (2, 8): "0.7.0",
        (2, 9): "0.9.0",
        (2, 10): "0.10.0",
        (2, 11): "0.16.0",
    }
    if pair not in codec_versions:
        raise RuntimeError(
            f"PyTorch {version} needs a verified TorchCodec pairing. Check "
            "https://github.com/meta-pytorch/torchcodec#installing-torchcodec; "
            "do not upgrade Colab's torch/CUDA stack blindly."
        )
    pins = {
        "datasets": ("4.4.1", "datasets"),
        "transformers": ("4.57.3", "transformers"),
        "scikit-learn": ("1.7.2", "sklearn"),
        "soundfile": ("0.13.1", "soundfile"),
    }
    requirements = []
    for name, (wanted, module) in pins.items():
        if installed_version(name) != wanted:
            if module in sys.modules:
                raise RuntimeError(
                    f"Restart the runtime before replacing imported {module}."
                )
            requirements.append(f"{name}=={wanted}")
    for name, wanted in {"numpy": "2.3.5", "pandas": "2.3.3"}.items():
        if installed_version(name) is None:
            requirements.append(f"{name}=={wanted}")
    providers = [name for name in ("xgboost", "xgboost-cpu") if installed_version(name)]
    if len(providers) > 1:
        raise RuntimeError(
            "Both xgboost and xgboost-cpu are installed and share the same import. "
            "Start a clean Colab runtime with just one provider; setup will not "
            "uninstall or replace existing XGBoost packages."
        )
    if providers:
        provider = providers[0]
        xgb_version = installed_version(provider)
        # 2.1.4 fixes sklearn's newer estimator tags; all supported versions
        # provide CPU hist/device parameters and native UBJSON persistence.
        try:
            compatible = SpecifierSet(">=2.1.4,<4", prereleases=False).contains(
                xgb_version
            )
        except InvalidVersion:
            compatible = False
        if not compatible:
            raise RuntimeError(
                f"Installed {provider}=={xgb_version} is unsupported: this notebook "
                "requires a stable XGBoost >=2.1.4,<4 with scikit-learn 1.7.2. "
                "Use a compatible Colab runtime; existing XGBoost will not be replaced."
            )
        print(f"Reusing installed {provider}=={xgb_version} (CPU histogram training).")
    else:
        # CPU-only fallback for environments without either provider, not Colab's
        # already-installed compatible XGBoost (which may include GPU support).
        requirements.append("xgboost-cpu==3.1.2")
    if requirements:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", *requirements],
            check=True,
        )
    codec = codec_versions[pair]
    if (installed_version("torchcodec") or "").split("+")[0] != codec:
        if "torchcodec" in sys.modules:
            raise RuntimeError(
                "Restart the runtime before replacing imported TorchCodec."
            )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--no-deps",
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
                f"torchcodec=={codec}",
            ],
            check=True,
        )
    subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.DEVNULL)


if NOTEBOOK_RUN:
    ensure_dependencies()

import copy
import csv
import hashlib
import io
import os
import random
import shutil
import tempfile
import time
import warnings
from collections import Counter
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from datasets import (
    Audio,
    ClassLabel,
    Dataset,
    DatasetDict,
    Features,
    Value,
    load_from_disk,
)
from sklearn.metrics import roc_curve
from torch import nn
from torch.nn import functional as F
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Model


# %% Configuration — edit paths, groups, budgets, and overrides here.
# Keep configuration's standard-library dependencies local to this cell.
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

SPLITS = ("train", "validation", "test")
LABEL_NAMES = ["bonafide", "spoof"]
TARGET_SR = 16_000
CONV_KERNELS = (10, 3, 3, 3, 3, 2, 2)
CONV_STRIDES = (5, 2, 2, 2, 2, 2, 2)


def in_colab():
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


@dataclass(frozen=True)
class RunConfig:
    mode: str = "smoke"
    seed: int = 42
    model_id: str = "facebook/wav2vec2-xls-r-300m"
    model_revision: str = "1a640f32ac3e39899438a2931f9924c02f080a54"
    layer: int = 6
    max_seconds: float | None = None
    extraction_batch_size: int = 1
    mlp_batch_size: int = 64
    sequence_batch_size: int = 16
    num_workers: int = 0
    cpu_threads: int = 2
    max_epochs: int = 20
    smoke_epochs: int = 2
    patience: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    head_device: str = "auto"
    run_tag: str = "v1"

    def validate(self):
        if self.mode not in ("smoke", "comparison"):
            raise ValueError("mode must be 'smoke' or 'comparison'.")
        integers = (
            self.seed,
            self.layer,
            self.extraction_batch_size,
            self.mlp_batch_size,
            self.sequence_batch_size,
            self.num_workers,
            self.cpu_threads,
            self.max_epochs,
            self.smoke_epochs,
            self.patience,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in integers
        ):
            raise ValueError(
                "Seeds, sizes, layer, workers, and epoch controls must be integers."
            )
        if (
            not 0 <= self.seed < 2**32
            or self.num_workers < 0
            or min(
                self.layer,
                self.extraction_batch_size,
                self.mlp_batch_size,
                self.sequence_batch_size,
            )
            < 1
        ):
            raise ValueError(
                "Invalid nonpositive size/layer or negative seed/worker count."
            )
        if min(self.cpu_threads, self.max_epochs, self.smoke_epochs, self.patience) < 1:
            raise ValueError("Thread and epoch controls must be positive.")
        for name in ("learning_rate", "grad_clip"):
            if (
                type(getattr(self, name)) not in (int, float)
                or not math.isfinite(getattr(self, name))
                or getattr(self, name) <= 0
            ):
                raise ValueError(f"{name} must be finite and positive.")
        if (
            type(self.weight_decay) not in (int, float)
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0
        ):
            raise ValueError("weight_decay must be finite and nonnegative.")
        if self.max_seconds is not None and (
            type(self.max_seconds) not in (int, float)
            or not math.isfinite(self.max_seconds)
            or self.max_seconds <= 0
        ):
            raise ValueError("An enabled max_seconds cap must be finite and positive.")
        if not self.run_tag or any(
            char
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for char in self.run_tag
        ):
            raise ValueError("run_tag must use only letters, numbers, '-' and '_'.")


CONFIG = RunConfig()  # Change mode='comparison' only after smoke checks pass.
MAX_SECONDS = None  # Optional explicit cap; no automatic cropping on OOM.
CONFIG = replace(CONFIG, max_seconds=MAX_SECONDS)
SOURCE_CONFIGS = {
    "asv": {"path": "", "language": "en"},  # Exact full save_to_disk folder on Drive.
    "itw": {"path": "", "language": "en"},
}
# Future source options: audio_column, label_column, id_column, label_map,
# speaker_column, num_samples_column, language_column, language, require_manifest.
# Do not set require_manifest=False for existing ASV/ITW artifacts.
GROUPS = {
    "asv": [{"source": "asv", "filters": {}}],
    "itw": [{"source": "itw", "filters": {}}],
}
# Example future group: [{'source': 'kathbath', 'filters': {'language': 'hi'}},
#                        {'source': 'synthetic_hi', 'filters': {}}].
PRESETS = {
    "smoke": {"train": 32, "validation": 16, "test": 16},
    "comparison": {"train": 400, "validation": 100, "test": 100},
}
GROUP_QUOTAS = {name: dict(PRESETS[CONFIG.mode]) for name in GROUPS}
ENABLED_CLASSIFIERS = (
    "dummy",
    "logistic_regression",
    "linear_svm",
    "rbf_svm",
    "knn",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "gaussian_nb",
    "xgboost",
    "mlp",
    "asp",
    "temporal_cnn",
)
CLASSIFIER_OVERRIDES = {}  # e.g. {'random_forest': {'n_estimators': 300}}
MOUNT_GOOGLE_DRIVE = in_colab()
LOCAL_ROOT = Path("/content/model_work" if MOUNT_GOOGLE_DRIVE else "./model_work")
DRIVE_ROOT = Path("/content/drive/MyDrive/ds-ai-lab/model_experiments")
DRIVE_QUOTA_CONFIRMED = False  # Check actual account quota before NEW publication.
RUN_TEST_EVALUATION = False  # Final evaluation only, after model settings are fixed.


# %% Shared, side-effect-free helpers.
def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def json_digest(value):
    # Default separators deliberately match the existing ITW sidecar contract.
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, allow_nan=False, default=json_default
        ).encode()
    ).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=json_default,
        )


def read_json(path):
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON constant: {value}")

    return json.loads(
        Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant
    )


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(seed, *parts):
    return int(json_digest([int(seed), *parts])[:16], 16) % (2**32)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def package_versions():
    names = (
        "torch",
        "torchcodec",
        "datasets",
        "transformers",
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "xgboost-cpu",
        "soundfile",
        "joblib",
        "scipy",
    )
    return {
        "python": sys.version.split()[0],
        **{name: installed_version(name) for name in names},
    }


def conv_frame_count(num_samples, kernels=CONV_KERNELS, strides=CONV_STRIDES):
    length = int(num_samples)
    for kernel, stride in zip(kernels, strides, strict=True):
        if length < kernel:
            return 0
        length = (length - kernel) // stride + 1
    return length


def expect_error(callback, error=ValueError):
    try:
        callback()
    except error:
        return
    raise AssertionError(f"Expected {error.__name__}.")


# %% Metric contracts — deliberately include coarse/tied-score operating points.
def metric_inputs(labels, scores, require_both=True):
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    if (
        labels.ndim != 1
        or scores.ndim != 1
        or not len(labels)
        or len(labels) != len(scores)
    ):
        raise ValueError("Expected equally sized nonempty label/score vectors.")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Labels must be 0/1 and every score must be finite.")
    labels = labels.astype(np.int64)
    if require_both and len(np.unique(labels)) != 2:
        raise ValueError(
            "EER/threshold selection requires both bonafide and spoof examples."
        )
    return labels, scores


def equal_error_rate(labels, scores):
    labels, scores = metric_inputs(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1, drop_intermediate=False)
    fnr = 1.0 - tpr
    difference = fpr - fnr
    right = int(np.flatnonzero(difference >= 0)[0])
    if difference[right] == 0:
        return float(fpr[right])
    left = right - 1
    fraction = -difference[left] / (difference[right] - difference[left])
    return float(fpr[left] + fraction * (fpr[right] - fpr[left]))


def operating_threshold(labels, scores):
    labels, scores = metric_inputs(labels, scores)
    fpr, tpr, thresholds = roc_curve(
        labels, scores, pos_label=1, drop_intermediate=False
    )
    thresholds[0] = np.nextafter(scores.max(), np.inf)
    if not np.isfinite(thresholds).all():
        raise ValueError(
            "Cannot construct a finite all-negative threshold for these scores."
        )
    fnr = 1.0 - tpr
    best = min(
        range(len(thresholds)),
        key=lambda i: (abs(fpr[i] - fnr[i]), (fpr[i] + fnr[i]) / 2, -thresholds[i]),
    )
    return {
        "threshold": float(thresholds[best]),
        "fpr": float(fpr[best]),
        "fnr": float(fnr[best]),
    }


def metric_summary(labels, scores, threshold):
    labels, scores = metric_inputs(labels, scores, require_both=False)
    if not math.isfinite(threshold):
        raise ValueError("The saved operating threshold must be finite.")
    # Float64 comparison preserves the finite sentinel above tied float32 scores.
    predictions = scores >= np.float64(threshold)
    positive = labels == 1
    tp = int(np.sum(predictions & positive))
    fp = int(np.sum(predictions & ~positive))
    fn = int(np.sum(~predictions & positive))
    tn = int(np.sum(~predictions & ~positive))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "eer": equal_error_rate(labels, scores)
        if positive.any() and (~positive).any()
        else None,
        "accuracy": (tp + tn) / len(labels),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        "support": len(labels),
        "bonafide": int((~positive).sum()),
        "spoof": int(positive.sum()),
        "predicted_spoof": int(predictions.sum()),
        "fpr": fp / (fp + tn) if fp + tn else None,
        "fnr": fn / (fn + tp) if fn + tp else None,
    }


def run_metric_checks():
    labels = np.array([0, 0, 1, 1])
    assert equal_error_rate(labels, [0.1, 0.2, 0.8, 0.9]) == 0.0
    assert equal_error_rate(labels, [0.9, 0.8, 0.2, 0.1]) == 1.0
    assert equal_error_rate(labels, [0.5] * 4) == 0.5
    assert equal_error_rate(labels, [0.1, 0.5, 0.5, 0.9]) == 0.25
    point = operating_threshold(labels, [0.5] * 4)
    assert math.isfinite(point["threshold"]) and point["threshold"] > 0.5
    metrics = metric_summary(
        labels, np.full(4, 0.5, dtype=np.float32), point["threshold"]
    )
    assert metrics["predicted_spoof"] == 0 and metrics["accuracy"] == 0.5
    assert metrics["precision"] == metrics["recall"] == metrics["f1"] == 0.0
    assert metric_summary([1, 1], [0.7, 0.9], 0.8)["eer"] is None
    expect_error(lambda: equal_error_rate([1, 1], [0.7, 0.9]))
    expect_error(lambda: equal_error_rate(labels, [0.1, np.nan, 0.8, 0.9]))
    before = operating_threshold(labels, [0.1, 0.2, 0.8, 0.9])
    metric_summary([1, 0, 1, 0], [0.9, 0.1, 0.7, 0.2], before["threshold"])
    assert before == operating_threshold(labels, [0.1, 0.2, 0.8, 0.9])
    print(
        "PASS: EER, attainable thresholds, spoof-positive metrics, and undefined groups"
    )


# %% Source configuration and deterministic original-row selection.
from itertools import combinations
from pathlib import PurePosixPath


@dataclass(frozen=True)
class SourceSpec:
    path: str
    audio_column: str = "audio"
    label_column: str = "key"
    id_column: str = "audio_file_name"
    label_map: dict = field(default_factory=lambda: {0: 0, 1: 1})
    speaker_column: str | None = "speaker_id"
    num_samples_column: str | None = "num_samples"
    language_column: str | None = None
    language: str | None = None
    require_manifest: bool = True

    def __post_init__(self):
        for name in ("path", "audio_column", "label_column", "id_column"):
            _data_string(getattr(self, name), name)
        for name in (
            "speaker_column",
            "num_samples_column",
            "language_column",
            "language",
        ):
            if getattr(self, name) is not None:
                _data_string(getattr(self, name), name)
        if self.language_column is not None and self.language is not None:
            raise ValueError(
                "Configure a language column OR a constant language, not both."
            )
        if self.audio_column in {
            self.label_column,
            self.id_column,
            self.speaker_column,
            self.num_samples_column,
            self.language_column,
        }:
            raise ValueError("Audio and metadata column mappings must be distinct.")
        if type(self.require_manifest) is not bool:
            raise ValueError("require_manifest must be an explicit boolean.")
        if not isinstance(self.label_map, dict) or not self.label_map:
            raise ValueError("An explicit, nonempty label_map is required.")
        for original, canonical in self.label_map.items():
            if (
                type(original) not in (str, int)
                or (type(original) is str and not original.strip())
                or type(canonical) is not int
                or canonical not in (0, 1)
            ):
                raise ValueError(
                    "label_map must map explicit string/integer labels to integer 0/1."
                )


def _data_string(value, context):
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or "\x00" in value
    ):
        raise ValueError(f"{context}: expected a nonempty, unmodified string.")
    return value


def _data_label(value, spec, context):
    # bool and float equality with 0/1 must not turn ambiguous labels into valid ones.
    if type(value) not in (str, int):
        raise ValueError(f"{context}: ambiguous/unmapped label {value!r}.")
    for original, canonical in spec.label_map.items():
        if type(value) is type(original) and value == original:
            return canonical
    raise ValueError(
        f"{context}: unmapped label {value!r}; configure label_map explicitly."
    )


def _data_ref(row):
    return row["source"], row["split"], row["row_index"]


def _data_sort_key(row):
    return row.get("group", ""), row["source"], row["clip_id"], row["row_index"]


def _data_filter_matches(row, filters):
    for name, expected in filters.items():
        if name in row and name != "attributes":
            actual = row[name]
        elif name in row["attributes"]:
            actual = row["attributes"][name]
        else:
            raise ValueError(
                f"{row['source']}: filter {name!r} has no configured metadata."
            )
        if actual is None and name in ("language", "speaker", "num_samples"):
            raise ValueError(
                f"{row['source']}: filter {name!r} requires known metadata."
            )
        if actual != expected:
            return False
    return True


def select_rows(metadata, groups, quotas, seed=42):
    """Balanced group -> split quotas; select original rows, never create splits.

    Every per-label pool has one seeded permutation. Smaller quotas select prefixes
    of that permutation; the returned reporting rows are then canonically sorted.
    """
    if set(metadata) != set(SPLITS) or not groups or set(groups) != set(quotas):
        raise ValueError(
            "Require train/validation/test metadata and identical nonempty group/quota keys."
        )
    available_sources, references = set(), set()
    for split in SPLITS:
        for row in metadata[split]:
            if (
                row.get("split") != split
                or type(row.get("row_index")) is not int
                or row["row_index"] < 0
                or type(row.get("label")) is not int
                or row["label"] not in (0, 1)
                or not isinstance(row.get("attributes"), dict)
            ):
                raise ValueError(f"{split}: invalid canonical metadata row.")
            _data_string(row.get("source"), "source")
            _data_string(row.get("clip_id"), "clip_id")
            if _data_ref(row) in references:
                raise ValueError("Repeated original row reference in metadata.")
            references.add(_data_ref(row))
            available_sources.add(row["source"])
    pools, shortages = {}, []
    for group in sorted(groups):
        _data_string(group, "group")
        members = groups[group]
        if not isinstance(members, list) or not members:
            raise ValueError(f"{group}: configure a nonempty list of source members.")
        if not isinstance(quotas[group], dict) or set(quotas[group]) != set(SPLITS):
            raise ValueError(
                f"{group}: quotas must explicitly name all three original splits."
            )
        for member in members:
            if not isinstance(member, dict) or set(member) - {"source", "filters"}:
                raise ValueError(
                    f"{group}: members accept only source and equality filters."
                )
            if member.get("source") not in available_sources:
                raise ValueError(f"{group}: unknown source {member.get('source')!r}.")
            filters = member.get("filters", {})
            if not isinstance(filters, dict) or any(
                not isinstance(key, str)
                or type(value) not in (str, int, float, bool, type(None))
                or (type(value) is float and not math.isfinite(value))
                for key, value in filters.items()
            ):
                raise ValueError(
                    f"{group}: filters must be explicit scalar equality predicates."
                )
        for split in SPLITS:
            total = quotas[group][split]
            if type(total) is not int or total <= 0 or total % 2:
                raise ValueError(
                    f"{group}/{split}: quota must be a positive even integer, got {total!r}."
                )
            # Union overlapping member views inside one group, without weighting a row twice.
            eligible = {}
            for member in members:
                for row in metadata[split]:
                    if row["source"] == member["source"] and _data_filter_matches(
                        row, member.get("filters", {})
                    ):
                        eligible[_data_ref(row)] = row
            for label in (0, 1):
                pool = sorted(
                    (row for row in eligible.values() if row["label"] == label),
                    key=_data_sort_key,
                )
                pools[group, split, label] = pool
                if len(pool) < total // 2:
                    shortages.append(
                        f"{group}/{split}/label={label}: requested={total // 2}, available={len(pool)}"
                    )
    if shortages:
        raise ValueError(
            "Insufficient group quotas: "
            + "; ".join(shortages)
            + ". Reduce explicit quotas or fix group membership/filters; no rows were decoded."
        )
    selection, used = {split: [] for split in SPLITS}, {}
    for group in sorted(groups):
        for split in SPLITS:
            for label in (0, 1):
                pool = list(pools[group, split, label])
                random.Random(
                    stable_seed(seed, "selection", group, split, str(label))
                ).shuffle(pool)
                for row in pool[: quotas[group][split] // 2]:
                    reference = _data_ref(row)
                    if reference in used:
                        raise ValueError(
                            f"Overlapping group selections {used[reference]!r}/{group!r} reuse original row {reference}; configure disjoint views."
                        )
                    used[reference] = group
                    selection[split].append({**copy.deepcopy(row), "group": group})
    return {split: sorted(selection[split], key=_data_sort_key) for split in SPLITS}


# %% Read completion inventories and immutable ITW assignments before decoding audio.
def _data_sha256(value, context):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context}: expected a lowercase SHA256 digest.")
    return value


def _data_relative_path(value):
    _data_string(value, "artifact file path")
    path = PurePosixPath(value)
    if (
        not path.parts
        or path.is_absolute()
        or str(path) != value
        or "\\" in value
        or ":" in value
        or any(part in (".", "..") for part in path.parts)
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"Unsafe relative artifact path: {value!r}.")
    return value


def _data_load_manifest(root, spec):
    marker, pending = root / "manifest.json", root / ".manifest.pending"
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"{root}: source must be an existing, non-symlink directory.")
    if pending.exists() or pending.is_symlink():
        raise ValueError(f"{root}: incomplete artifact (.manifest.pending exists).")
    if not marker.exists() and not marker.is_symlink():
        if spec.require_manifest:
            raise ValueError(
                f"{root}: missing completion manifest; use a full prepared artifact."
            )
        # An explicit opt-out permits no marker, never an invalid marker or unsafe files.
        if any(path.is_symlink() for path in root.rglob("*")):
            raise ValueError(f"{root}: source files must not contain symlinks.")
        return None
    if marker.is_symlink() or not marker.is_file():
        raise ValueError(f"{root}: invalid completion manifest.")
    manifest = read_json(marker)
    if (
        not isinstance(manifest, dict)
        or manifest.get("status") != "complete"
        or manifest.get("kind") != "full"
    ):
        raise ValueError(
            f"{root}: manifest must declare status=complete, kind=full; smoke/partial inputs are not allowed."
        )
    _data_sha256(manifest.get("fingerprint"), f"{root}/manifest fingerprint")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError(f"{root}: missing manifest file inventory.")
    for relative, entry in files.items():
        _data_relative_path(relative)
        if relative in ("manifest.json", ".manifest.pending"):
            raise ValueError(f"{root}: invalid inventory marker entry.")
        if (
            not isinstance(entry, dict)
            or set(entry) != {"size", "sha256"}
            or type(entry["size"]) is not int
            or entry["size"] < 0
        ):
            raise ValueError(
                f"{root}/{relative}: file inventory entries must be exactly size and sha256."
            )
        _data_sha256(entry["sha256"], f"{root}/{relative}")
    actual = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(
                f"{root}: portable source artifacts must not contain symlinks."
            )
        if path.is_file() and path != marker:
            relative = path.relative_to(root).as_posix()
            actual[relative] = {
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
    if actual != files:
        raise ValueError(
            f"{root}: manifest inventory has missing/extra/corrupt files; do not reuse this artifact."
        )
    stored_metadata = manifest.get("metadata")
    if not isinstance(stored_metadata, dict):
        raise ValueError(f"{root}: manifest metadata must be an object.")
    counts = stored_metadata.get("counts")
    if (
        not isinstance(counts, dict)
        or set(counts) != set(SPLITS)
        or any(type(count) is not int or count <= 0 for count in counts.values())
    ):
        raise ValueError(
            f"{root}: manifest metadata must contain positive counts for exactly the three splits."
        )
    return manifest


def _data_load_protocol(root, manifest):
    path = root / "split_assignments.json"
    if not path.exists() and not path.is_symlink():
        if manifest is not None and manifest.get("assignment_digest") is not None:
            raise ValueError(f"{root}: frozen split protocol is missing.")
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{root}: invalid frozen split protocol file.")
    protocol = read_json(path)
    if (
        not isinstance(protocol, dict)
        or protocol.get("version") != "iwd-split-sidecar-v1"
    ):
        raise ValueError(f"{root}: unsupported ITW split protocol schema.")
    body = {key: value for key, value in protocol.items() if key != "assignment_digest"}
    if protocol.get("assignment_digest") != json_digest(body):
        raise ValueError(f"{root}: frozen protocol assignment digest mismatch.")
    if (
        manifest is not None
        and manifest.get("assignment_digest") != protocol["assignment_digest"]
    ):
        raise ValueError(f"{root}: manifest/protocol assignment digest mismatch.")
    expected_aliases = {
        "audio_file_name": "file",
        "speaker_id": "speaker",
        "key": {"bona-fide": 0, "spoof": 1},
    }
    if json_digest(protocol.get("aliases")) != json_digest(expected_aliases):
        raise ValueError(
            f"{root}: ITW protocol aliases must preserve original file/speaker and integer 0/1 labels."
        )
    records, mapping = (
        protocol.get("canonical_metadata"),
        protocol.get("speaker_to_split"),
    )
    if not isinstance(records, list) or not records or not isinstance(mapping, dict):
        raise ValueError(
            f"{root}: missing canonical metadata/speaker assignments in protocol."
        )
    seen, speakers = set(), set()
    for row in records:
        if not isinstance(row, dict) or set(row) != {"file", "speaker", "label"}:
            raise ValueError(
                f"{root}: protocol records require exactly file/speaker/label."
            )
        for name, value in row.items():
            _data_string(value, f"protocol {name}")
        filename = _data_relative_path(row["file"])
        if (
            "/" in filename
            or not filename.endswith(".wav")
            or filename in seen
            or row["label"] not in expected_aliases["key"]
        ):
            raise ValueError(
                f"{root}: invalid/duplicate canonical protocol record {row!r}."
            )
        seen.add(filename)
        speakers.add(row["speaker"])
    if (
        records != sorted(records, key=lambda row: row["file"])
        or json_digest(records) != protocol.get("canonical_metadata_sha256")
        or set(mapping) != speakers
        or any(split not in SPLITS for split in mapping.values())
    ):
        raise ValueError(
            f"{root}: canonical protocol metadata digest/order or speaker assignments differ."
        )
    if "source_metadata_csv" in protocol:
        text = protocol["source_metadata_csv"]
        if not isinstance(text, str) or hashlib.sha256(
            text.encode()
        ).hexdigest() != protocol.get("source_metadata_sha256"):
            raise ValueError(f"{root}: protocol original CSV digest mismatch.")
        reader = csv.DictReader(
            io.StringIO(text.lstrip("\ufeff"), newline=""), strict=True
        )
        if (
            reader.fieldnames != ["file", "speaker", "label"]
            or sorted(list(reader), key=lambda row: row["file"]) != records
        ):
            raise ValueError(
                f"{root}: protocol original CSV/canonical membership mismatch."
            )
    if "release_attribution" in protocol:
        text = protocol["release_attribution"]
        if (
            not isinstance(text, str)
            or not text.strip()
            or hashlib.sha256(text.encode()).hexdigest()
            != protocol.get("release_attribution_sha256")
        ):
            raise ValueError(f"{root}: protocol release attribution digest mismatch.")
    return protocol


def _data_optional_identity(value, context):
    if value is None:
        return None
    if type(value) is int:
        return str(value)
    return _data_string(value, context)


def _data_sample_count(value, context):
    if value is not None and (type(value) is not int or value <= 0):
        raise ValueError(f"{context}: num_samples must be a positive integer or null.")
    return value


def _data_summary(rows_by_split):
    identifiers = {
        split: {row["clip_id"] for row in rows_by_split[split]} for split in SPLITS
    }
    speakers = {
        split: {
            row["speaker"] for row in rows_by_split[split] if row["speaker"] is not None
        }
        for split in SPLITS
    }
    return {
        "counts": {split: len(rows_by_split[split]) for split in SPLITS},
        "class_counts": {
            split: {
                name: sum(row["label"] == label for row in rows_by_split[split])
                for label, name in enumerate(LABEL_NAMES)
            }
            for split in SPLITS
        },
        "speaker_counts": {split: len(speakers[split]) for split in SPLITS},
        "overlaps": {
            f"{a}/{b}": {
                "utterance_ids": len(identifiers[a] & identifiers[b]),
                "speakers": len(speakers[a] & speakers[b]),
            }
            for a, b in combinations(SPLITS, 2)
        },
    }


def _data_verify_metadata(manifest, raw_by_split, canonical, protocol, source):
    summary = _data_summary(canonical)
    digest_verified = False
    if manifest is not None:
        stored = manifest["metadata"]
        for name in ("counts", "class_counts", "speaker_counts", "overlaps"):
            if name in stored and json_digest(stored[name]) != json_digest(
                summary[name]
            ):
                raise ValueError(f"{source}: manifest metadata {name} mismatch.")
        if (
            stored.get("kind", "full") != "full"
            or stored.get("complete_source_coverage", True) is not True
        ):
            raise ValueError(
                f"{source}: manifest metadata describes incomplete/smoke coverage."
            )
        columns = (
            ["file", "speaker", "label", "audio_file_name", "speaker_id", "key"]
            if protocol is not None
            else ["speaker_id", "audio_file_name", "system_id", "key"]
        )
        if "metadata_sha256" in stored:
            expected = stored["metadata_sha256"]
            if not isinstance(expected, dict) or set(expected) != set(SPLITS):
                raise ValueError(f"{source}: invalid upstream metadata digest mapping.")
            for split in SPLITS:
                _data_sha256(expected[split], f"{source}/{split} metadata")
            # ASV's optional system_id participates only in its upstream hash; it is
            # never required by canonical metadata or by sources using other schemas.
            if all(
                all(name in row for name in columns)
                for rows in raw_by_split.values()
                for row in rows
            ):
                for split in SPLITS:
                    digest = hashlib.sha256()
                    for row in raw_by_split[split]:
                        digest.update(
                            json.dumps(
                                {name: row[name] for name in columns}, sort_keys=True
                            ).encode()
                            + b"\n"
                        )
                    if digest.hexdigest() != expected[split]:
                        raise ValueError(
                            f"{source}/{split}: manifest metadata digest mismatch."
                        )
                digest_verified = True
    if protocol is not None:
        by_file = {row["file"]: row for row in protocol["canonical_metadata"]}
        actual_ids = set()
        label_map = {"bona-fide": 0, "spoof": 1}
        for split in SPLITS:
            for row, original in zip(canonical[split], raw_by_split[split]):
                expected = by_file.get(row["clip_id"])
                if expected is None:
                    raise ValueError(
                        f"{source}/{split}: clip absent from immutable ITW protocol: {row['clip_id']!r}."
                    )
                aliases = {
                    **expected,
                    "audio_file_name": expected["file"],
                    "speaker_id": expected["speaker"],
                    "key": label_map[expected["label"]],
                }
                if (
                    any(original.get(name) != value for name, value in aliases.items())
                    or type(original.get("key")) is not int
                    or row["speaker"] != expected["speaker"]
                    or row["label"] != aliases["key"]
                    or protocol["speaker_to_split"][expected["speaker"]] != split
                ):
                    raise ValueError(
                        f"{source}/{split}: immutable ITW protocol ID/speaker/label/alias/split mismatch for {row['clip_id']!r}."
                    )
                actual_ids.add(row["clip_id"])
        if actual_ids != set(by_file) or sum(summary["counts"].values()) != len(
            by_file
        ):
            raise ValueError(
                f"{source}: full artifact membership differs from immutable ITW protocol."
            )
    return summary, digest_verified


def load_sources(specs):
    """Load metadata-only views and verify upstream inventories; decode no audio."""
    if not isinstance(specs, dict) or not specs:
        raise ValueError("Configure at least one named SourceSpec.")
    sources, metadata, provenance = {}, {split: [] for split in SPLITS}, {}
    for source in sorted(specs):
        _data_string(source, "source ID")
        spec = specs[source]
        if not isinstance(spec, SourceSpec):
            raise ValueError(f"{source}: expected SourceSpec.")
        root = Path(spec.path).expanduser()
        manifest = _data_load_manifest(root, spec)
        protocol = _data_load_protocol(root, manifest)
        loaded = load_from_disk(str(root))
        if not isinstance(loaded, DatasetDict) or set(loaded) != set(SPLITS):
            raise ValueError(
                f"{source}: require exactly train/validation/test DatasetDict splits; never resplit or alias them."
            )
        encoded, raw_by_split, canonical = {}, {}, {}
        for split in SPLITS:
            dataset = loaded[split].with_format(None)
            required = {spec.audio_column, spec.label_column, spec.id_column}
            required.update(
                name
                for name in (
                    spec.speaker_column,
                    spec.num_samples_column,
                    spec.language_column,
                )
                if name is not None
            )
            if not required <= set(dataset.column_names) or not isinstance(
                dataset.features.get(spec.audio_column), Audio
            ):
                raise ValueError(
                    f"{source}/{split}: missing configured columns or non-Audio audio feature."
                )
            if not len(dataset):
                raise ValueError(f"{source}/{split}: empty original split.")
            audio_columns = {
                name
                for name, feature in dataset.features.items()
                if isinstance(feature, Audio)
            }
            for name in sorted(audio_columns):
                dataset = dataset.cast_column(
                    name, Audio(sampling_rate=None, num_channels=None, decode=False)
                )
            encoded[split] = dataset
            raw_by_split[split] = list(
                dataset.select_columns(
                    [name for name in dataset.column_names if name not in audio_columns]
                )
            )
            canonical[split], seen = [], set()
            for index, original in enumerate(raw_by_split[split]):
                context = f"{source}/{split}/{index}"
                try:
                    json.dumps(original, sort_keys=True, allow_nan=False)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{context}: non-audio metadata must be finite JSON values."
                    ) from exc
                clip_id = _data_string(original[spec.id_column], f"{context} clip ID")
                if clip_id in seen:
                    raise ValueError(
                        f"{source}/{split}: duplicate original clip ID {clip_id!r}."
                    )
                seen.add(clip_id)
                count = _data_sample_count(
                    original[spec.num_samples_column]
                    if spec.num_samples_column
                    else None,
                    context,
                )
                if count is not None and original.get("duration_s") is not None:
                    duration = original["duration_s"]
                    if type(duration) not in (int, float) or not math.isclose(
                        duration, count / TARGET_SR, rel_tol=1e-6, abs_tol=1e-8
                    ):
                        raise ValueError(
                            f"{context}: duration/sample-count metadata mismatch."
                        )
                row = {
                    "source": source,
                    "split": split,
                    "row_index": index,
                    "clip_id": clip_id,
                    "label": _data_label(original[spec.label_column], spec, context),
                    "speaker": _data_optional_identity(
                        original[spec.speaker_column], context
                    )
                    if spec.speaker_column
                    else None,
                    "language": (
                        _data_optional_identity(original[spec.language_column], context)
                        if spec.language_column
                        else spec.language
                    ),
                    "num_samples": count,
                    "attributes": copy.deepcopy(original),
                }
                canonical[split].append(row)
        summary, digest_verified = _data_verify_metadata(
            manifest, raw_by_split, canonical, protocol, source
        )
        sources[source] = DatasetDict(encoded)
        for split in SPLITS:
            metadata[split].extend(canonical[split])
        spec_details = asdict(spec)
        spec_details["label_map"] = [
            {"original": original, "label": canonical_label}
            for original, canonical_label in sorted(
                spec.label_map.items(),
                key=lambda item: (type(item[0]).__name__, str(item[0])),
            )
        ]
        provenance[source] = {
            "path": str(root),
            "source_spec": spec_details,
            "manifest_digest": json_digest(manifest) if manifest is not None else None,
            "manifest_sha256": file_sha256(root / "manifest.json")
            if manifest is not None
            else None,
            "fingerprint": manifest.get("fingerprint")
            if manifest is not None
            else None,
            "source_details": copy.deepcopy(manifest.get("source"))
            if manifest is not None
            else None,
            "preprocess": copy.deepcopy(manifest.get("preprocess"))
            if manifest is not None
            else None,
            "upstream_metadata": copy.deepcopy(manifest.get("metadata"))
            if manifest is not None
            else None,
            "upstream_metadata_digest_verified": digest_verified,
            "assignment_digest": protocol["assignment_digest"]
            if protocol is not None
            else None,
            "protocol_digest": protocol["assignment_digest"]
            if protocol is not None
            else None,
            "protocol_sha256": file_sha256(root / "split_assignments.json")
            if protocol is not None
            else None,
            "metadata_digest": json_digest(canonical),
            "metadata": summary,
            "overlaps": summary["overlaps"],
            "identity_scope": "Speaker/clip overlaps are within this source only; unrelated source namespaces do not establish global identity disjointness.",
            "warnings": (
                [
                    "No upstream completion manifest was required; selected audio is fingerprinted during preflight."
                ]
                if manifest is None
                else [
                    "Upstream metadata digest columns use an unknown schema; file hashes and metadata counts were verified, and the digest is retained without claiming it was independently reproduced."
                ]
                if "metadata_sha256" in manifest["metadata"] and not digest_verified
                else []
            ),
        }
    return sources, metadata, provenance


# %% Selected-audio decoder, full-waveform leakage audit, and explicit center crops.
def decode_record(sources, specs, row):
    """Return original mono float32 samples; never resample, downmix, or normalize."""
    source, split, index = row.get("source"), row.get("split"), row.get("row_index")
    context = f"{source}/{split}/{row.get('clip_id', index)}"
    if (
        source not in specs
        or source not in sources
        or split not in SPLITS
        or split not in sources[source]
        or type(index) is not int
        or index < 0
        or index >= len(sources[source][split])
    ):
        raise ValueError(f"{context}: invalid original source/split/row reference.")
    spec, dataset = specs[source], sources[source][split]
    attributes = row.get("attributes")
    if (
        not isinstance(attributes, dict)
        or type(row.get("label")) is not int
        or row["label"] not in (0, 1)
    ):
        raise ValueError(f"{context}: invalid canonical metadata/label.")
    columns = set(attributes) | {spec.audio_column, spec.id_column, spec.label_column}
    columns.update(
        name
        for name in (spec.speaker_column, spec.num_samples_column, spec.language_column)
        if name is not None
    )
    if not columns <= set(dataset.column_names):
        raise ValueError(f"{context}: selected metadata columns changed.")
    # Read raw bytes even if a caller passes misleading resampling feature metadata.
    encoded = dataset.select_columns(sorted(columns)).cast_column(
        spec.audio_column, Audio(sampling_rate=None, num_channels=None, decode=False)
    )
    original = encoded[index]
    count = _data_sample_count(
        original[spec.num_samples_column] if spec.num_samples_column else None, context
    )
    speaker = (
        _data_optional_identity(original[spec.speaker_column], context)
        if spec.speaker_column
        else None
    )
    language = (
        _data_optional_identity(original[spec.language_column], context)
        if spec.language_column
        else spec.language
    )
    if (
        original[spec.id_column] != row.get("clip_id")
        or _data_label(original[spec.label_column], spec, context) != row["label"]
        or speaker != row.get("speaker")
        or language != row.get("language")
        or count != row.get("num_samples")
        or json_digest({name: original[name] for name in attributes})
        != json_digest(attributes)
    ):
        raise ValueError(
            f"{context}: selected source metadata changed after selection."
        )
    payload = original[spec.audio_column]
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("bytes"), (bytes, bytearray, memoryview))
        or not payload["bytes"]
    ):
        raise ValueError(
            f"{context}: embedded audio bytes are required; external-path decoding is forbidden."
        )
    audio_bytes = bytes(payload["bytes"])
    try:
        info = sf.info(io.BytesIO(audio_bytes))
    except (RuntimeError, ValueError, TypeError) as exc:
        raise ValueError(f"{context}: invalid embedded audio header.") from exc
    if info.samplerate != TARGET_SR or info.channels != 1 or info.frames <= 0:
        raise ValueError(
            f"{context}: original audio must be nonempty mono {TARGET_SR} Hz, got {info.samplerate} Hz/{info.channels} channels/{info.frames} samples."
        )
    try:
        samples = (
            Audio(sampling_rate=None, num_channels=None)
            .decode_example({"bytes": audio_bytes, "path": None})
            .get_all_samples()
        )
    except Exception as exc:
        raise ValueError(
            f"{context}: real Hugging Face/TorchCodec decode failed; verify the installed torch/torchcodec/FFmpeg compatibility before running extraction."
        ) from exc
    tensor = samples.data
    if (
        samples.sample_rate != TARGET_SR
        or not isinstance(tensor, torch.Tensor)
        or tensor.ndim != 2
        or tensor.shape[0] != 1
    ):
        raise ValueError(
            f"{context}: decoder changed the original sample rate/channel shape."
        )
    waveform = np.ascontiguousarray(tensor[0].detach().cpu().numpy(), dtype=np.float32)
    if waveform.ndim != 1 or not len(waveform) or not np.isfinite(waveform).all():
        raise ValueError(f"{context}: empty/non-finite/invalid decoded waveform.")
    if len(waveform) != info.frames or (count is not None and len(waveform) != count):
        raise ValueError(
            f"{context}: decoded sample count differs from original header/stored num_samples."
        )
    return waveform


def _data_waveform_sha256(waveform):
    # Explicit little-endian contiguous float32 is portable across host byte orders.
    return hashlib.sha256(
        np.asarray(waveform, dtype="<f4").tobytes(order="C")
    ).hexdigest()


def _data_cap_samples(max_seconds):
    if max_seconds is None:
        return None
    if (
        type(max_seconds) not in (int, float)
        or max_seconds <= 0
        or (type(max_seconds) is float and not math.isfinite(max_seconds))
    ):
        raise ValueError(
            "max_seconds must be None or an explicitly positive finite duration."
        )
    scaled = max_seconds * TARGET_SR
    # Very large explicit caps are harmless: short clips still remain intact.
    count = (
        math.floor(max_seconds) * TARGET_SR
        if type(scaled) is float and not math.isfinite(scaled)
        else math.floor(scaled)
    )
    if count < 1:
        raise ValueError("max_seconds is shorter than one audio sample.")
    return count


def prepare_selection(selection, sources, specs, max_seconds=None):
    """Decode only selected clips; freeze original identity and report exact audits."""
    cap = _data_cap_samples(max_seconds)
    if set(selection) != set(SPLITS):
        raise ValueError("Selection must retain exactly the three original splits.")
    references = set()
    # Validate every reference before any decode; do not mutate caller-owned rows.
    for split in SPLITS:
        for row in selection[split]:
            if (
                row.get("split") != split
                or type(row.get("row_index")) is not int
                or row["row_index"] < 0
            ):
                raise ValueError(f"{split}: invalid original row reference.")
            _data_string(row.get("source"), "source")
            _data_string(row.get("clip_id"), "clip ID")
            _data_string(row.get("group"), "sampling group")
            if type(row.get("label")) is not int or row["label"] not in (0, 1):
                raise ValueError("Selected labels must be integer 0/1.")
            reference = _data_ref(row)
            if reference in references:
                raise ValueError(
                    f"Selected original row {reference} occurs more than once (overlapping group views)."
                )
            references.add(reference)
    prepared, by_hash = {split: [] for split in SPLITS}, {}
    for split in SPLITS:
        for row in sorted(selection[split], key=_data_sort_key):
            waveform = decode_record(sources, specs, row)
            digest = _data_waveform_sha256(waveform)
            reference = {
                name: row[name]
                for name in (
                    "source",
                    "split",
                    "row_index",
                    "clip_id",
                    "label",
                    "group",
                )
            }
            previous = by_hash.setdefault(digest, [])
            if previous and (
                previous[0]["label"] != row["label"] or previous[0]["split"] != split
            ):
                raise ValueError(
                    f"Exact full-waveform duplicate crosses splits or conflicts in label: {previous[0]} / {reference}; resolve upstream without silently dropping rows."
                )
            previous.append(reference)
            original_samples = len(waveform)
            used_samples = (
                original_samples if cap is None else min(original_samples, cap)
            )
            crop_start = (original_samples - used_samples) // 2
            frame_count = int(conv_frame_count(used_samples))
            if frame_count < 1:
                raise ValueError(
                    f"{row['source']}/{split}/{row['clip_id']}: {used_samples} used samples produce no encoder frame; use at least one valid frontend frame, not a training-mask minimum."
                )
            prepared[split].append(
                {
                    **copy.deepcopy(row),
                    "original_samples": original_samples,
                    "used_samples": used_samples,
                    "crop_start": crop_start,
                    "frame_count": frame_count,
                    "audio_sha256": digest,
                }
            )
    counts, durations, detailed_counts = {}, {}, {}
    for split in SPLITS:
        rows = prepared[split]
        original_lengths = [row["original_samples"] for row in rows]
        used_lengths = [row["used_samples"] for row in rows]
        counts[split] = {
            "total": len(rows),
            "by_source": dict(sorted(Counter(row["source"] for row in rows).items())),
            "by_group": dict(sorted(Counter(row["group"] for row in rows).items())),
            "by_label": {
                name: sum(row["label"] == label for row in rows)
                for label, name in enumerate(LABEL_NAMES)
            },
        }
        compound = Counter((row["source"], row["group"], row["label"]) for row in rows)
        detailed_counts[split] = [
            {
                "source": source,
                "group": group,
                "label": label,
                "label_name": LABEL_NAMES[label],
                "count": count,
            }
            for (source, group, label), count in sorted(compound.items())
        ]
        durations[split] = {
            "original_seconds": sum(original_lengths) / TARGET_SR,
            "used_seconds": sum(used_lengths) / TARGET_SR,
            "min_original_seconds": min(original_lengths) / TARGET_SR if rows else None,
            "max_original_seconds": max(original_lengths) / TARGET_SR if rows else None,
            "cropped_clips": sum(
                row["used_samples"] < row["original_samples"] for row in rows
            ),
            "frames": sum(row["frame_count"] for row in rows),
        }
    audit = {
        "counts": counts,
        "source_group_label_counts": detailed_counts,
        "durations": durations,
        "duplicates": [
            {
                "audio_sha256": digest,
                "split": rows[0]["split"],
                "label": rows[0]["label"],
                "rows": rows,
            }
            for digest, rows in sorted(by_hash.items())
            if len(rows) > 1
        ],
        "crop_policy": {
            "kind": "none" if max_seconds is None else "center",
            "max_seconds": max_seconds,
        },
        "totals": {
            "clips": sum(len(rows) for rows in prepared.values()),
            "original_samples": sum(
                row["original_samples"] for rows in prepared.values() for row in rows
            ),
            "used_samples": sum(
                row["used_samples"] for rows in prepared.values() for row in rows
            ),
            "frames": sum(
                row["frame_count"] for rows in prepared.values() for row in rows
            ),
        },
        "limitations": [
            "Exact full-waveform hashes do not detect near duplicates or establish global speaker disjointness.",
            "Cross-source speaker identity is unresolved unless supplied upstream; unrelated speaker namespaces are not identity evidence.",
            "Pairing real-only/fake-only sources does not remove recording/source-channel confounding.",
        ],
    }
    return prepared, audit


def read_prepared_waveform(sources, specs, row, max_seconds=None):
    """Recheck original content, then use the frozen offsets rather than recropping."""
    cap = _data_cap_samples(max_seconds)
    waveform = decode_record(sources, specs, row)
    names = ("original_samples", "used_samples", "crop_start", "frame_count")
    if any(type(row.get(name)) is not int for name in names):
        raise ValueError(
            "Prepared sample/crop/frame metadata must contain integer offsets and lengths."
        )
    original, used, start, frames = (row[name] for name in names)
    _data_sha256(row.get("audio_sha256"), "prepared original audio")
    if original != len(waveform) or row["audio_sha256"] != _data_waveform_sha256(
        waveform
    ):
        raise ValueError(
            f"{row['source']}/{row['split']}/{row['clip_id']}: original audio content/length changed after preflight."
        )
    if (
        used <= 0
        or used > original
        or start < 0
        or start + used > original
        or start != (original - used) // 2
        or frames < 1
        or frames != int(conv_frame_count(used))
    ):
        raise ValueError(
            "Prepared center-crop offsets/length/frame count are invalid or changed."
        )
    if cap is not None and used != min(original, cap):
        raise ValueError(
            "Requested duration cap differs from the immutable prepared crop; rerun explicit preflight."
        )
    return np.ascontiguousarray(waveform[start : start + used], dtype=np.float32)


# %% Offline data-contract checks (definitions only; no corpus or network access).
def _data_expect_error(operation, text=None):
    try:
        operation()
    except ValueError as exc:
        if text is not None:
            assert text.lower() in str(exc).lower(), str(exc)
    else:
        raise AssertionError("Expected a data-contract ValueError.")


def _check_data_selection():
    metadata = {split: [] for split in SPLITS}
    for split in SPLITS:
        for source, labels in (("both", (0, 1)), ("real", (0,)), ("fake", (1,))):
            for label in labels:
                for index in range(6):
                    metadata[split].append(
                        {
                            "source": source,
                            "split": split,
                            "row_index": label * 6 + index,
                            "clip_id": f"{source}-{split}-{label}-{index}",
                            "label": label,
                            "speaker": f"speaker-{split}-{index}",
                            "language": "hi" if source != "both" else None,
                            "num_samples": 800,
                            "attributes": {"variant": "original"},
                        }
                    )
    groups = {
        "two_class": [{"source": "both", "filters": {}}],
        "paired": [
            {"source": "real", "filters": {"language": "hi"}},
            {"source": "fake", "filters": {"language": "hi"}},
        ],
    }
    quotas = {group: {split: 4 for split in SPLITS} for group in groups}
    before = copy.deepcopy(metadata)
    selected = select_rows(metadata, groups, quotas)
    assert selected == select_rows(metadata, groups, quotas)
    assert selected == select_rows(
        {split: list(reversed(rows)) for split, rows in metadata.items()},
        {group: list(reversed(groups[group])) for group in reversed(groups)},
        quotas,
    )
    assert metadata == before
    larger = select_rows(
        metadata, groups, {group: dict.fromkeys(SPLITS, 8) for group in groups}
    )
    for split in SPLITS:
        assert len(selected[split]) == 8
        assert Counter((row["group"], row["label"]) for row in selected[split]) == {
            ("two_class", 0): 2,
            ("two_class", 1): 2,
            ("paired", 0): 2,
            ("paired", 1): 2,
        }
        small_ids = {(row["source"], row["row_index"]) for row in selected[split]}
        large_ids = {(row["source"], row["row_index"]) for row in larger[split]}
        assert small_ids <= large_ids
        assert all(row["split"] == split for row in selected[split])
    _data_expect_error(
        lambda: select_rows(
            metadata,
            {"one_class": [{"source": "real"}]},
            {"one_class": dict.fromkeys(SPLITS, 2)},
        ),
        "available",
    )
    overlapping = {group: [{"source": "both"}] for group in ("a", "b")}
    _data_expect_error(
        lambda: select_rows(
            metadata,
            overlapping,
            {group: dict.fromkeys(SPLITS, 12) for group in overlapping},
        ),
        "overlap",
    )
    for bad_total in (0, 3, 100, True):
        _data_expect_error(
            lambda total=bad_total: select_rows(
                metadata,
                groups,
                {group: dict.fromkeys(SPLITS, total) for group in groups},
            )
        )
    _data_expect_error(
        lambda: select_rows(
            metadata,
            {"bad": [{"source": "unknown"}]},
            {"bad": dict.fromkeys(SPLITS, 2)},
        ),
        "source",
    )
    _data_expect_error(
        lambda: select_rows(
            metadata,
            {"bad": [{"source": "both", "filters": {"language": "hi"}}]},
            {"bad": dict.fromkeys(SPLITS, 2)},
        ),
        "metadata",
    )


def _data_fixture_write_json(path, value):
    """Overwrite disposable test-owned fixtures, never production artifacts."""
    Path(path).write_text(
        json.dumps(value, sort_keys=True, allow_nan=False), encoding="utf-8"
    )


def _data_fixture_wav(waveform, sample_rate=TARGET_SR, subtype="PCM_16"):
    buffer = io.BytesIO()
    sf.write(buffer, np.asarray(waveform), sample_rate, format="WAV", subtype=subtype)
    return buffer.getvalue()


def _data_fixture_dataset(itw=False):
    result = {}
    for split_index, split in enumerate(SPLITS):
        rows = []
        for index in range(4):
            token = 1 + 4 * split_index + index
            waveform = (0.2 * np.sin(np.arange(1200) * (0.017 + token * 0.003))).astype(
                np.float32
            )
            row = {
                "audio": {"bytes": _data_fixture_wav(waveform), "path": None},
                "key": index % 2,
                "audio_file_name": f"{split}-{index}.wav",
                "speaker_id": f"{split}-speaker-{index // 2}",
                "num_samples": len(waveform),
                "duration_s": len(waveform) / TARGET_SR,
                "preprocess_version": "synthetic-itw" if itw else "synthetic-other",
            }
            if itw:
                row.update(
                    file=row["audio_file_name"],
                    speaker=row["speaker_id"],
                    label="bona-fide" if row["key"] == 0 else "spoof",
                )
            rows.append(row)
        features = Features(
            {
                "audio": Audio(sampling_rate=TARGET_SR, num_channels=1),
                "key": ClassLabel(names=LABEL_NAMES),
                "audio_file_name": Value("string"),
                "speaker_id": Value("string"),
                "num_samples": Value("int64"),
                "duration_s": Value("float32"),
                "preprocess_version": Value("string"),
            }
        )
        if itw:
            features.update(
                {name: Value("string") for name in ("file", "speaker", "label")}
            )
        result[split] = Dataset.from_list(rows, features=features)
    return DatasetDict(result)


def _data_fixture_protocol(data):
    records, mapping = [], {}
    for split in SPLITS:
        for row in data[split].select_columns(["file", "speaker", "label"]):
            records.append(row)
            mapping[row["speaker"]] = split
    records.sort(key=lambda row: row["file"])
    protocol = {
        "version": "iwd-split-sidecar-v1",
        "source": {"id": "synthetic"},
        "canonical_metadata": records,
        "canonical_metadata_sha256": json_digest(records),
        "speaker_to_split": mapping,
        "aliases": {
            "audio_file_name": "file",
            "speaker_id": "speaker",
            "key": {"bona-fide": 0, "spoof": 1},
        },
    }
    protocol["assignment_digest"] = json_digest(protocol)
    return protocol


def _data_fixture_manifest(data, root, protocol=None):
    # Independent fixture construction follows the two preprocessing notebooks' schema.
    metadata = {"counts": {}, "class_counts": {}}
    if protocol is not None:
        _data_fixture_write_json(root / "split_assignments.json", protocol)
        metadata["metadata_sha256"] = {}
    for split in SPLITS:
        counts = Counter(data[split]["key"])
        metadata["counts"][split] = len(data[split])
        metadata["class_counts"][split] = {
            name: counts[label] for label, name in enumerate(LABEL_NAMES)
        }
        if protocol is not None:
            digest = hashlib.sha256()
            for row in data[split].select_columns(
                ["file", "speaker", "label", "audio_file_name", "speaker_id", "key"]
            ):
                digest.update(json.dumps(row, sort_keys=True).encode() + b"\n")
            metadata["metadata_sha256"][split] = digest.hexdigest()
    manifest = {
        "status": "complete",
        "kind": "full",
        "fingerprint": "a" * 64,
        "source": {"id": "synthetic"},
        "preprocess": {"version": "synthetic"},
        "metadata": metadata,
        "files": {
            path.relative_to(root).as_posix(): {
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        },
    }
    if protocol is not None:
        manifest["assignment_digest"] = protocol["assignment_digest"]
    _data_fixture_write_json(root / "manifest.json", manifest)
    return manifest


def _data_fixture_replace(data, split, index, updates):
    encoded = data[split].cast_column("audio", Audio(decode=False))
    rows = list(encoded)
    rows[index] = {**rows[index], **updates}
    return DatasetDict(
        {
            name: Dataset.from_list(rows, features=encoded.features)
            if name == split
            else dataset
            for name, dataset in data.items()
        }
    )


def _check_data_sources():
    temp_root = "/tmp/opencode" if Path("/tmp/opencode").is_dir() else None
    with tempfile.TemporaryDirectory(
        prefix="xlsr-data-sources-", dir=temp_root
    ) as temporary:
        root = Path(temporary) / "ordinary"
        data = _data_fixture_dataset()
        data.save_to_disk(str(root))
        required = SourceSpec(str(root))
        optional = replace(required, require_manifest=False, language="hi")
        _data_expect_error(lambda: load_sources({"fixture": required}), "manifest")
        sources, metadata, provenance = load_sources({"fixture": optional})
        assert set(sources["fixture"]) == set(SPLITS)
        assert sources["fixture"]["train"].features["audio"].decode is False
        assert metadata["train"][0]["language"] == "hi"
        assert (
            metadata["train"][0]["attributes"]["preprocess_version"]
            == "synthetic-other"
        )
        assert metadata["train"][0]["row_index"] == 0
        assert "audio" not in metadata["train"][0]["attributes"]
        assert provenance["fixture"]["manifest_digest"] is None
        # Paired source folders may each have one class; the GROUP supplies coverage.
        paired_specs = {}
        for name, indices in (("real", [0, 2]), ("fake", [1, 3])):
            paired_root = Path(temporary) / name
            DatasetDict(
                {split: dataset.select(indices) for split, dataset in data.items()}
            ).save_to_disk(str(paired_root))
            paired_specs[name] = SourceSpec(
                str(paired_root), require_manifest=False, language="hi"
            )
        _, paired_metadata, _ = load_sources(paired_specs)
        paired = select_rows(
            paired_metadata,
            {
                "Hindi": [
                    {"source": name, "filters": {"language": "hi"}}
                    for name in ("real", "fake")
                ]
            },
            {"Hindi": dict.fromkeys(SPLITS, 4)},
        )
        assert Counter((item["source"], item["label"]) for item in paired["train"]) == {
            ("real", 0): 2,
            ("fake", 1): 2,
        }
        string_root = Path(temporary) / "string-labels"
        string_data = DatasetDict(
            {
                split: dataset.add_column("verdict", ["real", "fake", "real", "fake"])
                for split, dataset in data.items()
            }
        )
        string_data.save_to_disk(str(string_root))
        _, string_metadata, _ = load_sources(
            {
                "mapped": SourceSpec(
                    str(string_root),
                    require_manifest=False,
                    label_column="verdict",
                    label_map={"real": 0, "fake": 1},
                    speaker_column=None,
                    num_samples_column=None,
                )
            }
        )
        assert [item["label"] for item in string_metadata["train"]] == [0, 1, 0, 1]
        assert string_metadata["train"][0]["speaker"] is None
        assert string_metadata["train"][0]["num_samples"] is None
        assert string_metadata["train"][0]["language"] is None
        float_root = Path(temporary) / "ambiguous-labels"
        DatasetDict(
            {
                split: dataset.cast_column("key", Value("float32"))
                for split, dataset in data.items()
            }
        ).save_to_disk(str(float_root))
        _data_expect_error(
            lambda: load_sources(
                {"ambiguous": SourceSpec(str(float_root), require_manifest=False)}
            ),
            "label",
        )
        _data_fixture_write_json(
            root / "manifest.json", {"status": "writing", "kind": "full"}
        )
        _data_expect_error(lambda: load_sources({"fixture": optional}), "complete")
        manifest = _data_fixture_manifest(data, root)
        _, _, provenance = load_sources({"fixture": required})
        assert provenance["fixture"]["fingerprint"] == "a" * 64
        assert provenance["fixture"]["manifest_digest"] == json_digest(manifest)
        for bad in (
            replace(required, label_column="absent"),
            replace(required, label_map={0: 0}),
        ):
            _data_expect_error(lambda spec=bad: load_sources({"fixture": spec}))
        for change in ("smoke", "counts", "unsafe", "schema", "checksum"):
            broken = copy.deepcopy(manifest)
            first = next(iter(broken["files"]))
            if change == "smoke":
                broken["kind"] = "smoke"
            elif change == "counts":
                broken["metadata"]["counts"]["train"] += 1
            elif change == "unsafe":
                broken["files"]["../outside"] = broken["files"].pop(first)
            elif change == "schema":
                broken["files"][first]["bytes"] = broken["files"][first]["size"]
            else:
                broken["files"][first]["sha256"] = "0" * 64
            _data_fixture_write_json(root / "manifest.json", broken)
            _data_expect_error(lambda: load_sources({"fixture": optional}))
        _data_fixture_write_json(root / "manifest.json", manifest)
        malformed_metadata = {**manifest, "metadata": []}
        _data_fixture_write_json(root / "manifest.json", malformed_metadata)
        _data_expect_error(lambda: load_sources({"fixture": optional}), "metadata")
        _data_fixture_write_json(root / "manifest.json", manifest)
        (root / ".manifest.pending").write_text("incomplete")
        _data_expect_error(lambda: load_sources({"fixture": optional}), "incomplete")
        # A source with missing original splits may not be repaired automatically.
        missing_root = Path(temporary) / "missing-split"
        DatasetDict({"train": data["train"], "test": data["test"]}).save_to_disk(
            str(missing_root)
        )
        _data_expect_error(
            lambda: load_sources(
                {"missing": SourceSpec(str(missing_root), require_manifest=False)}
            ),
            "split",
        )
        _data_expect_error(lambda: SourceSpec(str(root), label_map={"real": True}))
        _data_expect_error(
            lambda: SourceSpec(str(root), language="hi", language_column="language")
        )

        itw = _data_fixture_dataset(itw=True)
        protocol = _data_fixture_protocol(itw)
        itw_root = Path(temporary) / "itw"
        itw.save_to_disk(str(itw_root))
        itw_manifest = _data_fixture_manifest(itw, itw_root, protocol)
        _, itw_metadata, provenance = load_sources({"itw": SourceSpec(str(itw_root))})
        assert len(itw_metadata["train"]) == 4
        assert provenance["itw"]["assignment_digest"] == protocol["assignment_digest"]
        # Even internally valid file inventories cannot authorize metadata outside the frozen protocol.
        for case, updates in (
            ("id", {"file": "unknown.wav", "audio_file_name": "unknown.wav"}),
            ("speaker", {"speaker_id": "invented-speaker"}),
            ("label", {"key": 1}),
        ):
            changed = _data_fixture_replace(itw, "train", 0, updates)
            changed_root = Path(temporary) / case
            changed.save_to_disk(str(changed_root))
            _data_fixture_manifest(changed, changed_root, protocol)
            _data_expect_error(
                lambda path=changed_root: load_sources({"itw": SourceSpec(str(path))}),
                "protocol",
            )
        swapped = DatasetDict(
            {
                "train": itw["validation"],
                "validation": itw["train"],
                "test": itw["test"],
            }
        )
        swapped_root = Path(temporary) / "swapped"
        swapped.save_to_disk(str(swapped_root))
        _data_fixture_manifest(swapped, swapped_root, protocol)
        _data_expect_error(
            lambda: load_sources({"itw": SourceSpec(str(swapped_root))}), "protocol"
        )
        broken_protocol = copy.deepcopy(protocol)
        broken_protocol["speaker_to_split"]["train-speaker-0"] = "test"
        _data_fixture_manifest(itw, itw_root, broken_protocol)
        _data_expect_error(
            lambda: load_sources({"itw": SourceSpec(str(itw_root))}), "digest"
        )
        assert itw_manifest["assignment_digest"] == protocol["assignment_digest"]


def _check_data_audio():
    temp_root = "/tmp/opencode" if Path("/tmp/opencode").is_dir() else None
    with tempfile.TemporaryDirectory(
        prefix="xlsr-data-audio-", dir=temp_root
    ) as temporary:
        root = Path(temporary) / "source"
        _data_fixture_dataset().save_to_disk(str(root))
        specs = {"fixture": SourceSpec(str(root), require_manifest=False)}
        sources, metadata, _ = load_sources(specs)
        selection = select_rows(
            metadata,
            {"all": [{"source": "fixture"}]},
            {"all": dict.fromkeys(SPLITS, 4)},
        )
        row = selection["train"][0]
        payload = sources["fixture"]["train"][0]["audio"]["bytes"]
        expected, sample_rate = sf.read(io.BytesIO(payload), dtype="float32")
        waveform = decode_record(
            sources, specs, row
        )  # Actual HF/TorchCodec decoder, no mock.
        assert (
            sample_rate == 16000
            and waveform.dtype == np.float32
            and waveform.shape == (1200,)
        )
        np.testing.assert_array_equal(waveform, expected)
        # Misleading HF feature settings cannot resample/downmix the original payload.
        misleading = {
            "fixture": DatasetDict(
                {
                    split: dataset.cast_column(
                        "audio", Audio(sampling_rate=8000, num_channels=1, decode=False)
                    )
                    for split, dataset in sources["fixture"].items()
                }
            )
        }
        np.testing.assert_array_equal(decode_record(misleading, specs, row), expected)
        snapshot = copy.deepcopy(selection)
        prepared, audit = prepare_selection(selection, sources, specs)
        assert selection == snapshot and "audio_sha256" not in selection["train"][0]
        assert prepared["train"][0]["original_samples"] == 1200
        assert prepared["train"][0]["frame_count"] == 3
        assert audit["counts"]["train"]["by_source"] == {"fixture": 4}
        assert audit["counts"]["train"]["by_group"] == {"all": 4}
        assert audit["counts"]["train"]["by_label"] == {"bonafide": 2, "spoof": 2}
        assert math.isclose(audit["durations"]["train"]["original_seconds"], 0.3)
        assert audit["duplicates"] == []
        capped, _ = prepare_selection(selection, sources, specs, max_seconds=0.05)
        cropped_row = capped["train"][0]
        assert (
            cropped_row["original_samples"],
            cropped_row["used_samples"],
            cropped_row["crop_start"],
            cropped_row["frame_count"],
        ) == (1200, 800, 200, 2)
        np.testing.assert_array_equal(
            read_prepared_waveform(sources, specs, cropped_row), expected[200:1000]
        )
        np.testing.assert_array_equal(
            read_prepared_waveform(sources, specs, cropped_row, max_seconds=0.05),
            expected[200:1000],
        )
        _data_expect_error(
            lambda: read_prepared_waveform(sources, specs, cropped_row, max_seconds=0.1)
        )
        one_frame, _ = prepare_selection(selection, sources, specs, max_seconds=0.025)
        assert one_frame["train"][0]["used_samples"] == 400
        assert one_frame["train"][0]["frame_count"] == 1  # No training-mask minimum.
        unchanged, _ = prepare_selection(selection, sources, specs, max_seconds=0.2)
        assert (
            unchanged["train"][0]["used_samples"] == 1200
            and unchanged["train"][0]["crop_start"] == 0
        )
        for field_name, value in (
            ("original_samples", 1201),
            ("used_samples", 801),
            ("crop_start", 201),
            ("frame_count", 3),
            ("audio_sha256", "0" * 64),
        ):
            tampered = {**cropped_row, field_name: value}
            _data_expect_error(
                lambda record=tampered: read_prepared_waveform(sources, specs, record)
            )
        for cap in (0, -1, True, float("inf"), float("nan"), 0.0001):
            _data_expect_error(
                lambda value=cap: prepare_selection(
                    selection, sources, specs, max_seconds=value
                )
            )
        bad_payloads = [
            {"bytes": _data_fixture_wav(expected, sample_rate=8000), "path": None},
            {
                "bytes": _data_fixture_wav(np.column_stack((expected, expected))),
                "path": None,
            },
            {"bytes": _data_fixture_wav(np.array([], dtype=np.float32)), "path": None},
            {"bytes": _data_fixture_wav(expected[:-1]), "path": None},
            {"bytes": b"not a WAV", "path": None},
            {"bytes": None, "path": "/nonexistent/external.wav"},
        ]
        for nonfinite in (float("nan"), float("inf")):
            bad_waveform = expected.copy()
            bad_waveform[17] = nonfinite
            bad_payloads.append(
                {
                    "bytes": _data_fixture_wav(bad_waveform, subtype="FLOAT"),
                    "path": None,
                }
            )
        for bad_payload in bad_payloads:
            changed = {
                "fixture": _data_fixture_replace(
                    sources["fixture"], "train", 0, {"audio": bad_payload}
                )
            }
            _data_expect_error(
                lambda current=changed: decode_record(current, specs, row)
            )
        changed = {
            "fixture": _data_fixture_replace(
                sources["fixture"],
                "train",
                0,
                {"audio": {"bytes": _data_fixture_wav(expected * 0.5), "path": None}},
            )
        }
        _data_expect_error(
            lambda: read_prepared_waveform(changed, specs, cropped_row), "content"
        )
        # Original rows in different groups cannot be smuggled past select_rows.
        repeated = copy.deepcopy(selection)
        repeated["train"].append({**copy.deepcopy(row), "group": "other"})
        _data_expect_error(
            lambda: prepare_selection(repeated, sources, specs), "original row"
        )
        for target_split, target_index in (("validation", 0), ("train", 1)):
            duplicate = {
                "fixture": _data_fixture_replace(
                    sources["fixture"],
                    target_split,
                    target_index,
                    {"audio": {"bytes": payload, "path": None}},
                )
            }
            _data_expect_error(
                lambda current=duplicate: prepare_selection(selection, current, specs),
                "duplicate",
            )
        same_label_duplicate = {
            "fixture": _data_fixture_replace(
                sources["fixture"],
                "train",
                2,
                {"audio": {"bytes": payload, "path": None}},
            )
        }
        kept, duplicate_audit = prepare_selection(
            selection, same_label_duplicate, specs
        )
        assert len(kept["train"]) == 4 and len(duplicate_audit["duplicates"]) == 1
        assert {
            entry["clip_id"] for entry in duplicate_audit["duplicates"][0]["rows"]
        } == {"train-0.wav", "train-2.wav"}
        # Cropping must not redefine full-original waveform identity.
        different_ends = expected.copy()
        different_ends[:100] *= 0.5
        different_ends[-100:] *= 0.5
        crop_match = {
            "fixture": _data_fixture_replace(
                sources["fixture"],
                "train",
                2,
                {"audio": {"bytes": _data_fixture_wav(different_ends), "path": None}},
            )
        }
        _, crop_audit = prepare_selection(
            selection, crop_match, specs, max_seconds=0.05
        )
        assert crop_audit["duplicates"] == []


def run_data_checks():
    """Exercise the public data seams using embedded, disposable fixtures only."""
    _check_data_selection()
    _check_data_sources()
    _check_data_audio()


# DATA_HELPERS_END

# %% [markdown]
# ## Classifiers: cached-feature adapters and trainable heads
# All scores increase toward spoof (label 1). Only the small heads are trained.
# The ASP/CNN baselines use sqrt(variance + epsilon), epsilon=1e-5 by
# default, so a one-frame clip has finite standard-deviation gradients.

# %%
import hashlib as _classifier_hashlib
from contextlib import contextmanager
from dataclasses import field

from sklearn import __version__ as _classifier_sklearn_version
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from threadpoolctl import threadpool_limits
from torch.utils.data import (
    DataLoader as _ClassifierDataLoader,
    Dataset as _ClassifierDataset,
)
from xgboost import XGBClassifier, __version__ as _classifier_xgboost_version


CLASSIFIER_CONFIGS = {
    "dummy": {"strategy": "prior"},
    "logistic_regression": {"C": 1.0, "solver": "lbfgs", "max_iter": 1000},
    "linear_svm": {"C": 1.0, "dual": "auto", "max_iter": 5000},
    "rbf_svm": {"C": 1.0, "kernel": "rbf", "gamma": "scale", "probability": False},
    "knn": {"n_neighbors": 5, "weights": "distance", "metric": "euclidean"},
    "random_forest": {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 2},
    "extra_trees": {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 2},
    "hist_gradient_boosting": {
        "max_iter": 100,
        "learning_rate": 0.1,
        "max_leaf_nodes": 15,
        "l2_regularization": 1.0,
        "early_stopping": False,
    },
    "gaussian_nb": {"var_smoothing": 1e-9},
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "objective": "binary:logistic",
        "tree_method": "hist",
        "device": "cpu",
    },
    "mlp": {"hidden_dims": [256, 64], "dropout": 0.2},
    "asp": {
        "projection_dim": 128,
        "attention_dim": 64,
        "classifier_dim": 64,
        "projection_dropout": 0.1,
        "dropout": 0.2,
        "epsilon": 1e-5,
    },
    "temporal_cnn": {
        "projection_dim": 128,
        "classifier_dim": 64,
        "kernel_size": 3,
        "dilations": [1, 2],
        "conv_dropout": 0.1,
        "dropout": 0.2,
        "epsilon": 1e-5,
    },
}

DISPLAY_NAMES = {
    "dummy": "Dummy (prior)",
    "logistic_regression": "Logistic Regression",
    "linear_svm": "Linear SVM",
    "rbf_svm": "RBF SVM",
    "knn": "kNN",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "hist_gradient_boosting": "Histogram Gradient Boosting",
    "gaussian_nb": "Gaussian Naive Bayes",
    "xgboost": "XGBoost",
    "mlp": "PyTorch MLP",
    "asp": "PyTorch ASP",
    "temporal_cnn": "PyTorch Temporal CNN",
}
NEURAL_IDS = frozenset(("mlp", "asp", "temporal_cnn"))
_SCALED_CLASSIFIER_IDS = frozenset(
    ("logistic_regression", "linear_svm", "rbf_svm", "knn", "gaussian_nb", "mlp")
)


def _classifier_positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")
    return int(value)


def _head_dropout(value):
    if not isinstance(value, (float, int)) or not 0 <= value < 1:
        raise ValueError("Dropout must be in [0, 1).")
    return float(value)


def _head_epsilon(value):
    if not np.isfinite(value) or value <= 0:
        raise ValueError("Statistics epsilon must be positive and finite.")
    return float(value)


def _masked_head_input(frames, mask, hidden_size):
    if frames.ndim != 3 or frames.shape[-1] != hidden_size:
        raise ValueError(f"Expected frames [B, T, {hidden_size}].")
    if mask.dtype != torch.bool or mask.shape != frames.shape[:2]:
        raise ValueError("Expected a boolean mask with shape [B, T].")
    if frames.shape[0] == 0 or frames.shape[1] == 0 or not mask.any(dim=1).all():
        raise ValueError("Each sequence must contain at least one valid frame.")
    # Replace arbitrary padded values (including NaNs) before a projection.
    frames = frames.masked_fill(~mask.unsqueeze(-1), 0.0)
    if not torch.isfinite(frames).all():
        raise ValueError("Valid sequence frames must be finite.")
    return frames


def _masked_statistics(frames, mask, epsilon, weights=None):
    valid = mask.unsqueeze(-1)
    frames = frames.masked_fill(~valid, 0.0)
    if weights is None:
        weights = mask.to(frames.dtype) / mask.sum(dim=1, keepdim=True)
    weights = weights.unsqueeze(-1)
    mean = (weights * frames).sum(dim=1)
    centered = (frames - mean.unsqueeze(1)).masked_fill(~valid, 0.0)
    variance = (weights * centered.square()).sum(dim=1).clamp_min(0.0)
    return torch.cat((mean, torch.sqrt(variance + epsilon)), dim=-1)


class MLPHead(nn.Module):
    def __init__(self, input_dim, hidden_dims=(256, 64), dropout=0.2):
        super().__init__()
        self.input_dim = _classifier_positive_int(input_dim, "input_dim")
        if not isinstance(hidden_dims, (list, tuple)) or not hidden_dims:
            raise ValueError(
                "hidden_dims must be a nonempty list of positive integers."
            )
        layers = []
        previous = self.input_dim
        for width in hidden_dims:
            width = _classifier_positive_int(width, "hidden_dims entry")
            layers.extend(
                (
                    nn.Linear(previous, width),
                    nn.GELU(),
                    nn.Dropout(_head_dropout(dropout)),
                )
            )
            previous = width
        self.layers = nn.Sequential(*layers, nn.Linear(previous, 1))

    def forward(self, vectors):
        if vectors.ndim != 2 or vectors.shape[1] != self.input_dim:
            raise ValueError(f"Expected pooled vectors [B, {self.input_dim}].")
        return self.layers(vectors).squeeze(-1)


class ASPHead(nn.Module):
    def __init__(
        self,
        hidden_size,
        projection_dim=128,
        attention_dim=64,
        classifier_dim=64,
        projection_dropout=0.1,
        dropout=0.2,
        epsilon=1e-5,
    ):
        super().__init__()
        self.hidden_size = _classifier_positive_int(hidden_size, "hidden_size")
        projection_dim = _classifier_positive_int(projection_dim, "projection_dim")
        attention_dim = _classifier_positive_int(attention_dim, "attention_dim")
        classifier_dim = _classifier_positive_int(classifier_dim, "classifier_dim")
        self.epsilon = _head_epsilon(epsilon)
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_size, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(_head_dropout(projection_dropout)),
        )
        self.attention = nn.Sequential(
            nn.Linear(projection_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * projection_dim, classifier_dim),
            nn.GELU(),
            nn.Dropout(_head_dropout(dropout)),
            nn.Linear(classifier_dim, 1),
        )

    def forward(self, frames, mask):
        frames = _masked_head_input(frames, mask, self.hidden_size)
        projected = self.projection(frames).masked_fill(~mask.unsqueeze(-1), 0.0)
        logits = self.attention(projected).squeeze(-1).masked_fill(~mask, -torch.inf)
        weights = torch.softmax(logits, dim=1)
        pooled = _masked_statistics(projected, mask, self.epsilon, weights)
        return self.classifier(pooled).squeeze(-1)


class _MaskedTemporalBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout):
        super().__init__()
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            dilation=dilation,
            padding=dilation * (kernel_size - 1) // 2,
        )
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, frames, mask):
        # Both masks are essential: a later convolution must never read padding
        # populated by an earlier convolution's bias/neighboring valid frames.
        frames = frames.masked_fill(~mask.unsqueeze(-1), 0.0)
        frames = self.conv(frames.transpose(1, 2)).transpose(1, 2)
        frames = self.dropout(F.gelu(self.norm(frames)))
        return frames.masked_fill(~mask.unsqueeze(-1), 0.0)


class TemporalCNNHead(nn.Module):
    def __init__(
        self,
        hidden_size,
        projection_dim=128,
        classifier_dim=64,
        kernel_size=3,
        dilations=(1, 2),
        conv_dropout=0.1,
        dropout=0.2,
        epsilon=1e-5,
    ):
        super().__init__()
        self.hidden_size = _classifier_positive_int(hidden_size, "hidden_size")
        projection_dim = _classifier_positive_int(projection_dim, "projection_dim")
        classifier_dim = _classifier_positive_int(classifier_dim, "classifier_dim")
        kernel_size = _classifier_positive_int(kernel_size, "kernel_size")
        if kernel_size % 2 != 1:
            raise ValueError("CNN kernel_size must be odd to preserve sequence length.")
        if not isinstance(dilations, (list, tuple)) or not dilations:
            raise ValueError("dilations must be a nonempty list of positive integers.")
        self.epsilon = _head_epsilon(epsilon)
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_size, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [
                _MaskedTemporalBlock(
                    projection_dim,
                    kernel_size,
                    _classifier_positive_int(dilation, "dilation"),
                    _head_dropout(conv_dropout),
                )
                for dilation in dilations
            ]
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * projection_dim, classifier_dim),
            nn.GELU(),
            nn.Dropout(_head_dropout(dropout)),
            nn.Linear(classifier_dim, 1),
        )

    def forward(self, frames, mask):
        frames = _masked_head_input(frames, mask, self.hidden_size)
        frames = self.projection(frames).masked_fill(~mask.unsqueeze(-1), 0.0)
        for block in self.blocks:
            frames = block(frames, mask)
        return self.classifier(_masked_statistics(frames, mask, self.epsilon)).squeeze(
            -1
        )


@dataclass
class FittedClassifier:
    classifier_id: str
    params: dict
    model: object
    scaler: object = None
    history: list = field(default_factory=list)
    score_type: str = "probability_class_1"
    warnings: list = field(default_factory=list)
    loader_seeds: dict = field(default_factory=dict)
    trainable_parameters: int = 0
    fit_seconds: float = 0.0
    input_dim: int = 0
    hidden_size: int = 0
    feature_identity: str = ""
    producing_versions: dict = field(default_factory=dict)
    seed: int = 42
    best_epoch: object = None
    initial_state_sha256: str = ""


def _classifier_versions():
    versions = dict(package_versions())
    # Canonical keys are present even if the notebook reports distribution names
    # differently (e.g. a CPU-only XGBoost wheel).
    versions.update(
        {
            "sklearn": str(_classifier_sklearn_version),
            "torch": str(torch.__version__),
            "xgboost": str(_classifier_xgboost_version),
            "numpy": str(np.__version__),
        }
    )
    return versions


@contextmanager
def _classifier_runtime(config, neural=False):
    threads = min(
        2,
        _classifier_positive_int(
            2 if config is None else config.cpu_threads, "cpu_threads"
        ),
    )
    previous = torch.get_num_threads() if neural else None
    if neural:
        deterministic = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        benchmark = torch.backends.cudnn.benchmark
        cudnn_deterministic = torch.backends.cudnn.deterministic
    with threadpool_limits(limits=threads):
        try:
            if neural:
                torch.set_num_threads(threads)
                # Unsupported deterministic kernels warn visibly; the fit's
                # warning capture records them rather than promising cross-device
                # reproducibility or silently ignoring unsupported operations.
                torch.use_deterministic_algorithms(True, warn_only=True)
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
            yield threads
        finally:
            if neural:
                torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
                torch.backends.cudnn.benchmark = benchmark
                torch.backends.cudnn.deterministic = cudnn_deterministic
                if torch.get_num_threads() != previous:
                    torch.set_num_threads(previous)


@contextmanager
def _capture_classifier_warnings(messages):
    recorded = []
    try:
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            yield
    finally:
        # Replay outside catch_warnings, including on a failed fit.
        for warning in recorded:
            messages.append(f"{warning.category.__name__}: {warning.message}")
            warnings.warn(str(warning.message), warning.category, stacklevel=3)


def _cache_labels(cache, split, require_both=False):
    rows = cache.index(split)
    if not rows:
        raise ValueError(f"{split} contains no clips.")
    raw = [row["label"] for row in rows]
    if any(
        isinstance(label, bool)
        or not isinstance(label, (int, np.integer))
        or label not in (0, 1)
        for label in raw
    ):
        raise ValueError(f"{split} labels must be explicit integers 0/1.")
    labels = np.asarray(raw, dtype=np.int64)
    if require_both and not np.array_equal(np.unique(labels), [0, 1]):
        raise ValueError(f"{split} requires both labels 0 and 1.")
    return labels


def _pooled_matrix(values, input_dim, expected_count=None):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != input_dim or not len(values):
        raise ValueError(f"Expected nonempty pooled features [N, {input_dim}].")
    if expected_count is not None and len(values) != expected_count:
        raise ValueError("Pooled feature count does not match the split index.")
    if not np.isfinite(values).all():
        raise ValueError("Pooled features must be finite.")
    return values


def _finite_score_vector(scores, expected_count):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (expected_count,) or not np.isfinite(scores).all():
        raise ValueError(
            "Classifier scores must be a finite vector aligned with input rows."
        )
    return scores


def _binary_classifier_classes(model):
    classes = np.asarray(model.classes_)
    if classes.shape != (2,) or not np.array_equal(np.sort(classes), [0, 1]):
        raise ValueError(
            "The fitted classifier must have exactly the binary classes 0 and 1."
        )
    return classes


def _classical_constructor(classifier_id, params, seed, threads, train_count):
    constructors = {
        "dummy": DummyClassifier,
        "logistic_regression": LogisticRegression,
        "linear_svm": LinearSVC,
        "rbf_svm": SVC,
        "knn": KNeighborsClassifier,
        "random_forest": RandomForestClassifier,
        "extra_trees": ExtraTreesClassifier,
        "hist_gradient_boosting": HistGradientBoostingClassifier,
        "gaussian_nb": GaussianNB,
        "xgboost": XGBClassifier,
    }
    params = copy.deepcopy(params)
    if classifier_id in {
        "dummy",
        "logistic_regression",
        "linear_svm",
        "rbf_svm",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
        "xgboost",
    }:
        params["random_state"] = int(seed)
    if classifier_id in {
        "logistic_regression",
        "knn",
        "random_forest",
        "extra_trees",
        "xgboost",
    }:
        # lbfgs may otherwise run its single binary fit in a joblib process,
        # where convergence warnings cannot be captured by the parent.
        params["n_jobs"] = 1 if classifier_id == "logistic_regression" else threads
    if classifier_id == "knn":
        k = _classifier_positive_int(params["n_neighbors"], "n_neighbors")
        if k > train_count:
            raise ValueError(
                f"kNN n_neighbors={k} exceeds {train_count} training clips."
            )
    if classifier_id == "rbf_svm" and (
        params.get("probability") is not False or params.get("kernel") != "rbf"
    ):
        raise ValueError(
            "The RBF SVM adapter requires kernel='rbf', probability=False."
        )
    if (
        classifier_id == "hist_gradient_boosting"
        and params.get("early_stopping") is not False
    ):
        raise ValueError("Disable internal validation splitting: early_stopping=False.")
    if classifier_id == "xgboost":
        if any(
            params.get(key) != value
            for key, value in {
                "tree_method": "hist",
                "device": "cpu",
                "objective": "binary:logistic",
            }.items()
        ):
            raise ValueError(
                "XGBoost requires CPU histogram trees and binary:logistic."
            )
        if params.get("early_stopping_rounds") is not None or params.get("callbacks"):
            raise ValueError(
                "XGBoost does not use internal validation/early stopping here."
            )
    return constructors[classifier_id](**params), params


def _classical_scores(fitted, vectors):
    if fitted.scaler is not None:
        vectors = fitted.scaler.transform(vectors)
    classes = _binary_classifier_classes(fitted.model)
    if fitted.score_type == "decision_margin_class_1":
        scores = np.asarray(fitted.model.decision_function(vectors))
        scores = scores if classes[1] == 1 else -scores
    elif fitted.score_type == "probability_class_1":
        column = int(np.flatnonzero(classes == 1)[0])
        scores = np.asarray(fitted.model.predict_proba(vectors))[:, column]
    else:
        raise ValueError(f"Unsupported classical score type: {fitted.score_type}")
    return _finite_score_vector(scores, len(vectors))


def fit_classifier(classifier_id, params, cache, config):
    """Fit one configured model using train (+ neural validation) only."""
    if classifier_id not in CLASSIFIER_CONFIGS:
        raise ValueError(f"Unknown classifier ID: {classifier_id!r}")
    merged = copy.deepcopy(CLASSIFIER_CONFIGS[classifier_id])
    merged.update(copy.deepcopy(params))
    labels = _cache_labels(cache, "train", require_both=True)
    input_dim = _classifier_positive_int(cache.pooled_size, "cache.pooled_size")
    hidden_size = _classifier_positive_int(cache.hidden_size, "cache.hidden_size")
    if input_dim != 2 * hidden_size:
        raise ValueError("The cache pooled dimension must equal twice its hidden size.")
    fitted = FittedClassifier(
        classifier_id=classifier_id,
        params=merged,
        model=None,
        input_dim=input_dim,
        hidden_size=hidden_size,
        feature_identity=str(cache.identity),
        producing_versions=_classifier_versions(),
        seed=int(config.seed),
    )
    started = time.perf_counter()
    with (
        _capture_classifier_warnings(fitted.warnings),
        _classifier_runtime(config, neural=classifier_id in NEURAL_IDS) as threads,
    ):
        if classifier_id in NEURAL_IDS:
            _fit_neural_classifier(fitted, cache, labels, config)
        else:
            vectors = _pooled_matrix(cache.pooled("train"), input_dim, len(labels))
            # Never inherit RNG state from feature extraction or another row.
            seed_everything(config.seed)
            fitted.model, fitted.params = _classical_constructor(
                classifier_id, merged, config.seed, threads, len(labels)
            )
            seed_everything(config.seed)
            if classifier_id in _SCALED_CLASSIFIER_IDS:
                fitted.scaler = StandardScaler().fit(vectors)
                vectors = fitted.scaler.transform(vectors)
            fitted.model.fit(vectors, labels)
            _binary_classifier_classes(fitted.model)
            fitted.score_type = (
                "decision_margin_class_1"
                if classifier_id in {"linear_svm", "rbf_svm"}
                else "probability_class_1"
            )
    fitted.fit_seconds = float(time.perf_counter() - started)
    return fitted


def predict_scores(fitted, cache, split, config):
    """Score in canonical cache order; this function does not fit/calibrate."""
    if fitted.input_dim != cache.pooled_size or fitted.hidden_size != cache.hidden_size:
        raise ValueError("Feature cache dimensions do not match the fitted classifier.")
    count = len(cache.index(split))
    with _classifier_runtime(config, neural=fitted.classifier_id in NEURAL_IDS):
        if fitted.classifier_id in NEURAL_IDS:
            return _predict_neural_cache(fitted, cache, split, config)
        vectors = _pooled_matrix(cache.pooled(split), fitted.input_dim, count)
        return _classical_scores(fitted, vectors)


def predict_arrays(fitted, pooled=None, sequences=None, device="cpu", batch_size=16):
    """Score pre-extracted raw-probe features without loading an encoder."""
    _classifier_positive_int(batch_size, "batch_size")
    with _classifier_runtime(None, neural=fitted.classifier_id in NEURAL_IDS):
        if fitted.classifier_id in NEURAL_IDS:
            return _predict_neural_arrays(fitted, pooled, sequences, device, batch_size)
        if pooled is None or sequences is not None:
            raise ValueError("Classical classifiers require pooled vectors only.")
        return _classical_scores(fitted, _pooled_matrix(pooled, fitted.input_dim))


def _new_neural_head(classifier_id, params, input_dim, hidden_size):
    # Explicit allowlist; artifact JSON never names an import or callable.
    if classifier_id == "mlp":
        return MLPHead(input_dim, **params)
    if classifier_id == "asp":
        return ASPHead(hidden_size, **params)
    if classifier_id == "temporal_cnn":
        return TemporalCNNHead(hidden_size, **params)
    raise ValueError(f"Not a supported neural classifier: {classifier_id!r}")


def _resolve_head_device(device):
    if str(device) == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Head device must be auto, cpu, or cuda.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested for the head but is unavailable; select cpu explicitly."
        )
    return device


def _sequence_array(sequence, hidden_size):
    sequence = np.array(sequence, dtype=np.float32, copy=True)
    if sequence.ndim != 2 or sequence.shape[1] != hidden_size or not len(sequence):
        raise ValueError(f"Expected a nonempty sequence [T, {hidden_size}].")
    if not np.isfinite(sequence).all():
        raise ValueError("Sequence features must be finite.")
    return sequence


class _ClassifierFeatureDataset(_ClassifierDataset):
    def __init__(self, cache, split, sequence, scaler=None):
        self.cache, self.split, self.is_sequence = cache, split, sequence
        self.labels = _cache_labels(cache, split)
        self.vectors = None
        if not sequence:
            self.vectors = _pooled_matrix(
                cache.pooled(split), cache.pooled_size, len(self.labels)
            )
            if scaler is not None:
                self.vectors = scaler.transform(self.vectors)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if self.is_sequence:
            values = _sequence_array(
                self.cache.sequence(self.split, index), self.cache.hidden_size
            )
        else:
            # Copies of read-only memmap slices are normal, writable, non-inference
            # tensors; autograd is allowed to save them for head weight gradients.
            values = np.array(self.vectors[index], dtype=np.float32, copy=True)
        return torch.from_numpy(values), int(self.labels[index]), int(index)


def _pooled_classifier_collate(rows):
    vectors, labels, indices = zip(*rows)
    return (
        torch.stack(vectors),
        None,
        torch.tensor(labels, dtype=torch.float32),
        indices,
    )


def _sequence_classifier_collate(rows):
    sequences, labels, indices = zip(*rows)
    lengths = [len(sequence) for sequence in sequences]
    frames = torch.zeros(
        len(rows), max(lengths), sequences[0].shape[1], dtype=torch.float32
    )
    mask = torch.zeros(len(rows), max(lengths), dtype=torch.bool)
    for index, sequence in enumerate(sequences):
        frames[index, : len(sequence)] = sequence
        mask[index, : len(sequence)] = True
    return frames, mask, torch.tensor(labels, dtype=torch.float32), indices


def _classifier_worker_seed(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def _classifier_loader(fitted, cache, split, config, seed, shuffle=False):
    is_sequence = fitted.classifier_id != "mlp"
    batch_size = config.sequence_batch_size if is_sequence else config.mlp_batch_size
    batch_size = _classifier_positive_int(batch_size, "neural batch_size")
    workers = config.num_workers
    if (
        isinstance(workers, bool)
        or not isinstance(workers, (int, np.integer))
        or workers < 0
    ):
        raise ValueError("num_workers must be a nonnegative integer.")
    # A separate generator also supplies DataLoader's iterator base_seed.
    # Thus even num_workers=0 evaluation does not consume global/dropout RNG.
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return _ClassifierDataLoader(
        _ClassifierFeatureDataset(cache, split, is_sequence, fitted.scaler),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(workers),
        generator=generator,
        worker_init_fn=_classifier_worker_seed,
        collate_fn=_sequence_classifier_collate
        if is_sequence
        else _pooled_classifier_collate,
        drop_last=False,
        persistent_workers=False,
    )


def _head_batch_logits(model, values, mask, device):
    values = values.to(device)
    return model(values) if mask is None else model(values, mask.to(device))


def _evaluate_neural_loader(model, loader, device):
    model.eval()
    score_parts, label_parts = [], []
    total_loss, total_count = 0.0, 0
    with torch.no_grad():
        for values, mask, labels, _ in loader:
            logits = _head_batch_logits(model, values, mask, device)
            if logits.shape != labels.shape or not torch.isfinite(logits).all():
                raise ValueError(
                    "Neural evaluation returned invalid/non-finite logits."
                )
            loss = F.binary_cross_entropy_with_logits(
                logits, labels.to(device), reduction="sum"
            )
            if not torch.isfinite(loss):
                raise ValueError("Neural validation loss is not finite.")
            score_parts.append(logits.detach().cpu().numpy())
            label_parts.append(labels.numpy())
            total_loss += float(loss.detach().cpu())
            total_count += len(labels)
    if total_count == 0:
        raise ValueError("Cannot evaluate an empty split.")
    scores = _finite_score_vector(np.concatenate(score_parts), total_count)
    return (
        scores,
        np.concatenate(label_parts).astype(np.int64),
        total_loss / total_count,
    )


def _head_state_sha256(model):
    digest = _classifier_hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        digest.update(key.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _fit_neural_classifier(fitted, cache, labels, config):
    validation_labels = _cache_labels(cache, "validation", require_both=True)
    if config.mode not in {"smoke", "comparison"}:
        raise ValueError("mode must be smoke or comparison.")
    max_epochs = _classifier_positive_int(config.max_epochs, "max_epochs")
    if config.mode == "smoke":
        max_epochs = min(
            max_epochs, _classifier_positive_int(config.smoke_epochs, "smoke_epochs")
        )
    patience = _classifier_positive_int(config.patience, "patience")
    for name in ("learning_rate", "grad_clip"):
        if not np.isfinite(getattr(config, name)) or getattr(config, name) <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    if not np.isfinite(config.weight_decay) or config.weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative.")
    device = _resolve_head_device(config.head_device)
    seed_everything(config.seed)
    fitted.model = _new_neural_head(
        fitted.classifier_id, fitted.params, fitted.input_dim, fitted.hidden_size
    )
    fitted.initial_state_sha256 = _head_state_sha256(fitted.model)
    fitted.trainable_parameters = sum(
        p.numel() for p in fitted.model.parameters() if p.requires_grad
    )
    fitted.score_type = "raw_spoof_logit"
    fitted.model.to(device)
    optimizer = torch.optim.AdamW(
        fitted.model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()
    # Reset once at the start of this classifier's fit, not once per epoch.
    seed_everything(config.seed)
    if fitted.classifier_id == "mlp":
        vectors = _pooled_matrix(cache.pooled("train"), fitted.input_dim, len(labels))
        fitted.scaler = StandardScaler().fit(vectors)
    fitted.loader_seeds = {
        role: int(
            stable_seed(config.seed, "classifier", fitted.classifier_id, "loader", role)
        )
        for role in ("train", "validation")
    }
    train_loader = _classifier_loader(
        fitted, cache, "train", config, fitted.loader_seeds["train"], shuffle=True
    )
    validation_loader = _classifier_loader(
        fitted,
        cache,
        "validation",
        config,
        fitted.loader_seeds["validation"],
        shuffle=False,
    )
    best_key, best_state = (float("inf"), float("inf"), float("inf")), None
    epochs_without_eer_improvement = 0
    for epoch in range(1, max_epochs + 1):
        fitted.model.train()
        train_loss, train_count = 0.0, 0
        order_digest = _classifier_hashlib.sha256()
        for values, mask, batch_labels, indices in train_loader:
            order_digest.update(np.asarray(indices, dtype="<i8").tobytes())
            optimizer.zero_grad(set_to_none=True)
            logits = _head_batch_logits(fitted.model, values, mask, device)
            if logits.shape != batch_labels.shape or not torch.isfinite(logits).all():
                raise ValueError("Neural training returned invalid/non-finite logits.")
            loss = criterion(logits, batch_labels.to(device))
            if not torch.isfinite(loss):
                raise ValueError("Neural training loss is not finite.")
            loss.backward()
            for parameter in fitted.model.parameters():
                if parameter.requires_grad and (
                    parameter.grad is None or not torch.isfinite(parameter.grad).all()
                ):
                    raise ValueError(
                        "A trainable head parameter has missing/non-finite gradients."
                    )
            nn.utils.clip_grad_norm_(
                fitted.model.parameters(), config.grad_clip, error_if_nonfinite=True
            )
            optimizer.step()
            train_loss += float(loss.detach().cpu()) * len(batch_labels)
            train_count += len(batch_labels)
        scores, observed_labels, validation_loss = _evaluate_neural_loader(
            fitted.model, validation_loader, device
        )
        if not np.array_equal(observed_labels, validation_labels):
            raise ValueError("Validation loader changed canonical clip ordering.")
        validation_eer = float(equal_error_rate(observed_labels, scores))
        if not np.isfinite(validation_eer) or not 0 <= validation_eer <= 1:
            raise ValueError("Validation EER must be a finite fraction.")
        eer_improved = validation_eer < best_key[0]
        key = (validation_eer, validation_loss, epoch)
        selected = key < best_key
        if selected:
            best_key = key
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in fitted.model.state_dict().items()
            }
            fitted.best_epoch = epoch
        epochs_without_eer_improvement = (
            0 if eer_improved else epochs_without_eer_improvement + 1
        )
        fitted.history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss / train_count,
                "validation_loss": float(validation_loss),
                "validation_eer": validation_eer,
                "checkpoint_selected": bool(selected),
                "train_order_sha256": order_digest.hexdigest(),
            }
        )
        if epochs_without_eer_improvement >= patience:
            break
    if best_state is None:
        raise RuntimeError("Neural fitting did not produce a valid checkpoint.")
    fitted.model.load_state_dict(best_state, strict=True)
    fitted.model.zero_grad(set_to_none=True)
    fitted.model.to("cpu").eval()


def _predict_neural_cache(fitted, cache, split, config):
    device = _resolve_head_device(config.head_device)
    role_seed = fitted.loader_seeds.get(
        "validation",
        stable_seed(
            fitted.seed, "classifier", fitted.classifier_id, "loader", "validation"
        ),
    )
    loader = _classifier_loader(fitted, cache, split, config, role_seed, shuffle=False)
    previous_device = next(fitted.model.parameters()).device
    previous_training = fitted.model.training
    try:
        fitted.model.to(device)
        scores, _, _ = _evaluate_neural_loader(fitted.model, loader, device)
        return scores
    finally:
        fitted.model.to(previous_device).train(previous_training)


def _predict_neural_arrays(fitted, pooled, sequences, device, batch_size):
    device = _resolve_head_device(device)
    is_pooled = fitted.classifier_id == "mlp"
    if is_pooled:
        if pooled is None or sequences is not None:
            raise ValueError("MLP inference requires pooled vectors only.")
        vectors = _pooled_matrix(pooled, fitted.input_dim)
        count = len(vectors)
    else:
        if sequences is None or pooled is not None:
            raise ValueError(
                "Sequence heads require a sequence or a list of sequences only."
            )
        if isinstance(sequences, np.ndarray) and sequences.ndim == 2:
            sequences = [sequences]
        if not hasattr(sequences, "__len__") or len(sequences) == 0:
            raise ValueError("Supply one or more frame sequences.")
        count = len(sequences)
    previous_device = next(fitted.model.parameters()).device
    previous_training = fitted.model.training
    parts = []
    try:
        fitted.model.to(device).eval()
        with torch.no_grad():
            for start in range(0, count, batch_size):
                stop = min(count, start + batch_size)
                if is_pooled:
                    batch = vectors[start:stop]
                    if fitted.scaler is not None:
                        batch = fitted.scaler.transform(batch)
                    values = torch.from_numpy(
                        np.array(batch, dtype=np.float32, copy=True)
                    )
                    mask = None
                else:
                    # Construct/move only this batch, never the entire frame cache.
                    rows = [
                        (
                            torch.from_numpy(
                                _sequence_array(sequences[index], fitted.hidden_size)
                            ),
                            0,
                            index,
                        )
                        for index in range(start, stop)
                    ]
                    values, mask, _, _ = _sequence_classifier_collate(rows)
                logits = _head_batch_logits(fitted.model, values, mask, device)
                parts.append(logits.detach().cpu().numpy())
        return _finite_score_vector(np.concatenate(parts), count)
    finally:
        fitted.model.to(previous_device).train(previous_training)


def _classifier_artifact_layout(classifier_id):
    if classifier_id in NEURAL_IDS:
        files = ["weights.pt", "training_history.json"]
        if classifier_id == "mlp":
            files.append("scaler.npz")
        return "torch_state_dict", files
    if classifier_id == "xgboost":
        return "xgboost_native", ["model.ubj"]
    if classifier_id in CLASSIFIER_CONFIGS:
        return "sklearn_joblib", ["model.joblib"]
    raise ValueError(f"Unsupported artifact classifier ID: {classifier_id!r}")


def _write_new_classifier_json(path, value):
    # Deliberately independent of write_json's overwrite policy.
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(text + "\n")


def _save_numeric_head_scaler(scaler, path):
    if (
        not isinstance(scaler, StandardScaler)
        or not scaler.with_mean
        or not scaler.with_std
    ):
        raise ValueError(
            "MLP persistence requires its training-fitted default StandardScaler."
        )
    with Path(path).open("xb") as handle:
        np.savez(
            handle,
            mean=np.asarray(scaler.mean_, dtype=np.float64),
            var=np.asarray(scaler.var_, dtype=np.float64),
            scale=np.asarray(scaler.scale_, dtype=np.float64),
            n_features=np.asarray(scaler.n_features_in_, dtype=np.int64),
            n_samples_seen=np.asarray(scaler.n_samples_seen_, dtype=np.int64),
        )


def _load_numeric_head_scaler(path, input_dim):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {
            "mean",
            "var",
            "scale",
            "n_features",
            "n_samples_seen",
        }:
            raise ValueError("Unexpected numeric scaler schema.")
        mean = np.asarray(archive["mean"], dtype=np.float64)
        variance = np.asarray(archive["var"], dtype=np.float64)
        scale = np.asarray(archive["scale"], dtype=np.float64)
        n_features = np.asarray(archive["n_features"])
        n_samples = np.asarray(archive["n_samples_seen"])
    for name, vector in (("mean", mean), ("variance", variance), ("scale", scale)):
        if vector.shape != (input_dim,) or not np.isfinite(vector).all():
            raise ValueError(f"Invalid persisted scaler {name}.")
    if (variance < 0).any() or (scale <= 0).any():
        raise ValueError("Persisted scaler variance/scale is invalid.")
    if (
        n_features.shape != ()
        or n_features.dtype.kind not in "iu"
        or int(n_features) != input_dim
    ):
        raise ValueError("Persisted scaler feature count is incompatible.")
    if n_samples.shape != () or n_samples.dtype.kind not in "iu" or int(n_samples) < 1:
        raise ValueError("Persisted scaler training count is invalid.")
    scaler = StandardScaler()
    scaler.mean_, scaler.var_, scaler.scale_ = mean, variance, scale
    scaler.n_features_in_, scaler.n_samples_seen_ = int(n_features), np.int64(n_samples)
    return scaler


def save_fitted(fitted, path):
    """Write only NEW model-owned files; the caller publishes completion last."""
    path = Path(path)
    kind, artifact_names = _classifier_artifact_layout(fitted.classifier_id)
    path.mkdir(parents=True, exist_ok=True)
    for name in ["model_config.json", *artifact_names]:
        if (path / name).exists() or (path / name).is_symlink():
            raise FileExistsError(
                f"Refusing to overwrite model artifact: {path / name}"
            )
    if fitted.model is None:
        raise ValueError("Cannot save an unfitted classifier.")
    has_scaler = fitted.scaler is not None
    if has_scaler != (fitted.classifier_id in _SCALED_CLASSIFIER_IDS):
        raise ValueError(
            "Fitted preprocessing does not match the classifier's input contract."
        )
    metadata = {
        "schema_version": 1,
        "artifact_kind": kind,
        "classifier_id": fitted.classifier_id,
        "params": fitted.params,
        "input_dim": fitted.input_dim,
        "hidden_size": fitted.hidden_size,
        "feature_identity": fitted.feature_identity,
        "score_type": fitted.score_type,
        "has_scaler": has_scaler,
        "seed": fitted.seed,
        "producing_versions": fitted.producing_versions or _classifier_versions(),
        "warnings": fitted.warnings,
        "loader_seeds": fitted.loader_seeds,
        "trainable_parameters": fitted.trainable_parameters,
        "fit_seconds": fitted.fit_seconds,
        "best_epoch": fitted.best_epoch,
        "initial_state_sha256": fitted.initial_state_sha256,
        "label_mapping": {"0": "bonafide", "1": "spoof"},
    }
    # Validate JSON before writing any model files, not after an expensive save.
    json.dumps(metadata, allow_nan=False)
    if kind == "torch_state_dict":
        state = {
            name: tensor.detach().cpu()
            for name, tensor in fitted.model.state_dict().items()
        }
        if any(not torch.isfinite(tensor).all() for tensor in state.values()):
            raise ValueError("Cannot persist non-finite head weights.")
        with (path / "weights.pt").open("xb") as handle:
            torch.save(state, handle)
        if has_scaler:
            _save_numeric_head_scaler(fitted.scaler, path / "scaler.npz")
        _write_new_classifier_json(path / "training_history.json", fitted.history)
    elif kind == "xgboost_native":
        # UBJSON native model bytes, not a pickle of the Python wrapper.
        native_model = fitted.model.get_booster().save_raw(raw_format="ubj")
        with (path / "model.ubj").open("xb") as handle:
            handle.write(native_model)
    else:
        with (path / "model.joblib").open("xb") as handle:
            joblib.dump({"model": fitted.model, "scaler": fitted.scaler}, handle)
    metadata["artifacts"] = {name: file_sha256(path / name) for name in artifact_names}
    _write_new_classifier_json(path / "model_config.json", metadata)


def _check_classifier_producing_versions(metadata, classifier_id):
    produced = metadata.get("producing_versions")
    if not isinstance(produced, dict):
        raise ValueError("Model artifact is missing producing package versions.")
    relevant = {"torch"} if classifier_id in NEURAL_IDS else {"sklearn"}
    if classifier_id == "mlp":
        relevant.add("sklearn")
    if classifier_id == "xgboost":
        relevant = {"xgboost"}
    current = _classifier_versions()
    for name in sorted(relevant):
        if produced.get(name) != current[name]:
            raise ValueError(
                f"Incompatible {name} model artifact: produced with {produced.get(name)!r}, "
                f"running {current[name]!r}. Use the producing environment."
            )


def load_fitted(path, trusted=False):
    """Load an allowlisted architecture; joblib additionally requires explicit trust.

    Weights-only/native/numeric formats reduce code-execution risk; this helper
    is not a sandbox for untrusted files or protection against resource exhaustion.
    """
    path = Path(path)
    metadata = read_json(path / "model_config.json")
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
        raise ValueError("Unsupported classifier artifact schema.")
    classifier_id = metadata.get("classifier_id")
    if not isinstance(classifier_id, str) or classifier_id not in CLASSIFIER_CONFIGS:
        raise ValueError("Artifact must identify a supported classifier enum.")
    kind, names = _classifier_artifact_layout(classifier_id)
    if metadata.get("artifact_kind") != kind:
        raise ValueError("Artifact kind does not match its classifier enum.")
    if kind == "sklearn_joblib" and trusted is not True:
        raise ValueError(
            "Joblib can execute code. Load only your own trusted output with trusted=True."
        )
    _check_classifier_producing_versions(metadata, classifier_id)
    artifacts = metadata.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(names):
        raise ValueError("Unexpected classifier artifact file inventory.")
    for name in names:
        if not (path / name).is_file() or file_sha256(path / name) != artifacts[name]:
            raise ValueError(f"Missing/corrupt classifier artifact: {name}")
    input_dim = _classifier_positive_int(
        metadata.get("input_dim"), "artifact input_dim"
    )
    hidden_size = _classifier_positive_int(
        metadata.get("hidden_size"), "artifact hidden_size"
    )
    if input_dim != 2 * hidden_size:
        raise ValueError(
            "Invalid pooled/sequence dimension contract in model artifact."
        )
    if metadata.get("has_scaler") is not (classifier_id in _SCALED_CLASSIFIER_IDS):
        raise ValueError("Artifact scaler contract is incompatible.")
    if not isinstance(metadata.get("params"), dict):
        raise ValueError("Artifact constructor parameters must be a JSON object.")
    expected_score_type = (
        "raw_spoof_logit"
        if classifier_id in NEURAL_IDS
        else "decision_margin_class_1"
        if classifier_id in {"linear_svm", "rbf_svm"}
        else "probability_class_1"
    )
    if metadata.get("score_type") != expected_score_type:
        raise ValueError("Artifact score orientation/type contract is incompatible.")
    fitted = FittedClassifier(
        classifier_id=classifier_id,
        params=metadata["params"],
        model=None,
        input_dim=input_dim,
        hidden_size=hidden_size,
        feature_identity=str(metadata.get("feature_identity", "")),
        producing_versions=metadata["producing_versions"],
        score_type=expected_score_type,
        seed=int(metadata["seed"]),
        warnings=list(metadata.get("warnings", [])),
        loader_seeds=dict(metadata.get("loader_seeds", {})),
        trainable_parameters=int(metadata.get("trainable_parameters", 0)),
        fit_seconds=float(metadata.get("fit_seconds", 0.0)),
        best_epoch=metadata.get("best_epoch"),
        initial_state_sha256=str(metadata.get("initial_state_sha256", "")),
    )
    if kind == "torch_state_dict":
        with torch.random.fork_rng(devices=[]):
            fitted.model = _new_neural_head(
                classifier_id, fitted.params, input_dim, hidden_size
            )
        state = torch.load(path / "weights.pt", map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or any(
            not torch.is_tensor(tensor) or not torch.isfinite(tensor).all()
            for tensor in state.values()
        ):
            raise ValueError("Invalid/non-finite weights-only head state.")
        fitted.model.load_state_dict(state, strict=True)
        fitted.model.eval()
        if metadata["has_scaler"]:
            fitted.scaler = _load_numeric_head_scaler(path / "scaler.npz", input_dim)
        fitted.history = read_json(path / "training_history.json")
        if not isinstance(fitted.history, list) or not all(
            isinstance(row, dict) for row in fitted.history
        ):
            raise ValueError("Neural training history must be a list of epoch records.")
        actual_count = sum(
            p.numel() for p in fitted.model.parameters() if p.requires_grad
        )
        if actual_count != fitted.trainable_parameters:
            raise ValueError(
                "Persisted trainable parameter count does not match architecture."
            )
    elif kind == "xgboost_native":
        fitted.model = XGBClassifier(**fitted.params)
        fitted.model.load_model(path / "model.ubj")
        if fitted.model.objective != "binary:logistic":
            raise ValueError("The native XGBoost model is not binary:logistic.")
        _binary_classifier_classes(fitted.model)
    else:
        payload = joblib.load(path / "model.joblib")
        if not isinstance(payload, dict) or set(payload) != {"model", "scaler"}:
            raise ValueError("Unexpected trusted joblib payload schema.")
        fitted.model, fitted.scaler = payload["model"], payload["scaler"]
        if (fitted.scaler is not None) != metadata["has_scaler"]:
            raise ValueError("Trusted joblib scaler does not match its metadata.")
        if fitted.scaler is not None and not isinstance(fitted.scaler, StandardScaler):
            raise ValueError("Unexpected trusted preprocessing class.")
        _binary_classifier_classes(fitted.model)
    return fitted


# %%
def run_classifier_checks():
    """Offline CPU checks only: tiny dimensions/reduced fits, never real XLS-R.

    The registry defaults are NOT edited. Temporary artifacts belong to these
    checks; no Drive directory, corpus, or pretrained model is accessed.
    """
    assert len(CLASSIFIER_CONFIGS) == 13 and len(NEURAL_IDS) == 3
    config = replace(
        RunConfig(),
        mode="smoke",
        seed=42,
        cpu_threads=2,
        mlp_batch_size=8,
        sequence_batch_size=4,
        num_workers=0,
        max_epochs=2,
        smoke_epochs=2,
        patience=5,
        learning_rate=1e-3,
        weight_decay=1e-4,
        grad_clip=1.0,
        head_device="cpu",
    )

    class SyntheticClassifierCache:
        hidden_size, pooled_size = 16, 32
        identity = "synthetic-cold-cache"

        def __init__(self):
            rng = np.random.default_rng(123)
            self.rows, self.frames, self.vectors, self.accesses = {}, {}, {}, []
            for split, count in (("train", 16), ("validation", 8), ("test", 8)):
                self.rows[split] = [
                    {"clip_id": f"{split}-{i}", "label": i % 2} for i in range(count)
                ]
                self.frames[split] = [
                    rng.normal(
                        2 * (i % 2) - 1 + (0.4 if split != "train" else 0),
                        0.3,
                        (1 + i % 7, self.hidden_size),
                    ).astype(np.float32)
                    for i in range(count)
                ]
                self.vectors[split] = np.stack(
                    [
                        np.concatenate(
                            (frames.mean(axis=0), frames.std(axis=0, ddof=0))
                        )
                        for frames in self.frames[split]
                    ]
                )

        def index(self, split):
            self.accesses.append(("index", split))
            return self.rows[split]

        def pooled(self, split):
            self.accesses.append(("pooled", split))
            return self.vectors[split]

        def sequence(self, split, index):
            self.accesses.append(("sequence", split))
            return self.frames[split][index]

    def synthetic_params(classifier_id):
        params = copy.deepcopy(CLASSIFIER_CONFIGS[classifier_id])
        if classifier_id in {"random_forest", "extra_trees", "xgboost"}:
            params["n_estimators"] = 4
        if classifier_id == "hist_gradient_boosting":
            params.update(max_iter=3, min_samples_leaf=2, max_leaf_nodes=5)
        if classifier_id == "mlp":
            params["hidden_dims"] = [12, 6]
        if classifier_id in {"asp", "temporal_cnn"}:
            params.update(projection_dim=8, classifier_dim=6)
        if classifier_id == "asp":
            params["attention_dim"] = 5
        return params

    def expect_value_error(call):
        try:
            call()
        except ValueError:
            return
        raise AssertionError("Expected ValueError for an invalid classifier contract.")

    def replace_fixture_json(path, value):
        # Only deliberately corrupt OWN TemporaryDirectory fixtures. Never
        # weaken the notebook's exclusive-write helper for production artifacts.
        Path(path).write_text(json.dumps(value, allow_nan=False), encoding="utf-8")

    def array_probe(fitted, cache, batch_size=2):
        if fitted.classifier_id in {"asp", "temporal_cnn"}:
            return predict_arrays(
                fitted, sequences=cache.frames["validation"], batch_size=batch_size
            )
        return predict_arrays(
            fitted, pooled=cache.vectors["validation"], batch_size=batch_size
        )

    cache = SyntheticClassifierCache()
    with _classifier_runtime(config, neural=True):
        # Public head seams: padding invariance, including the multi-block CNN,
        # batching a short clip beside a longer one, and one-frame backprop.
        for classifier_id in ("asp", "temporal_cnn"):
            seed_everything(config.seed)
            constructor = ASPHead if classifier_id == "asp" else TemporalCNNHead
            head = constructor(
                cache.hidden_size, **synthetic_params(classifier_id)
            ).eval()
            short = torch.from_numpy(cache.frames["train"][2]).unsqueeze(0)
            short_mask = torch.ones(short.shape[:2], dtype=torch.bool)
            expected = head(short, short_mask).detach().numpy()
            padded = F.pad(short, (0, 0, 0, 11), value=float("nan"))
            padded_mask = torch.arange(padded.shape[1])[None, :] < short.shape[1]
            np.testing.assert_allclose(
                expected,
                head(padded, padded_mask).detach().numpy(),
                atol=2e-6,
                rtol=2e-5,
            )
            long = torch.linspace(-1, 1, 17 * cache.hidden_size).reshape(
                17, cache.hidden_size
            )
            values, mask, _, _ = _sequence_classifier_collate(
                [
                    (short[0], 0, 0),
                    (long, 1, 1),
                ]
            )
            np.testing.assert_allclose(
                expected, head(values, mask)[:1].detach().numpy(), atol=2e-6, rtol=2e-5
            )
            one = (
                torch.from_numpy(cache.frames["train"][0]).unsqueeze(0).requires_grad_()
            )
            head.zero_grad(set_to_none=True)
            head(one, torch.ones(1, 1, dtype=torch.bool)).sum().backward()
            assert one.grad is not None and torch.isfinite(one.grad).all()
            assert all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in head.parameters()
            )
            expect_value_error(lambda: head(one, torch.zeros(1, 1, dtype=torch.bool)))

        references = {}
        with tempfile.TemporaryDirectory(prefix="xlsr-classifier-checks-") as temporary:
            root = Path(temporary)
            for classifier_id in CLASSIFIER_CONFIGS:
                before_fit = len(cache.accesses)
                fitted = fit_classifier(
                    classifier_id, synthetic_params(classifier_id), cache, config
                )
                touched = {split for _, split in cache.accesses[before_fit:]}
                assert touched <= (
                    {"train", "validation"}
                    if classifier_id in NEURAL_IDS
                    else {"train"}
                )
                rng_before = torch.random.get_rng_state().clone()
                python_before, numpy_before = random.getstate(), np.random.get_state()
                scores = predict_scores(fitted, cache, "validation", config)
                assert torch.equal(rng_before, torch.random.get_rng_state())
                assert random.getstate() == python_before
                assert np.array_equal(numpy_before[1], np.random.get_state()[1])
                assert numpy_before[2:] == np.random.get_state()[2:]
                assert scores.shape == (8,) and np.isfinite(scores).all()
                if classifier_id == "dummy":
                    np.testing.assert_allclose(scores, 0.5)
                if classifier_id not in NEURAL_IDS and classifier_id != "dummy":
                    assert scores[1::2].mean() > scores[::2].mean(), classifier_id
                if fitted.scaler is not None:
                    np.testing.assert_allclose(
                        fitted.scaler.mean_,
                        cache.vectors["train"].mean(axis=0, dtype=np.float64),
                        atol=1e-12,
                        rtol=1e-12,
                    )
                    np.testing.assert_allclose(
                        fitted.scaler.var_,
                        cache.vectors["train"].var(axis=0, dtype=np.float64),
                        atol=1e-12,
                        rtol=1e-12,
                    )
                    assert fitted.scaler.n_samples_seen_ == 16
                if classifier_id in NEURAL_IDS:
                    assert len(fitted.history) == 2 and fitted.trainable_parameters > 0
                    best = min(
                        fitted.history,
                        key=lambda row: (
                            row["validation_eer"],
                            row["validation_loss"],
                            row["epoch"],
                        ),
                    )
                    assert fitted.best_epoch == best["epoch"]
                    assert (
                        abs(
                            equal_error_rate([i % 2 for i in range(8)], scores)
                            - best["validation_eer"]
                        )
                        < 1e-7
                    )
                    assert fitted.initial_state_sha256 != _head_state_sha256(
                        fitted.model
                    )
                    assert (
                        fitted.loader_seeds["train"]
                        != fitted.loader_seeds["validation"]
                    )
                    references[classifier_id] = (fitted, scores)
                destination = root / classifier_id
                destination.mkdir()
                (destination / "metadata.json").write_text(
                    "caller-owned", encoding="utf-8"
                )
                save_fitted(fitted, destination)
                if classifier_id not in NEURAL_IDS and classifier_id != "xgboost":
                    expect_value_error(lambda: load_fitted(destination))
                loaded = load_fitted(destination, trusted=True)
                np.testing.assert_allclose(
                    scores, array_probe(loaded, cache), atol=2e-6, rtol=2e-5
                )
                assert (
                    fitted.history == loaded.history
                    and fitted.loader_seeds == loaded.loader_seeds
                )
                assert (destination / "metadata.json").read_text(
                    encoding="utf-8"
                ) == "caller-owned"
                try:
                    save_fitted(fitted, destination)
                except FileExistsError:
                    pass
                else:
                    raise AssertionError("Model save overwrote an existing artifact.")

            # Cache/extraction noise and a different preceding classifier/order
            # must not affect initial weights, shuffle order, or fitted scores.
            cache.identity = "synthetic-cache-hit-other-path"
            for offset, classifier_id in enumerate(("temporal_cnn", "asp", "mlp")):
                fit_classifier("dummy", {}, cache, config)
                torch.rand(113 + offset)
                nn.Linear(5, 5)  # Simulated encoder construction on a cold-cache path.
                np.random.random(23 + offset)
                random.random()
                rerun = fit_classifier(
                    classifier_id, synthetic_params(classifier_id), cache, config
                )
                original, original_scores = references[classifier_id]
                assert rerun.initial_state_sha256 == original.initial_state_sha256
                assert rerun.loader_seeds == original.loader_seeds
                assert rerun.history == original.history
                np.testing.assert_allclose(
                    array_probe(rerun, cache), original_scores, atol=2e-6, rtol=2e-5
                )

            # Version and enum checks precede any joblib/weights deserialization.
            configuration = root / "mlp" / "model_config.json"
            metadata = read_json(configuration)
            for package in ("torch", "sklearn"):
                producing_version = metadata["producing_versions"][package]
                metadata["producing_versions"][package] = "not-this-runtime"
                replace_fixture_json(configuration, metadata)
                expect_value_error(lambda: load_fitted(root / "mlp", trusted=True))
                metadata["producing_versions"][package] = producing_version
            metadata["classifier_id"] = "__import__('os').system"
            replace_fixture_json(configuration, metadata)
            expect_value_error(lambda: load_fitted(root / "mlp", trusted=True))
            weights = root / "asp" / "weights.pt"
            with weights.open("r+b") as handle:
                first_byte = handle.read(1)
                handle.seek(0)
                handle.write(bytes([first_byte[0] ^ 1]))
            expect_value_error(lambda: load_fitted(root / "asp", trusted=True))

        expect_value_error(
            lambda: fit_classifier("knn", {"n_neighbors": 17}, cache, config)
        )
        # Negligible synthetic learning rate leaves float32 weights unchanged,
        # providing an exact EER/loss tie: keep epoch 1 and stop after patience.
        tied = fit_classifier(
            "mlp",
            synthetic_params("mlp"),
            cache,
            replace(
                config, mode="comparison", max_epochs=8, patience=2, learning_rate=1e-40
            ),
        )
        assert tied.best_epoch == 1 and len(tied.history) == 3
        assert all(
            row["validation_eer"] == tied.history[0]["validation_eer"]
            for row in tied.history
        )
        assert all(
            row["validation_loss"] == tied.history[0]["validation_loss"]
            for row in tied.history
        )
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always")
            unconverged = fit_classifier(
                "logistic_regression", {"max_iter": 1, "tol": 1e-14}, cache, config
            )
        assert any("ConvergenceWarning" in message for message in unconverged.warnings)
        assert any(
            warning.category.__name__ == "ConvergenceWarning" for warning in observed
        )
        assert all(split != "test" for _, split in cache.accesses)
    print(
        "Classifier checks passed: all 13 reduced synthetic fits; finite gradients/scores; "
        "padding/batch invariance; train-only scaling; best checkpoint; RNG/order isolation; "
        "safe save/reload, version/checksum/trust/overwrite guards; visible convergence warnings."
    )


# %% Feature contracts — small fixtures, no pretrained model allocation.
def run_feature_checks():
    assert conv_frame_count(399) == 0 and conv_frame_count(400) == 1
    assert conv_frame_count(720) == 2 and conv_frame_count(16_000) == 49
    np.testing.assert_array_equal(
        pool_frames(np.array([[1, 3], [3, 7]], dtype=np.float32)), [2, 5, 1, 2]
    )
    np.testing.assert_array_equal(
        pool_frames(np.array([[2, 9]], dtype=np.float32)), [2, 9, 0, 0]
    )
    selection = {
        split: [
            {
                "source": "fixture",
                "split": split,
                "row_index": i,
                "clip_id": f"{split}-{i}",
                "label": i,
                "group": "fixture",
                "language": "en",
                "speaker": None,
                "frame_count": i + 1,
                "audio_sha256": f"{split}-{i}",
            }
            for i in range(2)
        ]
        for split in SPLITS
    }
    assets = {
        "hidden_size": 2,
        "conv_kernel": list(CONV_KERNELS),
        "conv_stride": list(CONV_STRIDES),
    }
    payload = feature_identity_payload(selection, {}, RunConfig(), assets)
    assert payload == feature_identity_payload(
        selection, {}, replace(RunConfig(), max_epochs=7), assets
    )
    changed = copy.deepcopy(selection)
    changed["train"][0]["language"] = "hi"
    assert json_digest(payload) != json_digest(
        feature_identity_payload(changed, {}, RunConfig(), assets)
    )
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        batches = [
            (split, i, np.full((i + 1, 2), i + 1, dtype=np.float32))
            for split in SPLITS
            for i in range(2)
        ]
        write_feature_cache(root / "cache", selection, payload, 2, iter(batches))
        cache = FeatureCache(root / "cache", expected_identity=json_digest(payload))
        np.testing.assert_array_equal(
            cache.pooled("train"), [[1, 1, 0, 0], [2, 2, 0, 0]]
        )
        assert cache.sequence("train", 1).shape == (2, 2)
        expect_error(lambda: FeatureCache(root / "cache", expected_identity="wrong"))
        copy_artifact(root / "cache", root / "restored")
        restored = FeatureCache(root / "restored")
        np.testing.assert_array_equal(restored.pooled("train"), cache.pooled("train"))
        cache.close()
        restored.close()
        incomplete = root / "incomplete"
        incomplete.mkdir()
        expect_error(lambda: FeatureCache(incomplete))
        with (root / "restored" / "train" / "pooled.npy").open("ab") as handle:
            handle.write(b"corrupt")
        expect_error(lambda: FeatureCache(root / "restored"))
    print("PASS: frontend lengths, valid-frame pooling, cache identity/copy/corruption")


# %% Packed feature storage — immutable, checksummed, classifier-independent.
FEATURE_SCHEMA = "frozen-xlsr-frames-v1"


def pool_frames(frames):
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 2 or not min(frames.shape) or not np.isfinite(frames).all():
        raise ValueError("Expected a nonempty finite [time, hidden] feature sequence.")
    return np.concatenate(
        (
            frames.mean(axis=0, dtype=np.float64),
            frames.std(axis=0, dtype=np.float64, ddof=0),
        )
    ).astype(np.float32)


def feature_identity_payload(selection, provenance, config, assets):
    versions = package_versions()
    return {
        "schema": FEATURE_SCHEMA,
        "selection": copy.deepcopy(selection),
        "sources": copy.deepcopy(provenance),
        "assets": copy.deepcopy(assets),
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "layer": config.layer,
        "max_seconds": config.max_seconds,
        "crop_policy": "center-if-explicitly-capped",
        "dtype": "float32",
        "pooling": "mean-population-std-v1",
        "encoder_mode": "frozen-eval-no-grad-no-masks",
        "producing_versions": {
            name: versions[name]
            for name in (
                "torch",
                "torchcodec",
                "datasets",
                "transformers",
                "numpy",
                "soundfile",
            )
        },
    }


def safe_artifact_path(root, relative):
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("Invalid artifact-relative filename.")
    fragment = Path(relative)
    if fragment.is_absolute() or any(part in (".", "..") for part in fragment.parts):
        raise ValueError("Unsafe artifact-relative filename.")
    root = Path(root)
    path = root / fragment
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Artifact path escapes its root.")
    for candidate in (path, *path.parents):
        if candidate == root:
            break
        if candidate.is_symlink():
            raise ValueError("Artifacts may not contain symlinks.")
    return path


def artifact_inventory(root):
    root = Path(root)
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Artifacts may not contain symlinks.")
        if path.is_file() and path != root / "manifest.json":
            files[path.relative_to(root).as_posix()] = {
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
    return files


def verify_artifact(root, allowed_status=("complete",)):
    root = Path(root)
    if not (root / "manifest.json").is_file():
        raise ValueError(
            f"{root}: incomplete artifact (no completion manifest); use a new location."
        )
    manifest = read_json(root / "manifest.json")
    if manifest.get("status") not in allowed_status or not manifest.get("files"):
        raise ValueError(f"{root}: artifact is not complete/compatible.")
    for relative, entry in manifest["files"].items():
        path = safe_artifact_path(root, relative)
        if (
            not path.is_file()
            or path.stat().st_size != entry["size"]
            or file_sha256(path) != entry["sha256"]
        ):
            raise ValueError(f"{root}: checksum/inventory failure for {relative}.")
    if artifact_inventory(root) != manifest["files"]:
        raise ValueError(f"{root}: unexpected or missing artifact files.")
    return manifest


def finish_artifact(root, metadata):
    manifest = {
        **metadata,
        "files": artifact_inventory(root),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(
        Path(root) / "manifest.json", manifest
    )  # Completion marker is written LAST.
    return manifest


def ensure_space(path, required_bytes, purpose):
    parent = Path(path)
    while not parent.exists():
        parent = parent.parent
    free = shutil.disk_usage(parent).free
    if free < required_bytes:
        raise RuntimeError(
            f"{purpose}: need approximately {required_bytes / 2**30:.2f} GiB; "
            f"only {free / 2**30:.2f} GiB is available at {parent}."
        )


def copy_artifact(source, destination, allowed_status=("complete",)):
    source, destination = Path(source), Path(destination)
    manifest = verify_artifact(source, allowed_status)
    if destination.exists():
        existing = verify_artifact(destination, allowed_status)
        if existing != manifest:
            raise ValueError(
                f"{destination}: different existing artifact; choose a new location/tag."
            )
        return destination
    required = sum(entry["size"] for entry in manifest["files"].values())
    ensure_space(destination.parent, int(required * 1.2) + 1024**2, "Artifact staging")
    destination.mkdir(parents=True, exist_ok=False)
    for relative in manifest["files"]:
        target = safe_artifact_path(destination, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(safe_artifact_path(source, relative), target)
    if artifact_inventory(destination) != manifest["files"]:
        raise ValueError(
            f"{destination}: copied inventory differs; no completion marker published."
        )
    write_json(destination / "manifest.json", manifest)
    return destination


def write_jsonl(path, rows):
    with Path(path).open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, sort_keys=True, allow_nan=False, default=json_default)
                + "\n"
            )


def write_feature_cache(
    path, selection, identity_payload, hidden_size, feature_batches, diagnostics=None
):
    path = Path(path)
    if path.exists():
        raise ValueError(
            f"{path}: never overwrite an existing/incomplete cache; choose a new work location."
        )
    if set(selection) != set(SPLITS) or any(not selection[split] for split in SPLITS):
        raise ValueError("Every cache split must contain selected recordings.")
    if identity_payload.get("selection") != selection:
        raise ValueError("Cache identity and selected membership differ.")
    path.mkdir(parents=True, exist_ok=False)
    arrays, indexes, seen = {}, {}, set()
    try:
        for split in SPLITS:
            folder = path / split
            folder.mkdir()
            offset, rows = 0, []
            for row in selection[split]:
                count = row["frame_count"]
                if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                    raise ValueError(
                        "Each recording needs a positive integer frame count."
                    )
                rows.append({**row, "frame_offset": offset})
                offset += count
            indexes[split] = rows
            arrays[split] = (
                np.lib.format.open_memmap(
                    folder / "frames.npy",
                    mode="w+",
                    dtype="float32",
                    shape=(offset, hidden_size),
                ),
                np.lib.format.open_memmap(
                    folder / "pooled.npy",
                    mode="w+",
                    dtype="float32",
                    shape=(len(rows), 2 * hidden_size),
                ),
            )
        for split, index, values in feature_batches:
            if (
                split not in indexes
                or not 0 <= index < len(indexes[split])
                or (split, index) in seen
            ):
                raise ValueError(
                    "Feature stream contains an unknown or duplicate recording."
                )
            row = indexes[split][index]
            values = np.asarray(values, dtype=np.float32)
            if (
                values.shape != (row["frame_count"], hidden_size)
                or not np.isfinite(values).all()
            ):
                raise ValueError(
                    f"{split}/{row['clip_id']}: invalid extracted feature shape/values."
                )
            frames, pooled = arrays[split]
            start = row["frame_offset"]
            frames[start : start + len(values)] = values
            pooled[index] = pool_frames(values)
            seen.add((split, index))
        if len(seen) != sum(len(rows) for rows in indexes.values()):
            raise ValueError(
                "Feature stream ended before every selected recording was written."
            )
        for split, rows in indexes.items():
            write_jsonl(path / split / "index.jsonl", rows)
    finally:
        for matrices in arrays.values():
            for matrix in matrices:
                matrix.flush()
                matrix._mmap.close()
    finish_artifact(
        path,
        {
            "status": "complete",
            "kind": FEATURE_SCHEMA,
            "identity": json_digest(identity_payload),
            "identity_payload": identity_payload,
            "hidden_size": hidden_size,
            "counts": {split: len(selection[split]) for split in SPLITS},
            "diagnostics": diagnostics or {},
        },
    )
    return path


class FeatureCache:
    def __init__(self, path, expected_identity=None):
        self.root = Path(path)
        self.manifest = verify_artifact(self.root)
        manifest = self.manifest
        self.identity = manifest.get("identity")
        if (
            manifest.get("kind") != FEATURE_SCHEMA
            or json_digest(manifest.get("identity_payload")) != self.identity
            or (expected_identity is not None and self.identity != expected_identity)
        ):
            raise ValueError("Cache configuration/content identity mismatch.")
        self.hidden_size = manifest["hidden_size"]
        self.pooled_size = 2 * self.hidden_size
        self._indexes, self._arrays = {}, {}
        try:
            for split in SPLITS:
                with (self.root / split / "index.jsonl").open(
                    encoding="utf-8"
                ) as handle:
                    rows = [json.loads(line) for line in handle]
                original = [
                    {key: value for key, value in row.items() if key != "frame_offset"}
                    for row in rows
                ]
                if (
                    original != manifest["identity_payload"]["selection"][split]
                    or len(rows) != manifest["counts"][split]
                ):
                    raise ValueError(
                        "Cache index is not aligned with its frozen selection."
                    )
                offset = 0
                for row in rows:
                    if (
                        row["frame_offset"] != offset
                        or row["frame_count"] < 1
                        or row["label"] not in (0, 1)
                    ):
                        raise ValueError("Invalid cache offset/label/frame count.")
                    offset += row["frame_count"]
                frames = np.load(
                    self.root / split / "frames.npy", mmap_mode="r", allow_pickle=False
                )
                pooled = np.load(
                    self.root / split / "pooled.npy", mmap_mode="r", allow_pickle=False
                )
                self._arrays[split] = (frames, pooled)
                if (
                    frames.dtype != np.float32
                    or pooled.dtype != np.float32
                    or frames.shape != (offset, self.hidden_size)
                    or pooled.shape != (len(rows), self.pooled_size)
                    or not np.isfinite(pooled).all()
                ):
                    raise ValueError(
                        "Cache matrices have invalid shapes, precision, or pooled values."
                    )
                self._indexes[split] = rows
        except Exception:
            self.close()
            raise

    def index(self, split):
        return copy.deepcopy(self._indexes[split])

    def pooled(self, split):
        return self._arrays[split][1]

    def sequence(self, split, index):
        row = self._indexes[split][index]
        start = row["frame_offset"]
        return self._arrays[split][0][start : start + row["frame_count"]]

    def close(self):
        for matrices in self._arrays.values():
            for matrix in matrices:
                if (
                    getattr(matrix, "_mmap", None) is not None
                    and not matrix._mmap.closed
                ):
                    matrix._mmap.close()


# %% Encoder contract — random tiny model only; no Hub calls or 300M allocation.
def run_encoder_checks():
    seed_everything(42)
    torch.set_num_threads(2)
    tiny_config = Wav2Vec2Config(
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        conv_dim=(4,) * 7,
        conv_kernel=CONV_KERNELS,
        conv_stride=CONV_STRIDES,
        feat_extract_norm="layer",
        do_stable_layer_norm=True,
        num_conv_pos_embeddings=8,
        num_conv_pos_embedding_groups=2,
    )
    model = Wav2Vec2Model(tiny_config)
    extractor = Wav2Vec2FeatureExtractor(sampling_rate=TARGET_SR, do_normalize=True)
    waves = [
        np.sin(np.arange(length) * 0.07).astype(np.float32) for length in (1200, 1900)
    ]
    before = [parameter.detach().clone() for parameter in model.parameters()]
    sequences = encode_waveforms(model, extractor, waves, 1, "cpu")
    assert [len(sequence) for sequence in sequences] == [3, 5]
    single = encode_waveforms(model, extractor, waves[:1], 1, "cpu")[0]
    np.testing.assert_allclose(single, sequences[0], atol=2e-5, rtol=2e-5)
    assert not model.training and not model.config.apply_spec_augment
    assert all(
        not parameter.requires_grad and parameter.grad is None
        for parameter in model.parameters()
    )
    head = nn.Linear(32, 1)
    optimizer = torch.optim.SGD(head.parameters(), lr=0.01)
    old_weight = head.weight.detach().clone()
    loss = (
        head(torch.tensor(np.stack([pool_frames(sequence) for sequence in sequences])))
        .square()
        .mean()
    )
    loss.backward()
    optimizer.step()
    assert not torch.equal(old_weight, head.weight)
    assert all(
        torch.equal(old, current)
        for old, current in zip(before, model.parameters(), strict=True)
    )
    assert all(parameter.grad is None for parameter in model.parameters())
    expect_error(lambda: encode_waveforms(model, extractor, [np.ones(399)], 1, "cpu"))
    print(
        "PASS: frozen encoder, attention masks, layer/frame lengths, and head-only gradients"
    )


# %% Frozen encoder adapter — selected hidden states, never classifier training.
def load_encoder_assets(config):
    config.validate()
    model_config = Wav2Vec2Config.from_pretrained(
        config.model_id, revision=config.model_revision
    )
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        config.model_id, revision=config.model_revision
    )
    if (
        model_config.hidden_size != 1024
        or model_config.feat_extract_norm != "layer"
        or tuple(model_config.conv_kernel) != CONV_KERNELS
        or tuple(model_config.conv_stride) != CONV_STRIDES
        or extractor.sampling_rate != TARGET_SR
        or not extractor.do_normalize
        or not 1 <= config.layer <= model_config.num_hidden_layers
    ):
        raise ValueError(
            "Checkpoint/layer differs from the configured XLS-R-300M input contract."
        )
    model_config.apply_spec_augment = False
    signature = {
        "hidden_size": model_config.hidden_size,
        "conv_kernel": list(model_config.conv_kernel),
        "conv_stride": list(model_config.conv_stride),
        "config": model_config.to_dict(),
        "feature_extractor": extractor.to_dict(),
    }
    return model_config, extractor, signature


def encode_waveforms(model, extractor, waveforms, layer, device):
    if not waveforms or not 1 <= layer <= model.config.num_hidden_layers:
        raise ValueError(
            "Expected nonempty waveforms and a valid transformer layer (1-based)."
        )
    arrays = [np.asarray(waveform, dtype=np.float32) for waveform in waveforms]
    lengths = []
    for waveform in arrays:
        if waveform.ndim != 1 or not len(waveform) or not np.isfinite(waveform).all():
            raise ValueError("Expected a nonempty finite mono waveform.")
        frames = conv_frame_count(
            len(waveform), model.config.conv_kernel, model.config.conv_stride
        )
        if frames < 1:
            raise ValueError(
                "Recording is too short for one frontend frame; no row will be dropped."
            )
        lengths.append(frames)
    model.requires_grad_(False)
    model.eval()
    model.config.apply_spec_augment = False
    model.to(device=device, dtype=torch.float32)
    batch = extractor(
        arrays,
        sampling_rate=TARGET_SR,
        padding=True,
        truncation=False,
        return_attention_mask=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        output = model(
            input_values=batch["input_values"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = output.hidden_states[layer]
        sequences = [
            hidden[i, :length].detach().cpu().numpy().copy()
            for i, length in enumerate(lengths)
        ]
    if any(
        sequence.shape != (length, model.config.hidden_size)
        or not np.isfinite(sequence).all()
        for sequence, length in zip(sequences, lengths, strict=True)
    ):
        raise ValueError("Encoder returned invalid hidden-state frames.")
    return sequences


def encoder_feature_batches(
    sources,
    specs,
    selection,
    config,
    model_config,
    extractor,
    diagnostics,
    probe_holder=None,
):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Select a Colab GPU runtime for pretrained XLS-R extraction; CPU fallback is disabled."
        )
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model = None
    batch_refs = []
    try:
        model = Wav2Vec2Model.from_pretrained(
            config.model_id, revision=config.model_revision, config=model_config
        ).to(device="cuda", dtype=torch.float32)
        model.requires_grad_(False)
        model.eval()
        references = [
            (split, i, row)
            for split in SPLITS
            for i, row in enumerate(selection[split])
        ]
        references.sort(key=lambda item: (item[2]["used_samples"], item[0], item[1]))
        for start in range(0, len(references), config.extraction_batch_size):
            batch_refs = references[start : start + config.extraction_batch_size]
            waves = [
                read_prepared_waveform(sources, specs, row, config.max_seconds)
                for _, _, row in batch_refs
            ]
            sequences = encode_waveforms(model, extractor, waves, config.layer, "cuda")
            for (split, index, _), sequence in zip(batch_refs, sequences, strict=True):
                if probe_holder is not None and split == "validation" and index == 0:
                    # Retain one fresh raw-audio -> encoder result outside the persisted
                    # diagnostics. It verifies reloaded heads without another encoder pass.
                    probe_holder["frames"] = sequence.copy()
                yield split, index, sequence
            if (
                start == 0
                or start + len(batch_refs) == len(references)
                or start % 32 == 0
            ):
                print(f"XLS-R: {start + len(batch_refs)}/{len(references)} recordings")
        torch.cuda.synchronize()
        diagnostics.update(
            {
                "extraction_seconds": time.perf_counter() - started,
                "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
                "gpu": torch.cuda.get_device_name(),
                "attention_backend": getattr(
                    model.config, "_attn_implementation", None
                ),
                "encoder_trainable_parameters": sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ),
            }
        )
    except torch.cuda.OutOfMemoryError as error:
        names = [f"{row['source']}/{row['clip_id']}" for _, _, row in batch_refs]
        raise RuntimeError(
            f"XLS-R GPU memory exhausted for {names}, MAX_SECONDS={config.max_seconds}. "
            "Reduce extraction_batch_size or explicitly enable MAX_SECONDS. No recording "
            "was silently cropped/dropped. Use a new work location for an incomplete cache."
        ) from error
    finally:
        del model
        torch.cuda.empty_cache()


def feature_storage_bytes(selection, hidden_size):
    frames = sum(row["frame_count"] for rows in selection.values() for row in rows)
    clips = sum(map(len, selection.values()))
    return frames * hidden_size * 4 + clips * hidden_size * 2 * 4


def acquire_feature_cache(
    sources, specs, selection, provenance, config, assets, local_root, drive_root
):
    model_config, extractor, signature = assets
    payload = feature_identity_payload(selection, provenance, config, signature)
    identity = json_digest(payload)
    local = Path(local_root) / "feature_cache" / identity
    persistent = Path(drive_root) / "feature_cache" / identity
    if local.exists():
        cache = FeatureCache(local, identity)
        if persistent.exists():
            remote = verify_artifact(persistent)
            if remote != cache.manifest:
                raise ValueError(
                    "Local and Drive caches with the same identity differ."
                )
        else:
            copy_artifact(local, persistent)
        print(f"Reused features without loading XLS-R weights: {identity[:12]}")
        return cache
    if persistent.exists():
        remote = FeatureCache(persistent, identity)
        remote.close()
        copy_artifact(persistent, local)
        print(
            f"Staged verified Drive cache without loading XLS-R weights: {identity[:12]}"
        )
        return FeatureCache(local, identity)
    required = feature_storage_bytes(selection, signature["hidden_size"])
    ensure_space(
        local_root,
        int(required * 1.3) + 2 * 1024**3,
        "Features, model weights, and staging",
    )
    print(
        f"Selected unpadded feature payload: approximately {required / 2**30:.2f} GiB (plus overhead)."
    )
    diagnostics = {
        "extraction_batch_size": config.extraction_batch_size,
        "precision": "float32",
    }
    probe_holder = {}
    batches = encoder_feature_batches(
        sources,
        specs,
        selection,
        config,
        model_config,
        extractor,
        diagnostics,
        probe_holder,
    )
    write_feature_cache(
        local, selection, payload, signature["hidden_size"], batches, diagnostics
    )
    copy_artifact(local, persistent)
    cache = FeatureCache(local, identity)
    cache.raw_probe = probe_holder
    return cache


# %% Reporting contracts — all ranking uses validation, never test metrics.
def effective_classifier_configs(enabled, overrides):
    if not enabled or len(set(enabled)) != len(enabled):
        raise ValueError("Enable a nonempty, unique list of classifiers.")
    if set(enabled) - set(CLASSIFIER_CONFIGS) or set(overrides) - set(
        CLASSIFIER_CONFIGS
    ):
        raise ValueError("Unknown classifier ID in enabled classifiers or overrides.")
    parameters = {}
    for name in enabled:
        if not isinstance(overrides.get(name, {}), dict):
            raise ValueError(
                f"{name}: classifier overrides must be a parameter dictionary."
            )
        parameters[name] = {
            **copy.deepcopy(CLASSIFIER_CONFIGS[name]),
            **copy.deepcopy(overrides.get(name, {})),
        }
    json_digest(parameters)
    return parameters


def rank_results(results):
    successful = sorted(
        (row for row in results if row["status"] != "failed"),
        key=lambda row: (row["eer"], -row["f1"], row["classifier"]),
    )
    ranked = [{**row, "rank": rank} for rank, row in enumerate(successful, start=1)]
    ranked.extend(
        {**row, "rank": None}
        for row in sorted(results, key=lambda row: row["classifier"])
        if row["status"] == "failed"
    )
    return ranked


def grouped_metrics(index, scores, threshold, classifier_id):
    labels, scores = metric_inputs([row["label"] for row in index], scores)
    reports = []
    for dimension in ("group", "source", "language"):
        for value in sorted(
            {row.get(dimension) for row in index}, key=lambda item: str(item)
        ):
            positions = [
                i for i, row in enumerate(index) if row.get(dimension) == value
            ]
            reports.append(
                {
                    "classifier_id": classifier_id,
                    "dimension": dimension,
                    "value": value,
                    **metric_summary(labels[positions], scores[positions], threshold),
                }
            )
    return reports


def prediction_rows(index, scores, threshold, classifier_id):
    _, scores = metric_inputs([row["label"] for row in index], scores)
    fields = (
        "source",
        "group",
        "split",
        "row_index",
        "clip_id",
        "speaker",
        "language",
        "label",
    )
    return [
        {
            **{key: row.get(key) for key in fields},
            "classifier_id": classifier_id,
            "score": float(score),
            "prediction": int(score >= np.float64(threshold)),
        }
        for row, score in zip(index, scores, strict=True)
    ]


def write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False, mode="x")


def show_leaderboard(rows, test=False):
    table = pd.DataFrame(rows)
    columns = ["validation_rank", "validation_eer"] if test else ["rank"]
    columns += [
        "classifier",
        "status",
        "eer",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "support",
    ]
    shown = table.reindex(columns=columns).copy()
    percentages = ["eer", "accuracy", "precision", "recall", "f1"] + (
        ["validation_eer"] if test else []
    )
    for name in percentages:
        shown[name] = shown[name].map(
            lambda value: "—" if pd.isna(value) else f"{100 * value:.2f}"
        )
    shown = shown.rename(columns={name: name + " (%)" for name in percentages}).fillna(
        "—"
    )
    try:
        from IPython.display import display

        display(shown)
    except ImportError:
        print(shown.to_string(index=False))


def runtime_environment():
    return {
        "packages": package_versions(),
        "platform": sys.platform,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "numerical_policy": "float32; deterministic where supported; no cross-hardware identity guarantee",
    }


def configure_runtime(config):
    config.validate()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(config.cpu_threads)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    seed_everything(config.seed)


def run_identity(config, payload):
    return f"{config.run_tag}-{json_digest(payload)[:16]}"


def feature_contract(cache):
    payload = cache.manifest["identity_payload"]
    return {
        key: copy.deepcopy(payload[key])
        for key in (
            "model_id",
            "model_revision",
            "layer",
            "max_seconds",
            "crop_policy",
            "dtype",
            "assets",
        )
    }


def prepare_experiment(
    config,
    source_configs,
    groups,
    quotas,
    enabled,
    overrides,
    local_root,
    drive_root,
    quota_confirmed,
):
    configure_runtime(config)
    parameters = effective_classifier_configs(enabled, overrides)
    if not source_configs or any(
        not item.get("path") for item in source_configs.values()
    ):
        raise ValueError(
            "Edit SOURCE_CONFIGS: set each exact full DatasetDict path before continuing."
        )
    specs = {name: SourceSpec(**values) for name, values in source_configs.items()}
    sources, metadata, provenance = load_sources(specs)
    selected = select_rows(metadata, groups, quotas, config.seed)
    prepared, audit = prepare_selection(selected, sources, specs, config.max_seconds)
    assets = load_encoder_assets(
        config
    )  # Small config/feature-extractor files only, NOT weights.
    identity_payload = feature_identity_payload(prepared, provenance, config, assets[2])
    payload = {
        "config": asdict(config),
        "feature_identity": json_digest(identity_payload),
        "classifiers": parameters,
        "enabled_classifiers": list(enabled),
        "groups": groups,
        "quotas": quotas,
        "label_names": LABEL_NAMES,
        "selection_digest": json_digest(prepared),
    }
    identity = run_identity(config, payload)
    persistent_cache = Path(drive_root) / "feature_cache" / payload["feature_identity"]
    persistent_run = Path(drive_root) / "runs" / identity
    required = feature_storage_bytes(prepared, assets[2]["hidden_size"])
    if not quota_confirmed and (
        not persistent_cache.exists() or not persistent_run.exists()
    ):
        raise RuntimeError(
            f"New Drive publication needs approximately {required / 2**30:.2f} GiB of features "
            "plus model/report overhead. Verify your actual Drive account quota, then set "
            "DRIVE_QUOTA_CONFIRMED=True. No model weights have been loaded."
        )
    print(
        json.dumps(
            {
                "selection_counts": audit["counts"],
                "durations": audit["durations"],
                "source_overlaps": {
                    name: item["overlaps"] for name, item in provenance.items()
                },
            },
            indent=2,
        )
    )
    for name, item in provenance.items():
        for message in item.get("warnings", []):
            warnings.warn(f"{name}: {message}", stacklevel=2)
    return {
        "config": config,
        "sources": sources,
        "specs": specs,
        "selection": prepared,
        "provenance": provenance,
        "audit": audit,
        "assets": assets,
        "parameters": parameters,
        "payload": payload,
        "local_root": Path(local_root),
        "drive_root": Path(drive_root),
    }


def train_comparison(
    cache,
    config,
    classifier_parameters,
    payload,
    input_summary,
    local_root,
    drive_root=None,
    quota_confirmed=False,
):
    configure_runtime(config)
    if (
        payload.get("config") != asdict(config)
        or payload["feature_identity"] != cache.identity
        or payload["classifiers"] != classifier_parameters
    ):
        raise ValueError(
            "Live settings differ from the frozen preparation, features, or classifiers. "
            "Rerun the preparation/cache cells before training or reusing a run."
        )
    identity = run_identity(config, payload)
    root = Path(local_root) / "runs" / identity
    persistent = (
        Path(drive_root) / "runs" / identity if drive_root is not None else None
    )
    if persistent is not None and not persistent.exists() and not quota_confirmed:
        raise RuntimeError(
            "Confirm actual Drive account capacity before publishing a new run."
        )
    if not root.exists() and persistent is not None and persistent.exists():
        copy_artifact(persistent, root, allowed_status=("complete", "partial"))
    if root.exists():
        manifest = verify_artifact(root, allowed_status=("complete", "partial"))
        if manifest.get("kind") != "xlsr-classifier-run-v1" or manifest.get(
            "identity"
        ) != json_digest(payload):
            raise ValueError(
                "Existing run differs from this frozen configuration; use a new run tag."
            )
        if persistent is not None:
            copy_artifact(root, persistent, allowed_status=("complete", "partial"))
        if manifest["status"] == "partial" and config.mode == "smoke":
            raise RuntimeError(
                f"Smoke acceptance is incomplete for {manifest['failed']}; use a new run tag after fixing settings."
            )
        print(f"Reused sealed {manifest['status']} comparison (no retraining): {root}")
        return root
    train_labels = np.array([row["label"] for row in cache.index("train")])
    validation_index = cache.index("validation")
    validation_labels = np.array([row["label"] for row in validation_index])
    metric_inputs(train_labels, np.zeros(len(train_labels)))
    metric_inputs(validation_labels, np.zeros(len(validation_labels)))
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "config.json", payload)
    write_json(root / "environment.json", runtime_environment())
    write_json(root / "input_summary.json", input_summary)
    write_jsonl(
        root / "selection.jsonl",
        [row for split in SPLITS for row in cache.index(split)],
    )
    results, predictions, groups, successful, failures = [], [], [], [], []
    for name, parameters in classifier_parameters.items():
        fitted = restored = None
        folder = root / "models" / name
        try:
            print(f"Fitting {DISPLAY_NAMES[name]} ({config.mode}; one configuration)")
            fitted = fit_classifier(name, parameters, cache, config)
            scores = predict_scores(fitted, cache, "validation", config)
            _, scores = metric_inputs(validation_labels, scores)
            point = operating_threshold(validation_labels, scores)
            metrics = metric_summary(validation_labels, scores, point["threshold"])
            folder.mkdir(parents=True, exist_ok=False)
            save_fitted(fitted, folder)
            restored = load_fitted(folder, trusted=True)
            reloaded = predict_scores(restored, cache, "validation", config)
            np.testing.assert_allclose(scores, reloaded, atol=2e-5, rtol=2e-5)
            kwargs = (
                {"sequences": [cache.sequence("validation", 0)]}
                if name in ("asp", "temporal_cnn")
                else {"pooled": cache.pooled("validation")[:1]}
            )
            probe = predict_arrays(restored, **kwargs)
            np.testing.assert_allclose(probe, scores[:1], atol=2e-5, rtol=2e-5)
            raw_probe = getattr(cache, "raw_probe", None)
            if raw_probe is not None:
                frames = raw_probe["frames"]
                kwargs = (
                    {"sequences": [frames]}
                    if name in ("asp", "temporal_cnn")
                    else {"pooled": pool_frames(frames)[None]}
                )
                np.testing.assert_allclose(
                    predict_arrays(restored, **kwargs), scores[:1], atol=2e-5, rtol=2e-5
                )
            metadata = {
                "classifier_id": name,
                "feature_identity": cache.identity,
                "feature_contract": feature_contract(cache),
                "label_names": LABEL_NAMES,
                "threshold": point["threshold"],
                "score_type": fitted.score_type,
                "validation_operating_point": point,
                "validation_metrics": metrics,
                "fit_seconds": fitted.fit_seconds,
                "trainable_parameters": fitted.trainable_parameters,
                "loader_seeds": fitted.loader_seeds,
                "warnings": fitted.warnings,
                "reload_check": "passed",
                "raw_probe_check": "passed-fresh-extraction"
                if raw_probe is not None
                else "cached-probe-only; no encoder reload",
            }
            write_json(folder / "metadata.json", metadata)
            finish_artifact(
                folder,
                {
                    "status": "complete",
                    "kind": "xlsr-classifier-v1",
                    "feature_identity": cache.identity,
                },
            )
            results.append(
                {
                    "classifier_id": name,
                    "classifier": DISPLAY_NAMES[name],
                    "status": "warning" if fitted.warnings else "complete",
                    **metrics,
                    "threshold": point["threshold"],
                    "score_type": fitted.score_type,
                    "fit_seconds": fitted.fit_seconds,
                    "error": None,
                }
            )
            groups.extend(
                grouped_metrics(validation_index, scores, point["threshold"], name)
            )
            predictions.extend(
                prediction_rows(validation_index, scores, point["threshold"], name)
            )
            successful.append(name)
        except OSError:
            raise  # Storage/permission failures are not classifier failures; leave no completion marker.
        except Exception as error:
            failures.append(name)
            results.append(
                {
                    "classifier_id": name,
                    "classifier": DISPLAY_NAMES[name],
                    "status": "failed",
                    **{
                        key: None
                        for key in (
                            "eer",
                            "accuracy",
                            "precision",
                            "recall",
                            "f1",
                            "support",
                        )
                    },
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(f"FAILED {name}: {type(error).__name__}: {error}")
        finally:
            del fitted, restored
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    results = rank_results(results)
    write_csv(root / "validation_metrics.csv", results)
    write_json(root / "validation_metrics.json", results)
    write_csv(root / "grouped_validation_metrics.csv", groups)
    write_json(root / "grouped_validation_metrics.json", groups)
    write_csv(root / "validation_predictions.csv", predictions)
    winner = next((row["classifier_id"] for row in results if row["rank"] == 1), None)
    finish_artifact(
        root,
        {
            "status": "partial" if failures else "complete",
            "kind": "xlsr-classifier-run-v1",
            "identity": json_digest(payload),
            "feature_identity": cache.identity,
            "successful": successful,
            "failed": failures,
            "validation_winner": winner,
            "mode": config.mode,
        },
    )
    if persistent is not None:
        copy_artifact(root, persistent, allowed_status=("complete", "partial"))
    if failures and config.mode == "smoke":
        raise RuntimeError(
            f"Smoke acceptance failed for {failures}; inspect {root}. Use a NEW run tag after fixing settings."
        )
    return root


def final_test_evaluation(
    cache,
    comparison_root,
    local_root,
    drive_root=None,
    enabled=False,
    quota_confirmed=False,
):
    if not enabled:
        print(
            "Test scoring is disabled. Freeze settings, then explicitly enable the final test cell."
        )
        return None
    comparison_root = Path(comparison_root)
    run = verify_artifact(comparison_root, allowed_status=("complete", "partial"))
    if (
        run.get("kind") != "xlsr-classifier-run-v1"
        or run.get("feature_identity") != cache.identity
        or not run.get("successful")
    ):
        raise ValueError(
            "Final test scoring requires a sealed comparison with successful candidates "
            "bound to this exact feature cache. Incomplete publications are not accepted."
        )
    if run["status"] == "partial":
        print(
            f"Partial comparison: evaluating only frozen successful candidates; excluded failures: {run['failed']}"
        )
    payload = read_json(comparison_root / "config.json")
    config = RunConfig(**payload["config"])
    validation = read_json(comparison_root / "validation_metrics.json")
    evaluation = {
        "comparison_identity": run["identity"],
        "comparison_manifest_sha256": file_sha256(comparison_root / "manifest.json"),
        "feature_identity": cache.identity,
        "policy": "frozen-validation-thresholds-v1",
        "comparison_status": run["status"],
        "excluded_failed_classifiers": run["failed"],
    }
    identity = f"{comparison_root.name}-{json_digest(evaluation)[:12]}"
    # Sibling artifact: never append files to the sealed comparison run's inventory.
    root = Path(local_root) / "test_evaluations" / identity
    persistent = (
        Path(drive_root) / "test_evaluations" / identity
        if drive_root is not None
        else None
    )
    if persistent is not None and not persistent.exists() and not quota_confirmed:
        raise RuntimeError(
            "Confirm actual Drive account capacity before new test-result publication."
        )
    if not root.exists() and persistent is not None and persistent.exists():
        copy_artifact(persistent, root)
    if root.exists():
        existing = verify_artifact(root)
        if existing.get("identity") != json_digest(evaluation):
            raise ValueError(
                "Existing test evaluation differs from the frozen comparison."
            )
        if persistent is not None:
            copy_artifact(root, persistent)
        return root
    root.mkdir(parents=True, exist_ok=False)
    index = cache.index("test")
    labels = [row["label"] for row in index]
    results, predictions, groups = [], [], []
    for row in validation:  # Retain validation order; NEVER select a test winner.
        name = row["classifier_id"]
        if name not in run["successful"]:
            continue
        folder = comparison_root / "models" / name
        verify_artifact(folder)
        metadata = read_json(folder / "metadata.json")
        if metadata["feature_identity"] != cache.identity:
            raise ValueError("Classifier artifact and test feature cache differ.")
        fitted = load_fitted(folder, trusted=True)
        scores = predict_scores(fitted, cache, "test", config)
        threshold = metadata["threshold"]
        results.append(
            {
                "classifier_id": name,
                "classifier": row["classifier"],
                "status": row["status"],
                "validation_rank": row["rank"],
                "validation_eer": row["eer"],
                "threshold": threshold,
                **metric_summary(labels, scores, threshold),
            }
        )
        predictions.extend(prediction_rows(index, scores, threshold, name))
        groups.extend(grouped_metrics(index, scores, threshold, name))
        del fitted
    write_json(root / "config.json", evaluation)
    write_json(root / "test_metrics.json", results)
    write_csv(root / "test_metrics.csv", results)
    write_csv(root / "test_predictions.csv", predictions)
    write_csv(root / "grouped_test_metrics.csv", groups)
    write_json(root / "grouped_test_metrics.json", groups)
    finish_artifact(
        root,
        {
            "status": "complete",
            "kind": "xlsr-final-test-v1",
            "identity": json_digest(evaluation),
            "validation_winner": run["validation_winner"],
            "feature_identity": cache.identity,
            "comparison_status": run["status"],
            "excluded_failed_classifiers": run["failed"],
        },
    )
    if persistent is not None:
        copy_artifact(root, persistent)
    return root


def predict_preprocessed_waveform(
    model_folder, waveform, sample_rate=TARGET_SR, encoder=None, extractor=None
):
    """Explicit fresh-audio inference; caller supplies already-standardized mono audio.

    Normal notebook cache reuse does NOT invoke this helper or load encoder weights.
    Optional encoder/extractor injection supports offline verification with a tiny model.
    """
    model_folder = Path(model_folder)
    verify_artifact(model_folder)
    metadata = read_json(model_folder / "metadata.json")
    contract = metadata["feature_contract"]
    waveform = np.asarray(waveform, dtype=np.float32)
    if (
        sample_rate != TARGET_SR
        or waveform.ndim != 1
        or not len(waveform)
        or not np.isfinite(waveform).all()
    ):
        raise ValueError(
            "Inference requires nonempty finite mono 16 kHz preprocessed audio."
        )
    if contract["max_seconds"] is not None:
        maximum = int(contract["max_seconds"] * TARGET_SR)
        start = max(0, (len(waveform) - maximum) // 2)
        waveform = waveform[start : start + maximum]
    if (encoder is None) != (extractor is None):
        raise ValueError("Supply both encoder and extractor, or neither.")
    owned_encoder = encoder is None
    if owned_encoder:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Select a GPU runtime for an explicit fresh XLS-R inference pass."
            )
        extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            contract["model_id"], revision=contract["model_revision"]
        )
        encoder = Wav2Vec2Model.from_pretrained(
            contract["model_id"], revision=contract["model_revision"]
        ).to("cuda")
    try:
        if encoder.config.hidden_size != contract["assets"]["hidden_size"]:
            raise ValueError(
                "Inference encoder hidden size differs from the saved feature contract."
            )
        frames = encode_waveforms(
            encoder,
            extractor,
            [waveform],
            contract["layer"],
            next(encoder.parameters()).device,
        )[0]
        fitted = load_fitted(model_folder, trusted=True)
        kwargs = (
            {"sequences": [frames]}
            if fitted.classifier_id in ("asp", "temporal_cnn")
            else {"pooled": pool_frames(frames)[None]}
        )
        score = float(predict_arrays(fitted, **kwargs)[0])
        return {
            "score": score,
            "score_type": metadata["score_type"],
            "threshold": metadata["threshold"],
            "label": LABEL_NAMES[int(score >= metadata["threshold"])],
        }
    finally:
        if owned_encoder:
            del encoder
            torch.cuda.empty_cache()


# %% Offline orchestration checks — synthetic features and local temporary artifacts.
def run_pipeline_checks():
    configure_runtime(replace(RunConfig(), head_device="cpu"))
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        model_config = Wav2Vec2Config(
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=32,
            conv_dim=(4,) * 7,
            conv_kernel=CONV_KERNELS,
            conv_stride=CONV_STRIDES,
            feat_extract_norm="layer",
            do_stable_layer_norm=True,
            num_conv_pos_embeddings=8,
            num_conv_pos_embedding_groups=2,
        )
        encoder = Wav2Vec2Model(model_config)
        extractor = Wav2Vec2FeatureExtractor(sampling_rate=TARGET_SR, do_normalize=True)
        config = replace(
            RunConfig(),
            layer=1,
            smoke_epochs=1,
            head_device="cpu",
            run_tag="synthetic",
            model_id="synthetic/tiny",
            model_revision="offline",
        )
        selection, feature_batches, waves_by_split = {}, [], {}
        for split_number, split in enumerate(SPLITS):
            selection[split], waves_by_split[split] = [], []
            count = 8 if split == "train" else 4
            for i in range(count):
                wave = np.sin(
                    np.arange(1200 + (i % 2) * 320)
                    * (0.03 + 0.002 * (i + split_number * 10))
                ).astype(np.float32)
                frames = encode_waveforms(encoder, extractor, [wave], 1, "cpu")[0]
                selection[split].append(
                    {
                        "source": "synthetic",
                        "group": "synthetic",
                        "split": split,
                        "row_index": i,
                        "clip_id": f"{split}-{i}",
                        "label": i % 2,
                        "language": "hi",
                        "speaker": None,
                        "frame_count": len(frames),
                        "audio_sha256": hashlib.sha256(wave.tobytes()).hexdigest(),
                    }
                )
                feature_batches.append((split, i, frames))
                waves_by_split[split].append(wave)
        signature = {
            "hidden_size": 16,
            "conv_kernel": list(CONV_KERNELS),
            "conv_stride": list(CONV_STRIDES),
            "config": model_config.to_dict(),
            "feature_extractor": extractor.to_dict(),
        }
        identity_payload = feature_identity_payload(selection, {}, config, signature)
        write_feature_cache(
            root / "features", selection, identity_payload, 16, iter(feature_batches)
        )
        cache = FeatureCache(root / "features")
        cache.raw_probe = {
            "frames": next(
                values
                for split, i, values in feature_batches
                if split == "validation" and i == 0
            )
        }
        ids = ("dummy", "logistic_regression", "mlp", "asp", "temporal_cnn")
        parameters = effective_classifier_configs(ids, {})
        payload = {
            "config": asdict(config),
            "feature_identity": cache.identity,
            "classifiers": parameters,
            "enabled_classifiers": list(ids),
        }
        expect_error(
            lambda: train_comparison(
                cache,
                replace(config, learning_rate=config.learning_rate * 2),
                parameters,
                payload,
                {},
                root / "mismatched-config",
            )
        )
        assert not (root / "mismatched-config").exists()
        comparison = train_comparison(
            cache, config, parameters, payload, {"kind": "synthetic"}, root / "work"
        )
        manifest = verify_artifact(comparison)
        assert manifest["status"] == "complete" and len(manifest["successful"]) == 5
        validation = read_json(comparison / "validation_metrics.json")
        assert [row["rank"] for row in validation] == list(range(1, 6))
        assert [row["eer"] for row in validation] == sorted(
            row["eer"] for row in validation
        )
        assert (
            final_test_evaluation(cache, comparison, root / "work", enabled=False)
            is None
        )
        assert not (root / "work" / "test_evaluations").exists()
        original_digest = file_sha256(comparison / "manifest.json")
        evaluation = final_test_evaluation(
            cache, comparison, root / "work", enabled=True
        )
        assert original_digest == file_sha256(comparison / "manifest.json")
        results = read_json(evaluation / "test_metrics.json")
        assert [row["classifier_id"] for row in results] == [
            row["classifier_id"] for row in validation
        ]
        assert [row["threshold"] for row in results] == [
            row["threshold"] for row in validation
        ]
        assert (
            verify_artifact(evaluation)["validation_winner"]
            == manifest["validation_winner"]
        )
        for row in validation:
            folder = comparison / "models" / row["classifier_id"]
            restored = load_fitted(folder, trusted=True)
            expected = predict_scores(restored, cache, "validation", config)[0]
            prediction = predict_preprocessed_waveform(
                folder,
                waves_by_split["validation"][0],
                encoder=encoder,
                extractor=extractor,
            )
            np.testing.assert_allclose(
                prediction["score"], expected, atol=2e-5, rtol=2e-5
            )
        assert (
            train_comparison(cache, config, parameters, payload, {}, root / "work")
            == comparison
        )
        partial_config = replace(config, mode="comparison", run_tag="synthetic-partial")
        partial_parameters = effective_classifier_configs(
            ("dummy", "knn"), {"knn": {"n_neighbors": 100}}
        )
        partial_payload = {
            "config": asdict(partial_config),
            "feature_identity": cache.identity,
            "classifiers": partial_parameters,
            "enabled_classifiers": ["dummy", "knn"],
        }
        print(
            "Checking expected failure handling with an intentionally invalid kNN fixture."
        )
        partial = train_comparison(
            cache,
            partial_config,
            partial_parameters,
            partial_payload,
            {},
            root / "work",
            root / "mirror",
            True,
        )
        assert verify_artifact(partial, allowed_status=("partial",))["successful"] == [
            "dummy"
        ]
        partial_digest = file_sha256(partial / "manifest.json")
        partial_test = final_test_evaluation(
            cache, partial, root / "work", enabled=True
        )
        assert [
            row["classifier_id"]
            for row in read_json(partial_test / "test_metrics.json")
        ] == ["dummy"]
        assert verify_artifact(partial_test)["excluded_failed_classifiers"] == ["knn"]
        assert file_sha256(partial / "manifest.json") == partial_digest
        assert (
            train_comparison(
                cache,
                partial_config,
                partial_parameters,
                partial_payload,
                {},
                root / "work",
                root / "mirror",
                True,
            )
            == partial
        )
        cache.close()
    print(
        "PASS: immutable comparisons, validation ranking, raw-audio reload, and opt-in final test"
    )


def run_dependency_checks():
    """Exercise setup without installing packages or changing the live runtime."""
    from types import SimpleNamespace
    from unittest.mock import patch as mock_patch

    def setup_commands(
        providers, *, imported=False, missing_dataset=False, error=False
    ):
        versions = {
            "torch": "2.8.0+cu126",
            "torchcodec": "0.7.0+cpu",
            "datasets": "4.4.1",
            "transformers": "4.57.3",
            "scikit-learn": "1.7.2",
            "soundfile": "0.13.1",
            "numpy": "2.3.5",
            "pandas": "2.3.3",
            **providers,
        }
        if missing_dataset:
            versions.pop("datasets")
        runtime = SimpleNamespace(
            executable=sys.executable,
            modules={"xgboost": object()} if imported else {},
        )
        with (
            mock_patch.dict(
                ensure_dependencies.__globals__,
                installed_version=versions.get,
                sys=runtime,
            ),
            mock_patch.object(subprocess, "run") as commands,
            mock_patch("builtins.print"),
        ):
            if error:
                expect_error(ensure_dependencies, RuntimeError)
            else:
                ensure_dependencies()
        return [call.args[0] for call in commands.call_args_list]

    for provider in ("xgboost", "xgboost-cpu"):
        for version in ("2.1.4", "3.0.5", "3.1.2", "3.1.3", "2.1.4+colab"):
            for imported in (False, True):
                assert setup_commands({provider: version}, imported=imported) == [
                    ["ffmpeg", "-version"]
                ]
    assert setup_commands({"xgboost": "3.0.5"}, missing_dataset=True) == [
        [sys.executable, "-m", "pip", "install", "--quiet", "datasets==4.4.1"],
        ["ffmpeg", "-version"],
    ]
    assert setup_commands({}) == [
        [sys.executable, "-m", "pip", "install", "--quiet", "xgboost-cpu==3.1.2"],
        ["ffmpeg", "-version"],
    ]
    for providers in (
        *(
            {"xgboost": version}
            for version in ("1.7.6", "2.1.3", "4.0.0", "3.2.0rc1", "invalid")
        ),
        {"xgboost": "3.0.5", "xgboost-cpu": "3.1.2"},
    ):
        assert setup_commands(providers, missing_dataset=True, error=True) == []
    print(
        "PASS: installed XGBoost reuse, missing-only fallback, and no provider replacement"
    )


def run_configuration_checks():
    RunConfig().validate()
    for name in (
        "layer",
        "extraction_batch_size",
        "mlp_batch_size",
        "sequence_batch_size",
        "cpu_threads",
        "max_epochs",
        "smoke_epochs",
        "patience",
    ):
        expect_error(lambda key=name: replace(RunConfig(), **{key: 0}).validate())
    for values in (
        {"seed": -1},
        {"num_workers": -1},
        {"max_seconds": 0},
        {"max_seconds": True},
        {"learning_rate": float("nan")},
        {"weight_decay": -1},
        {"run_tag": "../bad"},
    ):
        expect_error(lambda kwargs=values: replace(RunConfig(), **kwargs).validate())
    print("PASS: configuration bounds and explicit duration policy")


def run_offline_checks():
    """Public offline acceptance entry point; no source corpus, Hub, or Drive access."""
    configure_runtime(replace(RunConfig(), head_device="cpu"))
    run_dependency_checks()
    run_configuration_checks()
    run_metric_checks()
    run_data_checks()
    run_feature_checks()
    run_encoder_checks()
    run_classifier_checks()
    run_pipeline_checks()
    print(
        "PASS: all offline/synthetic contracts (NOT a Colab/pretrained-model benchmark)"
    )


# %% [markdown]
# ## Run offline contracts first
# These checks use tiny random models, embedded synthetic WAVs, and temporary
# local files. They do not access prepared corpora, Google Drive, the Hub, or
# pretrained model weights. Passing them does not establish detection quality.

# %% Offline acceptance gate.
if NOTEBOOK_RUN:
    run_offline_checks()


# %% Mount Drive and freeze source/group/subset configuration.
if NOTEBOOK_RUN:
    if MOUNT_GOOGLE_DRIVE:
        from google.colab import drive

        drive.mount("/content/drive")
    experiment = prepare_experiment(
        CONFIG,
        SOURCE_CONFIGS,
        GROUPS,
        GROUP_QUOTAS,
        ENABLED_CLASSIFIERS,
        CLASSIFIER_OVERRIDES,
        LOCAL_ROOT,
        DRIVE_ROOT,
        DRIVE_QUOTA_CONFIRMED,
    )


# %% Reuse compatible features BEFORE loading any XLS-R weights.
if NOTEBOOK_RUN:
    feature_cache = acquire_feature_cache(
        experiment["sources"],
        experiment["specs"],
        experiment["selection"],
        experiment["provenance"],
        CONFIG,
        experiment["assets"],
        LOCAL_ROOT,
        DRIVE_ROOT,
    )


# %% Fit each configured classifier once and rank ONLY by validation EER.
if NOTEBOOK_RUN:
    comparison_root = train_comparison(
        feature_cache,
        CONFIG,
        experiment["parameters"],
        experiment["payload"],
        {
            "audio_audit": experiment["audit"],
            "source_provenance": experiment["provenance"],
        },
        LOCAL_ROOT,
        DRIVE_ROOT,
        DRIVE_QUOTA_CONFIRMED,
    )
    validation_results = read_json(comparison_root / "validation_metrics.json")
    show_leaderboard(validation_results)
    baseline = next(
        (
            row
            for row in validation_results
            if row["classifier_id"] == "dummy" and row["eer"] is not None
        ),
        None,
    )
    learners = [
        row
        for row in validation_results
        if row["classifier_id"] != "dummy" and row["eer"] is not None
    ]
    if baseline and (
        not learners or min(row["eer"] for row in learners) >= baseline["eer"]
    ):
        print(
            "CAUTION: no learned classifier beats the dummy validation EER in this run."
        )
    print(f"Validation artifacts: {DRIVE_ROOT / 'runs' / comparison_root.name}")


# %% [markdown]
# ## Final held-out evaluation — opt in only after freezing settings
# Set RUN_TEST_EVALUATION=True and rerun ONLY the following cell once model
# settings and the candidate list are fixed. Thresholds and ranking come from
# saved validation artifacts, never this test table. EER is interpolated from
# continuous scores; Accuracy/Precision/Recall/F1 use the frozen validation
# operating threshold. Spoof is positive; undefined precision/recall/F1 use 0,
# with support/predicted-positive counts retained in detailed reports.
#
# Test artifacts are siblings under `test_evaluations/`, not new files inside
# a sealed comparison run. No completed training manifest is rewritten.

# %% Explicit, separately persisted final test evaluation.
if NOTEBOOK_RUN:
    test_root = final_test_evaluation(
        feature_cache,
        comparison_root,
        LOCAL_ROOT,
        DRIVE_ROOT,
        enabled=RUN_TEST_EVALUATION,
        quota_confirmed=DRIVE_QUOTA_CONFIRMED,
    )
    if test_root is not None:
        show_leaderboard(read_json(test_root / "test_metrics.json"), test=True)
        print(
            f"Final test artifacts: {DRIVE_ROOT / 'test_evaluations' / test_root.name}"
        )


# %% [markdown]
# ## Interpretation and reruns
# - Smoke metrics only establish component integration, not model quality.
# - Balanced groups change corpus prevalence; pooled EER weights clips, not
#   groups equally when custom quotas differ. Group/source/language reports
#   expose coverage; a one-class subgroup has undefined EER, not zero EER.
# - One fixed configuration and one seed can shortlist models, not prove the
#   best full-data architecture. Confirm candidates on more data and seeds.
# - Training on both source datasets is not an unseen-dataset evaluation.
#   Multilingual data alone does not constitute an unseen-language protocol.
# - Preserving source splits and detecting exact duplicates cannot establish
#   globally reconciled speaker identities or eliminate near duplicates.
#   Real-only/fake-only source pairing does not remove channel/source shortcuts.
# - Margins/logits are not calibrated probabilities. EER interpolation may not
#   correspond to an attainable deterministic threshold on a small dataset.
# - Classifier-only edits reuse compatible features. New classifier settings
#   or a deliberate repeat need a new run identity/tag; incomplete artifacts
#   must be moved aside explicitly or a new work/output root configured.
# - Changing annotations, clips, layer, preprocessing/crop policy, checkpoint,
#   or extraction versions changes feature identity. No output is overwritten.
# - Fresh inference can be requested explicitly with
#   `predict_preprocessed_waveform(model_folder, waveform, sample_rate=16000)`.
#   That explicit call loads the pinned encoder; normal cache reuse does not.
