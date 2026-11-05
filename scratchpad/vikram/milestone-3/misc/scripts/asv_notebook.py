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
# # ASVspoof 2019 LA — preprocessing and on-the-fly augmentation
#
# **Source:** `Bisher/ASVspoof_2019_LA`. **Labels:** 0 = bonafide, 1 = spoof.
# This notebook preserves the supplied splits, saves **unaugmented** mono 16 kHz
# PCM16 WAV audio in a portable Hugging Face dataset, and provides training-only
# waveform augmentation and model-ready batches for XLS-R-300M.
#
# No classifier is trained and no pretrained model weights are downloaded.
# XLS-R SpecAugment masks learned time/features **inside the future model**;
# it is not spectrogram frequency masking and does not guarantee two disjoint
# masks. Log-mel plots below are visualizations only.
#
# ## Convert and run
# Locally: `pip install jupytext`, then
# `jupytext --to ipynb asv_notebook.py`. Upload the resulting notebook to Colab.
# Run cells in order on a CPU runtime. Colab's existing PyTorch is preserved.
#
# The source download is about **7.5 GB**, even for the smoke subset. Initially,
# `RUN_FULL_PREPROCESS=False`: only smoke processing and demonstrations run.
# After they pass, check local capacity and your **actual Drive account quota**,
# set `RUN_FULL_PREPROCESS=True` and, for a new Drive publication,
# `DRIVE_QUOTA_CONFIRMED=True`, and rerun. An existing verified full artifact is
# discovered first and can be reused without downloading the raw source again.
# Full preprocessing can require roughly 50–60 GB of local space; the processed
# copy may exceed Drive's free 15 GB allowance. No existing artifact is replaced.

# %% Setup — install before importing, and never replace Colab's PyTorch.
from __future__ import annotations

import importlib.metadata
import json
import subprocess
import sys

NOTEBOOK_RUN = __name__ == "__main__"


def installed_version(distribution):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def ensure_dependencies():
    """Bound critical APIs; install a CPU audio decoder matching existing torch."""
    torch_version = installed_version("torch")
    if torch_version is None:
        raise RuntimeError("Use Colab's PyTorch runtime, or install PyTorch first.")
    pair = tuple(int(part) for part in torch_version.split("+")[0].split(".")[:2])
    codec_versions = {
        (2, 8): "0.7.0",
        (2, 9): "0.9.0",
        (2, 10): "0.10.0",
        (2, 11): "0.16.0",
    }
    if pair not in codec_versions:
        raise RuntimeError(
            f"PyTorch {torch_version} needs a verified TorchCodec pairing. Check "
            "https://github.com/meta-pytorch/torchcodec#installing-torchcodec; "
            "do not blindly upgrade Colab's PyTorch/CUDA stack."
        )
    pins = {
        "datasets": "4.4.1",
        "transformers": "4.57.3",
        "librosa": "0.11.0",
        "soundfile": "0.13.1",
    }
    requirements = []
    for package, version in pins.items():
        if installed_version(package) != version:
            if package in sys.modules:
                raise RuntimeError(
                    f"Restart the runtime before replacing imported {package}."
                )
            requirements.append(f"{package}=={version}")
    for package in ("numpy", "scipy", "soxr", "matplotlib", "ipython"):
        if installed_version(package) is None:
            requirements.append(package)
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

import hashlib
import io
import itertools
import math
import random
import shutil
import tempfile
import warnings
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from datasets import (
    Audio,
    ClassLabel,
    Dataset,
    DatasetDict,
    Features,
    Value,
    load_dataset,
    load_from_disk,
)
from IPython.display import Audio as AudioPlayer
from IPython.display import display
from scipy.signal import fftconvolve, resample_poly
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Model


def package_versions():
    names = (
        "datasets",
        "transformers",
        "torch",
        "torchcodec",
        "numpy",
        "scipy",
        "librosa",
        "soundfile",
        "soxr",
    )
    return {
        "python": sys.version.split()[0],
        **{name: installed_version(name) for name in names},
    }


# %% Configuration — edit these controls before a full run.
def in_colab():
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


DATASET_ID = "Bisher/ASVspoof_2019_LA"
DATASET_REVISION = "aea92dd83a9c56e070c0b1e9f02e7c0d96216a4c"
MODEL_ID = "facebook/wav2vec2-xls-r-300m"
MODEL_REVISION = "1a640f32ac3e39899438a2931f9924c02f080a54"
EXPECTED_COUNTS = {"train": 25_380, "validation": 24_844, "test": 71_237}
LABEL_NAMES = ["bonafide", "spoof"]
TARGET_SR = 16_000
PCM16_STEP = 1.0 / 32768.0
CONV_KERNELS = (10, 3, 3, 3, 3, 2, 2)
CONV_STRIDES = (5, 2, 2, 2, 2, 2, 2)
SEED = 42
RUN_FULL_PREPROCESS = False
DRIVE_QUOTA_CONFIRMED = False
MOUNT_GOOGLE_DRIVE = in_colab()
RUN_TAG = "v1"  # Choose a NEW tag after an interrupted/incompatible artifact.
LOCAL_ROOT = Path("/content/asv_work" if MOUNT_GOOGLE_DRIVE else "./asv_work") / RUN_TAG
DRIVE_ROOT = Path("/content/drive/MyDrive/ds-ai-lab/asvspoof2019_la") / RUN_TAG
NUM_PROC = 1
WRITER_BATCH_SIZE = 32
SHOW_PREVIEWS = True


@dataclass(frozen=True)
class PreprocessConfig:
    sample_rate: int = TARGET_SR
    rms_dbfs: float = -20.0
    peak_ceiling: float = 0.99
    trim_silence: bool = False
    silence_db: float = 40.0
    frame_seconds: float = 0.02
    minimum_trim_seconds: float = 0.25
    version: str = "asv-standardized-v1"

    def __post_init__(self):
        if self.sample_rate != TARGET_SR:
            raise ValueError("XLS-R requires the configured target of 16 kHz.")
        values = (
            self.rms_dbfs,
            self.peak_ceiling,
            self.silence_db,
            self.frame_seconds,
            self.minimum_trim_seconds,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Preprocessing parameters must be finite.")
        if not PCM16_STEP < self.peak_ceiling < 1 or self.rms_dbfs > 0:
            raise ValueError("Invalid RMS target/peak ceiling.")
        if (
            self.silence_db <= 0
            or self.frame_seconds <= 0
            or self.minimum_trim_seconds < 0
        ):
            raise ValueError("Invalid silence-trimming parameters.")


PREPROCESS = PreprocessConfig()


# %% [markdown]
# ## 1. Deterministic audio standardization
# Audio is decoded with the current `datasets`/TorchCodec API. Mono conversion
# and 16 kHz resampling are lazy until the mapper encodes new embedded WAV bytes.
# Silence trimming is **off**; when enabled it affects only edges, retains partial
# frames, and keeps the original clip if trimming would leave less than 0.25 s.
# No recording is dropped for being shorter than one second.
#
# RMS normalization targets -20 dBFS, with one PCM16 step of peak headroom.
# High-crest-factor clips may remain quieter; silence gets a nullable RMS value.
# WAV avoids lossy codec compression, not the effects of gain/resampling or
# quantization. XLS-R later applies its own per-clip zero-mean/unit-variance step.


# %% Standardization helpers — no dependency on notebook execution order/state.
def valid_waveform(waveform):
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim != 1 or waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError("Expected a nonempty finite mono waveform.")
    return np.ascontiguousarray(waveform)


def mono_float(waveform):
    """Accept mono or channel-first [channels, samples] TorchCodec arrays."""
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 2:
        if not waveform.shape[0] or not np.isfinite(waveform).all():
            raise ValueError("Empty/non-finite multichannel audio.")
        waveform = waveform.mean(axis=0, dtype=np.float64).astype(np.float32)
    return valid_waveform(waveform)


def decode_hf_audio(audio):
    if not hasattr(audio, "get_all_samples"):
        raise TypeError(
            "Expected a datasets/TorchCodec AudioDecoder; check Audio decoding and installed versions."
        )
    samples = audio.get_all_samples()
    if samples.sample_rate != TARGET_SR:
        raise ValueError(
            f"Expected lazy resampling to {TARGET_SR}, got {samples.sample_rate}."
        )
    return mono_float(samples.data.detach().cpu().numpy())


def wav_bytes(waveform, sample_rate=TARGET_SR):
    buffer = io.BytesIO()
    sf.write(buffer, waveform, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def read_pcm16(payload):
    """Independent storage audit decoder, not a fallback for TorchCodec."""
    with sf.SoundFile(io.BytesIO(payload)) as stream:
        if (stream.format, stream.subtype, stream.channels, stream.samplerate) != (
            "WAV",
            "PCM_16",
            1,
            TARGET_SR,
        ):
            raise ValueError("Stored audio is not mono 16 kHz PCM16 WAV.")
        return valid_waveform(stream.read(dtype="float32"))


def rms_level(waveform):
    rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
    return rms, (20.0 * math.log10(rms) if rms else None)


def trim_edge_silence(waveform, config):
    waveform = valid_waveform(waveform)
    if not config.trim_silence or not np.any(waveform):
        return waveform
    frame = max(1, round(config.frame_seconds * config.sample_rate))
    starts = np.arange(0, len(waveform), frame)
    energy = np.add.reduceat(np.square(waveform, dtype=np.float64), starts)
    energy /= np.minimum(frame, len(waveform) - starts)
    voiced = np.flatnonzero(energy > energy.max() * 10.0 ** (-config.silence_db / 10.0))
    if voiced.size == 0:
        return waveform
    trimmed = waveform[
        starts[voiced[0]] : min(len(waveform), starts[voiced[-1]] + frame)
    ]
    if len(trimmed) < round(config.minimum_trim_seconds * config.sample_rate):
        warnings.warn(
            "Silence trimming would leave too little audio; retaining the original clip.",
            stacklevel=2,
        )
        return waveform
    return trimmed


def normalize_rms(waveform, config):
    rms, _ = rms_level(waveform)
    if rms == 0.0:
        return waveform.copy(), False
    desired_gain = 10.0 ** (config.rms_dbfs / 20.0) / rms
    safe_gain = (config.peak_ceiling - PCM16_STEP) / float(np.abs(waveform).max())
    return (waveform * min(desired_gain, safe_gain)).astype(
        np.float32
    ), safe_gain < desired_gain


def standardize_waveform(waveform, sample_rate, config=PREPROCESS):
    waveform = mono_float(waveform)
    if not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        raise ValueError("Sample rate must be a positive integer.")
    if sample_rate != config.sample_rate:
        factor = math.gcd(int(sample_rate), config.sample_rate)
        waveform = resample_poly(
            waveform, config.sample_rate // factor, sample_rate // factor
        ).astype(np.float32)
    waveform = trim_edge_silence(waveform, config)
    normalized, limited = normalize_rms(waveform, config)
    payload = wav_bytes(normalized, config.sample_rate)
    stored = read_pcm16(payload)
    _, dbfs = rms_level(stored)
    return {
        "audio": {"bytes": payload, "path": None},
        "duration_s": len(stored) / config.sample_rate,
        "num_samples": len(stored),
        "rms_dbfs": dbfs,
        "peak": float(np.abs(stored).max()),
        "is_silent": dbfs is None,
        "peak_limited": bool(limited),
        "preprocess_version": config.version,
    }


def standardize_row(row, config=PREPROCESS):
    try:
        return standardize_waveform(decode_hf_audio(row["audio"]), TARGET_SR, config)
    except Exception as exc:
        raise ValueError(
            f"Preprocessing {row.get('audio_file_name', '<unknown>')}: {exc}"
        ) from exc


def decoder_smoke_check():
    # Exercise actual lazy stereo -> mono AND 8 kHz -> 16 kHz conversion.
    time_axis = np.arange(8000) / 8000
    signal = 0.05 * np.sin(2 * np.pi * 220 * time_axis)
    buffer = io.BytesIO()
    sf.write(
        buffer,
        np.column_stack((signal, signal * 0.5)),
        8000,
        format="WAV",
        subtype="PCM_16",
    )
    fixture = Dataset.from_dict(
        {"audio": [{"bytes": buffer.getvalue(), "path": None}]},
        features=Features({"audio": Audio(sampling_rate=TARGET_SR, num_channels=1)}),
    )
    try:
        decoded = decode_hf_audio(fixture[0]["audio"])
    except Exception as exc:
        raise RuntimeError(
            "Actual TorchCodec decoding failed. Check the PyTorch/TorchCodec pairing, "
            "FFmpeg shared libraries, and restart after package changes. No alternate decoder was substituted."
        ) from exc
    assert decoded.shape == (TARGET_SR,) and np.any(decoded)
    print("PASS: datasets/TorchCodec stereo resampling smoke test")


# %% Synthetic standardization tests run before any source-data download.
def run_standardization_checks():
    """Contract tests: format, loudness, peak safety, trimming, and invalid input."""
    # These fixtures test both policies, independently of the operator's settings.
    config = PreprocessConfig(
        rms_dbfs=-20,
        peak_ceiling=0.99,
        trim_silence=False,
        silence_db=40,
        frame_seconds=0.02,
        minimum_trim_seconds=0.25,
    )
    time_axis = np.arange(TARGET_SR, dtype=np.float64) / TARGET_SR
    sine = (0.03 * np.sin(2 * np.pi * 440 * time_axis)).astype(np.float32)
    result = standardize_waveform(sine, TARGET_SR, config)
    audio = read_pcm16(result["audio"]["bytes"])
    assert audio.dtype == np.float32 and audio.ndim == 1 and len(audio) == TARGET_SR
    assert abs(result["rms_dbfs"] + 20.0) < 0.1 and result["peak"] <= 0.99
    assert result["audio"]["path"] is None and not result["is_silent"]
    assert result == standardize_waveform(sine, TARGET_SR, config)
    normalized, _ = normalize_rms(sine, config)
    assert np.max(np.abs(audio - normalized)) <= PCM16_STEP + 1e-7
    stereo = np.stack((sine, sine * 0.5))
    np.testing.assert_array_equal(
        read_pcm16(standardize_waveform(stereo, TARGET_SR, config)["audio"]["bytes"]),
        read_pcm16(
            standardize_waveform(stereo.mean(axis=0), TARGET_SR, config)["audio"][
                "bytes"
            ]
        ),
    )
    resampled = standardize_waveform(sine[::2], 8_000, config)
    assert resampled["num_samples"] == TARGET_SR
    silent = standardize_waveform(np.zeros(1000, np.float32), TARGET_SR, config)
    assert (
        silent["is_silent"]
        and silent["rms_dbfs"] is None
        and silent["num_samples"] == 1000
    )
    impulse = np.zeros(TARGET_SR, np.float32)
    impulse[0] = -1
    limited = standardize_waveform(impulse, TARGET_SR, config)
    assert limited["peak_limited"] and limited["peak"] <= config.peak_ceiling
    assert np.max(np.abs(read_pcm16(limited["audio"]["bytes"]))) <= 0.99
    trimmed_config = replace(config, trim_silence=True, minimum_trim_seconds=0)
    partial = np.concatenate((np.zeros(640, np.float32), np.ones(17, np.float32)))
    assert len(trim_edge_silence(partial, trimmed_config)) == 17
    np.testing.assert_array_equal(
        trim_edge_silence(np.zeros(657, np.float32), trimmed_config), np.zeros(657)
    )
    untrimmed = np.pad(sine, (640, 640))
    assert standardize_waveform(untrimmed, TARGET_SR, config)["num_samples"] == len(
        untrimmed
    )
    changed_policy = replace(config, trim_silence=True, rms_dbfs=-18)
    changed = standardize_waveform(untrimmed, TARGET_SR, changed_policy)
    assert changed["num_samples"] == TARGET_SR and abs(changed["rms_dbfs"] + 18) < 0.1
    for invalid in (np.array([], np.float32), np.array([np.nan], np.float32)):
        try:
            standardize_waveform(invalid, TARGET_SR, config)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid audio must not be silently retained.")
    print("PASS: synthetic standardization checks")


if NOTEBOOK_RUN:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    print(json.dumps(package_versions(), indent=2))
    decoder_smoke_check()
    run_standardization_checks()


# %% Small fixtures — these tests never contact the Hub.
def synthetic_dataset():
    """Exercise real Audio/ClassLabel/Arrow serialization with in-memory WAVs."""
    features = Features(
        {
            "speaker_id": Value("string"),
            "audio_file_name": Value("string"),
            "audio": Audio(sampling_rate=TARGET_SR, num_channels=1),
            "system_id": Value("string"),
            "key": ClassLabel(names=LABEL_NAMES),
        }
    )
    splits = {}
    for split, count in {"train": 16, "validation": 4, "test": 4}.items():
        rows = []
        for index in range(count):
            native_sr = 8000
            t = np.arange(int(native_sr * (0.6 + 0.1 * (index % 3)))) / native_sr
            y = 0.04 * np.sin(2 * np.pi * (220 + index * 7) * t)
            rows.append(
                {
                    "speaker_id": f"{split}_S{index % 3}",
                    "audio_file_name": f"{split}_{index}",
                    "audio": {"bytes": wav_bytes(y, native_sr), "path": None},
                    "system_id": "-" if index % 2 == 0 else "A01",
                    "key": index % 2,
                }
            )
        splits[split] = Dataset.from_list(rows, features=features)
    return DatasetDict(splits)


# %% Metadata, deterministic mapping, and portable artifact verification.
def json_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def preprocess_fingerprint(config):
    # Artifact compatibility is semantic; provenance versions remain in the manifest.
    # Mapping caches below additionally include package versions to prevent stale reuse.
    return json_digest(
        {
            "dataset": DATASET_ID,
            "revision": DATASET_REVISION,
            "preprocess": asdict(config),
        }
    )


def metadata_digest(split):
    digest = hashlib.sha256()
    for row in split.select_columns(
        ["speaker_id", "audio_file_name", "system_id", "key"]
    ):
        digest.update(json.dumps(row, sort_keys=True).encode() + b"\n")
    return digest.hexdigest()


def audit_metadata(data, expected_counts):
    if set(data) != set(expected_counts):
        raise ValueError(
            "Unexpected split names; do not merge or re-split the dataset."
        )
    summary = {"counts": {}, "class_counts": {}, "metadata_sha256": {}, "overlaps": {}}
    identifiers, speakers = {}, {}
    for name, split in data.items():
        if len(split) != expected_counts[name] or not len(split):
            raise ValueError(f"{name}: unexpected/empty row count {len(split)}.")
        required = {"speaker_id", "audio_file_name", "audio", "system_id", "key"}
        if not required.issubset(split.column_names):
            raise ValueError(f"{name}: missing required columns.")
        label_feature = split.features["key"]
        if (
            not isinstance(label_feature, ClassLabel)
            or label_feature.names != LABEL_NAMES
        ):
            raise ValueError(f"{name}: expected ClassLabel 0=bonafide, 1=spoof.")
        labels = Counter(split["key"])
        if set(labels) - {0, 1}:
            raise ValueError(f"{name}: invalid label values.")
        identifiers[name] = set(split["audio_file_name"])
        speakers[name] = set(split["speaker_id"])
        if len(identifiers[name]) != len(split):
            raise ValueError(
                f"{name}: duplicated utterance IDs; investigate without silently dropping rows."
            )
        summary["counts"][name] = len(split)
        summary["class_counts"][name] = {
            label: labels[index] for index, label in enumerate(LABEL_NAMES)
        }
        summary["metadata_sha256"][name] = metadata_digest(split)
    for a, b in itertools.combinations(data, 2):
        summary["overlaps"][f"{a}/{b}"] = {
            "utterance_ids": len(identifiers[a] & identifiers[b]),
            "speakers": len(speakers[a] & speakers[b]),
        }
    return summary


def audit_standardized(data, config, expected_counts):
    summary = audit_metadata(data, expected_counts)
    columns = [
        "num_samples",
        "duration_s",
        "peak",
        "rms_dbfs",
        "is_silent",
        "preprocess_version",
    ]
    for name, split in data.items():
        for row in split.select_columns(columns):
            if row["num_samples"] <= 0 or not np.isclose(
                row["duration_s"], row["num_samples"] / TARGET_SR, rtol=1e-6
            ):
                raise ValueError(f"{name}: invalid sample count/duration metadata.")
            if (
                not 0 <= row["peak"] <= config.peak_ceiling
                or row["preprocess_version"] != config.version
            ):
                raise ValueError(f"{name}: invalid peak/version metadata.")
            if row["is_silent"] != (row["rms_dbfs"] is None):
                raise ValueError(f"{name}: inconsistent silence metadata.")
        encoded = split.cast_column(
            "audio", Audio(sampling_rate=TARGET_SR, num_channels=1, decode=False)
        )
        indices = np.unique(
            np.concatenate(
                (
                    [0],
                    np.random.default_rng(SEED).choice(
                        len(split), min(6, len(split)), replace=False
                    ),
                )
            )
        )
        for index in indices:
            row = encoded[int(index)]
            payload = row["audio"]
            if not payload["bytes"] or payload["path"] is not None:
                raise ValueError(
                    f"{name}: missing embedded bytes or external audio path."
                )
            waveform = read_pcm16(payload["bytes"])
            _, dbfs = rms_level(waveform)
            if len(waveform) != row["num_samples"] or not np.isclose(
                float(np.abs(waveform).max()), row["peak"], atol=1e-7
            ):
                raise ValueError(f"{name}: stored audio differs from its metadata.")
            if (dbfs is None) != row["is_silent"] or (
                dbfs is not None and not np.isclose(dbfs, row["rms_dbfs"], atol=1e-4)
            ):
                raise ValueError(f"{name}: stored RMS differs from metadata.")
    return summary


def map_standardized(source, config, cache_dir, num_proc=1):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, split in source.items():
        features = Features(dict(split.features))
        features.update(
            {
                "audio": Audio(sampling_rate=TARGET_SR, num_channels=1),
                "duration_s": Value("float32"),
                "num_samples": Value("int32"),
                "rms_dbfs": Value("float32"),
                "peak": Value("float32"),
                "is_silent": Value("bool"),
                "peak_limited": Value("bool"),
                "preprocess_version": Value("string"),
            }
        )
        cache_key = json_digest(
            {
                "config": preprocess_fingerprint(config),
                "metadata": metadata_digest(split),
                "versions": package_versions(),
            }
        )
        result[name] = split.map(
            standardize_row,
            fn_kwargs={"config": config},
            batched=False,
            num_proc=None if num_proc == 1 else num_proc,
            writer_batch_size=WRITER_BATCH_SIZE,
            features=features,
            cache_file_name=str(cache_dir / f"{name}-{cache_key}.arrow"),
            desc=f"Standardizing {name} (no augmentation)",
        )
        if metadata_digest(result[name]) != metadata_digest(split):
            raise ValueError(
                f"{name}: preprocessing changed labels, metadata, or row order."
            )
    processed = DatasetDict(result)
    audit_standardized(
        processed, config, {name: len(split) for name, split in source.items()}
    )
    return processed


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_files(root):
    files = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_symlink():
            raise ValueError("Portable artifacts must not contain symlinks.")
        if path.is_file() and path != Path(root) / "manifest.json":
            files[path.relative_to(root).as_posix()] = {
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
    return files


def write_completion_marker(root, manifest):
    marker = Path(root) / "manifest.json"
    if marker.exists():
        raise FileExistsError(f"Will not replace {marker}.")
    pending = Path(root) / ".manifest.pending"
    with pending.open("x") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
    pending.replace(marker)


def verify_files(root, manifest):
    if not manifest.get("files") or artifact_files(root) != manifest["files"]:
        raise ValueError(
            f"{root}: missing/extra/corrupted artifact files; choose a new run directory."
        )


def load_artifact(path, config, expected_counts, kind="full"):
    path = Path(path)
    marker = path / "manifest.json"
    if not marker.is_file():
        raise ValueError(
            f"{path}: no completion marker; do not use a partial artifact."
        )
    manifest = json.loads(marker.read_text())
    if (
        manifest.get("status") != "complete"
        or manifest.get("kind") != kind
        or manifest.get("fingerprint") != preprocess_fingerprint(config)
    ):
        raise ValueError(f"{path}: incompatible artifact/configuration.")
    if manifest.get("metadata", {}).get("counts") != expected_counts:
        raise ValueError(
            f"{path}: wrong split sizes (a smoke artifact is not a full dataset)."
        )
    verify_files(path, manifest)
    data = load_from_disk(str(path))
    if audit_standardized(data, config, expected_counts) != manifest["metadata"]:
        raise ValueError(
            f"{path}: metadata audit does not match the completion marker."
        )
    return data


def save_artifact(data, path, config, kind="full"):
    path = Path(path)
    counts = (
        EXPECTED_COUNTS
        if kind == "full"
        else {name: len(split) for name, split in data.items()}
    )
    metadata = audit_standardized(data, config, counts)
    if path.exists():
        existing = load_artifact(path, config, counts, kind)
        if audit_metadata(existing, counts) != metadata:
            raise ValueError(f"{path}: existing artifact has different input metadata.")
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    data.save_to_disk(str(path), max_shard_size="500MB")
    reloaded = load_from_disk(str(path))
    if audit_standardized(reloaded, config, counts) != metadata:
        raise ValueError("Saved artifact failed reload validation.")
    manifest = {
        "status": "complete",
        "kind": kind,
        "fingerprint": preprocess_fingerprint(config),
        "source": {"id": DATASET_ID, "revision": DATASET_REVISION},
        "preprocess": asdict(config),
        "package_versions": package_versions(),
        "metadata": metadata,
        "files": artifact_files(path),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_completion_marker(path, manifest)
    return reloaded


def copy_artifact(
    source, destination, config, expected_counts, kind="full", check_local_capacity=True
):
    source, destination = Path(source), Path(destination)
    load_artifact(source, config, expected_counts, kind)
    manifest = json.loads((source / "manifest.json").read_text())
    if destination.exists():
        existing = load_artifact(destination, config, expected_counts, kind)
        if (
            json.loads((destination / "manifest.json").read_text())["files"]
            != manifest["files"]
        ):
            raise ValueError(
                "Destination is a different artifact; choose a new directory."
            )
        return existing
    if check_local_capacity:
        required_bytes = sum(item["size"] for item in manifest["files"].values())
        check_local_space(
            destination.parent,
            required_bytes * 1.1,
            "verified artifact restoration/copy",
        )
    destination.mkdir(parents=True, exist_ok=False)
    for relative in manifest["files"]:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)
    # IMPORTANT: verify destination bytes and reload BEFORE writing the marker.
    verify_files(destination, manifest)
    reloaded = load_from_disk(str(destination))
    if audit_standardized(reloaded, config, expected_counts) != manifest["metadata"]:
        raise ValueError(
            "Destination failed reload audit; no completion marker published."
        )
    write_completion_marker(destination, manifest)
    return reloaded


def restore_available_artifact(
    local_path, drive_path, config, expected_counts, use_drive, kind="full"
):
    """Resolve a verified artifact before considering any raw Hub download."""
    if Path(local_path).exists():
        return load_artifact(local_path, config, expected_counts, kind)
    if use_drive and Path(drive_path).exists():
        return copy_artifact(drive_path, local_path, config, expected_counts, kind)
    return None


# %% Artifact contract tests (small local data, never a full corpus).
def run_artifact_checks(base_dir=None):
    with tempfile.TemporaryDirectory(prefix="asv-check-", dir=base_dir) as temporary:
        root = Path(temporary)
        raw_fixture = synthetic_dataset()
        counts = {name: len(split) for name, split in raw_fixture.items()}
        processed = map_standardized(raw_fixture, PREPROCESS, root / "cache")
        for name in counts:
            assert list(processed[name]["key"]) == list(raw_fixture[name]["key"])
            assert list(processed[name]["audio_file_name"]) == list(
                raw_fixture[name]["audio_file_name"]
            )
        saved = save_artifact(processed, root / "local", PREPROCESS, kind="smoke")
        copied = copy_artifact(
            root / "local", root / "drive", PREPROCESS, counts, kind="smoke"
        )
        assert saved["train"][0]["num_samples"] == 9600
        np.testing.assert_array_equal(
            decode_hf_audio(saved["train"][0]["audio"]),
            decode_hf_audio(copied["train"][0]["audio"]),
        )
        manifest = json.loads((root / "drive" / "manifest.json").read_text())
        victim = root / "drive" / next(iter(manifest["files"]))
        with victim.open("ab") as handle:
            handle.write(b"corrupt")
        try:
            load_artifact(root / "drive", PREPROCESS, counts, kind="smoke")
        except ValueError:
            pass
        else:
            raise AssertionError("A corrupted artifact must not be reused.")
    print(
        "PASS: real Audio mapping, embedded WAV save/reload, verified copy, and corruption rejection"
    )


# %% [markdown]
# ## 2. Waveform augmentation — training only
# Five independent probabilities default to 0.5, identical for both labels.
# The expected chain has 2.5 effects; only 3.125% of chains skip all five.
# Order: pitch-preserving speed -> pitch -> zero-filled time shift -> reverb ->
# calibrated Gaussian noise -> linear peak protection. No RMS renormalization
# or PCM16 re-encoding is performed on these temporary float32 waveforms.
#
# Speed/pitch phase-vocoder artifacts can obscure spoof cues: validate strengths
# on validation only. Synthetic white noise is not environmental noise. Reverb
# is an energy-controlled synthetic proxy, not a measured/physical room model.
# Shifts discard up to 0.15 s at the opposite edge; there is no circular wrap.


# %% Configurable waveform transforms (they never receive a label).
@dataclass(frozen=True)
class AugmentationConfig:
    p_noise: float = 0.5
    p_shift: float = 0.5
    p_speed: float = 0.5
    p_pitch: float = 0.5
    p_reverb: float = 0.5
    snr_db: float = 15.0
    shift_seconds: float = 0.15
    speed_range: tuple[float, float] = (0.9, 1.1)
    pitch_semitones: float = 2.0
    rt60_seconds: float = 0.3
    drr_db: float = 6.0
    peak_ceiling: float = 0.99

    def __post_init__(self):
        probabilities = (
            self.p_noise,
            self.p_shift,
            self.p_speed,
            self.p_pitch,
            self.p_reverb,
        )
        values = (
            *probabilities,
            self.snr_db,
            self.shift_seconds,
            *self.speed_range,
            self.pitch_semitones,
            self.rt60_seconds,
            self.drr_db,
            self.peak_ceiling,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Augmentation parameters must be finite.")
        if any(not 0 <= value <= 1 for value in probabilities):
            raise ValueError("Augmentation probabilities must lie in [0, 1].")
        if not 0 < self.speed_range[0] <= self.speed_range[1]:
            raise ValueError("Speed range must be positive and ordered.")
        if self.shift_seconds < 0 or self.pitch_semitones < 0 or self.rt60_seconds <= 0:
            raise ValueError("Invalid shift, pitch, or RT60 range.")
        if not 0 < self.peak_ceiling < 1:
            raise ValueError("Peak ceiling must lie in (0, 1).")


AUGMENT = AugmentationConfig()


def change_speed(waveform, rate):
    waveform = valid_waveform(waveform)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("Speed rate must be finite and positive.")
    return (
        waveform.copy()
        if rate == 1
        else valid_waveform(librosa.effects.time_stretch(waveform, rate=rate))
    )


def shift_pitch(waveform, semitones):
    waveform = valid_waveform(waveform)
    if not math.isfinite(semitones):
        raise ValueError("Pitch shift must be finite.")
    if semitones == 0:
        return waveform.copy()
    return valid_waveform(
        librosa.effects.pitch_shift(
            waveform,
            sr=TARGET_SR,
            n_steps=semitones,
            bins_per_octave=12,
            res_type="soxr_hq",
        )
    )


def time_shift(waveform, seconds):
    waveform = valid_waveform(waveform)
    if not math.isfinite(seconds):
        raise ValueError("Time shift must be finite.")
    amount = int(
        np.clip(round(seconds * TARGET_SR), -(len(waveform) - 1), len(waveform) - 1)
    )
    shifted = np.zeros_like(waveform)
    if amount > 0:
        shifted[amount:] = waveform[:-amount]
    elif amount < 0:
        shifted[:amount] = waveform[-amount:]
    else:
        shifted[:] = waveform
    return shifted


def synthetic_rir(rng, rt60_seconds=0.3, drr_db=6.0):
    if (
        not math.isfinite(rt60_seconds)
        or rt60_seconds <= 0
        or not math.isfinite(drr_db)
    ):
        raise ValueError("Invalid synthetic room parameters.")
    n = max(2, math.ceil(2.0 * rt60_seconds * TARGET_SR))
    time_axis = np.arange(n) / TARGET_SR
    tail = rng.standard_normal(n) * 10.0 ** (-3.0 * time_axis / rt60_seconds)
    tail[0] = 0.0
    norm = np.linalg.norm(tail)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Invalid reverberation tail.")
    tail *= 10.0 ** (-drr_db / 20.0) / norm
    tail[0] = 1.0
    return tail


def estimate_rt60(tail):
    energy = np.cumsum(np.square(tail, dtype=np.float64)[::-1])[::-1]
    if len(energy) < 2 or energy[0] <= 0:
        raise ValueError("Cannot measure decay of an empty/zero tail.")
    db = 10 * np.log10(np.maximum(energy / energy[0], np.finfo(float).tiny))
    window = (db <= -5) & (db >= -25)
    if window.sum() < 2:
        raise ValueError("Insufficient decay region for a T20 fit.")
    slope, _ = np.polyfit(np.arange(len(tail))[window] / TARGET_SR, db[window], 1)
    if slope >= 0:
        raise ValueError("Non-decaying room response.")
    return float(-60.0 / slope)


def reverberate(waveform, rir):
    waveform = valid_waveform(waveform)
    # Causal alignment: mode='same' would center the convolution incorrectly.
    return valid_waveform(fftconvolve(waveform, rir, mode="full")[: len(waveform)])


def add_gaussian_noise(waveform, rng, snr_db):
    waveform = valid_waveform(waveform)
    if not math.isfinite(snr_db):
        raise ValueError("SNR must be finite.")
    signal_rms, _ = rms_level(waveform)
    if signal_rms == 0:
        return waveform.copy()
    noise = rng.standard_normal(len(waveform))
    noise_rms = float(np.sqrt(np.mean(noise**2)))
    if noise_rms == 0:
        raise ValueError("Cannot calibrate zero-power noise.")
    noise *= signal_rms / (10.0 ** (snr_db / 20.0) * noise_rms)
    return valid_waveform(waveform + noise)


def augment_waveform(waveform, rng, config=AUGMENT):
    waveform = valid_waveform(waveform).copy()
    effect = "speed"
    try:
        if rng.random() < config.p_speed:
            waveform = change_speed(waveform, rng.uniform(*config.speed_range))
        effect = "pitch"
        if rng.random() < config.p_pitch:
            waveform = shift_pitch(
                waveform, rng.uniform(-config.pitch_semitones, config.pitch_semitones)
            )
        effect = "time shift"
        if rng.random() < config.p_shift:
            waveform = time_shift(
                waveform, rng.uniform(-config.shift_seconds, config.shift_seconds)
            )
        effect = "reverberation"
        if rng.random() < config.p_reverb:
            waveform = reverberate(
                waveform, synthetic_rir(rng, config.rt60_seconds, config.drr_db)
            )
        effect = "noise"
        if rng.random() < config.p_noise:
            waveform = add_gaussian_noise(waveform, rng, config.snr_db)
        peak = float(np.abs(waveform).max())
        if peak > config.peak_ceiling:
            waveform *= (
                np.nextafter(np.float32(config.peak_ceiling), np.float32(0)) / peak
            )
        return valid_waveform(waveform)
    except Exception as exc:
        raise ValueError(f"{effect} augmentation failed: {exc}") from exc


# %% [markdown]
# ## 3. Future XLS-R classifier input contract
# These are variable-length **raw waveforms**, not spectrograms. Batches pad only
# to their longest recording, with valid-sample attention masks. The collator
# forwards only `input_values`, `attention_mask`, and int64 `labels` to a future
# two-logit `Wav2Vec2ForSequenceClassification`/`AutoModelForAudioClassification`.
# Metadata is for audits only; never feed speaker/system/filename to the model.
#
# Before each future training iterator, call `train_dataset.set_epoch(epoch)`.
# Within an epoch a row has a reproducible variant; epochs give new draws.
# Evaluation has no waveform RNG, and the future model also needs `model.eval()`
# to disable automatic SpecAugment. Explicit time masks must not be passed there.
# `default_rng` does not seed the global NumPy RNG used by Transformers masking.
#
# A 16 kHz recording needs 400 samples for one frontend frame and 3,280 samples
# for a length-10 time mask. Short clips are retained in storage but rejected by
# the training wrapper with their ID, never silently dropped or padded as speech.


# %% Length checks, PyTorch dataset wrapper, collator, and loader factory.
def conv_frame_count(num_samples):
    for kernel, stride in zip(CONV_KERNELS, CONV_STRIDES):
        if num_samples < kernel:
            return 0
        num_samples = (num_samples - kernel) // stride + 1
    return int(num_samples)


def require_model_length(num_samples, mask_time_length, identifier):
    frames = conv_frame_count(num_samples)
    if frames < max(1, mask_time_length):
        raise ValueError(
            f"{identifier}: {num_samples} samples give {frames} frontend frames; "
            f"need {max(1, mask_time_length)}. Change the global mask policy explicitly; "
            "do not drop this row or rely on another recording's batch padding."
        )


def audit_model_lengths(data, augmentation=AUGMENT, mask_time_length=10):
    """Check ALL saved length metadata without decoding or filtering recordings."""
    report = {}
    for name, split in data.items():
        minimum_frames = max(1, mask_time_length) if name == "train" else 1
        incompatible = []
        minimum_samples = None
        for row in split.select_columns(["audio_file_name", "num_samples"]):
            samples = int(row["num_samples"])
            minimum_samples = (
                samples if minimum_samples is None else min(minimum_samples, samples)
            )
            fastest = (
                round(samples / augmentation.speed_range[1])
                if name == "train" and augmentation.p_speed > 0
                else samples
            )
            if (
                conv_frame_count(samples) < minimum_frames
                or conv_frame_count(fastest) < minimum_frames
            ):
                incompatible.append(row["audio_file_name"])
        report[name] = {
            "rows_checked": len(split),
            "minimum_samples": minimum_samples,
            "incompatible_ids": incompatible,
        }
    return report


class SpoofAudioDataset(TorchDataset):
    def __init__(
        self,
        data,
        split_name,
        augment=False,
        augmentation=AUGMENT,
        base_seed=SEED,
        mask_time_length=10,
        preprocess_config=PREPROCESS,
    ):
        if split_name not in EXPECTED_COUNTS:
            raise ValueError("Unknown split.")
        if augment and split_name != "train":
            raise ValueError("Waveform augmentation is only permitted on train.")
        if base_seed < 0 or mask_time_length < 0:
            raise ValueError("Seed and mask length must be nonnegative.")
        required = {
            "audio",
            "key",
            "audio_file_name",
            "preprocess_version",
            "num_samples",
        }
        if not required.issubset(data.column_names):
            raise ValueError(
                "Load a standardized artifact, not the unprocessed source."
            )
        if (
            not isinstance(data.features["key"], ClassLabel)
            or data.features["key"].names != LABEL_NAMES
        ):
            raise ValueError("Unexpected label mapping.")
        self.data = data.cast_column(
            "audio", Audio(sampling_rate=TARGET_SR, num_channels=1)
        )
        self.split_name, self.augment, self.augmentation = (
            split_name,
            augment,
            augmentation,
        )
        self.base_seed, self.epoch = int(base_seed), 0
        self.mask_time_length = int(mask_time_length) if split_name == "train" else 0
        self.preprocess_config = preprocess_config

    def __len__(self):
        return len(self.data)

    def set_epoch(self, epoch):
        if not isinstance(epoch, (int, np.integer)) or epoch < 0:
            raise ValueError("Epoch must be a nonnegative integer.")
        self.epoch = int(epoch)

    def __getitem__(self, index):
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        identifier = f"{self.split_name}/{index}"
        try:
            row = self.data[index]
            identifier = f"{self.split_name}/{row['audio_file_name']}"
            if row["preprocess_version"] != self.preprocess_config.version:
                raise ValueError("Unexpected preprocessing version.")
            waveform = decode_hf_audio(row["audio"])
            if len(waveform) != row["num_samples"]:
                raise ValueError("Decoded length differs from stored metadata.")
            require_model_length(len(waveform), self.mask_time_length, identifier)
            if self.augment:
                if self.augmentation.p_speed > 0:
                    require_model_length(
                        round(len(waveform) / self.augmentation.speed_range[1]),
                        self.mask_time_length,
                        identifier,
                    )
                rng = np.random.default_rng(
                    np.random.SeedSequence([self.base_seed, self.epoch, 0, index])
                )
                waveform = augment_waveform(waveform, rng, self.augmentation)
                require_model_length(len(waveform), self.mask_time_length, identifier)
            return {
                "input_values": waveform,
                "labels": int(row["key"]),
                "duration_s": len(waveform) / TARGET_SR,
                "audio_file_name": row["audio_file_name"],
            }
        except Exception as exc:
            raise ValueError(f"{identifier}: {exc}") from exc


@dataclass
class AudioCollator:
    feature_extractor: Wav2Vec2FeatureExtractor

    def __call__(self, rows):
        labels = [int(row["labels"]) for row in rows]
        if not rows or any(label not in (0, 1) for label in labels):
            raise ValueError("Expected a nonempty batch with binary labels.")
        batch = self.feature_extractor(
            [valid_waveform(row["input_values"]) for row in rows],
            sampling_rate=TARGET_SR,
            padding=True,
            truncation=False,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_values": batch["input_values"],
            "attention_mask": batch["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.int64),
        }


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def make_dataloader(data, feature_extractor, batch_size=4, shuffle=None, num_workers=0):
    if shuffle is None:
        shuffle = data.split_name == "train"
    if data.split_name != "train" and shuffle:
        raise ValueError("Evaluation loaders must preserve deterministic row order.")
    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=AudioCollator(feature_extractor),
        persistent_workers=False,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(data.base_seed),
    )


# %% Small XLS-R configuration files only — no model weights or tokenizer.
def configure_specaugment(config):
    config.num_labels = 2
    config.id2label, config.label2id = (
        {0: "bonafide", 1: "spoof"},
        {"bonafide": 0, "spoof": 1},
    )
    config.apply_spec_augment = True
    config.mask_time_prob, config.mask_time_length, config.mask_time_min_masks = (
        0.075,
        10,
        2,
    )
    (
        config.mask_feature_prob,
        config.mask_feature_length,
        config.mask_feature_min_masks,
    ) = 0.02, 10, 2
    return config


def load_xlsr_assets():
    config = configure_specaugment(
        Wav2Vec2Config.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    )
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION
    )
    if (
        tuple(config.conv_kernel) != CONV_KERNELS
        or tuple(config.conv_stride) != CONV_STRIDES
    ):
        raise ValueError("Checkpoint frontend differs from the length-check contract.")
    if config.hidden_size != 1024 or config.feat_extract_norm != "layer":
        raise ValueError("Unexpected XLS-R configuration.")
    if extractor.sampling_rate != TARGET_SR or not extractor.do_normalize:
        raise ValueError("Unexpected XLS-R waveform feature-extractor configuration.")
    return config, extractor


# %% Full-metadata length audit contract, including non-leading records.
def run_model_length_checks():
    data = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "audio_file_name": ["safe", "late-short", "speed-edge"],
                    "num_samples": [64000, 1600, 3600],
                }
            ),
            "validation": Dataset.from_dict(
                {"audio_file_name": ["eval-short"], "num_samples": [399]}
            ),
            "test": Dataset.from_dict(
                {"audio_file_name": ["eval-safe"], "num_samples": [400]}
            ),
        }
    )
    report = audit_model_lengths(
        data, AugmentationConfig(speed_range=(0.9, 1.1), p_speed=0.5), 10
    )
    assert report["train"]["rows_checked"] == 3
    assert report["train"]["incompatible_ids"] == ["late-short", "speed-edge"]
    assert report["validation"]["incompatible_ids"] == ["eval-short"]
    assert report["test"]["incompatible_ids"] == []
    assert len(data["train"]) == 3  # The audit reports; it never filters rows.
    print(
        "PASS: every-recording length audit, fastest-speed edge, evaluation minimum, and row retention"
    )


# %% Dataset and collator contract checks.
def run_batch_checks(processed, feature_extractor, num_workers=0):
    noise_only = AugmentationConfig(
        p_noise=1, p_shift=0, p_speed=0, p_pitch=0, p_reverb=0
    )
    train = SpoofAudioDataset(
        processed["train"], "train", augment=True, augmentation=noise_only
    )
    train_loader = make_dataloader(
        train, feature_extractor, batch_size=2, shuffle=False, num_workers=num_workers
    )
    train.set_epoch(0)
    first = next(iter(train_loader))
    np.testing.assert_array_equal(train[0]["input_values"], train[0]["input_values"])
    train.set_epoch(1)
    second = next(iter(train_loader))
    assert not torch.equal(first["input_values"], second["input_values"])
    evaluation = SpoofAudioDataset(processed["validation"], "validation", augment=False)
    loader = make_dataloader(
        evaluation, feature_extractor, batch_size=2, num_workers=num_workers
    )
    batch = next(iter(loader))
    evaluation.set_epoch(3)
    repeated = next(iter(loader))
    assert set(batch) == {"input_values", "attention_mask", "labels"}
    assert (
        batch["input_values"].dtype == torch.float32
        and batch["labels"].dtype == torch.int64
    )
    assert batch["input_values"].shape == batch["attention_mask"].shape
    assert batch["labels"].shape == (2,) and set(batch["labels"].tolist()) <= {0, 1}
    for key in batch:
        torch.testing.assert_close(batch[key], repeated[key], rtol=0, atol=0)
    assert batch["attention_mask"].sum(-1).tolist() == [
        len(evaluation[i]["input_values"]) for i in range(2)
    ]
    assert torch.count_nonzero(batch["input_values"][batch["attention_mask"] == 0]) == 0
    for split_name in ("validation", "test"):
        try:
            SpoofAudioDataset(processed[split_name], split_name, augment=True)
        except ValueError:
            pass
        else:
            raise AssertionError("Evaluation augmentation must be rejected.")
    short_row = {
        "speaker_id": "S",
        "audio_file_name": "too-short",
        "system_id": "-",
        "key": 0,
        **standardize_waveform(np.ones(1600, np.float32) * 0.01, TARGET_SR),
    }
    short = Dataset.from_list([short_row], features=processed["train"].features)
    try:
        SpoofAudioDataset(short, "train", augment=False)[0]
    except ValueError as exc:
        assert "too-short" in str(exc)
    else:
        raise AssertionError(
            "A short training recording must not rely on another row's padding."
        )
    print(
        f"PASS: train-only/epoch-aware batches, evaluation determinism, padding, labels, short input ({num_workers} workers)"
    )


# %% Tiny model masking check — public forward/hooks, no pretrained weights.
def run_masking_checks():
    torch.manual_seed(SEED)
    config = configure_specaugment(
        Wav2Vec2Config(
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=128,
            conv_dim=(16,) * 7,
            conv_kernel=CONV_KERNELS,
            conv_stride=CONV_STRIDES,
            feat_extract_norm="layer",
            do_stable_layer_norm=True,
            num_conv_pos_embeddings=16,
            num_conv_pos_embedding_groups=2,
            hidden_dropout=0,
            attention_dropout=0,
            activation_dropout=0,
            feat_proj_dropout=0,
            final_dropout=0,
            layerdrop=0,
        )
    )
    tiny = Wav2Vec2Model(config)
    captured = {}

    def before_mask(_module, _args, output):
        captured["before"] = output[0].detach().clone()

    def after_mask(_module, args):
        captured["after"] = args[0].detach().clone()

    handles = [
        tiny.get_submodule("feature_projection").register_forward_hook(before_mask),
        tiny.get_submodule("encoder").register_forward_pre_hook(after_mask),
    ]
    inputs = torch.randn(2, TARGET_SR)
    attention = torch.ones_like(inputs, dtype=torch.long)
    attention[1, 12_800:] = 0
    inputs[1, 12_800:] = 0

    def forward():
        captured.clear()
        with torch.no_grad():
            output = tiny(inputs, attention_mask=attention)
        assert output.last_hidden_state.shape == (2, 49, 64)
        assert torch.isfinite(output.last_hidden_state).all()

    try:
        tiny.eval()
        forward()
        torch.testing.assert_close(
            captured["before"], captured["after"], rtol=0, atol=0
        )
        tiny.train()
        config.apply_spec_augment = False
        forward()
        torch.testing.assert_close(
            captured["before"], captured["after"], rtol=0, atol=0
        )
        config.apply_spec_augment, config.mask_feature_prob = True, 0
        np.random.seed(SEED)
        forward()
        time_masked = (captured["before"] != captured["after"]).any(dim=-1)
        assert time_masked.any()
        assert not time_masked[
            1, 39:
        ].any()  # padding must not generate time-mask starts
        config.mask_time_prob, config.mask_feature_prob = 0, 0.02
        np.random.seed(SEED)
        forward()
        changed = captured["before"] != captured["after"]
        for index in range(2):
            channels = changed[index].any(dim=0)
            assert (
                channels.any()
                and torch.count_nonzero(captured["after"][index, :, channels]) == 0
            )
    finally:
        for handle in handles:
            handle.remove()
    assert [conv_frame_count(n) for n in (399, 400, 3279, 3280, 64000)] == [
        0,
        1,
        9,
        10,
        199,
    ]
    print(
        "PASS: tiny-model train/eval mask gating, separate time/feature masking, padding, frontend lengths"
    )


# %% Augmentation contract checks.
def run_augmentation_checks():
    t = np.arange(TARGET_SR) / TARGET_SR
    clean = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    noisy = add_gaussian_noise(clean, np.random.default_rng(SEED), 15.0)
    snr = 10 * np.log10(
        np.mean(clean.astype(np.float64) ** 2)
        / np.mean((noisy.astype(np.float64) - clean) ** 2)
    )
    assert abs(snr - 15.0) < 0.1
    zero = np.zeros(TARGET_SR, np.float32)
    np.testing.assert_array_equal(
        add_gaussian_noise(zero, np.random.default_rng(SEED), 15), zero
    )
    values = np.arange(1, 11, dtype=np.float32)
    np.testing.assert_array_equal(
        time_shift(values, 2 / TARGET_SR), [0, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    )
    np.testing.assert_array_equal(
        time_shift(values, -2 / TARGET_SR), [3, 4, 5, 6, 7, 8, 9, 10, 0, 0]
    )
    faster = change_speed(clean, 1.1)
    assert abs(len(faster) - round(TARGET_SR / 1.1)) <= 1
    peak_frequency = np.argmax(np.abs(np.fft.rfft(faster))) * TARGET_SR / len(faster)
    assert abs(peak_frequency - 440) < 3  # speed must not also shift pitch
    higher = shift_pitch(clean, 2.0)
    assert len(higher) == len(clean)
    assert abs(np.argmax(np.abs(np.fft.rfft(higher))) - 494) < 4
    for seed in range(20):
        rir = synthetic_rir(np.random.default_rng(seed), 0.3, 6.0)
        assert abs(estimate_rt60(rir[1:]) - 0.3) < 0.05
        assert abs(-10 * np.log10(np.sum(rir[1:] ** 2)) - 6) < 0.001
        impulse = np.zeros(TARGET_SR, np.float32)
        impulse[0] = 1
        wet = reverberate(impulse, rir)
        assert len(wet) == len(impulse) and np.isclose(wet[0], 1)
    off = AugmentationConfig(p_noise=0, p_shift=0, p_speed=0, p_pitch=0, p_reverb=0)
    np.testing.assert_array_equal(
        augment_waveform(clean, np.random.default_rng(SEED), off), clean
    )
    all_effects = replace(off, p_noise=1, p_shift=1, p_speed=1, p_pitch=1, p_reverb=1)
    first = augment_waveform(clean, np.random.default_rng(SEED), all_effects)
    second = augment_waveform(clean, np.random.default_rng(SEED), all_effects)
    np.testing.assert_array_equal(first, second)
    assert np.isfinite(first).all() and np.max(np.abs(first)) <= 0.99 + 1e-7
    print(
        "PASS: noise SNR, shift direction, independent speed/pitch, RIR decay/energy, and augmentation composition"
    )


# %% [markdown]
# ## 4. Storage and dataset inspection
# Keep downloads/map caches on local Colab disk, not Drive. Processing is row-wise;
# audio is never expanded into one large in-memory array. Saved audio bytes have
# no external paths. Cache keys include input metadata, configuration and package
# versions. Saved artifact compatibility uses the preprocessing configuration;
# the exact producing package versions are retained as provenance, so an already
# encoded artifact is portable to a later compatible runtime.
#
# Completion markers are published **after** local reload verification and,
# separately, **after** Drive checksums/reload verification. Existing incomplete
# or incompatible directories are errors: choose a new `RUN_TAG`, do not overwrite.
# Drive's mounted free-space report is not your account quota; check the Drive UI.


# %% Restoration contract — a saved artifact does not require the raw download.
def run_restore_checks(base_dir=None):
    with tempfile.TemporaryDirectory(prefix="asv-restore-", dir=base_dir) as folder:
        root = Path(folder)
        data = map_standardized(synthetic_dataset(), PREPROCESS, root / "cache")
        counts = {name: len(split) for name, split in data.items()}
        save_artifact(data, root / "drive", PREPROCESS, kind="smoke")
        assert (
            restore_available_artifact(
                root / "absent",
                root / "absent-drive",
                PREPROCESS,
                counts,
                True,
                kind="smoke",
            )
            is None
        )
        available = shutil.disk_usage(root)
        with patch.object(
            shutil, "disk_usage", return_value=available._replace(free=0)
        ):
            try:
                restore_available_artifact(
                    root / "no-room",
                    root / "drive",
                    PREPROCESS,
                    counts,
                    True,
                    kind="smoke",
                )
            except OSError:
                assert not (root / "no-room").exists()
            else:
                raise AssertionError(
                    "Restoration must check capacity before creating its destination."
                )
        restored = restore_available_artifact(
            root / "local", root / "drive", PREPROCESS, counts, True, kind="smoke"
        )
        assert len(restored["train"]) == counts["train"]
        assert (
            restore_available_artifact(
                root / "local", root / "drive", PREPROCESS, counts, True, kind="smoke"
            )
            is not None
        )
    print(
        "PASS: verified restoration/reuse and insufficient-disk preflight without a raw download"
    )


# %% Inspection, budget estimates, and balanced *smoke-only* selection.
def check_local_space(directory, required_bytes, purpose):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    available = shutil.disk_usage(directory).free
    print(
        f"{purpose}: need ~{required_bytes / 1e9:.2f} GB; local free {available / 1e9:.2f} GB"
    )
    if available < required_bytes:
        raise OSError(
            f"Insufficient local disk for {purpose}. Change LOCAL_ROOT or free space explicitly."
        )


def balanced_indices(split, per_label, seed=SEED):
    labels = np.asarray(list(split["key"]))
    rng = np.random.default_rng(seed)
    indices = []
    for label in (0, 1):
        eligible = np.flatnonzero(labels == label)
        if len(eligible) < per_label:
            raise ValueError(
                f"Not enough {LABEL_NAMES[label]} rows for the smoke test."
            )
        indices.extend(rng.choice(eligible, per_label, replace=False).tolist())
    return sorted(indices)


def inspect_source(source, sample_size=32):
    report = {}
    estimated_payload = 0.0
    for split_id, (name, split) in enumerate(source.items()):
        indices = np.random.default_rng(SEED + split_id).choice(
            len(split), min(sample_size, len(split)), replace=False
        )
        native = split.cast_column("audio", Audio(decode=False))
        durations, headers = [], Counter()
        for index in indices:
            index = int(index)
            waveform = decode_hf_audio(split[index]["audio"])
            durations.append(len(waveform) / TARGET_SR)
            payload = native[index]["audio"]
            try:
                info = sf.info(
                    io.BytesIO(payload["bytes"])
                    if payload["bytes"] is not None
                    else payload["path"]
                )
                headers[
                    f"{info.format}/{info.subtype}/{info.channels}ch/{info.samplerate}Hz"
                ] += 1
            except (RuntimeError, OSError, TypeError):
                # Header inspection is not a production decoder fallback.
                headers["header unavailable; TorchCodec decode passed"] += 1
        estimated_payload += float(np.mean(durations)) * len(split) * TARGET_SR * 2
        report[name] = {
            "sample_count": len(durations),
            "native_headers": dict(headers),
            "mean_seconds": float(np.mean(durations)),
            "p95_seconds": float(np.percentile(durations, 95)),
            "minimum_sampled_seconds": min(durations),
        }
    estimated_artifact = estimated_payload * 1.15  # rough WAV/Arrow overhead allowance
    print(
        json.dumps(
            {
                "sampled_format_audit": report,
                "estimated_processed_GB": estimated_artifact / 1e9,
            },
            indent=2,
        )
    )
    print(
        "Estimates are for storage only, not for choosing augmentation strengths from test data."
    )
    return estimated_artifact


def plot_audio_examples(examples, title):
    fig, axes = plt.subplots(
        len(examples), 2, figsize=(13, 2.7 * len(examples)), squeeze=False
    )
    for (label, waveform), row in zip(examples, axes):
        row[0].plot(np.arange(len(waveform)) / TARGET_SR, waveform, linewidth=0.5)
        row[0].set(title=label, xlabel="Seconds", ylabel="Amplitude")
        mel = librosa.feature.melspectrogram(
            y=waveform, sr=TARGET_SR, n_fft=512, hop_length=160, n_mels=64
        )
        librosa.display.specshow(
            librosa.power_to_db(mel, ref=np.max),
            sr=TARGET_SR,
            hop_length=160,
            x_axis="time",
            y_axis="mel",
            ax=row[1],
        )
        row[1].set_title(f"{label} — log-mel (visualization only)")
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
    plt.close(fig)
    for label, waveform in examples:
        print(label)
        # Playback-only scaling if a raw resampling transient exceeds full scale.
        playback = waveform / max(1.0, float(np.abs(waveform).max()))
        display(AudioPlayer(playback, rate=TARGET_SR, normalize=False))


# %% Offline integration checks — run before downloading the actual corpus.
if NOTEBOOK_RUN:
    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
    run_artifact_checks(LOCAL_ROOT)
    run_restore_checks(LOCAL_ROOT)
    run_augmentation_checks()
    run_masking_checks()
    run_model_length_checks()


# %% Mount Drive; no automatic overwrites or inferred quota checks.
if NOTEBOOK_RUN:
    if MOUNT_GOOGLE_DRIVE:
        from google.colab import drive

        drive.mount("/content/drive")
        print(
            f"Drive output: {DRIVE_ROOT}. Check available account quota at drive.google.com."
        )
    else:
        print(
            "Outside Colab/Drive mount disabled: full artifacts stay under LOCAL_ROOT."
        )
    FINGERPRINT = preprocess_fingerprint(PREPROCESS)[:16]
    LOCAL_ARTIFACT = LOCAL_ROOT / f"standardized-{FINGERPRINT}"
    DRIVE_ARTIFACT = DRIVE_ROOT / f"standardized-{FINGERPRINT}"
    SOURCE_CACHE = LOCAL_ROOT / "source_cache"
    MAP_CACHE = LOCAL_ROOT / "map_cache"
    if (
        RUN_FULL_PREPROCESS
        and MOUNT_GOOGLE_DRIVE
        and not DRIVE_ARTIFACT.exists()
        and not DRIVE_QUOTA_CONFIRMED
    ):
        raise RuntimeError(
            "Check Drive account quota, then set DRIVE_QUOTA_CONFIRMED=True before publishing a new full artifact."
        )
    raw = None
    processed = (
        restore_available_artifact(
            LOCAL_ARTIFACT,
            DRIVE_ARTIFACT,
            PREPROCESS,
            EXPECTED_COUNTS,
            MOUNT_GOOGLE_DRIVE,
        )
        if RUN_FULL_PREPROCESS
        else None
    )


# %% Load all supplied splits — .select() later does NOT avoid this download.
if NOTEBOOK_RUN:
    if processed is not None:
        print(
            "Reusing verified full artifact; raw source download and mapping skipped."
        )
    else:
        if not SOURCE_CACHE.exists():
            check_local_space(
                LOCAL_ROOT, 7_543_123_064 * 2 * 1.25, "source download + Arrow cache"
            )
        raw = load_dataset(
            DATASET_ID, revision=DATASET_REVISION, cache_dir=str(SOURCE_CACHE)
        )
        raw = raw.cast_column("audio", Audio(sampling_rate=TARGET_SR, num_channels=1))
        SOURCE_AUDIT = audit_metadata(raw, EXPECTED_COUNTS)
        print(json.dumps(SOURCE_AUDIT, indent=2))
        ESTIMATED_ARTIFACT_BYTES = inspect_source(raw)
        # One actual example from each split, with no augmentation.
        for split_name, split in raw.items():
            example = standardize_row(split[0], PREPROCESS)
            assert read_pcm16(example["audio"]["bytes"]).size == example["num_samples"]


# %% [markdown]
# ## 5. Smoke-map, then opt into the full materialization
# The smoke artifact has **16 train + 2 validation + 2 test** rows, explicitly
# marked `kind="smoke"` and stored only locally. Only the training smoke subset
# is balanced; the complete source/saved splits are never balanced or re-split.
# The full artifact's strict count check prevents substituting the smoke artifact.
# No augmentation occurs in either preprocessing map or saved artifact.
# Reuse selects demonstrations from the verified full artifact and needs no raw download.

# %% Seeded smoke subset, deterministic encoding, local save/reload audit.
if NOTEBOOK_RUN:
    selection_input = raw if raw is not None else processed
    smoke_selection = DatasetDict(
        {
            name: split.select(
                balanced_indices(split, 8 if name == "train" else 1, SEED + index)
            )
            for index, (name, split) in enumerate(selection_input.items())
        }
    )
    SMOKE_COUNTS = {name: len(split) for name, split in smoke_selection.items()}
    SMOKE_DIR = LOCAL_ROOT / f"smoke-{FINGERPRINT}-seed{SEED}"
    if raw is None:
        smoke_source, smoke_processed = None, smoke_selection
        print("Using standardized demonstration rows from the verified full artifact.")
    else:
        smoke_source = smoke_selection
        if SMOKE_DIR.exists():
            smoke_processed = load_artifact(
                SMOKE_DIR, PREPROCESS, SMOKE_COUNTS, kind="smoke"
            )
            if audit_metadata(smoke_processed, SMOKE_COUNTS) != audit_metadata(
                smoke_source, SMOKE_COUNTS
            ):
                raise ValueError(
                    "Existing smoke artifact represents a different subset; choose a new RUN_TAG."
                )
        else:
            smoke_processed = map_standardized(
                smoke_source, PREPROCESS, MAP_CACHE / "smoke", num_proc=NUM_PROC
            )
            smoke_processed = save_artifact(
                smoke_processed, SMOKE_DIR, PREPROCESS, kind="smoke"
            )
        # Compare repeated output bytes, not source-vs-output hashes.
        stored_row = smoke_processed["train"].cast_column("audio", Audio(decode=False))[
            0
        ]
        assert (
            stored_row["audio"]["bytes"]
            == standardize_row(smoke_source["train"][0], PREPROCESS)["audio"]["bytes"]
        )
        print("PASS: actual-source smoke map, byte determinism, and reload")


# %% Full run — save locally, verify, copy to Drive, verify, publish marker last.
if NOTEBOOK_RUN:
    if RUN_FULL_PREPROCESS:
        if processed is None:
            check_local_space(
                LOCAL_ROOT,
                ESTIMATED_ARTIFACT_BYTES * 2 * 1.25,
                "processed map cache + saved staging copy",
            )
            processed = map_standardized(
                raw, PREPROCESS, MAP_CACHE / "full", num_proc=NUM_PROC
            )
            processed = save_artifact(processed, LOCAL_ARTIFACT, PREPROCESS)
        if (
            raw is not None
            and audit_metadata(processed, EXPECTED_COUNTS) != SOURCE_AUDIT
        ):
            raise ValueError("Full artifact metadata differs from the pinned source.")
        if MOUNT_GOOGLE_DRIVE:
            copy_artifact(
                LOCAL_ARTIFACT,
                DRIVE_ARTIFACT,
                PREPROCESS,
                EXPECTED_COUNTS,
                check_local_capacity=False,  # Drive account quota was confirmed separately.
            )
            print(f"Verified complete Drive artifact: {DRIVE_ARTIFACT}")
        active_data = processed
        print(
            "FULL STANDARDIZED ARTIFACT VERIFIED:",
            {name: len(split) for name, split in processed.items()},
        )
    else:
        active_data = smoke_processed
        print(
            "SMOKE MODE ONLY: full mapping/Drive publication skipped. Set RUN_FULL_PREPROCESS=True after checking storage."
        )


# %% [markdown]
# ## 6. XLS-R integration and demonstrations
# The next cell downloads **configuration/feature-extractor JSON only**. The
# future classifier must be constructed with `model_config` before training,
# then use `model.train()` for training and `model.eval()` for evaluation.
# This notebook deliberately does not instantiate or download XLS-R-300M.
#
# Native masks operate on learned features. `mask_time_prob=0.075`, length 10,
# minimum 2; feature probability 0.02, length 10, minimum 2. Counts are rounded,
# length-capped, and overlapping. At 4 s there are 199 frames, typically two
# selected time starts, not necessarily two connected regions. Features are
# hidden channels, not physical frequency bins. Keeping a frozen backbone in
# eval mode would also disable its automatic SpecAugment.
#
# Per-clip feature-extractor normalization ignores batch padding. Its output can
# exceed [-1, 1]: never waveform-clip the normalized model tensors.

# %% Build model-ready loaders; no optimizer, classifier weights, or training.
if NOTEBOOK_RUN:
    model_config, feature_extractor = load_xlsr_assets()
    TRAIN_MASK_LENGTH = (
        model_config.mask_time_length
        if model_config.apply_spec_augment and model_config.mask_time_prob > 0
        else 0
    )
    MODEL_LENGTH_AUDIT = audit_model_lengths(active_data, AUGMENT, TRAIN_MASK_LENGTH)
    print(
        json.dumps(
            {
                name: {
                    "rows_checked": value["rows_checked"],
                    "incompatible_count": len(value["incompatible_ids"]),
                    "first_incompatible_ids": value["incompatible_ids"][:20],
                }
                for name, value in MODEL_LENGTH_AUDIT.items()
            },
            indent=2,
        )
    )
    if any(value["incompatible_ids"] for value in MODEL_LENGTH_AUDIT.values()):
        raise ValueError(
            "Model-incompatible lengths found. All IDs are in MODEL_LENGTH_AUDIT; rows remain in the saved artifact. Adjust the global mask/speed policy explicitly before training."
        )
    run_batch_checks(smoke_processed, feature_extractor)
    train_dataset = SpoofAudioDataset(
        active_data["train"],
        "train",
        augment=True,
        augmentation=AUGMENT,
        mask_time_length=TRAIN_MASK_LENGTH,
    )
    validation_dataset = SpoofAudioDataset(
        active_data["validation"], "validation", augment=False
    )
    test_dataset = SpoofAudioDataset(active_data["test"], "test", augment=False)
    train_loader = make_dataloader(train_dataset, feature_extractor)
    validation_loader = make_dataloader(validation_dataset, feature_extractor)
    test_loader = make_dataloader(test_dataset, feature_extractor)
    train_dataset.set_epoch(0)  # Repeat before EACH future training iterator.
    demonstration_batch = next(iter(train_loader))
    print(
        {
            key: (tuple(value.shape), str(value.dtype))
            for key, value in demonstration_batch.items()
        }
    )


# %% Optional training-only before/after waveform, log-mel, and listening examples.
if NOTEBOOK_RUN and SHOW_PREVIEWS:
    preview_data = smoke_source if smoke_source is not None else smoke_processed
    for index in balanced_indices(preview_data["train"], 1):
        row = preview_data["train"][index]
        decoded = decode_hf_audio(row["audio"])
        examples = []
        if smoke_source is not None:
            examples.append(("Decoded original", decoded))
            standardized = read_pcm16(
                standardize_waveform(decoded, TARGET_SR, PREPROCESS)["audio"]["bytes"]
            )
        else:
            standardized = decoded
        augmented = augment_waveform(
            standardized, np.random.default_rng(SEED + index), AUGMENT
        )
        plot_audio_examples(
            examples
            + [
                ("Standardized PCM16", standardized),
                ("Random train-only augmentation", augmented),
            ],
            f"{LABEL_NAMES[row['key']]} — {row['audio_file_name']}",
        )


# %% [markdown]
# ## Runbook and limitations
# - Reuse a verified artifact with `load_artifact(path, PREPROCESS, EXPECTED_COUNTS)`;
#   a full notebook rerun with RUN_FULL_PREPROCESS=True discovers a compatible
#   local/Drive copy before downloading the raw source again. Local restoration
#   checks the manifest's size budget before creating its destination.
#   a missing/incompatible marker or changed shard is rejected. Full SHA-256
#   verification reads every shard and can take time on Drive. For later training,
#   copy to local disk and use the supplied wrappers/collator.
# - Configure probabilities/ranges in `AUGMENT`; do not augment validation/test.
#   Saved train rows are clean standardized recordings, **not** pre-augmented.
# - Reduce training batch size for variable-length memory pressure; no default
#   truncation, row dropping, resampling for class balance, or fixed 4 s crop.
# - MODEL_LENGTH_AUDIT checks every active recording's saved length, including
#   training's fastest configured speed. It holds all incompatible IDs; the print
#   preview is capped at 20. Smoke mode does not certify full-corpus lengths.
# - Keep raw HF cache and prior artifacts intact. On interruption/incompatibility,
#   choose a new `RUN_TAG`; this notebook never deletes an old output directory.
# - Class imbalance makes accuracy alone misleading. Later use EER, ROC-AUC,
#   per-class recall, validation-only tuning, and a held-out final test. min-tDCF
#   additionally needs compatible ASV-system scores/protocol/cost assumptions.
# - Passing these data checks is not evidence of a classification accuracy gain.
#   Full real-corpus/Drive execution and XLS-R-300M training must be verified in
#   the target runtime. All random waveform operations are reproducible per row
#   and epoch under the recorded package versions, not promised across hardware.

# %% Final summary — never label a smoke artifact as a completed full run.
if NOTEBOOK_RUN:
    summary_path = LOCAL_ARTIFACT if RUN_FULL_PREPROCESS else SMOKE_DIR
    completion = json.loads((summary_path / "manifest.json").read_text())
    print(
        json.dumps(
            {
                "mode": completion["kind"],
                "artifact": str(summary_path),
                "counts": completion["metadata"]["counts"],
                "labels": LABEL_NAMES,
                "source_revision": DATASET_REVISION,
                "preprocess_fingerprint": completion["fingerprint"],
                "augmentation": "on-the-fly train only; saved audio is unaugmented",
                "classifier_training": "not performed",
                "package_versions": package_versions(),
            },
            indent=2,
        )
    )
