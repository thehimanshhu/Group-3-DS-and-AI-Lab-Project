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
# # In-The-Wild — independent preprocessing and training-input notebook
#
# **Source:** `mueller91/In-The-Wild`, pinned ZIP release; **labels:**
# 0 = bonafide, 1 = spoof. This notebook is independent of ASVspoof: it does not
# import that notebook, combine corpora, train a classifier, or download model
# weights. Saved audio is **unaugmented**, mono 16 kHz PCM16 WAV in portable Arrow.
# Waveform effects are applied only on the fly to future training examples.
#
# Convert with `jupytext --to ipynb iwd_notebook.py`, upload to Colab, and run
# cells in order on CPU. Setup preserves Colab's PyTorch. The initial setting is
# **`RUN_FULL_PREPROCESS=False`**: 16 train + 2 validation + 2 test rows only.
# Even smoke mode needs the entire **8.16 GB ZIP** unless a verified processed
# artifact exists. Full preprocessing and Drive publication require explicit
# opt-in; verify local capacity and your actual Drive account quota first.
# No existing output is overwritten. Use a new `RUN_TAG` after interruption.
#
# The Hub viewer exposes audio only. Labels/speakers come exclusively from the
# ZIP's `meta.csv`, joined by exact basename, not row position or inferred labels.
# Its 31,779 clips contain 19,963 bona-fide and 11,816 spoof rows, and **54 exact
# speaker strings**, not the card's advertised 58. All 54 have both labels.
# The project defines a metadata-only, whole-speaker 70/15/15 split by **clip
# count**. It is not an official protocol. Source-video/attack IDs are absent:
# speaker disjointness does not establish source-video disjointness. Exact-byte
# duplicate checks below do not rule out near duplicates, shared source videos,
# or speaker aliases. Do not select a new partition by model performance.
# The HF card declares CC BY-SA 4.0, while the authors' site declares Apache 2.0.
# This discrepancy remains unresolved. The portable protocol preserves the exact
# release attribution and both license references; it does not grant redistribution rights.

# %% Setup — install before importing; never replace Colab's PyTorch.
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
    """Same bounded audio/model APIs and CPU decoder pairing as the ASV notebook."""
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

import csv
import hashlib
import io
import itertools
import math
import random
import shutil
import stat
import tempfile
import warnings
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
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
    load_from_disk,
)
from huggingface_hub import hf_hub_download, try_to_load_from_cache
from IPython.display import Audio as AudioPlayer
from IPython.display import display
from scipy.signal import fftconvolve, resample_poly
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Model


def package_versions():
    names = (
        "datasets",
        "huggingface_hub",
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


# %% Configuration — a full run is explicitly opt-in.
def in_colab():
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


DATASET_ID = "mueller91/In-The-Wild"
DATASET_REVISION = "eee168f92c367f8c82ff2cf42b6f61e362fd6211"
ARCHIVE_NAME = "release_in_the_wild.zip"
ARCHIVE_ROOT = "release_in_the_wild"
MODEL_ID = "facebook/wav2vec2-xls-r-300m"
MODEL_REVISION = "1a640f32ac3e39899438a2931f9924c02f080a54"
SPLIT_NAMES = ("train", "validation", "test")
LABEL_NAMES = ["bonafide", "spoof"]
LABEL_MAPPING = {"bona-fide": 0, "spoof": 1}
SOURCE = {
    "id": DATASET_ID,
    "revision": DATASET_REVISION,
    "archive": ARCHIVE_NAME,
    "archive_size": 8_161_489_023,
    "archive_sha256": "46665f30a6758b45642c659c7e0e80da2ba3d121b064716824a56a010ddf9e1a",
    "metadata_csv_sha256": "2e404150eff3cf171b766da81013a93248cbb219682fff420bbadb6b86a18795",
    "entries": 31_782,
    "rows": 31_779,
    "class_counts": {"bona-fide": 19_963, "spoof": 11_816},
    "speakers": 54,
    "wav_uncompressed_bytes": 8_722_611_294,
    "kind": "corpus",
}
SOURCE_LICENSE_PROVENANCE = {
    "hf_card": {
        "license": "CC-BY-SA-4.0",
        "url": f"https://huggingface.co/datasets/{DATASET_ID}/blob/{DATASET_REVISION}/README.md",
    },
    "authors_site": {
        "license": "Apache-2.0",
        "url": "https://deepfake-total.com/in_the_wild",
    },
    "status": "Unresolved discrepancy; preserve attribution and clarify rights before redistribution.",
}
TARGET_SR = 16_000
PCM16_STEP = 1.0 / 32768.0
CONV_KERNELS = (10, 3, 3, 3, 3, 2, 2)
CONV_STRIDES = (5, 2, 2, 2, 2, 2, 2)
SEED = 42
RUN_FULL_PREPROCESS = False
DRIVE_QUOTA_CONFIRMED = False
MOUNT_GOOGLE_DRIVE = in_colab()
RUN_TAG = "v1"  # NEW tag for an interrupted/incompatible output; never overwrite.
LOCAL_ROOT = Path("/content/iwd_work" if MOUNT_GOOGLE_DRIVE else "./iwd_work") / RUN_TAG
DRIVE_ROOT = Path("/content/drive/MyDrive/ds-ai-lab/in_the_wild") / RUN_TAG
WRITER_BATCH_SIZE = 32
SHOW_PREVIEWS = True
SMOKE_POLICY = {
    "version": "within-frozen-splits-v1",
    "seed": SEED,
    "train_per_label": 8,
    "evaluation_per_label": 1,
}
ALIASES = {"audio_file_name": "file", "speaker_id": "speaker", "key": LABEL_MAPPING}


@dataclass(frozen=True)
class PreprocessConfig:
    sample_rate: int = TARGET_SR
    rms_dbfs: float = -20.0
    peak_ceiling: float = 0.99
    trim_silence: bool = False
    silence_db: float = 40.0
    frame_seconds: float = 0.02
    minimum_trim_seconds: float = 0.25
    version: str = "iwd-standardized-v1"

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


@dataclass(frozen=True)
class SplitConfig:
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15)
    seed: int = SEED
    candidate_starts: int = 64
    max_improving_steps: int = 256
    minimum_speakers: int = 2
    max_clip_share_deviation: float = 0.02
    max_spoof_share_deviation: float = 0.05
    version: str = "iwd-whole-speaker-search-v1"

    def __post_init__(self):
        if len(self.ratios) != 3 or not all(
            math.isfinite(x) and x > 0 for x in self.ratios
        ):
            raise ValueError("Three finite positive split ratios are required.")
        if not math.isclose(sum(self.ratios), 1.0, abs_tol=1e-12):
            raise ValueError("Split ratios must sum to one.")
        for name in (
            "seed",
            "candidate_starts",
            "max_improving_steps",
            "minimum_speakers",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer.")
        if (
            self.candidate_starts < 1
            or self.max_improving_steps < 1
            or self.minimum_speakers < 2
        ):
            raise ValueError(
                "Search budgets must be positive; at least two speakers per split."
            )
        for value in (self.max_clip_share_deviation, self.max_spoof_share_deviation):
            if not math.isfinite(value) or not 0 <= value < 1:
                raise ValueError("Acceptance deviations must lie in [0, 1).")
        if not self.version:
            raise ValueError(
                "Declare a split-algorithm version for behavioral changes."
            )


PREPROCESS = PreprocessConfig()
SPLIT_POLICY = SplitConfig()


# %% [markdown]
# ## 1. Deterministic standardization (same defaults as ASV)
# Public `Audio(...).decode_example` uses TorchCodec's mono conversion and lazy
# resampling before standardization. Optional edge trimming is disabled, retains
# partial frames, and keeps the original if less than 0.25 s would remain.
# RMS targets -20 dBFS with the same peak/PCM16-quantization guard as ASV.
# High-crest-factor clips can remain quieter; silence has nullable RMS metadata.
# There is no fixed crop, class resampling, or short/silent-row dropping.


# %% Standardization helpers.
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
            "Expected a datasets/TorchCodec AudioDecoder; check installed versions."
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
    t = np.arange(8000) / 8000
    signal = 0.05 * np.sin(2 * np.pi * 220 * t)
    buffer = io.BytesIO()
    sf.write(
        buffer,
        np.column_stack((signal, signal * 0.5)),
        8000,
        format="WAV",
        subtype="PCM_16",
    )
    payload = {"bytes": buffer.getvalue(), "path": None}
    feature = Audio(sampling_rate=TARGET_SR, num_channels=1)
    fixture = Dataset.from_dict(
        {"audio": [payload]}, features=Features({"audio": feature})
    )
    try:
        decoded = decode_hf_audio(feature.decode_example(payload))
        np.testing.assert_array_equal(decoded, decode_hf_audio(fixture[0]["audio"]))
    except Exception as exc:
        raise RuntimeError(
            "Actual TorchCodec decoding failed. Check PyTorch/TorchCodec, FFmpeg shared "
            "libraries, and restart after package changes. No alternate decoder was substituted."
        ) from exc
    assert decoded.shape == (TARGET_SR,) and np.any(decoded)
    print("PASS: public Audio.decode_example matches Dataset stereo resampling")


def run_standardization_checks():
    config = PreprocessConfig()
    t = np.arange(TARGET_SR, dtype=np.float64) / TARGET_SR
    sine = (0.03 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    result = standardize_waveform(sine, TARGET_SR, config)
    audio = read_pcm16(result["audio"]["bytes"])
    assert audio.dtype == np.float32 and audio.ndim == 1 and len(audio) == TARGET_SR
    assert abs(result["rms_dbfs"] + 20) < 0.1 and result["peak"] <= 0.99
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
    assert standardize_waveform(sine[::2], 8000, config)["num_samples"] == TARGET_SR
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
    changed = standardize_waveform(
        untrimmed, TARGET_SR, replace(config, trim_silence=True, rms_dbfs=-18)
    )
    assert changed["num_samples"] == TARGET_SR and abs(changed["rms_dbfs"] + 18) < 0.1
    for invalid in (np.array([], np.float32), np.array([np.nan], np.float32)):
        try:
            standardize_waveform(invalid, TARGET_SR, config)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid audio must not be silently retained.")
    print(
        "PASS: format, deterministic bytes, loudness, quantization, silence, trimming, invalid input"
    )


# %% [markdown]
# ## 2. Safe ZIP inventory and original metadata
# The pinned 8,161,489,023-byte ZIP64 file is downloaded with `hf_hub_download`
# to a local cache, then verified by full SHA256. The cache is read-only here:
# no extraction, `extractall`, corpus modification, or raw-audio Arrow duplicate.
# Central-directory checks reject unsafe/duplicate names, symlinks, unexpected
# entries, and a mismatch between WAV basenames and CSV files. CRC32/size groups
# are only candidate **archive-payload** fingerprints; candidate members are
# SHA256-compared uncompressed. Confirmed cross-label/cross-split byte duplicates
# stop execution for explicit resolution. Rows are never silently deduplicated.


# %% Metadata and checksum helpers (also usable in offline synthetic checks).
def json_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def stream_sha256(handle):
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path):
    with Path(path).open("rb") as handle:
        return stream_sha256(handle)


def safe_relative_name(name):
    if (
        not isinstance(name, str)
        or not name
        or "\\" in name
        or "\x00" in name
        or any(ord(char) < 32 for char in name)
    ):
        raise ValueError(f"Unsafe archive/artifact path: {name!r}")
    parts = name.rstrip("/").split("/")
    if any(part in ("", ".", "..") or ":" in part for part in parts):
        raise ValueError(f"Unsafe archive/artifact path: {name!r}")
    if PurePosixPath(name).is_absolute():
        raise ValueError(f"Absolute path rejected: {name!r}")
    return name


def canonical_metadata(records, expected_source=SOURCE):
    """Validate without normalization/dropping; exact original strings survive."""
    canonical, seen = [], set()
    for row in records:
        if set(row) != {"file", "speaker", "label"}:
            raise ValueError("Metadata must contain exactly file,speaker,label.")
        for key, value in row.items():
            if (
                not isinstance(value, str)
                or not value
                or not value.strip()
                or "\x00" in value
            ):
                raise ValueError(f"Missing/invalid original {key}: {value!r}")
            if value != value.strip():
                raise ValueError(
                    f"Unexpected edge whitespace in {key}; resolve explicitly."
                )
        filename = safe_relative_name(row["file"])
        if "/" in filename or not filename.endswith(".wav"):
            raise ValueError(f"Expected an exact flat WAV basename: {filename!r}")
        if filename in seen:
            raise ValueError(f"Duplicate/conflicting metadata for {filename!r}.")
        if row["label"] not in LABEL_MAPPING:
            raise ValueError(f"Unknown original label: {row['label']!r}")
        seen.add(filename)
        canonical.append(dict(row))
    canonical.sort(key=lambda row: row["file"])
    if not canonical:
        raise ValueError("Empty metadata.")
    counts = Counter(row["label"] for row in canonical)
    speakers = {}
    for row in canonical:
        speakers.setdefault(row["speaker"], set()).add(row["label"])
    if any(labels != set(LABEL_MAPPING) for labels in speakers.values()):
        raise ValueError(
            "Every observed speaker must have both original labels; investigate."
        )
    if expected_source is not None and (
        len(canonical) != expected_source["rows"]
        or dict(counts) != expected_source["class_counts"]
        or len(speakers) != expected_source["speakers"]
    ):
        raise ValueError(
            f"Metadata differs from expected source: rows={len(canonical)}, "
            f"labels={dict(counts)}, speakers={len(speakers)}."
        )
    return canonical


def parse_metadata_csv(csv_text, expected_source=SOURCE):
    reader = csv.DictReader(
        io.StringIO(csv_text.lstrip("\ufeff"), newline=""), strict=True
    )
    if reader.fieldnames != ["file", "speaker", "label"]:
        raise ValueError(f"Unexpected CSV fields: {reader.fieldnames!r}")
    return canonical_metadata(list(reader), expected_source)


def verify_source_archive(path, expected_source=SOURCE):
    path = Path(path)
    if path.stat().st_size != expected_source["archive_size"]:
        raise ValueError("Wrong pinned ZIP size; retain the cache and investigate.")
    if file_sha256(path) != expected_source["archive_sha256"]:
        raise ValueError("Wrong pinned ZIP SHA256; retain the cache and investigate.")


def inspect_archive(path, expected_source=SOURCE):
    """Central directory/CSV audit; only duplicate candidates require WAV reads."""
    verify_source_archive(path, expected_source)
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names, basenames, wav_members, candidates = set(), set(), {}, {}
        wav_size = 0
        allowed = {
            f"{ARCHIVE_ROOT}/",
            f"{ARCHIVE_ROOT}/meta.csv",
            f"{ARCHIVE_ROOT}/attribution.txt",
        }
        for info in infos:
            if info.orig_filename != info.filename:
                raise ValueError("Truncated/NUL-containing ZIP member name rejected.")
            name = safe_relative_name(info.filename)
            if name in names:
                raise ValueError(f"Duplicate ZIP member: {name}")
            names.add(name)
            if stat.S_ISLNK(info.external_attr >> 16) or info.flag_bits & 1:
                raise ValueError(f"Symlink/encrypted member rejected: {name}")
            if name in allowed:
                if info.is_dir() != name.endswith("/"):
                    raise ValueError(f"Invalid directory entry: {name}")
                continue
            parts = PurePosixPath(name).parts
            if (
                info.is_dir()
                or len(parts) != 2
                or parts[0] != ARCHIVE_ROOT
                or not parts[1].endswith(".wav")
                or info.file_size <= 0
            ):
                raise ValueError(f"Unexpected/unlabelled/unsafe archive entry: {name}")
            basename = parts[1]
            if basename in basenames:
                raise ValueError(f"Duplicate WAV basename: {basename}")
            basenames.add(basename)
            wav_members[basename] = name
            wav_size += info.file_size
            candidates.setdefault(f"{info.CRC:08x}:{info.file_size}", []).append(
                basename
            )
        if not allowed.issubset(names) or len(infos) != expected_source["entries"]:
            raise ValueError(
                "Archive central-directory layout/count differs from the pinned source."
            )
        if wav_size != expected_source["wav_uncompressed_bytes"]:
            raise ValueError("Unexpected total uncompressed WAV bytes.")
        meta_info = archive.getinfo(f"{ARCHIVE_ROOT}/meta.csv")
        if meta_info.file_size > 16 * 1024 * 1024:
            raise ValueError("Unexpectedly large metadata CSV.")
        csv_bytes = archive.read(meta_info)
        if (
            hashlib.sha256(csv_bytes).hexdigest()
            != expected_source["metadata_csv_sha256"]
        ):
            raise ValueError("Pinned source metadata CSV SHA256 mismatch.")
        csv_text = csv_bytes.decode("utf-8")
        records = parse_metadata_csv(csv_text, expected_source)
        attribution_info = archive.getinfo(f"{ARCHIVE_ROOT}/attribution.txt")
        if attribution_info.file_size > 64 * 1024:
            raise ValueError("Unexpectedly large release attribution.")
        attribution_bytes = archive.read(attribution_info)
        attribution_text = attribution_bytes.decode("utf-8")
        if not attribution_text.strip():
            raise ValueError("Missing release attribution text.")
        if set(wav_members) != {row["file"] for row in records}:
            raise ValueError(
                "Exact basename join failed: missing/unlabelled/unexpected WAVs."
            )
        candidate_groups = {
            key: sorted(files)
            for key, files in sorted(candidates.items())
            if len(files) > 1
        }
        confirmed, candidate_sha256 = [], {}
        for files in candidate_groups.values():
            by_sha256 = {}
            for filename in files:
                with archive.open(wav_members[filename]) as handle:
                    digest = stream_sha256(handle)
                candidate_sha256[filename] = digest
                by_sha256.setdefault(digest, []).append(filename)
            for digest, matches in sorted(by_sha256.items()):
                if len(matches) > 1:
                    confirmed.append({"sha256": digest, "files": sorted(matches)})
    audit = {
        "entries": len(infos),
        "wav_count": len(wav_members),
        "wav_uncompressed_bytes": expected_source["wav_uncompressed_bytes"],
        "payload_candidate_groups_crc32_size": candidate_groups,
        "payload_candidate_member_sha256": candidate_sha256,
        "confirmed_byte_duplicate_groups": sorted(
            confirmed, key=lambda row: row["files"]
        ),
    }
    print(
        json.dumps(
            {
                "archive_audit": audit,
                "observed_speakers": expected_source["speakers"],
                "card_speaker_count": 58 if expected_source == SOURCE else None,
                "duplicate_scope": "archive payload only; not acoustic/source-video deduplication",
            },
            indent=2,
        )
    )
    by_file = {row["file"]: row for row in records}
    for group in confirmed:
        if len({by_file[name]["label"] for name in group["files"]}) > 1:
            raise ValueError(
                f"Confirmed cross-label byte duplicates; resolve explicitly, preserving rows: {group}"
            )
    return {
        "records": records,
        "csv_text": csv_text,
        "source_metadata_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "release_attribution": attribution_text,
        "release_attribution_sha256": hashlib.sha256(attribution_bytes).hexdigest(),
        "archive_audit": audit,
    }


# %% [markdown]
# ## 3. Freeze a project-defined, whole-speaker partition
# Sorted original metadata and stable speaker names feed 64 seeded candidate
# starts. Greedy starts are refined with at most 256 strictly improving
# whole-speaker moves/swaps each. The declared objective is the sum, across
# splits and total/bonafide/spoof counts, of squared deviations from target
# **shares**, divided by the target split ratio. No exact speaker quotas apply.
# Stable operation order and lexicographic assignment tie-breaking are used.
# Acceptable candidate endpoints are preferred; their declared objective chooses
# the best one. If none passes the gates, diagnostics describe the best endpoint.
# All search budgets, ratios, seed, version and acceptance gates are fingerprinted.
# Each split needs both labels and at least two speakers. Default gates are
# max absolute clip-share deviation 0.02 and spoof-prevalence deviation 0.05.
# Failure stops with diagnostics; there is no relaxed or clip-level fallback.
# `split_assignments.json` freezes the full partition even in a smoke artifact.
# It retains the original CSV/checksum, canonical original rows, exact aliases,
# speaker counts, split summary, duplicate audit and immutable assignment digest.


# %% Seeded bounded speaker search — metadata only.
def speaker_count_matrix(records):
    names = sorted({row["speaker"] for row in records})
    index = {speaker: i for i, speaker in enumerate(names)}
    counts = np.zeros((len(names), 3), dtype=np.int64)
    for row in records:
        i = index[row["speaker"]]
        counts[i, 0] += 1
        counts[i, LABEL_MAPPING[row["label"]] + 1] += 1
    return names, counts


def split_summary(records, mapping, config):
    names, speaker_counts = speaker_count_matrix(records)
    if set(mapping) != set(names) or set(mapping.values()) != set(SPLIT_NAMES):
        raise ValueError("Speaker assignment is incomplete or has unexpected splits.")
    totals = speaker_counts.sum(axis=0)
    global_spoof = float(totals[2] / totals[0])
    summary = {
        "counts": {},
        "class_counts": {},
        "speaker_counts": {},
        "clip_shares": {},
        "spoof_shares": {},
        "clip_share_deviations": {},
        "spoof_share_deviations": {},
        "global_spoof_share": global_spoof,
        "per_speaker": {},
        "coverage": len(records),
        "speaker_overlap": 0,
        "source_video_disjointness": "unproven; source IDs absent",
    }
    for speaker, vector in zip(names, speaker_counts):
        summary["per_speaker"][speaker] = {
            "split": mapping[speaker],
            "total": int(vector[0]),
            "bonafide": int(vector[1]),
            "spoof": int(vector[2]),
        }
    for i, split in enumerate(SPLIT_NAMES):
        members = [j for j, speaker in enumerate(names) if mapping[speaker] == split]
        vector = speaker_counts[members].sum(axis=0)
        if len(members) < config.minimum_speakers or min(vector) <= 0:
            raise ValueError(
                f"{split}: need both labels and >= {config.minimum_speakers} speakers."
            )
        share, spoof = float(vector[0] / totals[0]), float(vector[2] / vector[0])
        summary["counts"][split] = int(vector[0])
        summary["class_counts"][split] = {
            "bonafide": int(vector[1]),
            "spoof": int(vector[2]),
        }
        summary["speaker_counts"][split] = len(members)
        summary["clip_shares"][split] = share
        summary["spoof_shares"][split] = spoof
        summary["clip_share_deviations"][split] = abs(share - config.ratios[i])
        summary["spoof_share_deviations"][split] = abs(spoof - global_spoof)
    return summary


def enforce_split_gates(summary, config):
    if (
        max(summary["clip_share_deviations"].values())
        > config.max_clip_share_deviation + 1e-12
        or max(summary["spoof_share_deviations"].values())
        > config.max_spoof_share_deviation + 1e-12
    ):
        raise ValueError(
            "No acceptable frozen whole-speaker split. Never silently relax gates. "
            "Review diagnostics/config explicitly:\n" + json.dumps(summary, indent=2)
        )


def assign_speakers(records, config=SPLIT_POLICY):
    records = canonical_metadata(records, expected_source=None)
    speakers, vectors = speaker_count_matrix(records)
    n = len(speakers)
    if n < 3 * config.minimum_speakers:
        raise ValueError(
            "Too few speakers for three disjoint splits with the declared minimum."
        )
    totals = vectors.sum(axis=0).astype(np.float64)
    ratios = np.asarray(config.ratios, dtype=np.float64)
    targets = ratios[:, None] * totals[None, :]
    weights = 1.0 / (ratios[:, None] * totals[None, :] ** 2)
    rng = random.Random(config.seed)
    best = None
    best_key = None
    for _start in range(config.candidate_starts):
        # Randomized priorities, sorted stable inputs, no dependence on CSV row order.
        priorities = [int(vectors[i, 0]) * rng.uniform(0.65, 1.35) for i in range(n)]
        order = sorted(range(n), key=lambda i: (-priorities[i], speakers[i]))
        assignment = np.full(n, -1, dtype=np.int64)
        counts = np.zeros((3, 3), dtype=np.int64)
        speaker_counts = np.zeros(3, dtype=np.int64)
        for i in order:
            scores = []
            for target in range(3):
                proposed = counts.copy()
                proposed[target] += vectors[i]
                scores.append(float(np.sum((proposed - targets) ** 2 * weights)))
            target = min(range(3), key=lambda split: (scores[split], split))
            assignment[i] = target
            counts[target] += vectors[i]
            speaker_counts[target] += 1
        # Repair only speaker minima, never changing rows or relaxing the policy.
        for target in range(3):
            while speaker_counts[target] < config.minimum_speakers:
                choices = []
                for i in range(n):
                    donor = int(assignment[i])
                    if (
                        donor == target
                        or speaker_counts[donor] <= config.minimum_speakers
                    ):
                        continue
                    proposed = counts.copy()
                    proposed[donor] -= vectors[i]
                    proposed[target] += vectors[i]
                    choices.append(
                        (float(np.sum((proposed - targets) ** 2 * weights)), i)
                    )
                if not choices:
                    raise ValueError("Cannot satisfy speaker minima.")
                _, i = min(choices)
                donor = int(assignment[i])
                counts[donor] -= vectors[i]
                counts[target] += vectors[i]
                speaker_counts[donor] -= 1
                speaker_counts[target] += 1
                assignment[i] = target
        for _step in range(config.max_improving_steps):
            operations, src, dst, change = [], [], [], []
            for i in range(n):
                a = int(assignment[i])
                if speaker_counts[a] > config.minimum_speakers:
                    for b in range(3):
                        if a != b:
                            operations.append((i, -1, b))
                            src.append(a)
                            dst.append(b)
                            change.append(-vectors[i])
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = int(assignment[i]), int(assignment[j])
                    if a != b:
                        operations.append((i, j, b))
                        src.append(a)
                        dst.append(b)
                        change.append(vectors[j] - vectors[i])
            if not operations:
                break
            a, b, delta = np.asarray(src), np.asarray(dst), np.asarray(change)
            residual = counts - targets
            improvements = np.sum(
                (2 * residual[a] * delta + delta**2) * weights[a]
                + (-2 * residual[b] * delta + delta**2) * weights[b],
                axis=1,
            )
            selected = int(np.argmin(improvements))  # stable first operation on ties
            if improvements[selected] >= -1e-12:
                break
            i, j, target = operations[selected]
            donor = int(assignment[i])
            counts[donor] += delta[selected]
            counts[target] -= delta[selected]
            assignment[i] = target
            if j < 0:
                speaker_counts[donor] -= 1
                speaker_counts[target] += 1
            else:
                assignment[j] = donor
        score = float(np.sum((counts - targets) ** 2 * weights))
        clip_error = float(np.max(np.abs(counts[:, 0] / totals[0] - ratios)))
        spoof_error = float(
            np.max(np.abs(counts[:, 2] / counts[:, 0] - totals[2] / totals[0]))
        )
        acceptable = (
            clip_error <= config.max_clip_share_deviation + 1e-12
            and spoof_error <= config.max_spoof_share_deviation + 1e-12
        )
        key = (not acceptable, score, tuple(int(value) for value in assignment))
        if best_key is None or key < best_key:
            best_key, best = key, assignment.copy()
    mapping = {speaker: SPLIT_NAMES[int(best[i])] for i, speaker in enumerate(speakers)}
    summary = split_summary(records, mapping, config)
    enforce_split_gates(summary, config)
    return (
        mapping,
        summary,
        {
            "objective": best_key[1],
            "candidate_starts": config.candidate_starts,
            "max_improving_steps_per_start": config.max_improving_steps,
        },
    )


def audit_duplicate_payloads(archive_audit, records, mapping=None):
    by_file = {row["file"]: row for row in records}
    candidates = archive_audit["payload_candidate_groups_crc32_size"]
    seen_candidates = set()
    for key, files in candidates.items():
        crc, size = key.split(":")
        if len(crc) != 8 or int(crc, 16) < 0 or int(size) <= 0 or len(files) < 2:
            raise ValueError("Invalid CRC32/size candidate audit.")
        if (
            files != sorted(set(files))
            or not set(files).issubset(by_file)
            or seen_candidates & set(files)
        ):
            raise ValueError("Invalid/overlapping duplicate candidates.")
        seen_candidates.update(files)
    member_sha256 = archive_audit["payload_candidate_member_sha256"]
    if set(member_sha256) != seen_candidates or any(
        len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest)
        for digest in member_sha256.values()
    ):
        raise ValueError("Candidate member SHA256 evidence is missing/invalid.")
    expected_confirmed = []
    for files in candidates.values():
        by_digest = {}
        for filename in files:
            by_digest.setdefault(member_sha256[filename], []).append(filename)
        expected_confirmed.extend(
            {"sha256": digest, "files": sorted(matches)}
            for digest, matches in by_digest.items()
            if len(matches) > 1
        )
    if (
        sorted(expected_confirmed, key=lambda row: row["files"])
        != archive_audit["confirmed_byte_duplicate_groups"]
    ):
        raise ValueError(
            "Confirmed duplicate groups differ from candidate SHA256 evidence."
        )
    confirmed_seen = set()
    for group in archive_audit["confirmed_byte_duplicate_groups"]:
        files, digest = group["files"], group["sha256"]
        if (
            len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            or len(files) < 2
            or files != sorted(set(files))
            or not any(
                set(files).issubset(group_files) for group_files in candidates.values()
            )
            or confirmed_seen & set(files)
        ):
            raise ValueError("Invalid confirmed-byte-duplicate audit.")
        confirmed_seen.update(files)
        labels = {by_file[name]["label"] for name in files}
        splits = (
            {mapping[by_file[name]["speaker"]] for name in files}
            if mapping is not None
            else set()
        )
        if len(labels) > 1 or len(splits) > 1:
            raise ValueError(
                f"Confirmed cross-label/cross-split byte duplicates; manual resolution "
                f"required without silent row changes: {group}"
            )


def freeze_protocol(inspection, split_config=SPLIT_POLICY, expected_source=SOURCE):
    records = canonical_metadata(inspection["records"], expected_source)
    mapping, summary, search = assign_speakers(records, split_config)
    protocol = {
        "version": "iwd-split-sidecar-v1",
        "source": dict(expected_source),
        "source_metadata_csv": inspection["csv_text"],
        "source_metadata_sha256": inspection["source_metadata_sha256"],
        "release_attribution": inspection["release_attribution"],
        "release_attribution_sha256": inspection["release_attribution_sha256"],
        "license_provenance": SOURCE_LICENSE_PROVENANCE
        if expected_source == SOURCE
        else {"status": "Synthetic fixture; no corpus license assumed."},
        "canonical_metadata": records,
        "canonical_metadata_sha256": json_digest(records),
        "aliases": ALIASES,
        "split_config": asdict(split_config),
        "speaker_to_split": mapping,
        "summary": summary,
        "search": search,
        "archive_audit": inspection["archive_audit"],
    }
    protocol["assignment_digest"] = json_digest(protocol)
    audit_protocol(protocol, split_config, expected_source)
    return protocol


def audit_protocol(protocol, split_config=SPLIT_POLICY, expected_source=SOURCE):
    # Canonicalize tuple/list representation through JSON for portable config comparisons.
    if (
        protocol.get("version") != "iwd-split-sidecar-v1"
        or protocol.get("source") != expected_source
        or json_digest(protocol.get("split_config"))
        != json_digest(asdict(split_config))
        or protocol.get("aliases") != ALIASES
        or protocol.get("license_provenance")
        != (
            SOURCE_LICENSE_PROVENANCE
            if expected_source == SOURCE
            else {"status": "Synthetic fixture; no corpus license assumed."}
        )
    ):
        raise ValueError(
            "Incompatible source, aliases, licensing, or split-policy sidecar."
        )
    body = {key: value for key, value in protocol.items() if key != "assignment_digest"}
    if protocol.get("assignment_digest") != json_digest(body):
        raise ValueError("Frozen assignment integrity digest mismatch.")
    attribution = protocol.get("release_attribution")
    if (
        not isinstance(attribution, str)
        or not attribution.strip()
        or hashlib.sha256(attribution.encode("utf-8")).hexdigest()
        != protocol.get("release_attribution_sha256")
    ):
        raise ValueError("Release attribution is missing or its checksum differs.")
    csv_text = protocol["source_metadata_csv"]
    if (
        hashlib.sha256(csv_text.encode("utf-8")).hexdigest()
        != protocol["source_metadata_sha256"]
        or protocol["source_metadata_sha256"] != expected_source["metadata_csv_sha256"]
    ):
        raise ValueError("Original metadata CSV checksum mismatch.")
    records = parse_metadata_csv(csv_text, expected_source)
    if (
        records != protocol["canonical_metadata"]
        or json_digest(records) != protocol["canonical_metadata_sha256"]
    ):
        raise ValueError("Original-to-canonical metadata mismatch.")
    summary = split_summary(records, protocol["speaker_to_split"], split_config)
    if summary != protocol["summary"]:
        raise ValueError(
            "Frozen split summary differs from actual original metadata/assignment."
        )
    enforce_split_gates(summary, split_config)
    archive_audit = protocol["archive_audit"]
    if (
        archive_audit["entries"] != expected_source["entries"]
        or archive_audit["wav_count"] != expected_source["rows"]
        or archive_audit["wav_uncompressed_bytes"]
        != expected_source["wav_uncompressed_bytes"]
    ):
        raise ValueError("Archive inventory does not match expected provenance.")
    audit_duplicate_payloads(archive_audit, records, protocol["speaker_to_split"])
    return summary


def selected_records(protocol, kind="full"):
    if kind not in ("full", "smoke"):
        raise ValueError("Artifact kind must be full or smoke.")
    result = {name: [] for name in SPLIT_NAMES}
    for row in protocol["canonical_metadata"]:
        result[protocol["speaker_to_split"][row["speaker"]]].append(dict(row))
    if kind == "smoke":
        for split_id, name in enumerate(SPLIT_NAMES):
            rows = result[name]
            rng = np.random.default_rng(SMOKE_POLICY["seed"] + split_id)
            indices = []
            per_label = SMOKE_POLICY[
                "train_per_label" if name == "train" else "evaluation_per_label"
            ]
            for label in LABEL_MAPPING:
                eligible = [i for i, row in enumerate(rows) if row["label"] == label]
                if len(eligible) < per_label:
                    raise ValueError(
                        f"{name}: insufficient {label} rows for smoke selection."
                    )
                indices.extend(rng.choice(eligible, per_label, replace=False).tolist())
            result[name] = [rows[i] for i in sorted(indices)]
    return result


def lookup_fingerprint(
    config=PREPROCESS, split_config=SPLIT_POLICY, expected_source=SOURCE
):
    """Config-only directory lookup permits verified reuse before raw ZIP access."""
    return json_digest(
        {
            "source": expected_source,
            "preprocess": asdict(config),
            "split": asdict(split_config),
            "aliases": ALIASES,
            "smoke_policy": SMOKE_POLICY,
            "artifact_version": "iwd-artifact-v1",
        }
    )


def preprocess_fingerprint(
    config, protocol, split_config=SPLIT_POLICY, expected_source=SOURCE
):
    return json_digest(
        {
            "lookup": lookup_fingerprint(config, split_config, expected_source),
            "assignment_digest": protocol["assignment_digest"],
        }
    )


# %% Direct ZIP -> standardized Arrow: one archive handle per split, one WAV at a time.
def standardized_features():
    return Features(
        {
            "file": Value("string"),
            "speaker": Value("string"),
            "label": Value("string"),
            "audio_file_name": Value("string"),
            "speaker_id": Value("string"),
            "key": ClassLabel(names=LABEL_NAMES),
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


def standardized_archive_generator(archive_path, records, config):
    config = PreprocessConfig(**config)
    feature = Audio(sampling_rate=TARGET_SR, num_channels=1)
    with zipfile.ZipFile(archive_path) as archive:
        for original in records:
            filename = original["file"]
            try:
                payload = archive.read(f"{ARCHIVE_ROOT}/{filename}")
                decoded = feature.decode_example({"bytes": payload, "path": None})
                standardized = standardize_waveform(
                    decode_hf_audio(decoded), TARGET_SR, config
                )
                yield {
                    **original,
                    "audio_file_name": filename,
                    "speaker_id": original["speaker"],
                    "key": LABEL_MAPPING[original["label"]],
                    **standardized,
                }
            except Exception as exc:
                raise ValueError(f"Preprocessing ZIP member {filename}: {exc}") from exc


def map_standardized(
    archive_path,
    protocol,
    config,
    cache_dir,
    kind="full",
    split_config=SPLIT_POLICY,
    expected_source=SOURCE,
):
    audit_protocol(protocol, split_config, expected_source)
    selection = selected_records(protocol, kind)
    result = {}
    for name, records in selection.items():
        cache_key = json_digest(
            {
                "config": preprocess_fingerprint(
                    config, protocol, split_config, expected_source
                ),
                "kind": kind,
                "split": name,
                "metadata": json_digest(records),
                "versions": package_versions(),
                "generator_version": "iwd-zip-arrow-v1",
            }
        )
        # fingerprint is public in pinned datasets 4.4.1 (introduced in 4.3.0).
        # writer_batch_size is forwarded to GeneratorBasedBuilder, not an invented map API.
        result[name] = Dataset.from_generator(
            standardized_archive_generator,
            features=standardized_features(),
            cache_dir=str(Path(cache_dir) / name),
            keep_in_memory=False,
            gen_kwargs={
                "archive_path": str(archive_path),
                "records": tuple(records),
                "config": asdict(config),
            },
            fingerprint=cache_key,
            split=name,
            num_proc=None,
            writer_batch_size=WRITER_BATCH_SIZE,
        )
    processed = DatasetDict(result)
    audit_standardized(processed, config, protocol, kind, split_config, expected_source)
    return processed


# %% Metadata and portable artifact audits — counts derive from the frozen assignment.
def metadata_digest(split):
    digest = hashlib.sha256()
    for row in split.select_columns(
        ["file", "speaker", "label", "audio_file_name", "speaker_id", "key"]
    ):
        digest.update(json.dumps(row, sort_keys=True).encode() + b"\n")
    return digest.hexdigest()


def audit_metadata(
    data, protocol, kind="full", split_config=SPLIT_POLICY, expected_source=SOURCE
):
    audit_protocol(protocol, split_config, expected_source)
    planned = selected_records(protocol, kind)
    if set(data) != set(SPLIT_NAMES):
        raise ValueError("Expected exactly train/validation/test.")
    summary = {
        "kind": kind,
        "complete_source_coverage": kind == "full",
        "counts": {},
        "class_counts": {},
        "speaker_counts": {},
        "metadata_sha256": {},
        "overlaps": {},
    }
    identifiers, speakers = {}, {}
    for name in SPLIT_NAMES:
        split = data[name]
        if len(split) != len(planned[name]) or not len(split):
            raise ValueError(
                f"{name}: wrong/empty derived row count, or smoke/full substitution."
            )
        required = {
            "file",
            "speaker",
            "label",
            "audio_file_name",
            "speaker_id",
            "key",
            "audio",
        }
        if (
            not required.issubset(split.column_names)
            or "system_id" in split.column_names
        ):
            raise ValueError(
                f"{name}: missing original metadata/aliases or fabricated system_id."
            )
        feature = split.features["key"]
        if not isinstance(feature, ClassLabel) or feature.names != LABEL_NAMES:
            raise ValueError("Expected ClassLabel names ['bonafide', 'spoof'].")
        actual = split.select_columns(
            ["file", "speaker", "label", "audio_file_name", "speaker_id", "key"]
        )
        identifiers[name], speakers[name] = set(), set()
        labels = Counter()
        for row, original in zip(actual, planned[name]):
            expected = {
                **original,
                "audio_file_name": original["file"],
                "speaker_id": original["speaker"],
                "key": LABEL_MAPPING[original["label"]],
            }
            if (
                row != expected
                or protocol["speaker_to_split"].get(row["speaker"]) != name
            ):
                raise ValueError(
                    f"{name}: original metadata/label/alias/assignment/order mismatch: {row}"
                )
            if row["audio_file_name"] in identifiers[name]:
                raise ValueError(
                    f"{name}: duplicated audio ID; never silently drop rows."
                )
            identifiers[name].add(row["audio_file_name"])
            speakers[name].add(row["speaker_id"])
            labels[row["key"]] += 1
        if set(labels) != {0, 1}:
            raise ValueError(f"{name}: both labels required.")
        summary["counts"][name] = len(split)
        summary["class_counts"][name] = {
            label: labels[i] for i, label in enumerate(LABEL_NAMES)
        }
        summary["speaker_counts"][name] = len(speakers[name])
        summary["metadata_sha256"][name] = metadata_digest(split)
    for a, b in itertools.combinations(SPLIT_NAMES, 2):
        overlap = {
            "utterance_ids": len(identifiers[a] & identifiers[b]),
            "speakers": len(speakers[a] & speakers[b]),
        }
        if any(overlap.values()):
            raise ValueError(f"Cross-split speaker/ID leakage: {a}/{b}: {overlap}")
        summary["overlaps"][f"{a}/{b}"] = overlap
    if (
        kind == "full"
        and len(set.union(*identifiers.values())) != expected_source["rows"]
    ):
        raise ValueError("Full source coverage failed.")
    return summary


def audit_standardized(
    data,
    config,
    protocol,
    kind="full",
    split_config=SPLIT_POLICY,
    expected_source=SOURCE,
):
    summary = audit_metadata(data, protocol, kind, split_config, expected_source)
    columns = [
        "num_samples",
        "duration_s",
        "peak",
        "rms_dbfs",
        "is_silent",
        "peak_limited",
        "preprocess_version",
    ]
    for name in SPLIT_NAMES:
        split = data[name]
        for row in split.select_columns(columns):
            if row["num_samples"] <= 0 or not np.isclose(
                row["duration_s"], row["num_samples"] / TARGET_SR, rtol=1e-6
            ):
                raise ValueError(f"{name}: invalid length/duration metadata.")
            if (
                not 0 <= row["peak"] <= config.peak_ceiling
                or row["preprocess_version"] != config.version
            ):
                raise ValueError(f"{name}: invalid peak/version metadata.")
            if (
                row["is_silent"] != (row["rms_dbfs"] is None)
                or (row["rms_dbfs"] is not None and not math.isfinite(row["rms_dbfs"]))
                or not isinstance(row["peak_limited"], bool)
            ):
                raise ValueError(f"{name}: invalid silence/RMS/peak-limit metadata.")
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
                    f"{name}: embedded WAV bytes required, no external paths."
                )
            waveform = read_pcm16(payload["bytes"])
            _, dbfs = rms_level(waveform)
            if len(waveform) != row["num_samples"] or not np.isclose(
                float(np.abs(waveform).max()), row["peak"], atol=1e-7
            ):
                raise ValueError(f"{name}: audio/length/peak metadata mismatch.")
            if (dbfs is None) != row["is_silent"] or (
                dbfs is not None and not np.isclose(dbfs, row["rms_dbfs"], atol=1e-4)
            ):
                raise ValueError(f"{name}: audio/RMS mismatch.")
            # Reload through the actual model decoder as well as the independent PCM audit.
            redecoded = decode_hf_audio(data[name][int(index)]["audio"])
            np.testing.assert_allclose(redecoded, waveform, rtol=0, atol=PCM16_STEP)
    return summary


def artifact_files(root):
    root = Path(root)
    if root.is_symlink():
        raise ValueError("Artifact roots must not be symlinks.")
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Portable artifacts must not contain symlinks.")
        if path.is_file() and path != root / "manifest.json":
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
    files = manifest.get("files")
    if not files:
        raise ValueError("Missing artifact checksum inventory.")
    for relative in files:
        safe_relative_name(relative)
        if relative.endswith("/") or relative in ("manifest.json", ".manifest.pending"):
            raise ValueError("Invalid checksum inventory path.")
    if artifact_files(root) != files or "split_assignments.json" not in files:
        raise ValueError(
            f"{root}: missing/extra/corrupted files; choose a new RUN_TAG."
        )


def load_protocol(path, split_config=SPLIT_POLICY, expected_source=SOURCE):
    protocol = json.loads((Path(path) / "split_assignments.json").read_text())
    audit_protocol(protocol, split_config, expected_source)
    return protocol


def validate_reloaded(path, manifest, config, split_config, expected_source):
    protocol = load_protocol(path, split_config, expected_source)
    if (
        manifest.get("fingerprint")
        != preprocess_fingerprint(config, protocol, split_config, expected_source)
        or manifest.get("assignment_digest") != protocol["assignment_digest"]
    ):
        raise ValueError("Manifest/protocol fingerprint mismatch.")
    data = load_from_disk(str(path))
    if (
        audit_standardized(
            data, config, protocol, manifest["kind"], split_config, expected_source
        )
        != manifest["metadata"]
    ):
        raise ValueError(
            "Reloaded metadata/audio audit differs from completion marker."
        )
    return data


def load_artifact(
    path,
    config=PREPROCESS,
    split_config=SPLIT_POLICY,
    kind="full",
    expected_source=SOURCE,
):
    path = Path(path)
    marker = path / "manifest.json"
    if not marker.is_file() or marker.is_symlink() or path.is_symlink():
        raise ValueError(
            f"{path}: no safe completion marker; never reuse partial artifacts."
        )
    manifest = json.loads(marker.read_text())
    if (
        manifest.get("status") != "complete"
        or manifest.get("kind") != kind
        or kind not in ("full", "smoke")
        or manifest.get("source") != expected_source
        or manifest.get("lookup_fingerprint")
        != lookup_fingerprint(config, split_config, expected_source)
        or json_digest(manifest.get("preprocess")) != json_digest(asdict(config))
        or json_digest(manifest.get("split_config"))
        != json_digest(asdict(split_config))
        or manifest.get("smoke_policy") != SMOKE_POLICY
    ):
        raise ValueError(f"{path}: incompatible artifact/source/configuration/kind.")
    verify_files(path, manifest)
    return validate_reloaded(path, manifest, config, split_config, expected_source)


def save_artifact(
    data,
    path,
    config,
    protocol,
    kind="full",
    split_config=SPLIT_POLICY,
    expected_source=SOURCE,
):
    path = Path(path)
    metadata = audit_standardized(
        data, config, protocol, kind, split_config, expected_source
    )
    if path.exists():
        existing = load_artifact(path, config, split_config, kind, expected_source)
        if audit_metadata(
            existing, protocol, kind, split_config, expected_source
        ) != metadata or json_digest(
            load_protocol(path, split_config, expected_source)
        ) != json_digest(protocol):
            raise ValueError(
                "Existing artifact has different metadata/protocol; choose a new RUN_TAG."
            )
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    data.save_to_disk(str(path), max_shard_size="500MB")
    with (path / "split_assignments.json").open("x") as handle:
        json.dump(protocol, handle, indent=2, sort_keys=True, allow_nan=False)
    manifest = {
        "status": "complete",
        "kind": kind,
        "lookup_fingerprint": lookup_fingerprint(config, split_config, expected_source),
        "fingerprint": preprocess_fingerprint(
            config, protocol, split_config, expected_source
        ),
        "assignment_digest": protocol["assignment_digest"],
        "source": dict(expected_source),
        "preprocess": asdict(config),
        "split_config": asdict(split_config),
        "smoke_policy": SMOKE_POLICY,
        "package_versions": package_versions(),
        "metadata": metadata,
        "files": artifact_files(path),
        "completed_utc": datetime.now(UTC).isoformat(),
    }
    # No marker until inventory validation, sidecar validation and actual audio reload pass.
    verify_files(path, manifest)
    reloaded = validate_reloaded(path, manifest, config, split_config, expected_source)
    write_completion_marker(path, manifest)
    return reloaded


def copy_artifact(
    source,
    destination,
    config=PREPROCESS,
    split_config=SPLIT_POLICY,
    kind="full",
    check_local_capacity=True,
    expected_source=SOURCE,
):
    source, destination = Path(source), Path(destination)
    load_artifact(source, config, split_config, kind, expected_source)
    manifest = json.loads((source / "manifest.json").read_text())
    if destination.exists():
        existing = load_artifact(
            destination, config, split_config, kind, expected_source
        )
        other = json.loads((destination / "manifest.json").read_text())
        if (
            other["files"] != manifest["files"]
            or other["fingerprint"] != manifest["fingerprint"]
        ):
            raise ValueError("Different destination artifact; choose a new RUN_TAG.")
        return existing
    if check_local_capacity:
        required = sum(item["size"] for item in manifest["files"].values())
        check_local_space(
            destination.parent, required * 1.1, "verified artifact restoration/copy"
        )
    destination.mkdir(parents=True, exist_ok=False)
    for relative in manifest["files"]:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)
    verify_files(destination, manifest)
    reloaded = validate_reloaded(
        destination, manifest, config, split_config, expected_source
    )
    write_completion_marker(
        destination, manifest
    )  # Drive/local destination marker LAST.
    return reloaded


def restore_available_artifact(
    local_path,
    drive_path,
    config=PREPROCESS,
    split_config=SPLIT_POLICY,
    use_drive=False,
    kind="full",
    expected_source=SOURCE,
):
    """Verified config-only discovery; does not inspect/download the source ZIP."""
    if Path(local_path).exists():
        return load_artifact(local_path, config, split_config, kind, expected_source)
    if use_drive and Path(drive_path).exists():
        return copy_artifact(
            drive_path,
            local_path,
            config,
            split_config,
            kind,
            expected_source=expected_source,
        )
    return None


# %% Disk preflight and pinned download — called only after artifact discovery.
def check_local_space(directory, required_bytes, purpose):
    # Inspect the nearest existing parent, without creating the restoration target.
    directory = Path(directory)
    parent = directory
    while not parent.exists():
        parent = parent.parent
    available = shutil.disk_usage(parent).free
    print(
        f"{purpose}: need ~{required_bytes / 1e9:.2f} GB; local free {available / 1e9:.2f} GB"
    )
    if available < required_bytes:
        raise OSError(
            f"Insufficient local disk for {purpose}. Change LOCAL_ROOT or free space explicitly."
        )


def download_source(source_cache):
    # Existing *directories* do not prove the 8.16 GB file is cached.
    cached = try_to_load_from_cache(
        DATASET_ID,
        ARCHIVE_NAME,
        revision=DATASET_REVISION,
        repo_type="dataset",
        cache_dir=str(source_cache),
    )
    complete_cached_file = (
        isinstance(cached, str)
        and Path(cached).is_file()
        and Path(cached).stat().st_size == SOURCE["archive_size"]
    )
    # Do not count partial cache files as free/resumable bytes. Full source is verified below.
    required = (
        0 if complete_cached_file else SOURCE["archive_size"] * 1.15
    ) + 512 * 1024**2
    check_local_space(
        source_cache,
        required,
        "pinned ZIP download/cache overhead (even in smoke mode)",
    )
    path = hf_hub_download(
        repo_id=DATASET_ID,
        filename=ARCHIVE_NAME,
        repo_type="dataset",
        revision=DATASET_REVISION,
        cache_dir=str(source_cache),
    )
    verify_source_archive(path)
    return Path(path)


# %% Synthetic fixtures — separate provenance; never accepted as the corpus.
def fixture_metadata():
    return [
        {
            "file": f"fixture_{speaker:02d}_{clip:02d}.wav",
            "speaker": f"Fixture speaker {speaker:02d}",
            "label": "bona-fide" if clip % 2 == 0 else "spoof",
        }
        for speaker in range(20)
        for clip in range(10)
    ]


def synthetic_archive(
    path,
    metadata=None,
    csv_text=None,
    omit=(),
    extra_members=(),
    payload_overrides=None,
):
    """Small stereo 8 kHz ZIP/CSV fixture, including malicious variants for audits."""
    records = fixture_metadata() if metadata is None else metadata
    if csv_text is None:
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=["file", "speaker", "label"])
        writer.writeheader()
        writer.writerows(records)
        csv_text = stream.getvalue()
    with zipfile.ZipFile(
        path, "w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as archive:
        archive.writestr(f"{ARCHIVE_ROOT}/", b"")
        archive.writestr(f"{ARCHIVE_ROOT}/meta.csv", csv_text.encode("utf-8"))
        archive.writestr(
            f"{ARCHIVE_ROOT}/attribution.txt",
            b"Synthetic test data, not the real corpus.",
        )
        for index, row in enumerate(records):
            if row["file"] in omit:
                continue
            sr = 8000
            t = np.arange(round(sr * (0.6 + 0.1 * (index % 3)))) / sr
            y = 0.04 * np.sin(2 * np.pi * (220 + index * 3) * t)
            payload = wav_bytes(np.column_stack((y, y * 0.5)), sr)
            if payload_overrides and row["file"] in payload_overrides:
                payload = payload_overrides[row["file"]]
            archive.writestr(f"{ARCHIVE_ROOT}/{row['file']}", payload)
        for name, payload in extra_members:
            archive.writestr(name, payload)
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        wav_size = sum(
            info.file_size for info in infos if info.filename.endswith(".wav")
        )
    return {
        "id": "synthetic/iwd-fixture",
        "revision": "fixture-v1",
        "archive": Path(path).name,
        "archive_size": Path(path).stat().st_size,
        "archive_sha256": file_sha256(path),
        "metadata_csv_sha256": hashlib.sha256(csv_text.encode("utf-8")).hexdigest(),
        "entries": len(infos),
        "rows": len(records),
        "class_counts": dict(Counter(row["label"] for row in records)),
        "speakers": len({row["speaker"] for row in records}),
        "wav_uncompressed_bytes": wav_size,
        "kind": "synthetic-fixture",
    }


def expect_error(function, exception=ValueError, contains=None):
    try:
        function()
    except exception as exc:
        if contains is not None:
            assert contains in str(exc), str(exc)
        return
    raise AssertionError(f"Expected {exception.__name__}, not silent acceptance.")


def run_splitter_checks():
    records = fixture_metadata()
    config = SplitConfig(candidate_starts=8, max_improving_steps=32)
    first = assign_speakers(records, config)
    reordered = list(reversed(records))
    assert (
        first == assign_speakers(reordered, config) == assign_speakers(records, config)
    )
    mapping, summary, _ = first
    assert sum(summary["counts"].values()) == len(records)
    assert set(mapping) == {row["speaker"] for row in records}
    assert min(summary["speaker_counts"].values()) >= 2
    assert summary["speaker_overlap"] == 0
    for changed in (
        replace(config, seed=43),
        replace(config, candidate_starts=9),
        replace(config, max_improving_steps=33),
        replace(config, ratios=(0.6, 0.2, 0.2)),
        replace(config, max_clip_share_deviation=0.03),
        replace(config, version="test-v2"),
    ):
        assert lookup_fingerprint(PREPROCESS, changed) != lookup_fingerprint(
            PREPROCESS, config
        )
    # A different seed is fingerprinted, but need not change the selected optimum.
    expect_error(lambda: canonical_metadata(records + [dict(records[0])], None))
    for change in (
        {"speaker": ""},
        {"label": "real"},
        {"file": "../unsafe.wav"},
        {"label": None},
    ):
        invalid = [dict(row) for row in records]
        invalid[0].update(change)
        expect_error(lambda invalid=invalid: canonical_metadata(invalid, None))
    missing = [dict(row) for row in records]
    del missing[0]["speaker"]
    expect_error(lambda: canonical_metadata(missing, None))
    impossible = [row for row in records if row["speaker"] in sorted(mapping)[:6]]
    expect_error(lambda: assign_speakers(impossible, config), contains="No acceptable")
    print(
        "PASS: deterministic/reordered speaker splitting, coverage, gates, conflicts and config fingerprints"
    )


def run_archive_checks(base_dir=None):
    with tempfile.TemporaryDirectory(prefix="iwd-zip-check-", dir=base_dir) as folder:
        root = Path(folder)
        path = root / "fixture.zip"
        source = synthetic_archive(path)
        inspected = inspect_archive(path, source)
        policy = SplitConfig(candidate_starts=8, max_improving_steps=32)
        protocol = freeze_protocol(inspected, policy, source)
        with zipfile.ZipFile(path) as archive:
            assert protocol["release_attribution"].encode("utf-8") == archive.read(
                f"{ARCHIVE_ROOT}/attribution.txt"
            )
        changed_attribution = json.loads(json.dumps(protocol))
        changed_attribution["release_attribution"] += "changed"
        changed_attribution["assignment_digest"] = json_digest(
            {
                key: value
                for key, value in changed_attribution.items()
                if key != "assignment_digest"
            }
        )
        expect_error(
            lambda: audit_protocol(changed_attribution, policy, source),
            contains="attribution",
        )
        assert audit_protocol(protocol, policy, source)["coverage"] == 200
        expect_error(
            lambda: audit_protocol(protocol, policy), contains="Incompatible source"
        )
        before = file_sha256(path)
        row = selected_records(protocol, "smoke")["train"][0]
        generated = next(
            standardized_archive_generator(str(path), (row,), asdict(PREPROCESS))
        )
        with zipfile.ZipFile(path) as archive:
            payload = archive.read(f"{ARCHIVE_ROOT}/{row['file']}")
        feature = Audio(sampling_rate=TARGET_SR, num_channels=1)
        through_dataset = Dataset.from_dict(
            {"audio": [{"bytes": payload, "path": None}]},
            features=Features({"audio": feature}),
        )
        assert (
            generated["audio"]["bytes"]
            == standardize_row(through_dataset[0])["audio"]["bytes"]
        )
        assert (
            generated["file"] == row["file"]
            and generated["speaker_id"] == row["speaker"]
        )
        assert (
            generated["key"] == LABEL_MAPPING[row["label"]]
            and "system_id" not in generated
        )
        assert file_sha256(path) == before
        changed = json.loads(json.dumps(protocol))
        changed["speaker_to_split"][row["speaker"]] = "test"
        expect_error(
            lambda: audit_protocol(changed, policy, source), contains="integrity"
        )
        for index, extra in enumerate(
            (
                ("../escape.wav", b"bad"),
                (f"{ARCHIVE_ROOT}/nested/extra.wav", b"bad"),
                (f"{ARCHIVE_ROOT}/extra.mp3", b"bad"),
                (f"{ARCHIVE_ROOT}/meta.csv", b"bad"),
            )
        ):
            bad = root / f"unsafe-{index}.zip"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                bad_source = synthetic_archive(bad, extra_members=(extra,))
            expect_error(
                lambda bad=bad, bad_source=bad_source: inspect_archive(bad, bad_source)
            )
        missing = root / "missing.zip"
        missing_source = synthetic_archive(
            missing, omit=(fixture_metadata()[0]["file"],)
        )
        expect_error(
            lambda: inspect_archive(missing, missing_source), contains="basename join"
        )
        bad_csv = root / "bad-label.zip"
        bad_source = synthetic_archive(
            bad_csv, csv_text="file,speaker,label\nexample.wav,Speaker,real\n"
        )
        expect_error(
            lambda: inspect_archive(bad_csv, bad_source),
            contains="Unknown original label",
        )
        # Candidate confirmation; identical bytes across labels must stop, not deduplicate.
        duplicate = root / "duplicate.zip"
        pair = fixture_metadata()[:2]
        shared = wav_bytes(np.zeros(8000, np.float32), 8000)
        duplicate_source = synthetic_archive(
            duplicate, payload_overrides={row["file"]: shared for row in pair}
        )
        expect_error(
            lambda: inspect_archive(duplicate, duplicate_source), contains="cross-label"
        )
        same_label = root / "same-label-duplicate.zip"
        shared_names = [fixture_metadata()[i]["file"] for i in (0, 2)]
        same_source = synthetic_archive(
            same_label, payload_overrides={name: shared for name in shared_names}
        )
        same_inspection = inspect_archive(same_label, same_source)
        same_protocol = freeze_protocol(same_inspection, policy, same_source)
        assert (
            same_protocol["summary"]["coverage"]
            == len(same_inspection["records"])
            == 200
        )
        assert (
            len(same_inspection["archive_audit"]["confirmed_byte_duplicate_groups"])
            == 1
        )
        first_row = next(
            row
            for row in inspected["records"]
            if protocol["speaker_to_split"][row["speaker"]] == "train"
        )
        second_row = next(
            row
            for row in inspected["records"]
            if protocol["speaker_to_split"][row["speaker"]] == "test"
            and row["label"] == first_row["label"]
        )
        audit = dict(inspected["archive_audit"])
        files = sorted([first_row["file"], second_row["file"]])
        audit["payload_candidate_groups_crc32_size"] = {"00000000:16044": files}
        audit["payload_candidate_member_sha256"] = {
            filename: "0" * 64 for filename in files
        }
        audit["confirmed_byte_duplicate_groups"] = [
            {"sha256": "0" * 64, "files": files}
        ]
        expect_error(
            lambda: audit_duplicate_payloads(
                audit, inspected["records"], protocol["speaker_to_split"]
            ),
            contains="cross-label/cross-split",
        )
        # CRC32/size alone is not enough: distinct candidate hashes are not duplicates.
        audit["payload_candidate_member_sha256"][files[1]] = "1" * 64
        audit["confirmed_byte_duplicate_groups"] = []
        audit_duplicate_payloads(
            audit, inspected["records"], protocol["speaker_to_split"]
        )
    print(
        "PASS: ZIP/CSV/alias safety, real public decoder semantics, byte-duplicate leakage and immutable protocol"
    )


# %% Artifact/corruption/reuse tests — no raw access permitted on the restore path.
def run_artifact_checks(base_dir=None):
    with tempfile.TemporaryDirectory(
        prefix="iwd-artifact-check-", dir=base_dir
    ) as folder:
        root = Path(folder)
        archive = root / "fixture.zip"
        source = synthetic_archive(archive)
        policy = SplitConfig(candidate_starts=8, max_improving_steps=32)
        protocol = freeze_protocol(inspect_archive(archive, source), policy, source)
        data = map_standardized(
            archive, protocol, PREPROCESS, root / "cache", "smoke", policy, source
        )
        assert {name: len(split) for name, split in data.items()} == {
            "train": 16,
            "validation": 2,
            "test": 2,
        }
        saved = save_artifact(
            data, root / "local", PREPROCESS, protocol, "smoke", policy, source
        )
        save_artifact(
            data, root / "local", PREPROCESS, protocol, "smoke", policy, source
        )
        copy_artifact(
            root / "local",
            root / "drive",
            PREPROCESS,
            policy,
            "smoke",
            expected_source=source,
        )
        expect_error(
            lambda: load_artifact(root / "local", PREPROCESS, policy, "full", source)
        )
        full = map_standardized(
            archive, protocol, PREPROCESS, root / "full-cache", "full", policy, source
        )
        full = save_artifact(
            full, root / "fixture-full", PREPROCESS, protocol, "full", policy, source
        )
        assert {name: len(split) for name, split in full.items()} == protocol[
            "summary"
        ]["counts"]
        expect_error(
            lambda: load_artifact(root / "fixture-full", PREPROCESS, policy),
            contains="incompatible",
        )
        assert audit_metadata(
            smoke_from_full(full, protocol), protocol, "smoke", policy, source
        ) == audit_metadata(data, protocol, "smoke", policy, source)
        expect_error(lambda: load_artifact(root / "local", PREPROCESS, policy, "smoke"))
        expect_error(
            lambda: load_artifact(
                root / "local",
                replace(PREPROCESS, rms_dbfs=-18),
                policy,
                "smoke",
                source,
            )
        )
        incomplete = root / "incomplete"
        incomplete.mkdir()
        expect_error(
            lambda: save_artifact(
                data, incomplete, PREPROCESS, protocol, "smoke", policy, source
            ),
            contains="completion marker",
        )
        wrong = DatasetDict(dict(data))
        encoded = data["train"].cast_column(
            "audio", Audio(sampling_rate=TARGET_SR, num_channels=1, decode=False)
        )
        rows = list(encoded)
        rows[0]["speaker_id"] = "fabricated alias"
        wrong["train"] = Dataset.from_list(rows, features=data["train"].features)
        expect_error(
            lambda: audit_metadata(wrong, protocol, "smoke", policy, source),
            contains="mismatch",
        )
        with patch.dict(
            globals(),
            {
                "hf_hub_download": lambda *a, **k: (_ for _ in ()).throw(
                    AssertionError("raw access")
                ),
                "inspect_archive": lambda *a, **k: (_ for _ in ()).throw(
                    AssertionError("raw access")
                ),
            },
        ):
            assert (
                restore_available_artifact(
                    root / "absent",
                    root / "absent-drive",
                    PREPROCESS,
                    policy,
                    True,
                    "smoke",
                    source,
                )
                is None
            )
            available = shutil.disk_usage(root)
            with patch.object(
                shutil, "disk_usage", return_value=available._replace(free=0)
            ):
                expect_error(
                    lambda: restore_available_artifact(
                        root / "no-room",
                        root / "drive",
                        PREPROCESS,
                        policy,
                        True,
                        "smoke",
                        source,
                    ),
                    OSError,
                )
                assert not (root / "no-room").exists()
            restored = restore_available_artifact(
                root / "restored",
                root / "drive",
                PREPROCESS,
                policy,
                True,
                "smoke",
                source,
            )
            reused = restore_available_artifact(
                root / "restored",
                root / "drive",
                PREPROCESS,
                policy,
                True,
                "smoke",
                source,
            )
            assert len(restored["train"]) == len(reused["train"]) == 16
        np.testing.assert_array_equal(
            decode_hf_audio(saved["train"][0]["audio"]),
            decode_hf_audio(restored["train"][0]["audio"]),
        )
        local_extractor = Wav2Vec2FeatureExtractor(
            sampling_rate=TARGET_SR,
            do_normalize=True,
            return_attention_mask=True,
            padding_value=0.0,
        )
        run_batch_checks(saved, local_extractor)
        marker = root / "drive" / "manifest.json"
        manifest = json.loads(marker.read_text())
        victim = root / "drive" / "split_assignments.json"
        with victim.open("ab") as handle:
            handle.write(b"corrupt")
        expect_error(
            lambda: load_artifact(root / "drive", PREPROCESS, policy, "smoke", source),
            contains="corrupted",
        )
        assert "split_assignments.json" in manifest["files"]
    print(
        "PASS: Arrow save/reload, frozen subset/full distinction, source isolation, checksum corruption, disk-safe raw-free reuse"
    )


# %% [markdown]
# ## 4. Waveform augmentation — training only, unchanged ASV defaults
# Five independent probabilities are 0.5 for both labels: the expected chain has
# 2.5 effects, and 3.125% skip all five. Order: pitch-preserving speed -> pitch ->
# zero-filled shift -> causal synthetic reverb -> calibrated whole-clip Gaussian
# noise -> linear peak protection. No RMS renormalization or PCM16 re-encoding.
# Speed is librosa time-stretch 0.9–1.1, pitch +/-2 semitones, shift +/-0.15 s,
# noise SNR 15 dB, synthetic RIR RT60 0.3 s / DRR 6 dB. Shifts discard edge audio
# without circular wrapping; reverb preserves duration. These are proxies, not
# measured environmental noise/rooms, and phase-vocoder artifacts may obscure
# spoof cues. Tune only against validation, never the final held-out test.


# %% Configurable label-independent waveform transforms.
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
# ## 5. Future XLS-R classifier input contract
# Variable-length raw waveforms, not spectrograms, are padded only to the longest
# row in a batch. `AudioCollator` returns **only** `input_values`, valid-sample
# `attention_mask`, and int64 `labels` for a future two-logit
# `Wav2Vec2ForSequenceClassification` / `AutoModelForAudioClassification`.
# Original metadata and aliases are audits only; never feed them to the model.
#
# Before each training iterator call `train_dataset.set_epoch(epoch)`. Row/epoch
# variants are reproducible, and workers are nonpersistent so epoch state reaches
# them. Validation/test never create waveform RNGs or augment, regardless of epoch.
# Also call the future model's `eval()` to disable native SpecAugment, and never
# pass explicit time masks during evaluation. Transformers masking uses global
# NumPy RNG, not `default_rng`; seed it separately as below.
#
# A frontend frame requires 400 samples; a length-10 time mask needs 3,280.
# All short/silent rows stay in storage. Incompatible lengths raise with their ID,
# including the fastest configured training speed, rather than relying on other
# rows' padding, dropping examples, or padding silence as speech.


# %% Length checks, PyTorch wrapper, collator, and loader (ASV-compatible APIs).
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
    """Check ALL saved length metadata, without decoding or filtering recordings."""
    report = {}
    for name, split in data.items():
        minimum_frames = max(1, mask_time_length) if name == "train" else 1
        incompatible, minimum_samples = [], None
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
        if split_name not in SPLIT_NAMES:
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


# %% Configuration/feature-extractor JSON only — no model weights or tokenizer.
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
    assert report["test"]["incompatible_ids"] == [] and len(data["train"]) == 3
    print(
        "PASS: every-row lengths, fastest-speed edge, evaluation minimum and retained short rows"
    )


# %% Batch checks — evaluation is explicitly forbidden from creating a waveform RNG.
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
    for split_name in ("validation", "test"):
        evaluation = SpoofAudioDataset(processed[split_name], split_name, augment=False)
        loader = make_dataloader(
            evaluation, feature_extractor, batch_size=2, num_workers=num_workers
        )
        # num_workers=0 directly exercises the guard; fork workers inherit it too.
        with patch.object(
            np.random,
            "default_rng",
            side_effect=AssertionError("evaluation waveform RNG"),
        ):
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
        assert (
            torch.count_nonzero(batch["input_values"][batch["attention_mask"] == 0])
            == 0
        )
        expect_error(
            lambda split_name=split_name: SpoofAudioDataset(
                processed[split_name], split_name, augment=True
            )
        )
        expect_error(
            lambda evaluation=evaluation: make_dataloader(
                evaluation, feature_extractor, shuffle=True
            )
        )
    short_row = {
        "file": "too-short.wav",
        "speaker": "S",
        "label": "bona-fide",
        "speaker_id": "S",
        "audio_file_name": "too-short.wav",
        "key": 0,
        **standardize_waveform(np.ones(1600, np.float32) * 0.01, TARGET_SR),
    }
    short = Dataset.from_list([short_row], features=processed["train"].features)
    expect_error(
        lambda: SpoofAudioDataset(short, "train", augment=False)[0],
        contains="too-short.wav",
    )
    print(
        f"PASS: epoch-aware train-only batches, no evaluation waveform RNG, labels/padding/metadata isolation ({num_workers} workers)"
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
    attention[1, 12800:] = 0
    inputs[1, 12800:] = 0

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
        assert time_masked.any() and not time_masked[1, 39:].any()
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
        "PASS: tiny-model train/eval mask gating, native time/feature masking, padding and frontend lengths"
    )


# %% Augmentation checks — exact ASV strengths and semantics.
def run_augmentation_checks():
    t = np.arange(TARGET_SR) / TARGET_SR
    clean = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    noisy = add_gaussian_noise(clean, np.random.default_rng(SEED), 15.0)
    snr = 10 * np.log10(
        np.mean(clean.astype(np.float64) ** 2)
        / np.mean((noisy.astype(np.float64) - clean) ** 2)
    )
    assert abs(snr - 15) < 0.1
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
    assert abs(peak_frequency - 440) < 3
    higher = shift_pitch(clean, 2.0)
    assert (
        len(higher) == len(clean)
        and abs(np.argmax(np.abs(np.fft.rfft(higher))) - 494) < 4
    )
    for seed in range(20):
        rir = synthetic_rir(np.random.default_rng(seed), 0.3, 6)
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
        "PASS: whole-clip noise SNR, zero-filled shift, independent speed/pitch, RIR decay/DRR and composition"
    )


# %% Storage estimates and training-only visualizations (not model inputs).
def inspect_source(archive_path, protocol, sample_size=32):
    report, estimated_payload = {}, 0.0
    feature = Audio(sampling_rate=TARGET_SR, num_channels=1)
    with zipfile.ZipFile(archive_path) as archive:
        for split_id, (name, records) in enumerate(selected_records(protocol).items()):
            indices = np.random.default_rng(SEED + split_id).choice(
                len(records), min(sample_size, len(records)), replace=False
            )
            durations, headers = [], Counter()
            for index in indices:
                original = records[int(index)]
                payload = archive.read(f"{ARCHIVE_ROOT}/{original['file']}")
                waveform = decode_hf_audio(
                    feature.decode_example({"bytes": payload, "path": None})
                )
                durations.append(len(waveform) / TARGET_SR)
                info = sf.info(io.BytesIO(payload))
                headers[
                    f"{info.format}/{info.subtype}/{info.channels}ch/{info.samplerate}Hz"
                ] += 1
            estimated_payload += (
                float(np.mean(durations)) * len(records) * TARGET_SR * 2
            )
            report[name] = {
                "sample_count": len(durations),
                "native_headers": dict(headers),
                "mean_seconds": float(np.mean(durations)),
                "p95_seconds": float(np.percentile(durations, 95)),
                "minimum_sampled_seconds": min(durations),
            }
    estimated_artifact = estimated_payload * 1.15
    print(
        json.dumps(
            {
                "sampled_format_audit": report,
                "estimated_processed_GB": estimated_artifact / 1e9,
                "published_duration_hours_not_measured": 37.9,
                "published_duration_implied_PCM16_payload_GB_not_measured": 37.9
                * 3600
                * TARGET_SR
                * 2
                / 1e9,
            },
            indent=2,
        )
    )
    print(
        "Storage estimates only: not a Drive quota guarantee or test-set augmentation tuning."
    )
    return estimated_artifact


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


def smoke_from_full(data, protocol):
    result = {}
    for name, rows in selected_records(protocol, "smoke").items():
        index = {
            filename: i for i, filename in enumerate(data[name]["audio_file_name"])
        }
        result[name] = data[name].select([index[row["file"]] for row in rows])
    return DatasetDict(result)


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
        playback = waveform / max(1.0, float(np.abs(waveform).max()))
        display(AudioPlayer(playback, rate=TARGET_SR, normalize=False))


# %% Additional download preflight contract — an existing cache directory is insufficient.
def run_download_preflight_checks(base_dir=None):
    with tempfile.TemporaryDirectory(prefix="iwd-preflight-", dir=base_dir) as folder:
        cache = Path(folder) / "existing-cache"
        cache.mkdir()
        available = shutil.disk_usage(cache)
        with (
            patch.dict(
                globals(),
                {
                    "try_to_load_from_cache": lambda *a, **k: None,
                    "hf_hub_download": lambda *a, **k: (_ for _ in ()).throw(
                        AssertionError("download before capacity")
                    ),
                },
            ),
            patch.object(shutil, "disk_usage", return_value=available._replace(free=0)),
        ):
            expect_error(lambda: download_source(cache), OSError)
    print(
        "PASS: preflight runs before download even when the source-cache directory already exists"
    )


# %% Offline integration checks — no corpus, pretrained weights, or Hub contact.
if NOTEBOOK_RUN:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    print(json.dumps(package_versions(), indent=2))
    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
    decoder_smoke_check()
    run_standardization_checks()
    run_splitter_checks()
    run_archive_checks(LOCAL_ROOT)
    run_artifact_checks(LOCAL_ROOT)
    run_download_preflight_checks(LOCAL_ROOT)
    run_augmentation_checks()
    run_masking_checks()
    run_model_length_checks()


# %% [markdown]
# ## 6. Discover verified artifacts BEFORE accessing the raw ZIP
# A config-only lookup directory locates artifacts; compatibility additionally
# requires the source constants, full frozen assignment digest, original metadata,
# actual per-row aliases/labels/split membership, complete file SHA256 inventory,
# and reloaded PCM16/TorchCodec audio. A manifest alone is not sufficient.
# All metadata rows are checked; stored audio is deterministically sampled for
# PCM/decoder audits while every Arrow/sidecar file is fully checksummed.
# Local and Drive completion markers are published only **after** checksum and
# reload verification. Partial/incompatible outputs are errors, not cache misses.
# Restoration checks capacity before creating its destination. Keep raw source
# and Arrow caches local, not on Drive. Mounted free space is not account quota.

# %% Mount Drive, set independent IWD paths, and try full/smoke reuse first.
if NOTEBOOK_RUN:
    if MOUNT_GOOGLE_DRIVE:
        from google.colab import drive

        drive.mount("/content/drive")
        print(
            f"Drive output: {DRIVE_ROOT}. Check actual account quota at drive.google.com."
        )
    else:
        print("Drive mount disabled: outputs remain under LOCAL_ROOT.")
    LOOKUP = lookup_fingerprint(PREPROCESS, SPLIT_POLICY)[:16]
    LOCAL_ARTIFACT = LOCAL_ROOT / "saved" / f"in_the_wild-{LOOKUP}"
    DRIVE_ARTIFACT = DRIVE_ROOT / f"standardized-{LOOKUP}"
    SMOKE_DIR = LOCAL_ROOT / "saved" / f"in_the_wild-smoke-{LOOKUP}"
    DRIVE_SMOKE_DIR = DRIVE_ROOT / f"smoke-{LOOKUP}"
    SOURCE_CACHE = LOCAL_ROOT / "source_cache" / "in_the_wild"
    MAP_CACHE = LOCAL_ROOT / "arrow_cache" / "in_the_wild"
    if (
        RUN_FULL_PREPROCESS
        and MOUNT_GOOGLE_DRIVE
        and not DRIVE_ARTIFACT.exists()
        and not DRIVE_QUOTA_CONFIRMED
    ):
        raise RuntimeError(
            "Check Drive account quota, then set DRIVE_QUOTA_CONFIRMED=True before new full publication."
        )
    # Even smoke mode may reuse a verified full artifact without raw-source access.
    processed = restore_available_artifact(
        LOCAL_ARTIFACT,
        DRIVE_ARTIFACT,
        PREPROCESS,
        SPLIT_POLICY,
        MOUNT_GOOGLE_DRIVE,
        kind="full",
    )
    smoke_processed = restore_available_artifact(
        SMOKE_DIR,
        DRIVE_SMOKE_DIR,
        PREPROCESS,
        SPLIT_POLICY,
        MOUNT_GOOGLE_DRIVE,
        kind="smoke",
    )
    protocol = (
        load_protocol(LOCAL_ARTIFACT)
        if processed is not None
        else load_protocol(SMOKE_DIR)
        if smoke_processed is not None
        else None
    )
    if (
        processed is not None
        and smoke_processed is not None
        and protocol["assignment_digest"]
        != load_protocol(SMOKE_DIR)["assignment_digest"]
    ):
        raise ValueError(
            "Full/smoke frozen assignments disagree; choose a new RUN_TAG, never silently repartition."
        )
    ARCHIVE_PATH, SOURCE_INSPECTION, ESTIMATED_ARTIFACT_BYTES = None, None, None


# %% Download only if required — the default remains SMOKE-ONLY processing.
if NOTEBOOK_RUN:
    needs_source = processed is None and (
        RUN_FULL_PREPROCESS or smoke_processed is None
    )
    if needs_source:
        ARCHIVE_PATH = download_source(SOURCE_CACHE)
        SOURCE_INSPECTION = inspect_archive(ARCHIVE_PATH)
        if protocol is None:
            protocol = freeze_protocol(SOURCE_INSPECTION)
        else:
            # Freeze/reuse the same full partition from the previous smoke artifact.
            if (
                protocol["source_metadata_sha256"]
                != SOURCE_INSPECTION["source_metadata_sha256"]
                or protocol["canonical_metadata"] != SOURCE_INSPECTION["records"]
                or protocol["archive_audit"] != SOURCE_INSPECTION["archive_audit"]
                or protocol["release_attribution"]
                != SOURCE_INSPECTION["release_attribution"]
                or protocol["release_attribution_sha256"]
                != SOURCE_INSPECTION["release_attribution_sha256"]
            ):
                raise ValueError(
                    "Pinned source differs from frozen protocol; investigate, never reassign silently."
                )
            audit_protocol(protocol)
        print(
            json.dumps(
                {
                    "assignment_digest": protocol["assignment_digest"],
                    "frozen_split_summary": protocol["summary"],
                },
                indent=2,
            )
        )
        ESTIMATED_ARTIFACT_BYTES = inspect_source(ARCHIVE_PATH, protocol)
    else:
        print(
            "Reusing verified processed artifact(s); raw ZIP download/inventory/decoding skipped."
        )


# %% [markdown]
# ## 7. Smoke encode/reload, then explicitly opt into full materialization
# Smoke uses 8 rows per label in train and 1 per label in validation/test, all
# selected **within the already frozen full splits**. Only this diagnostic subset
# is balanced. A smoke artifact contains the full protocol but is marked as
# incomplete source coverage and can never substitute for full data. Full counts
# derive from assignments, not hardcoded 70% row counts. No saved audio is augmented.

# %% Seeded smoke subset and actual-source byte determinism.
if NOTEBOOK_RUN:
    if smoke_processed is None:
        check_local_space(
            LOCAL_ROOT, 512 * 1024**2, "smoke Arrow cache + saved copy allowance"
        )
        if processed is not None:
            smoke_processed = smoke_from_full(processed, protocol)
        else:
            smoke_processed = map_standardized(
                ARCHIVE_PATH, protocol, PREPROCESS, MAP_CACHE / "smoke", kind="smoke"
            )
        smoke_processed = save_artifact(
            smoke_processed, SMOKE_DIR, PREPROCESS, protocol, kind="smoke"
        )
    if ARCHIVE_PATH is not None:
        original = selected_records(protocol, "smoke")["train"][0]
        repeated = next(
            standardized_archive_generator(
                str(ARCHIVE_PATH), (original,), asdict(PREPROCESS)
            )
        )
        stored = smoke_processed["train"].cast_column("audio", Audio(decode=False))[0]
        assert stored["audio"]["bytes"] == repeated["audio"]["bytes"]
        print("PASS: actual ZIP -> standardized embedded PCM16 bytes are repeatable")


# %% Full run — local save/reload first, Drive checksum/reload next, marker LAST.
if NOTEBOOK_RUN:
    if RUN_FULL_PREPROCESS:
        if processed is None:
            check_local_space(
                LOCAL_ROOT,
                ESTIMATED_ARTIFACT_BYTES * 2 * 1.25,
                "full standardized Arrow cache + saved staging copy (estimate)",
            )
            processed = map_standardized(
                ARCHIVE_PATH, protocol, PREPROCESS, MAP_CACHE / "full"
            )
            processed = save_artifact(processed, LOCAL_ARTIFACT, PREPROCESS, protocol)
        if MOUNT_GOOGLE_DRIVE:
            copy_artifact(
                LOCAL_ARTIFACT,
                DRIVE_ARTIFACT,
                PREPROCESS,
                SPLIT_POLICY,
                check_local_capacity=False,
            )  # Account quota was explicitly confirmed.
            print(f"Verified complete Drive artifact: {DRIVE_ARTIFACT}")
        active_data = processed
        print(
            "FULL STANDARDIZED ARTIFACT VERIFIED:",
            {name: len(split) for name, split in processed.items()},
        )
    else:
        active_data = smoke_processed
        print(
            "SMOKE MODE ONLY: full processing and Drive publication skipped. Set RUN_FULL_PREPROCESS=True explicitly after checking storage."
        )


# %% [markdown]
# ## 8. XLS-R integration and demonstrations
# Only config/feature-extractor JSON is downloaded. Construct the future
# classifier using `model_config`; this notebook never loads XLS-R-300M weights.
# Native learned-feature masks have time probability 0.075, length 10, min 2;
# feature probability 0.02, length 10, min 2, exactly as ASV. Counts are rounded,
# capped and can overlap: there is no guarantee of two disjoint regions. Feature
# masking means hidden channels, not physical frequency bins. Frozen backbones
# kept in eval mode also disable automatic SpecAugment. Future training must use
# `model.train()`, evaluation `model.eval()`; do not pass explicit eval time masks.
# Feature-extractor per-clip normalization ignores batch padding and may exceed
# [-1, 1]; never waveform-clip normalized model tensors. Log-mel plots are visuals.

# %% Model-ready loaders — no optimizer, pretrained weights, classifier or training.
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
            "Model-incompatible lengths found. All IDs are in MODEL_LENGTH_AUDIT; "
            "rows remain in the saved artifact. Adjust the global mask/speed policy explicitly before training."
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
    train_dataset.set_epoch(0)  # Repeat BEFORE each future training iterator.
    demonstration_batch = next(iter(train_loader))
    print(
        {
            key: (tuple(value.shape), str(value.dtype))
            for key, value in demonstration_batch.items()
        }
    )


# %% Optional training-only before/after waveform, log-mel and listening examples.
if NOTEBOOK_RUN and SHOW_PREVIEWS:
    for index in balanced_indices(smoke_processed["train"], 1):
        row = smoke_processed["train"][index]
        standardized = decode_hf_audio(row["audio"])
        examples = []
        if ARCHIVE_PATH is not None:
            with zipfile.ZipFile(ARCHIVE_PATH) as archive:
                payload = archive.read(f"{ARCHIVE_ROOT}/{row['file']}")
            original = Audio(sampling_rate=TARGET_SR, num_channels=1).decode_example(
                {"bytes": payload, "path": None}
            )
            examples.append(("Decoded original", decode_hf_audio(original)))
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
# - Later: `data = load_artifact(path, PREPROCESS, SPLIT_POLICY, kind="full")`.
#   Use `SpoofAudioDataset`, `AudioCollator`, `make_dataloader` and `set_epoch` as
#   above. A compatible notebook rerun restores/reuses local/Drive data before
#   any raw ZIP access. Full checksums read every file; Drive validation takes time.
# - `split_assignments.json` is portable and frozen. It includes full original
#   CSV text to revalidate the source-metadata checksum **without the raw ZIP**.
#   It also retains verbatim release attribution with its SHA256 and the unresolved
#   HF CC BY-SA 4.0 / author-site Apache 2.0 discrepancy; do not infer sharing rights.
#   Do not change assignments after observing performance. Changed policy/search
#   budgets/seeds create different config lookups; algorithm changes require a
#   version bump. A new seed is not guaranteed to change an optimal partition.
# - Every saved row retains `file,speaker,label`, with audited exact aliases
#   `audio_file_name=file`, `speaker_id=speaker`, and explicit ClassLabel mapping.
#   No attack/system ID is fabricated. Metadata is never forwarded to the model.
# - The card says 58 speakers; verified exact strings total 54. Neither exact
#   speaker strings nor CRC32/SHA256 payload checks establish source disjointness.
#   Same-video contamination, speaker aliases and acoustic near duplicates remain
#   risks. There is no fuzzy hashing or acoustic near-duplicate search here.
# - Confirmed cross-label/cross-split byte duplicates stop for explicit resolution;
#   all original rows remain intact. No automatic deduplication or speaker moving.
# - Processing reads one WAV at a time and keeps only standardized Arrow plus its
#   saved copy, not an extracted/raw Arrow corpus. 37.9 published hours imply
#   ~4.37 GB PCM16 payload, not measured output; Arrow, caches and ZIP add overhead.
#   Sampled storage estimates do not guarantee capacity or a free Drive quota.
# - Reduce batch size for variable-length memory pressure; no fixed 4 s crop,
#   truncation, class resampling, short/silent-row dropping or automatic padding as
#   speech. MODEL_LENGTH_AUDIT reports all incompatible IDs and never filters them.
#   Smoke success does not certify full-corpus lengths.
# - Keep source cache and earlier artifacts intact. Incomplete/incompatible output
#   is an error: choose a new RUN_TAG; this notebook never deletes old artifacts.
# - Later evaluate EER, ROC-AUC and per-class recall, tune only on validation and
#   reserve test for final evaluation. min-tDCF additionally requires compatible
#   ASV-system scores/protocol/costs. These checks do not demonstrate accuracy gains.
# - All embedded synthetic checks are offline and separately source-tagged. Actual
#   corpus/Drive execution must be verified in the target runtime. Reproducibility
#   is per row/epoch under recorded versions, not promised across all hardware.

# %% Final summary — a smoke artifact is never a completed full corpus run.
if NOTEBOOK_RUN:
    summary_path = LOCAL_ARTIFACT if RUN_FULL_PREPROCESS else SMOKE_DIR
    completion = json.loads((summary_path / "manifest.json").read_text())
    print(
        json.dumps(
            {
                "mode": completion["kind"],
                "artifact": str(summary_path),
                "counts": completion["metadata"]["counts"],
                "complete_source_coverage": completion["metadata"][
                    "complete_source_coverage"
                ],
                "frozen_full_counts": protocol["summary"]["counts"],
                "labels": LABEL_NAMES,
                "source_revision": DATASET_REVISION,
                "assignment_digest": protocol["assignment_digest"],
                "artifact_fingerprint": completion["fingerprint"],
                "augmentation": "on-the-fly train only; saved audio is unaugmented",
                "classifier_training": "not performed",
                "package_versions": package_versions(),
            },
            indent=2,
        )
    )
