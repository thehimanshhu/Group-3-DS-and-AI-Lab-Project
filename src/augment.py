"""Label-independent waveform augmentation for preprocessed mono 16 kHz audio.

Import individual effects, or use ``apply(wave, sr, augment_cfg, rng)``. The
notebook-facing effects are called as ``effect(wave, sr, rng)`` (or
``effect(wave, rng)`` for RawBoost and the noise/gain helpers); optional keyword
arguments fix their sampled ranges. Every effect is disabled by default; enabled
effects have independent probability 0.5. Configuration is a plain,
YAML-compatible dictionary or AugmentationConfig:

    cfg = {
        "noise": {"enabled": True, "kind": "gaussian", "snr_db": [15, 20]},
        "gain": {"enabled": True, "gain_db": [-6, 6]},
    }
    rng = np.random.default_rng(42)
    augmented = apply(wave, 16000, cfg, rng)

The caller must restrict augmentation to training and supply per-clip/per-epoch
generators when needed. No global RNG, labels, loaders, models, downloads, or
filesystem writes are used. Imports of librosa, SciPy, and SoundFile are lazy.
MUSAN uses local WAV/FLAC files only; G.711 codecs use an optional FFmpeg binary.

Speed changes duration; other effects preserve length. Individual effects do
not apply a final loudness normalization. ``apply`` protects peaks after an
effect runs, but returns an unchanged copy when every effect is disabled/skipped.
``codec`` defaults to the full phone channel and chooses G.711 mu-law/A-law with
the caller's generator. Speed/pitch can obscure spoof cues: these strengths are
starting points, not validated training settings.
"""

from __future__ import annotations

import math
import os
import subprocess
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

TARGET_SR = 16_000
EFFECT_ORDER = (
    "speed",
    "pitch",
    "shift",
    "reverb",
    "rawboost",
    "noise",
    "gain",
    "channel",
)

_RAWBOOST_DEFAULTS = {
    "components": ("convolutive", "impulsive"),
    "n_bands": 5,
    "frequency_hz": (20.0, 8000.0),
    "bandwidth_hz": (100.0, 1000.0),
    "filter_taps": (10, 100),
    "filter_gain_db": (0.0, 0.0),
    "nonlinear_order": 5,
    "nonlinear_bias_db": (5.0, 20.0),
    "max_impulsive_percent": 10.0,
    "impulsive_gain": 2.0,
    "stationary_snr_db": (10.0, 40.0),
}
_DEFAULTS: dict[str, dict[str, object]] = {
    "speed": {"rate": (0.9, 1.1)},
    "pitch": {"semitones": (-2.0, 2.0)},
    "shift": {"seconds": (-0.15, 0.15)},
    "reverb": {"rt60_seconds": 0.3, "drr_db": 6.0},
    "rawboost": _RAWBOOST_DEFAULTS,
    "noise": {"kind": "gaussian", "snr_db": 15.0, "folder": None},
    "gain": {"gain_db": (-6.0, 6.0)},
    "channel": {"mode": "phone", "codec": "random", "ffmpeg_path": "ffmpeg"},
}


def _number(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number.")
    return float(value)


def _range(value, name, *, minimum=None, positive=False):
    """Accept a fixed scalar or an inclusive, ordered pair of numeric bounds."""
    if isinstance(value, (list, tuple)) and len(value) == 2:
        low, high = (_number(item, name) for item in value)
    else:
        low = high = _number(value, name)
    if low > high or (minimum is not None and low < minimum) or (positive and low <= 0):
        raise ValueError(f"{name} has invalid or unordered bounds.")
    return low, high


def _positive_integer(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _path(value, name):
    if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
        raise ValueError(f"{name} must be a nonempty local path.")
    return os.fsdecode(value)


def _options(name, supplied):
    if not isinstance(supplied, Mapping):
        raise ValueError(f"{name} configuration must be a dictionary.")
    defaults = {"enabled": False, "p": 0.5, **_DEFAULTS[name]}
    unknown = set(supplied) - set(defaults)
    if unknown:
        raise ValueError(
            f"{name}: unknown configuration keys {sorted(unknown, key=str)}."
        )
    options = {**deepcopy(defaults), **deepcopy(dict(supplied))}
    if type(options["enabled"]) is not bool:
        raise ValueError(f"{name}.enabled must be a boolean.")
    options["p"] = _number(options["p"], f"{name}.p")
    if not 0 <= options["p"] <= 1:
        raise ValueError(f"{name}.p must lie in [0, 1].")

    if name in ("speed", "pitch", "shift", "gain", "noise"):
        key = {
            "speed": "rate",
            "pitch": "semitones",
            "shift": "seconds",
            "gain": "gain_db",
            "noise": "snr_db",
        }[name]
        options[key] = _range(options[key], f"{name}.{key}", positive=name == "speed")
    if name == "reverb":
        options["rt60_seconds"] = _number(
            options["rt60_seconds"], "reverb.rt60_seconds"
        )
        options["drr_db"] = _number(options["drr_db"], "reverb.drr_db")
        if options["rt60_seconds"] <= 0:
            raise ValueError("reverb.rt60_seconds must be positive.")
    elif name == "noise":
        if options["kind"] not in ("gaussian", "musan"):
            raise ValueError("noise.kind must be 'gaussian' or 'musan'.")
        if options["folder"] is not None:
            options["folder"] = _path(options["folder"], "noise.folder")
        if (
            options["enabled"]
            and options["kind"] == "musan"
            and options["folder"] is None
        ):
            raise ValueError(
                "Enabled MUSAN noise requires noise.folder; nothing is downloaded."
            )
    elif name == "channel":
        if options["mode"] not in ("codec", "phone"):
            raise ValueError("channel.mode must be 'codec' or 'phone'.")
        if options["codec"] not in ("random", "mulaw", "alaw"):
            raise ValueError(
                "channel.codec must be 'random', 'mulaw' or 'alaw'; "
                "AMR is not supported."
            )
        options["ffmpeg_path"] = _path(options["ffmpeg_path"], "channel.ffmpeg_path")
    elif name == "rawboost":
        components = options["components"]
        if (
            not isinstance(components, (list, tuple))
            or not components
            or any(
                item not in ("convolutive", "impulsive", "stationary")
                for item in components
            )
            or len(set(components)) != len(components)
        ):
            raise ValueError(
                "rawboost.components must be a nonempty list of unique "
                "'convolutive', 'impulsive', 'stationary' names."
            )
        options["components"] = tuple(components)
        for key in ("n_bands", "nonlinear_order"):
            options[key] = _positive_integer(options[key], f"rawboost.{key}")
        for key in ("frequency_hz", "bandwidth_hz"):
            options[key] = _range(options[key], f"rawboost.{key}", positive=True)
        if options["frequency_hz"][1] > TARGET_SR / 2:
            raise ValueError("rawboost.frequency_hz must not exceed 8 kHz (Nyquist).")
        options["filter_taps"] = _range(
            options["filter_taps"], "rawboost.filter_taps", minimum=3
        )
        if any(not item.is_integer() for item in options["filter_taps"]):
            raise ValueError("rawboost.filter_taps bounds must be integers.")
        for key in ("filter_gain_db", "stationary_snr_db"):
            options[key] = _range(options[key], f"rawboost.{key}")
        options["nonlinear_bias_db"] = _range(
            options["nonlinear_bias_db"], "rawboost.nonlinear_bias_db", minimum=0
        )
        for key in ("max_impulsive_percent", "impulsive_gain"):
            options[key] = _number(options[key], f"rawboost.{key}")
            if options[key] < 0:
                raise ValueError(f"rawboost.{key} must be nonnegative.")
        if options["max_impulsive_percent"] > 100:
            raise ValueError("rawboost.max_impulsive_percent must not exceed 100.")
    return options


@dataclass
class AugmentationConfig:
    """Validated effect dictionaries; omitted effects stay disabled.

    Every effect accepts ``enabled`` (False) and ``p`` (0.5), in addition to the
    parameters below. Scalar ranges fix a value; two-element lists/tuples sample
    uniformly between bounds. Dictionaries are copied, never modified in place.

    - noise: kind='gaussian'|'musan', snr_db=15, folder=None.
    - gain: gain_db=[-6, 6].
    - speed: rate=[0.9, 1.1]; pitch: semitones=[-2, 2].
    - shift: seconds=[-0.15, 0.15].
    - reverb: rt60_seconds=0.3, drr_db=6 (synthetic, not a measured room).
    - rawboost: components=['convolutive', 'impulsive'] (upstream algorithm 5).
      Other upstream defaults: n_bands=5, frequency_hz=[20, 8000],
      bandwidth_hz=[100, 1000], filter_taps=[10, 100] (rounded up to odd),
      filter_gain_db=[0, 0], nonlinear_order=5, nonlinear_bias_db=[5, 20],
      max_impulsive_percent=10, impulsive_gain=2, stationary_snr_db=[10, 40].
      Components run serially in the configured order, behind one probability.
    - channel: mode='phone'|'codec', codec='random'|'mulaw'|'alaw',
      ffmpeg_path='ffmpeg'. 'random' selects mu-law or A-law with the pipeline
      generator; mode='phone' is the notebook default full phone chain.
    - peak_ceiling: 0.99, applied only after an effect is selected.

    Reconstruct the config after changing settings to validate the new values.
    ``dataclasses.asdict(cfg)`` gives a serializable configuration dictionary.
    """

    noise: dict = field(default_factory=dict)
    gain: dict = field(default_factory=dict)
    speed: dict = field(default_factory=dict)
    pitch: dict = field(default_factory=dict)
    shift: dict = field(default_factory=dict)
    reverb: dict = field(default_factory=dict)
    rawboost: dict = field(default_factory=dict)
    channel: dict = field(default_factory=dict)
    peak_ceiling: float = 0.99

    def __post_init__(self):
        for name in EFFECT_ORDER:
            setattr(self, name, _options(name, getattr(self, name)))
        self.peak_ceiling = _number(self.peak_ceiling, "peak_ceiling")
        if not 0 < self.peak_ceiling < 1:
            raise ValueError("peak_ceiling must lie in (0, 1).")

    @classmethod
    def from_dict(cls, values):
        if not isinstance(values, Mapping):
            raise ValueError("Augmentation configuration must be a dictionary.")
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown augmentation keys {sorted(unknown, key=str)}.")
        return cls(**values)


def _wave(wave):
    if np.iscomplexobj(wave):
        raise ValueError("Expected a real mono waveform.")
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim != 1 or not wave.size or not np.isfinite(wave).all():
        raise ValueError("Expected a nonempty, finite mono waveform.")
    return np.ascontiguousarray(wave)


def _sample_rate(sr):
    if (
        isinstance(sr, (bool, np.bool_))
        or not isinstance(sr, (int, np.integer))
        or sr != TARGET_SR
    ):
        raise ValueError("Input speech must already be preprocessed to mono 16 kHz.")


def _generator(rng):
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be an explicit numpy.random.Generator.")


def _protect_peaks(wave, ceiling=0.99):
    peak = float(np.max(np.abs(wave)))
    if peak > ceiling:
        wave = wave * (np.nextafter(np.float32(ceiling), np.float32(0)) / peak)
    return _wave(wave)


def _mix_noise(wave, noise, snr_db):
    snr_db = _number(snr_db, "snr_db")
    signal_rms = float(np.sqrt(np.mean(wave.astype(np.float64) ** 2)))
    if signal_rms == 0:
        return wave.copy()
    noise = np.asarray(noise, dtype=np.float64)
    noise_rms = float(np.sqrt(np.mean(noise**2)))
    if not np.isfinite(noise_rms) or noise_rms == 0:
        raise ValueError(
            "Cannot calibrate empty, silent, or non-finite noise to an SNR."
        )
    noise *= signal_rms / (10.0 ** (snr_db / 20.0) * noise_rms)
    return _wave(wave + noise)


def add_noise(wave, rng, snr_db=15.0):
    """Add Gaussian noise at whole-clip SNR in dB; leave silent speech unchanged."""
    wave = _wave(wave)
    _generator(rng)
    return _mix_noise(wave, rng.standard_normal(len(wave)), snr_db)


@lru_cache(maxsize=8)
def _noise_files(folder):
    root = Path(folder)
    if not root.is_dir():
        raise ValueError(f"MUSAN folder does not exist: {root}")
    files = tuple(
        sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in (".wav", ".flac")
        )
    )
    if not files:
        raise ValueError(f"No local WAV/FLAC noise files found in {root}.")
    return files


def add_musan_noise(wave, sr, rng, folder, snr_db=15.0):
    """Mix a random local WAV/FLAC segment, tiling short files as necessary.

    The caller chooses which MUSAN categories to put in the folder. Files are
    sampled uniformly (not category-balanced). The file inventory is cached per
    folder/process; only the needed segment is decoded, never downloaded or
    saved. Only noise assets are downmixed/resampled, not the input speech.
    """
    import soundfile as sf
    from scipy.signal import resample_poly

    wave = _wave(wave)
    _sample_rate(sr)
    _generator(rng)
    files = _noise_files(str(Path(_path(folder, "folder")).expanduser().resolve()))
    path = files[int(rng.integers(len(files)))]
    try:
        with sf.SoundFile(path) as stream:
            noise_sr = stream.samplerate
            frames = min(len(stream), math.ceil(len(wave) * noise_sr / sr))
            start = int(rng.integers(len(stream) - frames + 1))
            stream.seek(start)
            noise = stream.read(frames, dtype="float32", always_2d=True)
        noise = _wave(noise.mean(axis=1))
        if noise_sr != sr:
            factor = math.gcd(noise_sr, sr)
            noise = _wave(resample_poly(noise, sr // factor, noise_sr // factor))
        if len(noise) < len(wave):
            noise = np.tile(noise, math.ceil(len(wave) / len(noise)))
        start = int(rng.integers(len(noise) - len(wave) + 1))
        return _mix_noise(wave, noise[start : start + len(wave)], snr_db)
    except (ValueError, OSError, RuntimeError) as exc:
        raise ValueError(f"MUSAN noise file {path}: {exc}") from exc


def random_gain(wave, rng, gain_db=(-6.0, 6.0)):
    """Apply uniformly sampled gain in dB, without RMS normalization."""
    wave = _wave(wave)
    _generator(rng)
    bounds = _range(gain_db, "gain_db")
    return _wave(wave * 10.0 ** (rng.uniform(*bounds) / 20.0))


def speed(wave, sr, rng, rate=(0.9, 1.1)):
    """Pitch-preserving time stretch; output length is round(len(wave) / rate).

    ``rate`` may be a fixed number or an inclusive ``(low, high)`` pair sampled
    with ``rng``. The notebook form ``speed(wave, sr, rng)`` uses the default
    range. Librosa receives float64 data to avoid a NumPy/Numba cast warning;
    ``_wave`` converts the result back to float32.
    """
    wave = _wave(wave)
    _sample_rate(sr)
    _generator(rng)
    rate = rng.uniform(*_range(rate, "rate", positive=True))
    if round(len(wave) / rate) < 1:
        raise ValueError("rate must leave at least one audio sample.")
    if rate == 1:
        return wave.copy()
    import librosa

    return _wave(librosa.effects.time_stretch(wave.astype(np.float64), rate=rate))


def pitch(wave, sr, rng, semitones=(-2.0, 2.0)):
    """Shift pitch without changing waveform length.

    ``semitones`` may be fixed or an inclusive ``(low, high)`` pair sampled with
    ``rng``. The notebook form ``pitch(wave, sr, rng)`` uses the default range.
    """
    wave = _wave(wave)
    _sample_rate(sr)
    _generator(rng)
    semitones = rng.uniform(*_range(semitones, "semitones"))
    if semitones == 0:
        return wave.copy()
    import librosa

    return _wave(
        librosa.effects.pitch_shift(
            wave.astype(np.float64),
            sr=sr,
            n_steps=semitones,
            bins_per_octave=12,
            res_type="soxr_hq",
        )
    )


def time_shift(wave, sr, seconds):
    """Shift with zero fill, discarding the opposite edge; never wrap samples."""
    wave = _wave(wave)
    _sample_rate(sr)
    seconds = _number(seconds, "seconds")
    amount = min(round(abs(seconds) * sr), len(wave))
    shifted = np.zeros_like(wave)
    if amount == 0:
        shifted[:] = wave
    elif seconds > 0:
        shifted[amount:] = wave[:-amount]
    else:
        shifted[:-amount] = wave[amount:]
    return shifted


def reverb(wave, sr, rng, rt60_seconds=0.3, drr_db=6.0):
    """Causal synthetic reverb with controlled decay and direct/reverberant energy."""
    from scipy.signal import fftconvolve

    wave = _wave(wave)
    _sample_rate(sr)
    _generator(rng)
    rt60_seconds = _number(rt60_seconds, "rt60_seconds")
    drr_db = _number(drr_db, "drr_db")
    if rt60_seconds <= 0:
        raise ValueError("rt60_seconds must be positive.")
    length = max(2, math.ceil(2 * rt60_seconds * sr))
    times = np.arange(length) / sr
    response = rng.standard_normal(length) * 10.0 ** (-3 * times / rt60_seconds)
    response[0] = 0
    norm = float(np.linalg.norm(response))
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Cannot construct a nonzero synthetic reverberation tail.")
    response *= 10.0 ** (-drr_db / 20.0) / norm
    response[0] = 1
    return _wave(fftconvolve(wave, response, mode="full")[: len(wave)])


# RawBoost helpers adapt Tak et al.'s reference implementation:
# https://github.com/TakHemlata/RawBoost-antispoofing (ICASSP 2022).
# Changes: explicit Generator, validated parameters, safe silence, float32 output.
#
# MIT License -- Copyright (c) 2021 Hemlata
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


def _rawboost_notches(sr, rng, options, gain_bounds):
    from scipy.signal import firwin, freqz

    coefficients = np.ones(1, dtype=np.float64)
    for _ in range(options["n_bands"]):
        center = rng.uniform(*options["frequency_hz"])
        width = rng.uniform(*options["bandwidth_hz"])
        taps = int(rng.uniform(*options["filter_taps"])) | 1
        low = float(np.clip(center - width / 2, 0.001, sr / 2 - 0.002))
        high = float(np.clip(center + width / 2, low + 0.001, sr / 2 - 0.001))
        # For two cutoffs, True selects a band-stop filter with DC in the passband.
        notch = np.asarray(
            firwin(taps, [low, high], pass_zero=True, window="hamming", fs=sr),
            dtype=np.float64,
        )
        coefficients = np.asarray(np.convolve(notch, coefficients), dtype=np.float64)
    _, response = freqz(coefficients, fs=sr)
    peak = float(np.max(np.abs(response)))
    if not np.isfinite(peak) or peak == 0:
        raise ValueError("Degenerate RawBoost notch filter.")
    return coefficients * (10.0 ** (rng.uniform(*gain_bounds) / 20.0) / peak)


def _rawboost_filter(wave, coefficients):
    from scipy.signal import lfilter

    # Reference implementation's zero padding and delay compensation.
    padding = len(coefficients) + 1
    filtered = lfilter(coefficients, [1.0], np.pad(wave, (0, padding)))
    return filtered[padding // 2 : padding // 2 + len(wave)]


def _unit_peak_if_needed(wave):
    peak = float(np.max(np.abs(wave)))
    return wave / peak if peak > 1 else wave


def rawboost(wave, rng, components=("convolutive", "impulsive"), **parameters):
    """Apply selected RawBoost components serially, in the supplied order.

    The input must already be mono 16 kHz; RawBoost uses ``TARGET_SR``
    internally. The notebook form is ``rawboost(wave, rng)``. Default:
    convolutive -> impulsive (reference algorithm 5), not a claim of optimality
    for XLS-R. Add 'stationary' for filtered independent noise, or select any
    single component. Other keyword parameters and their defaults are documented
    on AugmentationConfig. No parallel-combination mode is used. RawBoost
    retains its component-specific peak scaling, not RMS normalization.
    """
    wave = _wave(wave)
    _generator(rng)
    sr = TARGET_SR
    unknown = set(parameters) - set(_RAWBOOST_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown RawBoost parameters {sorted(unknown, key=str)}.")
    options = _options("rawboost", {"components": components, **parameters})
    result = wave.astype(np.float64)
    if not np.any(result):
        return result.astype(np.float32)
    for component in options["components"]:
        if component == "convolutive":
            distorted = np.zeros_like(result)
            gain = np.asarray(options["filter_gain_db"])
            for order in range(1, options["nonlinear_order"] + 1):
                if order == 2:
                    # Legacy np.random.uniform accepted reversed bounds; Generator
                    # requires them ordered. Sorting preserves the reference range.
                    gain = np.sort(gain - np.asarray(options["nonlinear_bias_db"]))
                coefficients = _rawboost_notches(sr, rng, options, gain)
                distorted += _rawboost_filter(result**order, coefficients)
            result = _unit_peak_if_needed(distorted - distorted.mean())
        elif component == "impulsive":
            percent = rng.uniform(0, options["max_impulsive_percent"])
            count = int(len(result) * percent / 100)
            indices = rng.permutation(len(result))[:count]
            factors = rng.uniform(-1, 1, count) * rng.uniform(-1, 1, count)
            result[indices] += options["impulsive_gain"] * result[indices] * factors
            result = _unit_peak_if_needed(result)
        else:
            noise = rng.standard_normal(len(result))
            coefficients = _rawboost_notches(
                sr, rng, options, options["filter_gain_db"]
            )
            noise = _rawboost_filter(noise, coefficients)
            result = _mix_noise(
                result, noise, rng.uniform(*options["stationary_snr_db"])
            )
    return _wave(result)


def _ffmpeg(payload, arguments, executable):
    try:
        completed = subprocess.run(
            [executable, "-hide_banner", "-loglevel", "error", "-nostdin", *arguments],
            input=payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            f"FFmpeg unavailable or failed ({executable}): {exc}"
        ) from exc
    if completed.returncode != 0:
        details = completed.stderr.decode(errors="replace").strip()[-1500:]
        raise RuntimeError(f"FFmpeg codec round-trip failed: {details}")
    return completed.stdout


def _codec_roundtrip(wave, sr, name, ffmpeg_path):
    if name not in ("mulaw", "alaw"):
        raise ValueError("codec must be 'mulaw' or 'alaw'; AMR is not supported.")
    executable = _path(ffmpeg_path, "ffmpeg_path")
    # Protect before integer/companded encoding too, not just after the chain.
    wave = _protect_peaks(wave)
    encoded = _ffmpeg(
        np.asarray(wave, dtype="<f4").tobytes(),
        [
            "-f",
            "f32le",
            "-ar",
            str(sr),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-c:a",
            f"pcm_{name}",
            "-f",
            name,
            "pipe:1",
        ],
        executable,
    )
    decoded = _ffmpeg(
        encoded,
        [
            "-f",
            name,
            "-ar",
            str(sr),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-c:a",
            "pcm_f32le",
            "-f",
            "f32le",
            "pipe:1",
        ],
        executable,
    )
    result = _wave(np.frombuffer(decoded, dtype="<f4").copy())
    if len(result) != len(wave):
        raise RuntimeError(
            "G.711 round-trip changed length; refusing to pad or drop audio."
        )
    return result


def codec(wave, sr, rng, mode="phone", name="random", ffmpeg_path="ffmpeg"):
    """Phone-channel augmentation by default, or a bare G.711 round-trip.

    ``name='random'`` selects mu-law or A-law with ``rng``; pass 'mulaw' or
    'alaw' to fix the codec. ``mode='phone'`` band-limits to 300-3400 Hz and
    resamples through 8 kHz; ``mode='codec'`` applies G.711 at the input rate.
    The notebook form ``codec(wave, sr, rng)`` is the full phone chain.
    """
    wave = _wave(wave)
    _sample_rate(sr)
    _generator(rng)
    if mode not in ("codec", "phone"):
        raise ValueError("codec mode must be 'codec' or 'phone'.")
    if name == "random":
        name = ("mulaw", "alaw")[int(rng.integers(2))]
    if name not in ("mulaw", "alaw"):
        raise ValueError(
            "codec name must be 'random', 'mulaw' or 'alaw'; AMR is not supported."
        )
    if mode == "phone":
        return phone_channel(wave, sr, name, ffmpeg_path)
    return _codec_roundtrip(wave, sr, name, ffmpeg_path)


def phone_channel(wave, sr, name="mulaw", ffmpeg_path="ffmpeg"):
    """300-3400 Hz band-pass -> 8 kHz -> G.711 -> 16 kHz; preserve duration."""
    from scipy.signal import butter, resample_poly, sosfilt

    wave = _wave(wave)
    _sample_rate(sr)
    sections = butter(4, [300, 3400], btype="bandpass", fs=sr, output="sos")
    filtered = sosfilt(sections, wave)
    narrowband = _wave(resample_poly(filtered, 1, 2))
    degraded = _codec_roundtrip(narrowband, 8000, name, ffmpeg_path)
    # Odd input lengths round up when downsampling. Remove only that extra sample.
    return _wave(resample_poly(degraded, 2, 1)[: len(wave)])


def apply(wave, sr, augment_cfg, rng):
    """Compose enabled effects using EFFECT_ORDER and the caller's Generator.

    Accept AugmentationConfig or a nested dictionary. For example, an empty
    dictionary is a no-op; ``{"noise": {"enabled": True}}`` independently adds
    15 dB Gaussian noise with probability 0.5. Resource-dependent effects load
    their dependencies only when selected; failures name the effect and never
    silently fall back. No training/evaluation split logic belongs here.
    """
    wave = _wave(wave).copy()
    _sample_rate(sr)
    _generator(rng)
    config = (
        augment_cfg
        if isinstance(augment_cfg, AugmentationConfig)
        else AugmentationConfig.from_dict(augment_cfg)
    )
    applied = False
    for name in EFFECT_ORDER:
        options = getattr(config, name)
        if not options["enabled"] or options["p"] == 0 or rng.random() >= options["p"]:
            continue
        try:
            if name == "speed":
                wave = speed(wave, sr, rng, options["rate"])
            elif name == "pitch":
                wave = pitch(wave, sr, rng, options["semitones"])
            elif name == "shift":
                wave = time_shift(wave, sr, rng.uniform(*options["seconds"]))
            elif name == "reverb":
                wave = reverb(wave, sr, rng, options["rt60_seconds"], options["drr_db"])
            elif name == "rawboost":
                parameters = {
                    key: value
                    for key, value in options.items()
                    if key not in ("enabled", "p")
                }
                wave = rawboost(wave, rng, **parameters)
            elif name == "noise":
                snr = rng.uniform(*options["snr_db"])
                wave = (
                    add_noise(wave, rng, snr)
                    if options["kind"] == "gaussian"
                    else add_musan_noise(wave, sr, rng, options["folder"], snr)
                )
            elif name == "gain":
                wave = random_gain(wave, rng, options["gain_db"])
            else:
                wave = codec(
                    wave,
                    sr,
                    rng,
                    mode=options["mode"],
                    name=options["codec"],
                    ffmpeg_path=options["ffmpeg_path"],
                )
            applied = True
        except Exception as exc:
            raise ValueError(f"{name} augmentation failed: {exc}") from exc
    return _protect_peaks(wave, config.peak_ceiling) if applied else wave
