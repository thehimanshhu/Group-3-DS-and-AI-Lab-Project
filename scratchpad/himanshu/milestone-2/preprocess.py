"""Shared preprocessing for every clip, real or fake.

Every source is brought to the same format so that the detector cannot tell real from fake by
sample rate, sample format, edge silence or loudness:

1. decode and mix to mono
2. resample to 16 kHz (the rate XLS-R was pretrained on)
3. trim leading and trailing silence
4. normalise loudness to one RMS level
5. store as 16-bit PCM WAV and hash the result (SHA-1) for duplicate removal

Usage:
    python preprocess.py <input audio> <output.wav>
"""

import hashlib
import sys
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 16000
TARGET_RMS_DBFS = -20.0
SILENCE_DB = 40      # frames more than 40 dB below the loudest frame count as silence
FRAME_S = 0.02       # 20 ms frames for the silence check
MIN_SECONDS = 1.0    # shorter clips are dropped


def load_mono(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    return audio.mean(axis=1), sr


def resample(audio, sr, target_sr=TARGET_SR):
    if sr == target_sr:
        return audio
    factor = gcd(sr, target_sr)
    return resample_poly(audio, target_sr // factor, sr // factor).astype(np.float32)


def trim_silence(audio, sr):
    frame = int(FRAME_S * sr)
    n_frames = len(audio) // frame
    if n_frames == 0:
        return audio
    energy = (audio[:n_frames * frame].reshape(n_frames, frame) ** 2).mean(axis=1)
    energy_db = 10 * np.log10(energy + 1e-12)
    voiced = np.where(energy_db > energy_db.max() - SILENCE_DB)[0]
    return audio[voiced[0] * frame:(voiced[-1] + 1) * frame]


def normalise_loudness(audio):
    rms = np.sqrt(np.mean(audio ** 2))
    if rms == 0:
        return audio
    audio = audio * (10 ** (TARGET_RMS_DBFS / 20) / rms)
    peak = np.abs(audio).max()
    return audio / peak * 0.99 if peak > 0.99 else audio  # never clip


def preprocess(path):
    """Return (16-bit PCM samples at 16 kHz, sha1) or (None, reason) if the clip is dropped."""
    audio, sr = load_mono(path)
    audio = trim_silence(resample(audio, sr), TARGET_SR)
    if len(audio) < MIN_SECONDS * TARGET_SR:
        return None, "shorter than 1 s after trimming"
    pcm = np.round(normalise_loudness(audio) * 32767).astype(np.int16)
    return pcm, hashlib.sha1(pcm.tobytes()).hexdigest()


if __name__ == "__main__":
    pcm, info = preprocess(sys.argv[1])
    if pcm is None:
        print(f"dropped: {info}")
    else:
        sf.write(sys.argv[2], pcm, TARGET_SR, subtype="PCM_16")
        print(f"{len(pcm) / TARGET_SR:.2f} s, sha1 {info}")
