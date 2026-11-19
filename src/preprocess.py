"""Shared preprocessing for every clip, real or fake (Milestone 2, with the Milestone-3 m4a fix).

Every source is brought to the same format, so the detector can't tell real from fake by sample rate, format,
edge silence or loudness:

1. decode and mix to mono (m4a through ffmpeg)
2. resample to 16 kHz, the rate XLS-R was pretrained on
3. trim silence at the start and end (20 ms frames more than 40 dB below the loudest frame)
4. cut the first 30 ms and trim again (Milestone 4): almost every real Kathbath recording starts with a click, which
   the first trim lands on; no fake source has it, so it would give the label away
5. drop the clip if less than 1 s is left
6. normalise loudness to -20 dBFS RMS, with the peak at most 0.99
7. store as 16-bit PCM WAV; the SHA-1 of the samples finds exact duplicates

Usage from code:   wave = preprocess(open(path, "rb").read())
From the shell:    python -m src.preprocess <input audio> <output.wav>
"""
import hashlib
import io
import shutil
import subprocess
import sys
import tempfile
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 16000
TARGET_DBFS = -20.0
SILENCE_DB = 40
FRAME_S = 0.02
MIN_SECONDS = 1.0
START_CUT_S = 0.03   # the Kathbath click lies 5-10 ms after the first trim


def decode(data):
    """Raw file bytes -> (mono float32 audio, sample rate)."""
    try:
        audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except sf.LibsndfileError:  # m4a
        # ffmpeg reads from a file, not a pipe: some m4a files keep their index at the end, and from a pipe those
        # decode to nothing (this dropped 28 Kathbath test recordings in Milestone 3)
        ffmpeg = shutil.which("ffmpeg") or __import__("imageio_ffmpeg").get_ffmpeg_exe()
        with tempfile.NamedTemporaryFile(suffix=".m4a") as tmp:
            tmp.write(data)
            tmp.flush()
            wav = subprocess.run([ffmpeg, "-v", "error", "-i", tmp.name, "-f", "wav", "pipe:1"],
                                 capture_output=True, check=True).stdout
        audio, sr = sf.read(io.BytesIO(wav), dtype="float32", always_2d=True)
    return audio.mean(axis=1), sr


def resample(audio, sr):
    if sr == TARGET_SR:
        return audio
    g = gcd(sr, TARGET_SR)
    return resample_poly(audio, TARGET_SR // g, sr // g).astype(np.float32)


def trim_silence(audio):
    frame = int(FRAME_S * TARGET_SR)
    n = len(audio) // frame
    if n == 0:
        return audio
    energy_db = 10 * np.log10((audio[:n * frame].reshape(n, frame) ** 2).mean(axis=1) + 1e-12)
    voiced = np.where(energy_db > energy_db.max() - SILENCE_DB)[0]
    return audio[voiced[0] * frame:(voiced[-1] + 1) * frame]


def cut_start(audio):
    """Cut the first 30 ms of a trimmed clip and trim its start again (removes the Kathbath click)."""
    return trim_silence(audio[int(START_CUT_S * TARGET_SR):])


def normalise(audio):
    rms = np.sqrt(np.mean(audio ** 2))
    if rms == 0:
        return audio
    audio = audio * (10 ** (TARGET_DBFS / 20) / rms)
    peak = np.abs(audio).max()
    return audio / peak * 0.99 if peak > 0.99 else audio  # never clip


def to_pcm16(data):
    """Raw file bytes -> (int16 samples at 16 kHz, or None if the clip is dropped; original sample rate)."""
    audio, sr = decode(data)
    audio = cut_start(trim_silence(resample(audio, sr)))
    if len(audio) < MIN_SECONDS * TARGET_SR:
        return None, sr
    return np.round(normalise(audio) * 32767).astype(np.int16), sr


def preprocess(data):
    """Raw file bytes -> float32 audio at 16 kHz, exactly as a stored WAV reads back, or None if dropped."""
    pcm, _ = to_pcm16(data)
    return None if pcm is None else pcm.astype(np.float32) / 32768


def sha1(pcm):
    return hashlib.sha1(pcm.tobytes()).hexdigest()


if __name__ == "__main__":
    pcm, sr = to_pcm16(open(sys.argv[1], "rb").read())
    if pcm is None:
        print("dropped: shorter than 1 s after trimming")
    else:
        sf.write(sys.argv[2], pcm, TARGET_SR, subtype="PCM_16")
        print(f"{sr} Hz -> {TARGET_SR} Hz, {len(pcm) / TARGET_SR:.2f} s, sha1 {sha1(pcm)}")
