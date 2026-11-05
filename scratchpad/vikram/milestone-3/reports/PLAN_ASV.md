# PLAN_ASV — ASVspoof 2019 LA preprocessing and augmentation notebook

**Target file:** `scratchpad/vikram/milestone-3/notebook.py`  
**Notebook format:** Colab-oriented Python notebook using `# %%` / `# %% [markdown]` cells  
**Scope:** data loading, standardization, saved Hugging Face dataset, on-the-fly training augmentation, and XLS-R integration contracts  
**Out of scope:** training the classifier, downloading XLS-R weights, or producing a trained model

**Review status:** checked against the pinned dataset card, checkpoint configurations, current library documentation/source, and the numerical checks in Section 14. This is an implementation plan, not evidence that the notebook or full dataset pipeline has been run. Only this plan is being changed; `preprocess.py` and `data_augmentation.md` remain untouched.

---

## 1. Goal and definition of done

Produce a `notebook.py` that:

1. Loads `Bisher/ASVspoof_2019_LA` with the `datasets` library.
2. Preserves the official `train`, `validation`, and `test` splits and the binary labels:
   - `0 = bonafide`
   - `1 = spoof`
3. Standardizes every clip to:
   - mono
   - 16 kHz
   - optional leading/trailing silence trimming, disabled by default
   - RMS loudness normalized to approximately `-20 dBFS`, with peak protection
   - uncompressed 16-bit PCM WAV, with no lossy codec
4. Saves the standardized, unaugmented dataset as a portable Hugging Face dataset on Google Drive.
5. Implements the requested waveform augmentations for **training only**:
   - Gaussian noise at approximately 15 dB SNR
   - random time shift of ±0.15 s
   - pitch-preserving speed change of 0.9–1.1×
   - pitch shift of ±2 semitones
   - synthetic room reverberation with approximately 0.3 s RT60
6. Provides XLS-R configuration for training-only, model-level time/feature masking; two minimum mask-start selections are requested where lengths permit, not exactly two disjoint masks.
7. Provides a model-ready PyTorch dataset, collator, and DataLoader factory for a future XLS-R binary classifier.
8. Includes smoke tests and full-run audit checks.

**The saved dataset contains no augmented copies.** Augmentation is applied on the fly during training, as confirmed in the design discussion. Therefore the train split remains 25,380 rows; different stochastic variants are produced across epochs.

---

## 2. Verified facts and technical decisions

| Item | Decision / verified fact |
|---|---|
| Source dataset | `Bisher/ASVspoof_2019_LA`, revision `aea92dd83a9c56e070c0b1e9f02e7c0d96216a4c` |
| Splits and counts | `train`: 25,380; `validation`: 24,844; `test`: 71,237 |
| Dataset columns | `speaker_id`, `audio_file_name`, `audio`, `system_id`, `key` |
| Labels | `ClassLabel` with names `["bonafide", "spoof"]`; values 0 and 1 |
| Source audio | The card declares an `Audio` feature at 16 kHz. Original channel count, codec, bit depth, and actual duration distribution must be inspected; the card does not establish them. |
| Model | `facebook/wav2vec2-xls-r-300m`, config revision `1a640f32ac3e39899438a2931f9924c02f080a54` |
| Model input | Raw float waveform; not log-mel spectrograms |
| XLS-R feature-extractor normalization | `do_normalize=true`, zero-mean/unit-variance per clip |
| XLS-R feature-extractor norm | `feat_extract_norm="layer"`, so an attention mask must be supplied for padded batches |
| XLS-R default SpecAugment | `apply_spec_augment=true`, `mask_time_prob=0.075`, `mask_time_length=10`, `mask_feature_prob=0.0` |
| Binary head class | `Wav2Vec2ForSequenceClassification` or `AutoModelForAudioClassification` with `num_labels=2` |
| Output audio | 16-bit PCM WAV bytes embedded in the dataset, not external file paths |
| Processed size estimate | At a four-second mean: approximately 15.55 GB of PCM payload. Measure actual durations and add WAV/Arrow overhead; this is not a measured corpus size. |
| Local disk budget | Source download/cache plus processed map cache and saved staging copy; calculate from sampled durations with a safety margin before the full run. Roughly 50–60 GB is a planning estimate, not a guarantee. |
| Drive budget | One verified processed copy plus overhead; check the actual account quota separately. The processed dataset may exceed the free 15 GB Drive allowance. |

### 2.1 Important correction: SpecAugment placement

XLS-R cannot consume a masked spectrogram directly: it expects raw waveforms. Spectrogram masking followed by waveform reconstruction is possible, but was explicitly not selected because of reconstruction artifacts and extra complexity. XLS-R’s built-in SpecAugment instead masks **learned hidden features**:

- time masks replace hidden time steps with `masked_spec_embed`
- feature masks zero spans of learned hidden channels, not physical frequency bands

This plan follows the approved design decision: use XLS-R’s training-only, model-level masking. It is not equivalent to two frequency-band masks on a spectrogram. Mask counts are probabilistic, length-constrained, and can overlap; they are not guaranteed to form two distinct masked regions in every example.

Literal spectrogram masking/reconstruction or custom exact-count feature masking would be separate design changes and are outside this plan. No spectrograms are reconstructed for the current XLS-R pipeline.

### 2.2 Important correction: loudness normalization and XLS-R

The `-20 dBFS` RMS target is applied during preprocessing for consistency and peak safety. The XLS-R feature extractor subsequently applies per-clip zero-mean/unit-variance normalization, so absolute loudness is not preserved at the model input. This is expected.

The RMS step is retained to provide standardized saved audio and peak safety before PCM16 storage. It:

- gives every clip a consistent level before waveform augmentation
- prevents clipping
- does not make SNR calibration necessary or possible by itself: noise must be scaled to each current waveform's measured RMS regardless of its absolute level

RMS normalization is not an established accuracy improvement over LUFS for this task; that requires a validation experiment. Mono mixing, resampling, gain changes, and PCM16 quantization can alter spoof cues. WAV avoids an additional lossy codec, but does not make those transformations lossless relative to the source samples.

### 2.3 Important correction: no augmented dataset rows

The phrase “create more training samples” is implemented as stochastic, epoch-wise sampling, not as a materialized larger dataset. This matches the approved on-the-fly decision and avoids storing additional full copies of the audio.

---

## 3. Notebook cell plan

`notebook.py` will use Jupytext percent cells and must remain syntactically valid Python; use `subprocess` rather than shell magics for installs, and guard Colab-only imports. Convert it to an `.ipynb` with:

```bash
pip install jupytext
jupytext --to ipynb notebook.py
```

Runtime dependencies for the code cells:

- `datasets` with the `Audio(..., num_channels=...)` API (added in 4.4); select and record a tested release during implementation, rather than relying on an unbounded upgrade
- a `torchcodec` build compatible with the installed PyTorch/Python versions, and FFmpeg shared libraries (required by current `datasets` audio decoding)
- `transformers` with `Wav2Vec2ForSequenceClassification`
- `librosa`, its `soxr` resampling dependency, `soundfile`, `scipy`, `numpy`, `torch`, and `matplotlib`/IPython for previews
- `jupytext` for local conversion only

The setup cell must install dependencies before importing them, report package versions, and use TorchCodec's compatibility matrix rather than blindly upgrading Colab's PyTorch/CUDA stack. A successful import and `ffmpeg -version` are not sufficient: create a small WAV in memory and actually decode it through the selected `datasets`/TorchCodec path. Stop with actionable instructions for incompatible binaries or unavailable shared libraries; restart the runtime after replacing already-imported packages. Do not silently change decoder backends.

Suggested cell sequence:

| Cell | Type | Content |
|---|---|---|
| 1 | Markdown | Title, scope, label map, saved dataset contract, model contract |
| 2 | Markdown | Colab setup and runtime instructions |
| 3 | Code | Guarded dependency setup via `subprocess`, then imports, versions, seeds, and a synthetic WAV decoder smoke test |
| 4 | Code | Configuration dataclasses/constants and path setup |
| 5 | Code | Google Drive mount with a non-Colab fallback |
| 6 | Code | Check source-download disk budget, load dataset, audit metadata, then sample durations to estimate processed storage |
| 7 | Markdown | Preprocessing specification |
| 8 | Code | Audio standardization helpers: mono, resample, optional trim, RMS normalization, PCM16 WAV encoding |
| 9 | Code | Synthetic edge-case tests plus one actual example from each split |
| 10 | Code | Smoke-map 16 seeded train examples covering both labels; save/reload and inspect results |
| 11 | Code | Full `Dataset.map` for all three splits |
| 12 | Code | Save/reload locally and mark complete; copy data to Drive, verify it, and publish the Drive completion manifest last |
| 13 | Markdown | Augmentation specification and probabilities |
| 14 | Code | Synthetic RIR, noise, shift, speed, pitch, and composed waveform augmentation |
| 15 | Code | Deterministic augmentation tests and audio/spectrogram previews |
| 16 | Code | `SpoofAudioDataset`, collator, and DataLoader factory |
| 17 | Code | XLS-R configuration/feature extractor only, plus a tiny randomly initialized masking test; no 300M-model allocation |
| 18 | Code | End-to-end smoke test with a small DataLoader batch |
| 19 | Markdown | Runbook, acceptance checklist, limitations |
| 20 | Code | Print final audit summary from the already-written manifest |

---

## 4. Data loading and preprocessing

### 4.1 Load and pin the source

```python
DATASET_ID = "Bisher/ASVspoof_2019_LA"
DATASET_REVISION = "aea92dd83a9c56e070c0b1e9f02e7c0d96216a4c"
EXPECTED_COUNTS = {"train": 25_380, "validation": 24_844, "test": 71_237}
EXPECTED_LABELS = ["bonafide", "spoof"]
TARGET_SR = 16_000

raw = load_dataset(
    DATASET_ID,
    revision=DATASET_REVISION,
    cache_dir=str(LOCAL_HF_CACHE),
)
```

Use `datasets` audio decoding with explicit mono/16 kHz:

```python
raw = raw.cast_column(
    "audio",
    Audio(sampling_rate=TARGET_SR, num_channels=1),
)
```

Before processing, assert:

- the three split names exist and their row counts match the dataset card
- `key` is a `ClassLabel` and its names are exactly `["bonafide", "spoof"]`
- `speaker_id`, `audio_file_name`, and `system_id` are present
- each split has audio rows

Do not merge or re-split. Preserve the official train/validation/test boundaries.

Inspect metadata without decoding the full audio column. Check that utterance IDs are unique within each split and report any cross-split utterance/speaker overlap rather than silently deduplicating or moving rows. Keep `system_id` for audits only. Sample audio for format/storage checks; do not choose augmentation strengths or thresholds from test-set statistics or outcomes.

### 4.2 Decode to a mono float waveform

For the current `datasets`/TorchCodec API:

```python
samples = row["audio"].get_all_samples()
waveform = samples.data.cpu().numpy().astype(np.float32, copy=False)
sample_rate = samples.sample_rate
```

Defensive handling:

- if the decoded tensor is 2-D `[channels, samples]`, average over channels
- if it is 1-D, use it directly
- assert `sample_rate == 16_000`
- if a decoded array is empty or contains non-finite values, fail loudly
- do not drop rows silently; preserving split counts and labels is a requirement

`Audio(sampling_rate=16000, num_channels=1)` performs lazy mono conversion and resampling when accessed; it does not rewrite stored audio by itself. Materialization happens in the mapper when the decoded standardized waveform is encoded to WAV bytes. Do not assume the legacy `row["audio"]["array"]` interface, and do not accidentally resample a waveform that is already at 16 kHz.

### 4.3 Optional silence trimming — disabled by default

`TRIM_SILENCE = False` by default, because leading/trailing pauses can contain useful countermeasure cues and the source clips are already short.

When enabled, use the existing `preprocess.py` threshold/frame policy, with safe boundary handling:

- frame size: 20 ms
- silence threshold: 40 dB below the loudest frame
- trim only leading and trailing silence, not internal pauses
- handle all-zero audio and the final partial frame explicitly; do not discard a partial frame by integer frame division
- if trimming would leave less than 0.25 s, keep the untrimmed waveform and record a warning; this avoids creating clips shorter than the default model masking requirements after a 1.1× speed change

Do not remove files shorter than one second. This intentionally differs from `preprocess.py`’s `MIN_SECONDS` drop because the official dataset splits and row counts must be preserved. The source dataset’s official rows are preserved.

### 4.4 RMS loudness normalization

Configuration:

```python
TARGET_RMS_DBFS = -20.0
PEAK_CEILING = 0.99
PCM16_STEP = 1.0 / 32768.0
```

Algorithm:

1. Compute `rms = sqrt(mean(x ** 2))` in float64.
2. Compute the gain needed to reach `TARGET_RMS_DBFS`.
3. Cap the gain to a pre-encoding peak of `PEAK_CEILING - PCM16_STEP`, leaving one quantization step of headroom.
4. Apply the gain.
5. Record final RMS and peak from the decoded PCM16 result, not just from the pre-quantization float array.

A clip whose peak limit prevents reaching the RMS target is retained at the lower RMS level. This avoids clipping and is preferable to altering the signal with compression or limiting.

All-zero clips are retained, flagged, and skipped by gain normalization/noise addition. Store an undefined RMS dBFS as a nullable value, not a fabricated finite level or invalid JSON infinity. Non-finite or empty waveforms are errors, not silent clips.

### 4.5 WAV encoding

Encode each normalized waveform as **16-bit PCM WAV** in memory:

```python
buffer = io.BytesIO()
sf.write(buffer, waveform, TARGET_SR, format="WAV", subtype="PCM_16")
wav_bytes = buffer.getvalue()
```

Store the bytes in the `Audio` column as:

```python
{"bytes": wav_bytes, "path": None}
```

Use WAV, not Opus. Opus is lossy and could distort the subtle spoofing artifacts this task depends on.

PCM16 quantizes floats. The maximum float round-trip error is approximately one PCM16 step for the selected encoder; include an explicit tolerance in tests. Gain applied to source PCM16 samples means the output bytes generally will not equal the original bytes.

The stored dataset is portable because audio bytes are embedded; it does not depend on Colab download paths.

### 4.6 Added metadata columns

Keep all original columns and add audit columns:

- `duration_s` (`float32`)
- `num_samples` (`int32`)
- `rms_dbfs` (`float32`)
- `peak` (`float32`)
- `is_silent` and `peak_limited` (`bool`)
- `preprocess_version` (`string`, constant)

Do not use `speaker_id`, `system_id`, or `audio_file_name` as model features. They are retained for auditing only; using `system_id` or `speaker_id` would risk label leakage or shortcut learning.

### 4.7 Mapping and saving

Use deterministic row-wise mapping (`batched=False`) with an explicit output `Features` schema preserving `key` as `ClassLabel` and `audio` as `Audio(sampling_rate=16000, num_channels=1)`. Decode/re-encode one clip at a time, not all waveforms in RAM. Default:

```python
NUM_PROC = 1           # opt into multiprocessing only after its smoke test passes
WRITER_BATCH_SIZE = 32
```

The mapper receives decoded audio rows and returns:

- standardized WAV bytes
- duration and level metadata
- unchanged original metadata and label

Use an explicit map-cache path on local Colab disk keyed by the source revision, preprocessing configuration/version, and relevant package versions. Keep source/map caches local; do not write caches to Google Drive. The smoke test must prove that mapped `{"bytes": ..., "path": None}` values survive encoding as an `Audio` feature in the chosen `datasets` release. Use `Audio(decode=False)` on a read-only audit view to inspect stored bytes/path fields; normal decoded rows are decoder objects, not dictionaries.

Save locally with `DatasetDict.save_to_disk`, verify the reload, then write a manifest whose `status` is `"complete"`:

```text
/content/asvspoof2019_la_standardized_v1/
/content/drive/MyDrive/ds-ai-lab/asvspoof2019_la/standardized_v1/
```

The manifest is a completion marker written **last** on each copy, after the dataset has been saved and verified. It records source revision, preprocessing/config fingerprint, package versions, split/label counts, and a file inventory with sizes and SHA-256 checksums. For the Drive copy: copy the data files without the completion marker, verify their checksums and a `load_from_disk` audio sample, then publish the complete manifest. Copying a local `status="complete"` manifest before verifying the destination would falsely certify a damaged/partial copy.

Staging does not prevent an interrupted Drive transfer; the missing completion marker makes that partial transfer detectable. A directory without a complete matching manifest and verified files must not be used or silently reused. Do not overwrite an existing output directory; verify/reuse a complete compatible artifact or choose a new run directory.

Calculate local disk needs from the source cache and sampled duration estimates, allowing for both map-cache and saved copies plus a safety margin. `shutil.disk_usage('/content')` can check local capacity, but a mounted Drive filesystem's free-space report is not a reliable account quota check. Ask the notebook operator to check Drive's quota UI; do not promise that the dataset fits a free account. For later training, prefer copying the verified Drive artifact back to local Colab storage instead of repeatedly decoding from Drive.

---

## 5. On-the-fly training augmentation

### 5.1 Configuration and probabilities

Every waveform effect has an independent, configurable probability of `0.5`, applied identically to bonafide and spoof clips:

```python
P_NOISE = 0.5
P_SHIFT = 0.5
P_SPEED = 0.5
P_PITCH = 0.5
P_REVERB = 0.5
SNR_DB = 15.0
TIME_SHIFT_S = 0.15     # uniform in [-0.15, +0.15]
SPEED_RANGE = (0.9, 1.1)
PITCH_SEMITONES = 2.0   # uniform in [-2.0, +2.0]
RT60_S = 0.3
REVERB_DRR_DB = 6.0     # direct-to-reverberant impulse-response energy ratio
```

The augmentation function must not receive or inspect the label. Use the same per-effect probabilities and parameter distributions for both classes and leave labels unchanged. The observed application fractions need not be exactly equal in finite samples; there must be no label-dependent policy. Validate probabilities/ranges at configuration time.

Because five effects are independent at 0.5, the expected number of effects per clip is 2.5 and the probability of at least one effect is 96.875%. This is intentionally aggressive and should be validated later; all values remain configurable.

### 5.2 Fixed augmentation order

Use one deterministic order for every clip:

1. pitch-preserving speed change
2. pitch shift
3. time shift
4. reverberation
5. additive Gaussian noise
6. final peak-safety scaling

Fixed order makes tests and reproducibility possible. If both speed and pitch are selected, two phase-vocoder operations may introduce artifacts; that is a documented limitation, not a hidden behavior.

### 5.3 Speed change — pitch preserving

Use librosa:

```python
y = librosa.effects.time_stretch(y, rate=rate)
```

- `rate` is drawn uniformly from `[0.9, 1.1]`
- `rate > 1` speeds the clip up; `rate < 1` slows it down
- output length is approximately `len(y) / rate`
- this changes duration, which is allowed; no fixed four-second crop is used

Do not catch failures and silently return unchanged audio. Validate input/model length constraints first (Section 6.4), then raise an error identifying the row and effect if a selected transformation fails. Cast the output back to contiguous float32 and check that it is nonempty and finite.

### 5.4 Pitch shift

Use librosa:

```python
y = librosa.effects.pitch_shift(
    y, sr=TARGET_SR, n_steps=n_steps, bins_per_octave=12, res_type="soxr_hq"
)
```

- `n_steps` is drawn uniformly from `[-2.0, +2.0]` semitones
- output length is preserved by librosa
- set `bins_per_octave=12` and `res_type="soxr_hq"` explicitly, and verify the resampler dependency in setup

### 5.5 Time shift

Shift with zero padding, not circular wrapping:

```python
n = int(round(shift_seconds * TARGET_SR))
n = int(np.clip(n, -(len(y) - 1), len(y) - 1))
if n > 0:
    y = np.concatenate([np.zeros(n, dtype=np.float32), y[:-n]])
elif n < 0:
    y = np.concatenate([y[-n:], np.zeros(-n, dtype=np.float32)])
```

- `shift_seconds` is drawn uniformly from `[-0.15, +0.15]`
- preserve the current length
- zero-fill the opened edge; do not use `np.roll`
- samples shifted beyond the opposite edge are discarded (up to 0.15 s); preserving natural duration means no fixed four-second crop, not that a time shift can retain all edge samples

### 5.6 Synthetic reverberation

Generate an approximate direct-path-plus-diffuse-tail impulse response locally. Control **tail energy**, not just the RIR peak; the previous peak-normalized noise tail could dominate the direct path by a large factor. Starting defaults:

```python
rt60 = RT60_S
n = int(np.ceil(2.0 * rt60 * TARGET_SR))  # decay beyond the target RT60
t = np.arange(n) / TARGET_SR
tail = rng.standard_normal(n) * 10.0 ** (-3.0 * t / rt60)
tail[0] = 0.0
tail_norm = np.linalg.norm(tail)
if not np.isfinite(tail_norm) or tail_norm == 0.0:
    raise ValueError("Invalid synthetic reverberation tail")
tail *= 10.0 ** (-REVERB_DRR_DB / 20.0) / tail_norm
rir = tail.copy()
rir[0] = 1.0  # direct path, with finite, controlled reverberant energy
```

Apply as a causal convolution and preserve the input length:

```python
y = scipy.signal.fftconvolve(y, rir, mode="full")[:len(y)].astype(np.float32)
```

Do **not** use `mode="same"`: it centers the convolution and can shift/delay the direct path. `mode="full"[:len(y)]` keeps the direct path aligned and truncates the final tail to preserve duration.

The exponential specifies a nominal decay envelope, not by itself a measured room RT60. Verify approximate decay with a Schroeder energy-decay curve of the diffuse tail: integrate squared samples backward, fit the -5 to -25 dB region, and extrapolate the fitted slope to a 60 dB decay. Test several fixed seeds with a documented tolerance (initially 0.30 ± 0.05 s). Also verify that the direct-to-tail energy ratio is approximately 6 dB. This is a reproducible augmentation proxy, not a physically simulated or measured room.

### 5.7 Additive Gaussian noise

Generate white Gaussian noise and scale it to an approximately 15 dB SNR relative to the current waveform:

```python
signal_rms = float(np.sqrt(np.mean(np.square(y, dtype=np.float64))))
if signal_rms > 0.0:
    noise = rng.standard_normal(len(y))
    noise_rms = float(np.sqrt(np.mean(noise ** 2)))
    if noise_rms == 0.0:
        raise ValueError("Cannot calibrate zero-power noise")
    noise *= signal_rms / (10.0 ** (SNR_DB / 20.0) * noise_rms)
    y = (y + noise).astype(np.float32)
```

- compute powers in float64, guard zero signal/noise power, and cast the result back to float32
- SNR is defined over the whole current clip, not a speech-only VAD estimate
- final uniform gain preserves the component-power ratio. The feature extractor additionally subtracts the mixture mean, so do not claim it guarantees the exact same whole-clip SNR after that centering step
- test SNR at the noise-addition stage using the immediate pre-noise waveform as the clean reference; after peak scaling, scale that reference by the same factor before measuring

### 5.8 Final peak guard

After all effects:

```python
peak = max(abs(y))
if peak > 0.99:
    y *= 0.99 / peak
```

Do not re-normalize to the `-20 dBFS` RMS target after augmentation. Peak protection is enough, and re-normalizing would not restore a more meaningful loudness reference. Scaling preserves the noise-to-signal ratio.

### 5.9 Reproducibility and train-only guard

`SpoofAudioDataset.__getitem__` uses a per-example RNG:

```python
split_id = {"train": 0, "validation": 1, "test": 2}[split_name]
rng = np.random.default_rng(
    np.random.SeedSequence([base_seed, epoch, split_id, row_index])
)
```

Requirements:

- `set_epoch(epoch)` is called before creating each training DataLoader iterator
- validation and test wrappers use `augment=False` and must not instantiate an RNG
- constructing a loader with `augment=True` for a non-train split must raise `ValueError`
- use `persistent_workers=False` unless epoch propagation is explicitly solved; persistent workers can retain stale epoch seeds
- seed Python, NumPy's global RNG, and PyTorch separately; `default_rng` does not seed the global NumPy RNG used by the checked Transformers masking implementation
- seed the DataLoader's shuffle generator separately; use worker initialization when adding any other worker-local random transforms

This policy deliberately returns the same variant for the same row within one epoch and different stochastic variants across epochs. Validation/test reads must be identical and never run waveform augmentation. Unit tests should force a random effect such as noise to run when testing epoch variation; do not make a flaky assertion that every independently sampled chain must differ (all effects can be skipped).

---

## 6. SpecAugment for XLS-R

### 6.1 Configuration

Load only the small configuration file and configure it **before constructing a future model**. This notebook must not instantiate XLS-R-300M, even with random weights, just to demonstrate masking:

```python
MODEL_ID = "facebook/wav2vec2-xls-r-300m"
MODEL_REVISION = "1a640f32ac3e39899438a2931f9924c02f080a54"
model_config = Wav2Vec2Config.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
model_config.num_labels = 2
model_config.id2label = {0: "bonafide", 1: "spoof"}
model_config.label2id = {"bonafide": 0, "spoof": 1}
model_config.apply_spec_augment = True
model_config.mask_time_prob = 0.075
model_config.mask_time_length = 10
model_config.mask_time_min_masks = 2
model_config.mask_feature_prob = 0.02
model_config.mask_feature_length = 10
model_config.mask_feature_min_masks = 2
```

These values are starting defaults, not a guarantee of exactly two masks:

- before length caps, the sampler requests `max(min_masks, floor(mask_prob * valid_length / mask_length + U))`, where `U` is the implementation's random rounding term in `[0, 1)`
- `min_masks=2` is a requested minimum number of selected starts, subject to length caps; overlapping/adjacent spans can merge into a single connected masked region
- feature masks operate over the 1,024 hidden channels, not mel bins
- a 64,000-sample clip produces 199 hidden frames; these settings generally request two time starts and two (occasionally three) feature starts over 1,024 channels
- longer clips may produce more than two time spans

Document this clearly in the notebook markdown. If strict “exactly two” behavior is required, it must be implemented separately and is not part of the approved XLS-R-native design.

The future classifier should use `Wav2Vec2ForSequenceClassification` or `AutoModelForAudioClassification` with this config and a newly initialized two-class head, not a CTC transcription model/tokenizer. This is an integration contract only; do not execute the large `from_pretrained` model call here. SpecAugment will not modify saved waveforms or run merely because a DataLoader is iterated.

### 6.2 Training/inference behavior

For automatically sampled masks, the checked `Wav2Vec2Model._mask_hidden_states` implementation requires:

- `config.apply_spec_augment` is true
- the corresponding probability is positive
- the model is in training mode

Therefore:

- call `model.train()` for training
- call `model.eval()` for validation, test, and inference
- do not pass explicit `mask_time_indices` at evaluation: manually supplied time masks are a separate code path that can apply even in `eval()` mode
- if the encoder is frozen for head-only training but the base is kept in `eval()` to freeze dropout, SpecAugment will be disabled; either accept that or keep the base in `train()` while freezing parameters

### 6.3 Feature extractor and attention mask

Use:

```python
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    MODEL_ID, revision=MODEL_REVISION
)

batch = feature_extractor(
    waveforms,
    sampling_rate=16_000,
    padding=True,
    truncation=False,
    return_attention_mask=True,
    return_tensors="pt",
)
```

XLS-R uses `feat_extract_norm="layer"`, so pass `attention_mask` to the model. Do not drop it.

The feature extractor’s `do_normalize=True` performs zero-mean/unit-variance normalization. This is independent of the preprocessing RMS target and is expected.

Normalize each clip using only its valid samples; the attention mask must exclude batch padding from its statistics. Zero-filled time-shift regions inside a recording remain part of the valid recording, not batch padding. The resulting normalized tensors can exceed `[-1, 1]`; do not clip them back to a waveform peak ceiling.

### 6.4 Short-recording safeguards

Derive frame counts from the checkpoint's actual convolution kernels/strides, applying `L_out = floor((L_in - kernel) / stride) + 1` at every layer. With this checkpoint:

- at least 400 input samples (25 ms) are needed for one convolutional output frame
- at least 3,280 input samples (205 ms) are needed for `mask_time_length=10` frames
- two requested length-10 mask starts may still be capped or overlap on short sequences

Audit standardized lengths and check actual lengths again after speed augmentation. If a recording is too short for its intended model/masking path, raise an informative error with its split/ID; do not silently drop it, invent padding as valid speech, or make behavior depend on a longer example happening to share the batch. The row remains in the saved artifact. A changed global mask length, if needed, must be an explicit configuration decision and pass the smoke tests.

---

## 7. Dataset wrapper and DataLoader contract

### 7.1 `SpoofAudioDataset`

A lightweight wrapper over the saved Hugging Face dataset:

- `split_name`
- `augment: bool`
- `base_seed`
- `epoch`
- `preprocess_version`

`__getitem__` returns:

```python
{
    "input_values": np.ndarray,  # float32 mono waveform at 16 kHz
    "labels": int,               # 0 = bonafide, 1 = spoof
    "duration_s": float,
    "audio_file_name": str,
}
```

Decode the saved audio bytes to a waveform. Keep `input_values` as a variable-length 1-D array; no fixed-length crop.

Use `key` as the label source and never derive labels from filename prefixes or `system_id`. If returning `duration_s`, compute it from the possibly augmented waveform; the saved dataset's `duration_s` describes the unaugmented standardized audio. The waveform-augmentation flag is not a replacement for the future PyTorch model's `train()`/`eval()` state.

### 7.2 Collator

Custom collator:

1. Convert waveforms to a list.
2. Use `Wav2Vec2FeatureExtractor` with `padding=True` and `return_attention_mask=True`.
3. Return only `input_values` (float32), `attention_mask` (integer/bool), and `labels` (int64, shape `[batch]`) to the future two-logit classifier.
4. Do not truncate. Reduce batch size if variable lengths cause memory pressure; length bucketing or cropping would be a separate training decision.

### 7.3 DataLoaders

- train: `shuffle=True`, `augment=True`; call the wrapper's `set_epoch(epoch)` before creating each iterator
- validation/test: `shuffle=False`, `augment=False`
- `num_workers=0` for the initial notebook smoke test; opt into worker processes only after verifying decoding and epoch propagation
- `pin_memory=True` when using a GPU
- `persistent_workers=False` when using epoch-seeded augmentation
- no class resampling in this notebook

The ASVspoof LA splits are strongly imbalanced (roughly 10% bonafide / 90% spoof). Report the exact distribution in the audit and leave imbalance handling to later training; do not balance, resample, or relabel the saved dataset here.

---

## 8. Verification plan

### 8.1 Static checks

- `python -m py_compile notebook.py`
- `jupytext --to ipynb notebook.py` succeeds
- no secrets, tokens, or raw `.env` contents in the notebook
- all dependency installation is guarded and version-reported

### 8.2 Preprocessing unit checks

Run on one example from each split and assert:

- decoded audio is mono and float32
- sample rate is 16,000 Hz
- decoded PCM16 output peak is `<= 0.99`, using the pre-encoding quantization headroom
- output RMS is approximately `-20 dBFS` (e.g. ±0.1 dB) unless peak-limited or silent; audit metadata must match the stored decoded samples
- WAV bytes decode successfully with `soundfile`
- PCM16 round-trip error is within one quantization step
- optional trimming is off and does not change the default output
- labels remain 0/1 and names remain `bonafide`/`spoof`

Also test synthetic stereo/downmix behavior, a non-16 kHz input, all-zero audio, a final partial silence-trim frame, a high-crest-factor peak-limited signal, and invalid empty/NaN input. The test must demonstrate actual mono conversion/resampling, not only assert that this particular dataset already matches the target format.

### 8.3 Smoke-map checks

Before the full run:

- map 16 seeded train examples, selecting eight per label from metadata so source ordering cannot yield a one-class smoke test
- save to a temporary directory
- reload with `load_from_disk`
- decode several saved rows
- compare counts and labels with the source; compare the saved WAV bytes with a second deterministic run of the same preprocessing function
- display one waveform and log-mel before/after standardization

The log-mel plot is for visual inspection only. XLS-R still receives the raw standardized waveform; no log-mel features are saved or fed to the model.

Standardized audio is intentionally different from the source audio, so source-vs-output waveform hashes are not expected to match. The useful hash check is deterministic output for the same input and config.

Only after these pass should the full 121,461-row mapping run start.

### 8.4 Full-run and reload audit

For the full saved dataset:

- split counts exactly match 25,380 / 24,844 / 71,237
- label names and label values are unchanged
- every sampled audio payload has non-empty embedded WAV bytes and no external path
- sampled audio decodes at 16 kHz mono
- no non-finite samples
- duration metadata matches decoded sample counts
- class distribution is reported per split
- reload from the Drive copy and repeat the sample audit
- verify the manifest records source revision, preprocessing version, config, and package versions
- verify copied shard sizes/checksums before marking Drive complete; reject a missing shard, mismatched fingerprint, or incomplete manifest

### 8.5 Augmentation checks

- Noise-only test: measured pre-peak-guard SNR is 15 ± 0.1 dB; separately check that scaling both clean/noise components preserves it
- Speed-only test: output sample count matches `round(input_samples / rate)` within one sample
- Pitch-only test: output duration is preserved and samples are finite
- Shift-only test: no circular wrapping; shifted edge is zero-filled; length preserved
- Reverb-only test: direct impulse is aligned at sample zero, output length is preserved, tail energy is controlled by the DRR setting, and estimated tail RT60 is 0.30 ± 0.05 s across fixed seeds
- Composition test: all effects remain finite and peak-guarded
- Same RNG seed/epoch/index produces identical output; with noise forced on, selected different epoch seeds produce distinct output. Test probability-gating logic independently instead of requiring every random chain to differ
- Validation/test wrapper returns bitwise-identical arrays across repeated accesses
- `augment=True` on validation/test raises an error

### 8.6 SpecAugment checks

- config fields are set as specified
- `apply_spec_augment` is true
- `mask_time_min_masks == 2` and `mask_feature_min_masks == 2`
- test masking behavior on a **tiny, randomly initialized** `Wav2Vec2Model`, not a full/random 300M model or downloaded weights. Use compatible dimensions: hidden size 64, two attention heads, one Transformer layer, intermediate size 128, and seven 16-channel convolution layers with the checkpoint's kernels/strides; do not leave the default 12 heads with a 64-dimensional hidden size
- test time and feature masking separately on known synthetic hidden states; disable unrelated dropout and do not infer mask behavior solely from train/eval output differences
- masks have valid shapes/length bounds and can overlap; do not count contiguous regions as a proof of the requested minimum-start count
- without explicit mask indices, the masking branch is unchanged in `eval()` and applies configured masks in `train()`
- attention masks are passed through the feature extractor and model
- check short inputs explicitly; never rely on batch padding to satisfy a per-recording convolution/mask length requirement

Private helpers such as `_mask_hidden_states` may be inspected in a version-specific smoke test to isolate masking, but the production dataset/collator must not depend on private mask-generation APIs. The actual XLS-R-300M forward/training integration remains unverified until the separate model-training task.

### 8.7 DataLoader smoke test

Create a small train loader and a small validation loader. Assert:

- `input_values` shape is `[batch, max_samples_in_batch]`
- `attention_mask` has the same shape
- `labels` shape is `[batch]` and values are 0/1
- labels are int64 and mask sums equal actual post-augmentation lengths; padded entries remain zero after feature-extractor normalization
- train batches vary across chosen epochs when a random effect is forced on; hold row order fixed for this assertion so shuffling is not mistaken for augmentation
- validation batches are identical across epochs
- no `speaker_id`, `system_id`, or `audio_file_name` is forwarded to the model
- run the default zero-worker test first; any optional multi-worker test must verify `set_epoch` takes effect with `persistent_workers=False`

---

## 9. Error handling and storage behavior

- Fail loudly on decode errors, non-finite audio, empty arrays, or missing splits; do not silently drop rows.
- If a clip is silent, retain it and skip RMS/noise operations that would divide by zero.
- If an optional trim would remove almost everything, retain the original waveform and warn.
- If a librosa effect fails, raise with the row identifier; do not silently substitute unaugmented audio without recording it.
- Reuse a local/Drive artifact only after matching its manifest/config fingerprint and checking its file inventory and a reload sample. A matching manifest alone does not prove that all files still exist.
- Stop on a mismatched or incomplete existing output directory. Choose a new versioned/run directory; do not delete or overwrite prior work automatically.
- Do not edit original source files or manually clear the Hugging Face cache. Library-managed downloads/caches and new explicitly named map-cache files are expected.
- Do not commit dataset artifacts to git.

---

## 10. Colab runbook

1. Install and convert:
   ```bash
   pip install jupytext
   jupytext --to ipynb notebook.py
   ```
2. Open the generated `.ipynb` in Colab.
3. Select a GPU runtime only when moving on to model training; preprocessing is CPU-bound.
4. Run setup and the synthetic WAV decoder test; restart after replacing already-imported packages if needed.
5. Mount Google Drive, confirm the output path and actual account quota, and check local download capacity. Loading the non-streaming source is already a roughly 7.5 GB download; a later `.select(...)` smoke subset does not avoid that download.
6. Run the 16-example smoke map and inspect the previews.
7. Run the full preprocessing map and the reload audit.
8. Run the augmentation previews and DataLoader smoke test.
9. Keep the saved Drive dataset; do not re-run the expensive preprocessing unless the config or source revision changes.

Expected resource use:

- source download: approximately 7.5 GB
- processed WAV payload: `sum(duration_s) * 16_000 * 2` bytes plus container/Arrow overhead; 121,461 clips averaging four seconds would be about 15.55 GB
- local storage: source download/Arrow cache + map cache + saved copy + safety margin; estimate from seeded samples of each split, then check before the full map
- Drive storage: one verified saved copy plus margin; may exceed the free 15 GB quota
- processing time: use a larger representative pilot if estimating throughput; 16 examples are a correctness smoke test, not a reliable runtime/disk-size forecast

---

## 11. Risks and limitations

1. **SpecAugment is adapted, not literal.** Model-level hidden-feature masking is not the same as two time/frequency masks on a log-mel spectrogram.
2. **Mask counts are not exactly two.** Requested starts are probabilistically rounded and length-capped, and spans can overlap/merge. A requested minimum is not a guarantee of two visible regions.
3. **Synthetic noise and RIR are approximations.** White Gaussian noise is not environmental noise, and the synthetic RIR is not a measured room. Both are configurable and can be replaced later.
4. **Phase vocoder artifacts.** Independent speed and pitch changes can stack and distort spoofing cues. The 0.5 probabilities are aggressive.
5. **RMS is not perceptual loudness.** The target is RMS dBFS, not LUFS. XLS-R’s feature extractor normalizes per clip afterward anyway.
6. **Evaluation limits.** Accuracy alone is misleading under class imbalance. Future training should evaluate EER, ROC-AUC, and per-class recall, choose hyperparameters/thresholds on validation, and reserve test for final evaluation. min-tDCF additionally requires a compatible ASV system's scores/error rates and protocol/cost assumptions; it cannot be obtained from the binary classifier's predictions alone.
7. **Metadata leakage.** `system_id`, `speaker_id`, and `audio_file_name` must never become model inputs.
8. **Storage pressure.** Size/throughput estimates are not measured full-corpus results. Check quotas, stage locally, verify the Drive copy, then publish its completion marker.
9. **Dependency drift.** `datasets` audio decoding, TorchCodec, and Transformers APIs change. Pin the dataset revision, print package versions, and run the smoke tests before the full run.
10. **No augmented rows are saved.** Downstream users must call the augmentation wrapper; treating the saved train split as already augmented would silently remove augmentation.
11. **No fixed crop.** Variable-length batching is required; downstream training must pad and mask correctly.

---

## 12. Acceptance criteria

The plan is complete when the implemented `notebook.py` satisfies all of the following:

- [ ] Uses `datasets` to load `Bisher/ASVspoof_2019_LA` at the pinned revision.
- [ ] Preserves the three official splits and binary labels.
- [ ] Produces mono, 16 kHz, PCM16 WAV, RMS-normalized, peak-safe audio.
- [ ] Trims silence only when explicitly enabled; default is off.
- [ ] Saves a portable Hugging Face dataset with embedded audio bytes to Drive.
- [ ] Implements all five waveform augmentations on the fly for train only.
- [ ] Keeps validation and test completely unaugmented.
- [ ] Configures XLS-R model-level SpecAugment with training-only time/feature masking.
- [ ] Provides a model-ready dataset, collator, and DataLoader factory.
- [ ] Validates short inputs, PCM16 quantization headroom, controlled reverb energy/decay, and safe artifact reuse.
- [ ] Passes the static, smoke, full-run, reload, augmentation, and DataLoader checks.
- [ ] Documents the SpecAugment adaptation, class imbalance, metadata leakage risk, and storage requirements.
- [ ] Does not train a classifier or download full XLS-R weights.

---

## 13. References

- Pinned dataset card: <https://huggingface.co/datasets/Bisher/ASVspoof_2019_LA/blob/aea92dd83a9c56e070c0b1e9f02e7c0d96216a4c/README.md>
- XLS-R model: <https://huggingface.co/facebook/wav2vec2-xls-r-300m>
- Pinned model configuration: <https://huggingface.co/facebook/wav2vec2-xls-r-300m/blob/1a640f32ac3e39899438a2931f9924c02f080a54/config.json>
- Pinned waveform processor configuration: <https://huggingface.co/facebook/wav2vec2-xls-r-300m/blob/1a640f32ac3e39899438a2931f9924c02f080a54/preprocessor_config.json>
- XLS-R documentation: <https://huggingface.co/docs/transformers/model_doc/xls_r>
- Wav2Vec2 documentation: <https://huggingface.co/docs/transformers/model_doc/wav2vec2>
- Wav2Vec2 feature extractor: <https://huggingface.co/docs/transformers/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor>
- Hugging Face Datasets audio guide: <https://huggingface.co/docs/datasets/audio_process>
- TorchCodec installation/compatibility: <https://github.com/meta-pytorch/torchcodec#installing-torchcodec>
- librosa time stretching and pitch shifting: <https://librosa.org/doc/latest/effects.html>
- Jupytext conversion examples: <https://github.com/jupytext/jupytext#command-line>
- Transformers masking implementation inspected during review: <https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2/modeling_wav2vec2.py>

---

## 14. Review evidence and remaining verification

Completed during plan review (not notebook execution):

- Rechecked the pinned dataset card: split names/counts, column names, label order, declared 16 kHz feature, and 7,543,123,064-byte download size.
- Rechecked XLS-R config/processor files: 1,024 hidden channels, convolution kernels/strides, layer-normalized frontend, 16 kHz input, attention masks, waveform normalization, and default time/feature mask probabilities.
- Compared mask semantics with Transformers source: random rounding, length caps, overlap, training gate, and the explicit-time-mask evaluation exception.
- Checked current `datasets`, TorchCodec, librosa, and Jupytext documentation, including PyTorch/TorchCodec compatibility, duration semantics, and the notebook conversion command.
- Ran a transient NumPy numerical probe of the revised RIR: direct-to-tail energy ratio was 6.000 dB, and tail RT60 estimates over seeds 0–19 ranged from 0.290 to 0.310 s. This validates the proposed synthetic construction, not real-room acoustics or classifier accuracy.
- Verified convolution frame arithmetic: 400 samples give one frame; 3,280 give ten; 64,000 give 199. PCM step arithmetic also illustrated how rounding a negative peak at -0.99 can exceed the nominal 0.99 absolute ceiling without headroom; the actual SoundFile encoder round-trip test remains pending.
- Checked this document's 22 balanced fenced blocks, compiled all 19 Python examples for syntax without executing them, and checked section numbering and obsolete settings/placeholders. These static checks do not validate external API calls or runtime behavior.

Not yet verified: dependency setup in a live Colab runtime, decoding this mirror's actual audio, notebook conversion/execution, full preprocessing/save/reload, Drive quota/copy behavior, tiny-model masking tests, downstream XLS-R-300M training, or any accuracy benefit from augmentation. Required audio/ML packages are not installed in the review environment, and no packages were installed or dataset audio downloaded for this planning task. Those runtime checks are explicit implementation acceptance gates, not completed results.
