# PLAN_IWD — In-The-Wild preprocessing, speaker-disjoint splits, and augmentation notebook

**Target file:** `scratchpad/vikram/milestone-3/iwd_notebook.py`  
**Notebook format:** self-contained, Colab-oriented Python notebook using `# %%` / `# %% [markdown]` cells  
**Scope:** pinned archive loading, explicit metadata joins, project-defined speaker-disjoint splits, standardization, a saved Hugging Face dataset, on-the-fly training augmentation, and XLS-R integration contracts  
**Out of scope:** combining IWD with ASVspoof, training a classifier, downloading XLS-R weights, or producing a trained model

**Agreed design:** use IWD as an independent training dataset; target **70/15/15 by clip count**, keep speakers disjoint, and retain the ASV preprocessing and augmentation defaults. Produce a separate notebook without changing `PLAN_ASV.md`, `asv_notebook.py`, `asv_notebook.ipynb`, `preprocess.py`, or `data_augmentation.md`.

**Review status:** the pinned repository, ZIP directory, and complete source metadata were checked without downloading audio payloads. The accompanying notebook must distinguish synthetic/local checks from full-corpus and Colab execution. Acceptance gates below are not claims that the full dataset or Drive workflow has run.

---

## 1. Goal and definition of done

Produce an `iwd_notebook.py` that:

1. Downloads the pinned `mueller91/In-The-Wild` release archive using `huggingface_hub` and builds a Hugging Face `DatasetDict` using `datasets`.
2. Reads `meta.csv` explicitly and joins every audio member to its exact filename, speaker, and label. Do not rely on automatic folder-label inference.
3. Creates reproducible, **speaker-disjoint** `train`, `validation`, and `test` splits targeting 70%, 15%, and 15% of clips, respectively, with approximate preservation of the corpus class proportions.
4. Preserves all **31,779** source rows exactly once in the full artifact, without balancing, duplicating, or silently dropping recordings.
5. Exposes the ASV-compatible binary label contract:
   - `0 = bonafide`
   - `1 = spoof`
6. Standardizes every recording to mono, 16 kHz, optional edge-silence trimming **off by default**, approximately `-20 dBFS` RMS with peak protection, and uncompressed 16-bit PCM WAV.
7. Saves a portable, standardized, **unaugmented** Hugging Face dataset and its frozen split assignments to Google Drive, with a verified local staging copy and completion manifest.
8. Applies the same five waveform augmentations as ASV, **on the fly and for training only**: 15 dB Gaussian noise, ±0.15 s shift, pitch-preserving 0.9–1.1× speed, ±2-semitone pitch, and synthetic room reverberation with approximately 0.3 s RT60.
9. Provides XLS-R-native training-only time/feature masking configuration, a PyTorch dataset wrapper, a collator, and DataLoader factories for a future binary classifier.
10. Includes source/split audits, synthetic tests, save/reload and corruption checks, augmentation previews, tiny-model masking tests, and model-ready batch smoke tests.

The Hugging Face repository's single split is a packaging detail, **not an official training protocol**. These are project-defined splits for an independent IWD experiment. They must not be presented as the original evaluation-only benchmark protocol.

No augmented rows are saved. The actual derived training count is fixed by the frozen speaker assignment; stochastic variants change across epochs without changing that count.

---

## 2. Verified facts and technical decisions

| Item | Decision / verified fact |
|---|---|
| Source | `mueller91/In-The-Wild` |
| Pinned revision | `eee168f92c367f8c82ff2cf42b6f61e362fd6211` |
| Repository payload | `release_in_the_wild.zip`; no released labeled Parquet table or official train/validation/test protocol |
| Archive size | 8,161,489,023 bytes, approximately 8.16 GB |
| Archive SHA-256 | `46665f30a6758b45642c659c7e0e80da2ba3d121b064716824a56a010ddf9e1a` from the pinned Hub LFS metadata; verify the actual downloaded file during execution |
| ZIP inventory | 31,782 entries: 31,779 WAV members, `meta.csv`, `attribution.txt`, and one directory entry |
| Metadata location | `release_in_the_wild/meta.csv` |
| Metadata schema | `file`, `speaker`, `label` |
| Metadata SHA-256 | `2e404150eff3cf171b766da81013a93248cbb219682fff420bbadb6b86a18795` for the original CSV bytes |
| Metadata/audio join | All 31,779 exact basenames are unique and align one-to-one; no missing file, speaker, or label values were found |
| Original labels | `bona-fide`: 19,963; `spoof`: 11,816 |
| Canonical labels | Explicitly map `bona-fide` → `0`, `spoof` → `1`; `ClassLabel(names=["bonafide", "spoof"])` |
| Speaker groups | **54 distinct released speaker strings**; all 54 have both labels. The card/paper reports 58 people; retain and document this discrepancy rather than inventing four identities. |
| Source provenance | No source-video ID, attack ID, or official split column in `meta.csv` |
| HF viewer behavior | The observed `train` preview exposes only `audio`; it does not supply the labels/speakers needed for this task |
| Source audio | The earlier Milestone 2 audit sampled 2,000 mono 16 kHz WAV/FLOAT recordings. This is prior sampled evidence, not a new full-corpus format guarantee. |
| Uncompressed source WAV payload | 8,722,611,294 bytes, from the ZIP directory; do not extract a second full source copy |
| Published duration | Approximately 37.9 h; not a newly measured sum of decoded durations |
| Processed payload estimate | 37.9 h × 3,600 × 16,000 × 2 ≈ **4.37 GB**, plus WAV/Arrow/metadata overhead; measure during execution |
| Splitting | Whole-speaker, deterministic, seed 42; target 70/15/15 **clips**, not 70/15/15 speakers |
| Model | `facebook/wav2vec2-xls-r-300m`, revision `1a640f32ac3e39899438a2931f9924c02f080a54` |
| Model input/output | Raw variable-length float waveforms with attention masks; future two-logit audio-classification head |
| Output audio | Embedded PCM16 WAV bytes, no external file paths |
| Licensing | The HF card declares CC BY-SA 4.0; the authors' website declares Apache 2.0. Record both sources and preserve attribution; this plan does not resolve the discrepancy or authorize redistribution. |

### 2.1 Consistency with ASV

Keep the same signal processing, augmentation strengths/order/probabilities, peak/quantization safeguards, waveform normalization contract, masking semantics, worker/epoch policy, and error handling as the ASV notebook. Do not introduce MP3/Opus round-trips, µ-law, new environmental noise corpora, fixed crops, or IWD-specific augmentation changes into this baseline.

Necessary differences are source ingestion, label spelling, available metadata, derived splitting, provenance/fingerprints, and storage estimates. IWD has no `system_id`; do not fabricate one to imitate ASV.

### 2.2 Loudness and masking retain the same caveats

The `-20 dBFS` target is RMS, not LUFS. XLS-R's feature extractor subsequently performs per-clip zero-mean/unit-variance normalization; absolute waveform loudness is therefore not preserved at model input. Noise must still be calibrated against the current waveform's measured RMS.

XLS-R-native SpecAugment masks learned time steps and hidden channels, **not physical spectrogram frequency bands**. It does not reconstruct a masked spectrogram or guarantee exactly two distinct masked regions. Mono conversion, gain, resampling, PCM16 quantization, and augmentation can change spoof cues; consistency is not evidence of improved accuracy.

---

## 3. Notebook cell plan

The notebook must remain valid Python. Use guarded `subprocess` installation instead of shell magics, and guard Colab imports. Convert locally with:

```bash
pip install jupytext
jupytext --to ipynb iwd_notebook.py
```

Use the same bounded dependency versions as the existing ASV notebook where possible: `datasets==4.4.1`, `transformers==4.57.3`, `librosa==0.11.0`, and `soundfile==0.13.1`. Also require `huggingface_hub`, NumPy, SciPy, PyTorch, a compatible TorchCodec build, FFmpeg shared libraries, `soxr`, Matplotlib, and IPython. Jupytext is for conversion only. Record all producing versions in the manifest.

Preserve the runtime's installed PyTorch/CUDA stack. Select TorchCodec using its compatibility matrix, not an unbounded upgrade. Installing packages after importing them requires a restart. Actually decode an in-memory WAV through `datasets`/TorchCodec; successful imports or `ffmpeg -version` alone are insufficient. Do not silently substitute decoder backends.

The logical sequence is:

| Stage | Content |
|---|---|
| 1 | Scope, custom split protocol, labels, conversion/run instructions, and storage warning |
| 2 | Guarded dependency setup, imports, package versions, configuration, and seeds |
| 3 | Standardization helpers and synthetic stereo/resampling/PCM16/trim tests |
| 4 | Archive/metadata validation, grouped splitting, frozen assignment helpers, and synthetic archive/split checks |
| 5 | Portable artifact save/load/copy/reuse helpers and corruption/incomplete-copy tests |
| 6 | Waveform augmentation, reproducibility, and numerical checks |
| 7 | XLS-R configuration, length safeguards, dataset/collator/loaders, and tiny-model tests |
| 8 | Drive mount and verified-artifact discovery **before any raw source download** |
| 9 | If needed, capacity preflight, pinned ZIP download/SHA-256, and complete metadata inventory |
| 10 | Construct/freeze the full speaker assignment and print the protocol audit |
| 11 | Seeded source-format/duration pilot for storage, plus one standardized example per split |
| 12 | Process 16 training + 2 validation + 2 test smoke examples; save/reload and compare deterministic bytes |
| 13 | Explicitly opt into full generation, local save/reload, verified Drive publication, and manifest-last completion |
| 14 | Model-length audit, loader smoke tests, and train-only previews |
| 15 | Runbook, limitations, and final summary clearly identifying smoke versus full mode |

Default to `RUN_FULL_PREPROCESS=False`, zero DataLoader workers, and one preprocessing process. Smoke mode still needs the full 8.16 GB archive unless reusing a verified artifact; selecting 20 rows does not avoid that download. Synthetic/offline tests must run before the source download.

---

## 4. Data loading, splitting, and preprocessing

### 4.1 Pin and verify the archive

```python
DATASET_ID = "mueller91/In-The-Wild"
DATASET_REVISION = "eee168f92c367f8c82ff2cf42b6f61e362fd6211"
ARCHIVE_NAME = "release_in_the_wild.zip"
EXPECTED_TOTAL = 31_779
EXPECTED_LABEL_COUNTS = {"bona-fide": 19_963, "spoof": 11_816}
SOURCE_LABEL_MAP = {"bona-fide": 0, "spoof": 1}
TARGET_SR = 16_000

archive_path = hf_hub_download(
    repo_id=DATASET_ID,
    filename=ARCHIVE_NAME,
    repo_type="dataset",
    revision=DATASET_REVISION,
    cache_dir=str(SOURCE_CACHE),
)
```

Before this call, check available local disk for the still-needed download with a safety margin. An existing cache directory does **not** prove the archive is complete. After download, verify the actual size and full SHA-256 against the pinned values. Never modify the returned cache file or manually clear the HF cache.

Open the ZIP read-only with Python's `zipfile`, which handles ZIP64. Do not use `extractall`, extract a second corpus, or manually implement a ZIP parser in the production notebook. The metadata-only HTTP-range probe used during planning is evidence gathering, not the notebook's download mechanism.

Validate member paths before use: reject absolute/parent-traversal paths, ambiguous/duplicate names or basenames, symlink-like members, unsupported unexpected payloads, empty WAV files, and a missing or duplicated `meta.csv`. Preserve archive attribution in provenance. Report archive CRC32/uncompressed-size duplicate candidates; resolve candidates with actual member SHA-256 rather than calling CRC32 a cryptographic proof.

### 4.2 Explicit metadata join and common label contract

Read `meta.csv` with explicit string fields, without guessing labels from directory names, filename prefixes, row ordering, or speaker identity. Assert:

- columns are `file`, `speaker`, and `label`
- 31,779 rows and unique exact filenames
- no missing/blank required values
- labels are exactly `bona-fide` and `spoof`, with the verified counts
- 54 distinct released speaker strings and the expected per-speaker label coverage
- audio members and metadata filenames join one-to-one with no leftover/unlabeled WAVs

Preserve source values. Do not silently merge speaker-name variants, strip meaningful content, or guess missing identities. Use exact released speaker strings as grouping keys and document that this is not independent identity verification.

Keep `file`, `speaker`, and original string `label`, and add:

| Output field | Meaning |
|---|---|
| `audio_file_name` | Exact alias of `file` |
| `speaker_id` | Exact alias of `speaker`; an audit/grouping field, never a model input |
| `key` | `ClassLabel(names=["bonafide", "spoof"])`, from the explicit source label map |
| `audio` | Standardized embedded WAV bytes |

`label` remains the original string; `key` is the canonical model-label source. Do not mistake Hugging Face's inferred audio-only `train` split for a labeled table or infer binary labels from its row positions.

### 4.3 Project-defined speaker-disjoint 70/15/15 protocol

Build a full metadata table **before** audio preprocessing or smoke selection. Aggregate each speaker's total, bona-fide, and spoof counts. Assign the entire speaker group to exactly one split.

Requirements:

1. Target clip proportions `{"train": 0.70, "validation": 0.15, "test": 0.15}`. Speaker-count proportions are not targets.
2. Approximately preserve the global spoof share, `11_816 / 31_779`, in every split; do not force a 50/50 class balance.
3. All source files occur exactly once; no speaker or filename overlap across splits; both labels and at least two speakers in each split.
4. Use seed 42, canonical sorted inputs, and deterministic tie-breaking. Default to **64 seeded candidate starts**, each refined by at most **256 strictly improving whole-speaker moves/swaps**. The versioned objective sums squared deviations from target clip shares for total/bonafide/spoof counts, weighted by the inverse target split ratio. Prefer acceptable endpoints, then the lowest objective, then the lexicographically smallest speaker assignment. Do not claim a globally optimal partition.
5. Default acceptance limits: each split's clip share is within **0.02 absolute proportion** (two percentage points) of its target, and its spoof share is within **0.05 absolute proportion** of the corpus spoof share. Validate the **full protocol**, not the deliberately balanced smoke subset.
6. If the configured search cannot meet those constraints, stop with diagnostics. Never silently relax tolerances, change the seed, split a speaker, drop rows, or fall back to clip-level stratification.
7. Only filename/speaker/label metadata may guide partitioning. No waveform statistics, model performance, or test outcomes may guide assignment selection.

Speaker concentration makes naive sampling of 70% of speaker names unsuitable for targeting 70% of clips. No exact per-split row counts or speaker counts should be invented before the algorithm has run.

For orientation only, unconstrained clip targets are approximately 22,245 training and 4,767 each validation/test. These are **not** required exact counts; the complete speaker assignment is authoritative.

### 4.4 Freeze and audit the split assignments

Persist `split_assignments.json` as a portable sidecar inside every artifact. It records:

- source identity/revision/archive checksum and raw metadata checksum
- verbatim release attribution and its SHA-256, plus both license references and the unresolved-discrepancy note; synthetic fixtures are marked separately, without assuming corpus licensing
- split-policy version, seed, ratios, search budget/objective settings, and acceptance limits
- canonical source metadata rows and the speaker-to-split mapping
- actual split row, class, and speaker counts and deviations from the targets
- a deterministic assignment digest over stable content, excluding timestamps/absolute paths

The combined artifact fingerprint must cover preprocessing, source, split policy, and the actual assignment digest. A separate configuration-only lookup fingerprint may name the output directory so compatible local/Drive artifacts can be found before downloading the source.

On reuse, validate actual rows against the stored assignment, not just a summary count. A different mapping can have the same 70/15/15 counts. Audit filename membership, speaker ownership, labels, original/canonical aliases, complete coverage for full artifacts, and the assignment digest.

Freeze one protocol and reuse it for all augmentation/model comparisons. Do not run many seeds and report the one with the best validation/test scores. A deliberate protocol change needs a new policy/config fingerprint and output directory.

Speaker-disjointness does **not** establish source-video-disjointness or eliminate acoustic near-duplicates: source IDs are absent. Confirmed conflicting/cross-split byte duplicates require an explicit decision; do not silently deduplicate or move rows. Do not claim unseen-generator evaluation when generator IDs are unavailable.

### 4.5 Decode and standardize one recording at a time

Read one WAV member into memory, then use the same `datasets`/TorchCodec mono/16 kHz decoding contract as ASV. An `Audio(sampling_rate=16_000, num_channels=1)` feature can decode the embedded source bytes before the standardization helper is called. Source decoder output is channel-first; do not confuse it with SoundFile's sample-first stereo layout.

For the selected API:

```python
samples = decoded_audio.get_all_samples()
waveform = samples.data.cpu().numpy().astype(np.float32, copy=False)
sample_rate = samples.sample_rate
```

- Accept one-dimensional mono or channel-first `[channels, samples]` decoder output; average channels defensively when needed.
- Assert 16 kHz after lazy decoding/resampling; do not resample an already converted waveform again.
- Validate a nonempty, finite, contiguous float32 mono waveform.
- Fail with split/filename context on decode/transform errors; never silently drop a row or substitute another decoder.

Use a generator that opens the ZIP once per split and progressively writes standardized examples to Arrow, with an explicit `Features` schema. This avoids extracting the archive or materializing a full raw-audio Arrow copy. Do not capture open file handles in multiprocessing state. Keep initial processing single-process and bound the Arrow writer batch size (32, as in ASV).

Generation-cache identity must include source revision/checksum, metadata and assignment digests, preprocessing/config version, and relevant producing package versions. An explicit version-compatible generator fingerprint or an equivalently keyed cache directory must prevent accidental stale reuse. Saving a feature with a sampling-rate declaration does not itself rewrite samples: the generator must encode the processed waveform into new WAV bytes.

### 4.6 Optional edge-silence trimming — disabled

Retain the ASV defaults:

- `TRIM_SILENCE = False`
- 20 ms frames; silence threshold 40 dB below the loudest frame
- trim leading/trailing silence only; preserve internal pauses and final partial frames
- keep all-zero audio unchanged
- if enabled trimming would leave less than 0.25 s, retain the original and warn

No one-second minimum-duration filter or fixed four-second crop is allowed. Pauses may contain countermeasure cues, and IWD already has natural channel/silence variation. Changing the trim policy is an explicit configuration/experiment change.

### 4.7 RMS normalization, PCM16 encoding, and metadata

Use the ASV normalization algorithm:

```python
TARGET_RMS_DBFS = -20.0
PEAK_CEILING = 0.99
PCM16_STEP = 1.0 / 32768.0
```

1. Compute RMS in float64.
2. Compute gain for the RMS target.
3. Cap gain using `PEAK_CEILING - PCM16_STEP` before encoding.
4. Encode mono 16 kHz `WAV` / `PCM_16` in memory.
5. Decode the stored PCM16 result to compute the authoritative sample count, RMS, and peak.

```python
buffer = io.BytesIO()
sf.write(buffer, waveform, TARGET_SR, format="WAV", subtype="PCM_16")
stored_audio = {"bytes": buffer.getvalue(), "path": None}
```

Keep peak-limited clips below the requested RMS target rather than compressing/limiting them. Retain all-zero audio, flag it, skip normalization/noise divisions, and store undefined RMS dBFS as `None`, not JSON infinity. Empty/non-finite arrays are errors.

Keep the original metadata and aliases and add the same standardization audit fields as ASV: `duration_s` (`float32`), `num_samples` (`int32`), nullable `rms_dbfs` (`float32`), `peak` (`float32`), `is_silent`/`peak_limited` (`bool`), and an IWD-specific `preprocess_version` (`string`).

PCM16 quantizes floats; use an explicit approximately one-step round-trip tolerance. Do not expect processed bytes to equal the source float-WAV bytes. Do not use MP3/Opus for storage: WAV avoids a new lossy codec but does not undo the online source's unknown codec history.

### 4.8 Saving, Drive publication, and reuse

Keep source and generation caches local. Save a portable `DatasetDict` locally with embedded audio and `split_assignments.json`, reload/audit it, compute file sizes and SHA-256 inventory, then publish `manifest.json` with `status="complete"` **last**.

Use IWD-specific local and Drive roots, for example:

```text
/content/iwd_work/<run-tag>/saved/in_the_wild-<configuration-fingerprint>/
/content/iwd_work/<run-tag>/saved/in_the_wild-smoke-<configuration-fingerprint>/
/content/drive/MyDrive/ds-ai-lab/in_the_wild/<run-tag>/standardized-<configuration-fingerprint>/
```

The manifest must include artifact kind (`smoke` or `full`), source/protocol identity, assignment and preprocessing fingerprints, configs, producing package versions, split/class/speaker counts, metadata digests, and a checksum inventory covering the assignment sidecar and dataset shards.

For a Drive copy: copy files **without** the completion marker, verify destination checksums and a reload/sample audit against the protocol, then publish the complete marker. A directory without a matching complete marker is not reusable. An interrupted transfer must never look complete.

Discover and verify an existing compatible local/Drive artifact before downloading the raw archive. A complete artifact contains enough protocol metadata to audit itself without the ZIP. Restore a verified Drive copy to local disk for later training; check capacity before creating the destination. Mounted Drive free-space reports are not account-quota checks.

Never overwrite/delete an existing directory automatically. Reuse only after full compatibility, inventory, assignment, and reload checks; otherwise stop and choose a new run tag. A smoke artifact must never be accepted as a full dataset.

---

## 5. On-the-fly training augmentation

### 5.1 Exact baseline settings

All effects use an independent probability of 0.5, identically for both labels:

```python
P_NOISE = P_SHIFT = P_SPEED = P_PITCH = P_REVERB = 0.5
SNR_DB = 15.0
TIME_SHIFT_S = 0.15
SPEED_RANGE = (0.9, 1.1)
PITCH_SEMITONES = 2.0
RT60_S = 0.3
REVERB_DRR_DB = 6.0
```

The augmentation function receives no label. Validate probabilities and parameter ranges. Finite-sample application fractions need not be identical by class; the policy/distributions must be label-independent.

This policy applies an expected 2.5 effects per recording; the probability of at least one effect is 96.875%. It is intentionally identical to ASV, **not** justified as optimal for IWD. Already-wild recordings may be over-augmented. Tune only in a separate validation-controlled experiment, not by inspecting test results.

### 5.2 Order and behavior

Always use:

1. **Pitch-preserving speed:** `librosa.effects.time_stretch(y, rate=rate)`, uniform `[0.9, 1.1]`; length approximately `round(len(y) / rate)`.
2. **Pitch:** `librosa.effects.pitch_shift(y, sr=16000, n_steps=n_steps, bins_per_octave=12, res_type="soxr_hq")`, uniform ±2 semitones; preserve length.
3. **Time shift:** uniform ±0.15 s, zero-filled, length-preserving, no circular wrap. Clamp the offset to avoid discarding the entire waveform. Shifted edge samples are discarded.
4. **Reverberation:** direct impulse plus an exponentially decaying random tail, RT60 ≈ 0.3 s and DRR 6 dB. Normalize tail **energy**, not its maximum. Use causal `fftconvolve(..., mode="full")[:len(y)]`.
5. **Gaussian noise:** calibrate noise RMS to the current waveform's whole-clip RMS for approximately 15 dB SNR; float64 power calculations and a zero-signal guard.
6. **Peak guard:** uniform scaling only when needed to stay at/below 0.99. Do not RMS-renormalize, quantize, or save temporary augmented waveforms.

For the synthetic room response:

```python
n = int(np.ceil(2.0 * RT60_S * TARGET_SR))
t = np.arange(n) / TARGET_SR
tail = rng.standard_normal(n) * 10.0 ** (-3.0 * t / RT60_S)
tail[0] = 0.0
tail *= 10.0 ** (-REVERB_DRR_DB / 20.0) / np.linalg.norm(tail)
rir = tail.copy()
rir[0] = 1.0
```

Guard invalid/zero tail norms. Check the diffuse tail's Schroeder energy-decay curve using a -5 to -25 dB fit and extrapolate to 60 dB; target 0.30 ± 0.05 s across fixed seeds. Check approximately 6 dB direct-to-tail energy. This is a reproducible proxy, not a measured or physical room simulation.

Two selected phase-vocoder operations may compound artifacts. Truncating the reverberation tail preserves duration; it is not a lossless operation. Calibrate/test SNR at the noise-addition stage; after a peak guard, scale the clean reference by the same factor. Feature-extractor mean subtraction is a later operation and need not preserve the exact whole-clip SNR definition.

### 5.3 Epoch reproducibility and train-only guards

Use the ASV per-row RNG design: seed a generator from base seed, epoch, split ID, and row index. Stable saved row ordering is part of that contract. Repeated reads of a row in one epoch produce the same augmentation; a new epoch gives new draws.

- Call `set_epoch(epoch)` before creating each future training iterator.
- Evaluation wrappers use `augment=False` and instantiate no waveform RNG.
- Reject `augment=True` for validation/test.
- Keep `persistent_workers=False`; initially use `num_workers=0`.
- Seed Python, NumPy global RNG, PyTorch, and DataLoader shuffling separately. NumPy `default_rng` does not seed Transformers' checked masking implementation.
- Test epoch variation with a stochastic effect forced on; do not assert that every randomly gated chain must differ.

---

## 6. SpecAugment for XLS-R

### 6.1 Configuration, not a large-model allocation

Load only the pinned small configuration and waveform feature-extractor files:

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

Configure before constructing a future `Wav2Vec2ForSequenceClassification` or `AutoModelForAudioClassification`, with a new two-class head. Do not instantiate XLS-R-300M, even with random weights, or use a CTC transcription model/tokenizer in this notebook.

The sampler requests minimum starts subject to probabilistic rounding and valid-length caps. Masks may overlap/merge. Feature masks operate on 1,024 learned hidden channels, not mel bins. A 64,000-sample recording produces 199 convolutional frames; the baseline generally requests two time starts and two (occasionally three) feature starts, not exactly two visible regions. Longer clips may have more time masks.

### 6.2 Training and evaluation

Automatic model-level masks require enabled SpecAugment, positive corresponding probabilities, and `model.train()`. Use `model.eval()` for validation/test/inference, and do not pass explicit `mask_time_indices` there: explicit masks are a separate path that can apply in evaluation.

Freezing parameters does not itself disable masks; keeping a frozen backbone in `eval()` does. Document this integration choice for future training. Iterating a DataLoader does not apply XLS-R's internal masking.

### 6.3 Feature extraction and short inputs

Use the pinned `Wav2Vec2FeatureExtractor` with `sampling_rate=16000`, `padding=True`, `truncation=False`, `return_attention_mask=True`, and `return_tensors="pt"`. XLS-R has `feat_extract_norm="layer"`; forward the attention mask.

Its `do_normalize=True` normalizes each recording using valid samples, excluding batch padding. Zero-filled time-shift regions inside the recording are still valid samples. Model tensors can exceed `[-1, 1]`; never apply waveform peak clipping after feature extraction.

Derive convolution frame counts from the pinned kernels/strides, not a rounded duration assumption:

- 400 input samples produce one frame.
- 3,280 samples produce ten frames, sufficient for the configured time-mask length.
- 64,000 samples produce 199 frames.

Audit **every** saved length using metadata; check training lengths at the fastest configured speed and actual post-augmentation lengths. Validation/test need the convolution minimum even though training masking is disabled. Keep short rows in storage but raise a filename/split-specific integration error; never silently filter them or count another row's batch padding as valid speech. Any shorter global mask/speed policy is an explicit configuration decision.

---

## 7. Dataset wrapper and DataLoader contract

### 7.1 Dataset wrapper

Retain the ASV wrapper's split, augmentation flag/config, base seed, epoch, preprocessing version, and short-input checks. Decode saved bytes through the validated Hugging Face audio path and return:

```python
{
    "input_values": waveform,       # contiguous 1-D float32, variable length
    "labels": int(row["key"]),      # 0 = bonafide, 1 = spoof
    "duration_s": len(waveform) / TARGET_SR,
    "audio_file_name": row["audio_file_name"],
}
```

Returned duration describes the possibly augmented waveform; stored duration describes the standardized unaugmented recording. Never derive labels from speaker/file identity. Metadata fields are for audit/debugging only.

### 7.2 Collator

Pad only to the longest recording in the batch using the feature extractor. Forward only:

- `input_values`: float32, `[batch, max_samples]`
- `attention_mask`: integer/bool with the same shape
- `labels`: int64, `[batch]`

No source `label` string, speaker, filename, duration, protocol field, or other metadata reaches the future classifier. Do not truncate. Reduce batch size for variable-length memory pressure; cropping/length bucketing is a separate training decision.

### 7.3 Loaders

- Training: shuffle, augmentation on, set the epoch before each iterator.
- Validation/test: no shuffling or augmentation; deterministic repeated reads.
- Initial `num_workers=0`; optional workers only after decoder/epoch-propagation tests.
- `persistent_workers=False`, worker initialization for other random transforms, and an independently seeded shuffle generator.
- Pin memory when using a GPU.
- No weighted/random class sampler, oversampling, or balancing in this preprocessing notebook.

Report the actual real-majority class distribution per derived split. Do not copy ASV's approximately 90%-spoof distribution or its official split counts into this notebook.

---

## 8. Verification plan

### 8.1 Static and conversion checks

- Compile `iwd_notebook.py` for syntax without executing its setup/data cells.
- Convert with `jupytext --to ipynb iwd_notebook.py` and validate the resulting notebook structure in a temporary location; generating a tracked `.ipynb` is not required for these deliverables.
- Compile code-cell/Python examples where appropriate.
- Ensure there are no secrets, shell magics, unguarded Colab imports, classifier-training calls, or large-model weight downloads.
- Ensure ASV/reference files are unchanged.

### 8.2 Archive and split unit checks

Use a small generated ZIP/CSV, never the 8.16 GB source, to test:

- exact 1:1 joins, original/canonical label mapping, and safe member reading
- rejection of duplicate members/basenames, unsafe paths, missing metadata/audio, unknown labels, blank speakers, and malformed metadata
- deterministic speaker assignment under repeated execution and reordered input metadata
- complete row coverage, no cross-split filenames/speakers, both labels per split, and configured ratio/balance limits
- policy/seed/config changes affect fingerprints; do **not** require every different seed to produce a different mapping
- sidecar digest/config/member-tampering rejection
- verbatim attribution preservation and rejection of altered attribution even if the overall assignment digest is recomputed
- impossible constraints fail clearly without a clip-split fallback or row dropping

Synthetic fixtures have explicitly separate expected source counts/provenance. They must never pass the strict full-IWD artifact gate.

### 8.3 Standardization unit checks

Retain ASV's synthetic stereo/downmix and non-16 kHz resampling tests, all-zero input, final partial trim frame, optional trim fallback, high-crest-factor peak limiting, PCM16 round-trip tolerance, deterministic output bytes, and empty/NaN rejection.

Stored audio must be mono 16 kHz PCM16 WAV, float32 when decoded, peak-safe at `<= 0.99`, and approximately `-20 dBFS` RMS (e.g. ±0.1 dB) unless silent/peak-limited. Audit values must match the decoded stored audio, not just pre-quantization floats.

### 8.4 Actual-source smoke generation

After the complete metadata protocol is frozen:

- select eight training clips per label, plus one per label in validation and test, all from their assigned splits
- generate/save/reload **16 + 2 + 2** rows locally with `kind="smoke"`
- preserve metadata/labels and verify embedded nonempty bytes with `path=None`
- compare stored WAV bytes to a second deterministic standardization of the same source member
- decode one actual example per split and inspect train-only waveform/log-mel/listening previews

The smoke subset is balanced for coverage, not for estimating corpus class balance. Log-mel plots are visualizations only; XLS-R still receives raw waveforms. Twenty examples are not a reliable throughput forecast.

### 8.5 Full-run, protocol, and reload audit

For the complete artifact:

- 31,779 total rows, 19,963 bonafide, 11,816 spoof, and 54 released speaker groups
- exact disjoint union of source filenames, unchanged source values/aliases/labels, and actual derived counts matching the frozen assignment
- zero filename/speaker overlap, both labels per split, and protocol tolerances satisfied
- full per-row length/level/version metadata audit; sampled WAV-format, embedded-byte, duration/RMS/peak, and TorchCodec decode checks
- matching source/metadata/protocol/config digests and file inventory
- local reload and Drive-copy reload/sample audit succeed
- reject missing/extra/corrupted shards, changed assignments, mismatched config, partial copies, and smoke-as-full reuse

A file-checksum pass covers every artifact file, but a sampled audio audit is not an independent second decode of every waveform. Report that distinction honestly.

### 8.6 Augmentation and model checks

Retain the ASV tests:

- Gaussian noise SNR 15 ± 0.1 dB at the injection stage; silence remains finite/unchanged for noise.
- Speed output length within one sample of `round(input_samples / rate)`, with preserved tone pitch.
- Pitch-shift duration preservation and expected tone-frequency movement.
- Zero-filled shifts, no wrapping, correct direction/length.
- Causal direct impulse alignment, controlled RIR energy/DRR, and 0.30 ± 0.05 s tail RT60 across fixed seeds.
- Same seeded chain gives identical finite peak-guarded output; test all-off/all-on gating separately.
- Forced stochastic training effect changes across selected epochs; evaluation remains bitwise identical and rejects augmentation.
- Tiny random `Wav2Vec2Model`: hidden size 64, two heads, one Transformer layer, intermediate size 128, seven 16-channel convolutions with the pinned kernels/strides. No 300M allocation or pretrained weights.
- Isolate time/feature mask behavior with dropout disabled, verify train/eval gating and valid lengths/padding, and do not count connected spans as a minimum-start proof.
- Length checks include non-leading short rows and fastest-speed boundaries without filtering.
- Loader batch keys/shapes/dtypes/mask lengths/padding are correct; model inputs contain no metadata. Hold row order fixed when testing epoch variation.

Production preprocessing/collation must not depend on private Transformers mask-generation helpers. A tiny-model smoke test does not verify later XLS-R-300M training or any accuracy benefit.

---

## 9. Error handling and storage behavior

- Fail with useful archive member, split, and filename context on corruption, missing metadata, invalid labels, empty/non-finite audio, transformation failures, or incompatible model lengths.
- Retain silent/short source rows in the saved dataset; flag silence and avoid undefined gain/noise operations. Do not silently repair, substitute, filter, or relabel recordings.
- Do not relax split constraints automatically or reuse a different same-sized partition.
- Keep snapshots immutable and caches local. Do not delete source files, manually clear caches, or overwrite previous artifacts.
- Publish completion markers last, independently for local and Drive copies. Verify the assignment sidecar and dataset files before trusting the marker.
- Require an explicit full-run opt-in and actual Drive-quota confirmation before a new full publication.
- No dataset artifacts, generated caches, runtime outputs, tokens, or trained models belong in git.

---

## 10. Colab runbook

1. Convert `iwd_notebook.py` using Jupytext and open the generated notebook in Colab.
2. Use a CPU runtime for preprocessing; a GPU is not required until separate training work.
3. Review paths, run tag, split/preprocessing settings, and default smoke-only mode.
4. Run dependency setup, actual TorchCodec decoder smoke test, and synthetic checks. Restart after replacing imported packages.
5. Mount Drive if enabled. Discover/reuse a verified compatible artifact before any source download; invalid existing artifacts are errors, not silently bypassed.
6. If no artifact is available, check local capacity, download/verify the 8.16 GB pinned ZIP, and audit its complete metadata.
7. Generate and freeze the speaker assignment. Inspect actual row/class/speaker counts and protocol tolerances; do not change it based on model outcomes.
8. Run source-format/storage pilots and the 20-row smoke save/reload. Inspect only training examples for augmentation previews.
9. Check actual Drive quota and local capacity. Opt into full generation and publish only after all local/Drive verification gates pass.
10. Run full metadata/model-length audits and model-ready loader checks. Keep the verified artifact for subsequent training instead of repeatedly preprocessing.

Storage budgeting:

- source download: 8.16 GB; no full archive extraction or raw-audio Arrow copy
- processed PCM16 payload: approximately 4.37 GB using published duration, plus measured WAV/Arrow/metadata overhead
- local peak: source cache + processed generation cache + local saved copy + smoke/protocol data + safety margin; a low-to-mid-20-GB planning allowance is plausible for a fresh run, **not a guarantee**
- Drive: one verified processed copy plus overhead; check account quota, including any existing ASV artifact
- incompatible reruns/new cache versions can increase disk requirements; estimate still-needed space before each materialization/copy

Use representative pilots for throughput estimates; do not turn the 20-row correctness smoke test into a promised full-run time or size. Duration/header sampling is for resource estimates, not for choosing preprocessing or augmentations using the test split.

---

## 11. Risks and limitations

1. **Custom training protocol:** IWD was introduced for external evaluation. Training on a project-defined subset changes the experimental question; results are not directly the original evaluation-only benchmark.
2. **Speaker metadata is limited:** the release exposes 54 name strings versus 58 people reported publicly; aliases/identity errors cannot be independently resolved here.
3. **No source/generator protocol:** absent video/attack IDs prevent a source-video or unseen-generator disjointness guarantee. Speaker separation and byte checks do not detect all acoustic near-duplicates.
4. **Concentrated, imbalanced groups:** a few speakers dominate clip counts and class mixes differ by speaker. Ratios and stratification are approximate; report per-speaker coverage, not just global counts.
5. **Aggressive augmentation:** preserving ASV's defaults controls one experimental variable but may damage already-wild spoof cues. White noise and synthetic RIRs are approximations; speed and pitch can compound phase-vocoder artifacts.
6. **Normalization/quantization:** RMS is not perceptual loudness; XLS-R normalizes again. PCM16 conversion from source float WAV is not lossless.
7. **Adapted SpecAugment:** hidden-channel masking is not literal frequency-band masking and minimum starts do not mean exactly two visible regions.
8. **Evaluation discipline:** future training should report EER, ROC-AUC, balanced accuracy, and per-class recall, tune on validation only, and reserve test for final evaluation. min-tDCF also needs a compatible ASV-system/protocol/cost setup; binary detector predictions alone are insufficient.
9. **Storage/copy failures:** account quota and Colab capacity vary. Completion markers detect incomplete publication but do not prevent interruption; checksum validation on Drive can be slow.
10. **Dependency/runtime drift:** record tested versions, preserve the PyTorch stack, and validate decoder binaries and the pinned Transformers behavior in the actual runtime.
11. **English-only data and license uncertainty:** this does not establish Indian-language generalization or resolve the conflicting published licenses.
12. **No training outcome claim:** saved audio is unaugmented; the wrapper and future model training state are essential. Passing these checks does not prove classifier accuracy or augmentation benefit.

---

## 12. Acceptance criteria

- [ ] Pins/verifies the correct archive and explicitly joins all source metadata/audio rows.
- [ ] Retains 31,779 recordings, original labels/metadata, and the canonical ASV-compatible label map.
- [ ] Creates speaker-disjoint, seed-42 70/15/15 clip-target splits with approximate class proportions and no silent fallback.
- [ ] Freezes, fingerprints, saves, and validates the exact split assignment, not only its summary counts.
- [ ] Produces mono 16 kHz peak-safe RMS-normalized PCM16 WAV with trimming off by default.
- [ ] Saves portable embedded audio and a protocol sidecar locally and to Drive with verified manifest-last completion.
- [ ] Discovers/reuses a verified artifact without requiring the raw ZIP.
- [ ] Keeps smoke/full artifacts distinct and checks capacity/quota before expensive operations.
- [ ] Implements the five exact ASV waveform augmentations on the fly for training only.
- [ ] Keeps validation/test unaugmented and configures training-only XLS-R native masks.
- [ ] Supplies variable-length dataset/collator/loaders with masks, canonical labels, and no metadata leakage.
- [ ] Passes static/conversion, archive/split, standardization, artifact/reload, augmentation, tiny-model, length, and batch checks in their appropriate environments.
- [ ] Documents the custom protocol, 54/58 discrepancy, source-provenance limits, class/speaker imbalance, licensing discrepancy, and storage requirements.
- [ ] Leaves ASV/reference files unchanged and does not train a classifier or download large model weights.

Unchecked full-runtime criteria remain acceptance gates, not completed results.

---

## 13. References

- Existing baseline: `PLAN_ASV.md` and `asv_notebook.py` in this directory.
- Earlier project audit: `scratchpad/vikram/milestone-2/InTheWild_report.md`.
- Pinned source card: <https://huggingface.co/datasets/mueller91/In-The-Wild/blob/eee168f92c367f8c82ff2cf42b6f61e362fd6211/README.md>
- Pinned repository inventory/LFS metadata: <https://huggingface.co/api/datasets/mueller91/In-The-Wild/tree/eee168f92c367f8c82ff2cf42b6f61e362fd6211?recursive=true>
- Pinned archive: <https://huggingface.co/datasets/mueller91/In-The-Wild/resolve/eee168f92c367f8c82ff2cf42b6f61e362fd6211/release_in_the_wild.zip>
- Authors' dataset page: <https://deepfake-total.com/in_the_wild>
- Müller et al., *Does Audio Deepfake Detection Generalize?*: <https://arxiv.org/abs/2203.16263>
- Hub download guide: <https://huggingface.co/docs/huggingface_hub/guides/download>
- Datasets audio loading/metadata: <https://huggingface.co/docs/datasets/audio_load>
- Datasets generator creation: <https://huggingface.co/docs/datasets/create_dataset>
- Datasets audio processing: <https://huggingface.co/docs/datasets/audio_process>
- Python ZIP/ZIP64 support: <https://docs.python.org/3/library/zipfile.html>
- Pinned XLS-R configuration: <https://huggingface.co/facebook/wav2vec2-xls-r-300m/blob/1a640f32ac3e39899438a2931f9924c02f080a54/config.json>
- Pinned waveform processor: <https://huggingface.co/facebook/wav2vec2-xls-r-300m/blob/1a640f32ac3e39899438a2931f9924c02f080a54/preprocessor_config.json>
- XLS-R/Wav2Vec2 documentation: <https://huggingface.co/docs/transformers/model_doc/xls_r> and <https://huggingface.co/docs/transformers/model_doc/wav2vec2>
- TorchCodec compatibility: <https://github.com/meta-pytorch/torchcodec#installing-torchcodec>
- librosa effects: <https://librosa.org/doc/latest/effects.html>
- Jupytext percent format/conversion: <https://jupytext.readthedocs.io/en/latest/formats-scripts.html>

---

## 14. Review evidence and remaining verification

Verified before implementation:

- Checked the pinned Hub revision, three-file repository inventory, archive size, and published LFS SHA-256.
- Inspected the current HF viewer: its single `train` preview has only an audio feature, so it cannot be trusted to supply source labels/speaker groups.
- Used a transient, read-only HTTP-range probe of the ZIP64 directory and compressed `meta.csv`; approximately 3.51 MB was fetched, **no audio payloads were requested, and no dataset files were written**.
- Independently verified 31,779 metadata/audio rows, exact filename alignment/uniqueness, label counts, no missing values, 54 released speaker strings, and both labels for all 54.
- Checked the earlier Milestone 2 audit and existing ASV implementation. Its old 80/10/10 proposal is superseded **for this experiment** by the approved 70/15/15 protocol; the earlier report remains untouched.
- Consulted current Datasets/Hub documentation for explicit metadata joins, embedded-audio decoding, progressive generator construction, pinned downloads, and immutable cached files.
- Rechecked the pinned XLS-R configuration/processor JSON: frontend kernels/strides, 1,024 hidden channels, layer normalization, 16 kHz input, attention-mask return, and per-clip normalization. Rechecked TorchCodec's current PyTorch/Python compatibility table and CPU-wheel/FFmpeg requirements; this is documentation evidence, not a live Colab decoder test.
- Recorded the HF-versus-author-site license discrepancy rather than silently choosing one.

Completed local implementation checks (not full notebook execution):

- Python syntax compilation, Ruff 0.17.3, and all **27 code-cell** compilations passed. Jupytext 1.19.6 conversion/round-trip and `nbformat` validation passed for the **27 code + 10 Markdown** cells; the converted notebook was written only under `/tmp/opencode/`.
- Checked all 14 plan sections, balanced fenced blocks, and the syntax of all eight Python examples.
- Verified unchanged ASV/reference file SHA-256 values, matching preprocessing/augmentation/masking defaults, and 31 identical shared implementation-definition ASTs. The preprocessing version intentionally differs for IWD.
- Executed actual selected notebook helpers through AST extraction without setup cells or ML imports: deterministic/reordered splitting, source-download capacity preflight, original label/metadata joins, safe ZIP member handling, SHA-confirmed duplicate rejection, frozen protocol/source isolation, verbatim attribution preservation/tamper rejection, and checksum-corruption rejection.
- Independently executed the splitter on all 31,779 original metadata rows, checked the pinned CSV checksum, and repeated it with reversed input ordering. The default result is:

| Split | Clips | Bonafide | Spoof | Speaker groups |
|---|---:|---:|---:|---:|
| train | 22,247 | 13,977 | 8,270 | 34 |
| validation | 4,769 | 2,995 | 1,774 | 8 |
| test | 4,763 | 2,991 | 1,772 | 12 |

These are metadata-only results for the recorded default policy, not counts from an exported/full audio artifact. Every source row/speaker is covered once, and both acceptance gates pass. The largest clip-share deviation is approximately 0.0122 percentage points and the largest spoof-prevalence deviation approximately 0.0217 percentage points.

- NumPy-only tests passed for actual RMS normalization, silent/noisy input handling, 15 dB SNR, zero-filled shifts, convolution-frame arithmetic, and synthetic RIR DRR/RT60 across 20 seeds. Shared noise/shift/RIR outputs matched ASV exactly for the tested inputs/seeds.
- Independent document review found no actionable issues. Full notebook review found one provenance gap—release attribution was not being retained—which was corrected by storing/checking the verbatim attribution and licensing provenance in the checksummed protocol.

Remaining verification: the local environment has no PyTorch/audio stack and insufficient space for the source corpus. An isolated attempt to install audio-test packages timed out; it did not modify project dependencies. Actual TorchCodec decoding, PCM16/SoundFile runtime checks, Arrow generation/save/reload, librosa speed/pitch, SciPy convolution, tiny-model masking, PyTorch batches, full-corpus duplicate checks, live Colab dependency setup, Drive publication, and future XLS-R-300M training remain unverified here. Their synthetic checks are embedded in the notebook and remain runtime acceptance gates. No full source archive, real audio payloads, or model weights were downloaded, and no processed-corpus artifact was produced. Only source metadata and temporary synthetic fixtures were used.
