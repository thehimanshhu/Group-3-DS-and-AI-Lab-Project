# Implementation plan: frozen XLS-R classifier comparison notebook

**Date:** 2026-11-05  
**Design status:** shared understanding confirmed by the user.  
**Deliverable now:** this plan only; implementation and training have not been performed.  
**Future implementation target:** `scratchpad/vikram/milestone-3/model_notebook.py`  
**Research:** [Frozen XLS-R classifier research findings](xlsr-classifier-research.md)

## 1. Goal and scope

Build one self-contained, Google Colab-compatible Python notebook that uses a completely frozen `facebook/wav2vec2-xls-r-300m` encoder and compares interchangeable downstream classifiers. It must support a quick integration smoke check and a larger, configurable subset experiment to shortlist promising models.

Convert the source notebook with:

```bash
uvx jupytext --to ipynb model_notebook.py
```

Success means that data loading, deterministic sampling, frozen feature extraction, all enabled classifiers, evaluation, caching, and save/reload inference work together. No minimum accuracy or EER is promised. A subset winner is a candidate for subsequent larger-data evaluation, not an established best full-dataset model.

### Included

- Separate preprocessed Hugging Face dataset folders on Google Drive.
- Existing `train`, `validation`, and `test` partitions, combined only with corresponding partitions.
- Explicit sampling groups supporting both two-class datasets and combinations of single-class sources.
- Nine traditional/XGBoost classifiers, three PyTorch heads, and a dummy baseline.
- One editable configuration per classifier; no automatic hyperparameter search.
- Persistent feature caches, trained heads, fitted preprocessing, thresholds, predictions, metrics, and configuration.

### Excluded

- Reprocessing or downloading raw datasets; creating or changing dataset splits.
- Fine-tuning XLS-R, LoRA, waveform augmentation, or encoder SpecAugment.
- Full AASIST, mixture-of-experts, multi-layer fusion, automatic layer selection, or model ensembles.
- Automatic repeated-seed experiments, deployment/UI work, or rewriting existing preprocessing notebooks.
- Committing a generated `.ipynb` unless separately requested. Conversion verification can use a temporary output directory.

## 2. Existing repository integration

Follow the percent-cell header and execution conventions in:

- `scratchpad/vikram/milestone-3/asv_notebook.py`
- `scratchpad/vikram/milestone-3/itw_notebook.py`

Both save portable Hugging Face datasets with embedded, unaugmented mono 16 kHz PCM16 WAV audio. Their canonical labels are `key`: `0 = bonafide`, `1 = spoof`. Useful shared columns include `audio_file_name`, `speaker_id`, `num_samples`, `duration_s`, and `preprocess_version`.

Do not import either preprocessing notebook: importing its orchestration, corpus-specific constants, or augmentation defaults would couple the new notebook to the wrong workflow. In particular:

- ASV and ITW have different valid `preprocess_version` values; do not require identical strings across sources.
- ITW has no ASV `system_id`; generator/attack metadata is optional.
- Existing ASV/ITW full artifacts have `manifest.json` with `status = complete`, `kind = full`, a fingerprint, and file inventory.
- ITW also has `split_assignments.json`; retain its assignment digest and verify selected records against the frozen assignment.
- A preprocessing smoke artifact is not the full input corpus. The model notebook's smoke mode samples from full prepared artifacts.

## 3. Notebook layout

Keep all implementation helpers and lightweight contract checks in `model_notebook.py`, organized into clear `# %%` and `# %% [markdown]` cells. Keep the file valid Python: no `!pip` or notebook-only magics.

| Stage | Cell content |
|---|---|
| 1 | Purpose, labels, frozen-backbone explanation, research-inspired head caveats, conversion and Colab GPU instructions |
| 2 | Guarded dependency setup before imports; versions and runtime diagnostics |
| 3 | Configuration, source registry, sampling groups, classifier registry, and seeds |
| 4 | Metadata, sampling, metric, pooling, cache, and head helpers |
| 5 | Offline/synthetic contract checks; no Drive, corpus, or pretrained-weight access |
| 6 | Guarded Drive mount, input validation, canonical metadata views, and immutable subset selection |
| 7 | Selected-audio validation, leakage checks, duration/frame counts, and storage preflight |
| 8 | Compatible feature-cache discovery, or frozen XLS-R extraction and verified cache publication |
| 9 | Feature datasets/loaders, train-only scaling, classical classifiers, and PyTorch heads |
| 10 | Smoke execution or comparison execution; validation-based checkpoint/threshold selection |
| 11 | Validation leaderboard, grouped reports, run artifacts, and save/reload checks |
| 12 | Separate opt-in final test evaluation using already-fixed artifacts and thresholds |
| 13 | Run summary, failures/warnings, interpretation limits, and rerun instructions |

Use the existing `NOTEBOOK_RUN = __name__ == "__main__"` pattern for side-effecting cells. Definitions must be importable without mounting Drive, installing packages, downloading weights, or training models. A converted Colab notebook runs its execution cells in order.

## 4. Configuration and runtime defaults

| Control | Default / rule |
|---|---|
| Run mode | `smoke`; change explicitly to `comparison` |
| Seed | `42`, one run per classifier |
| Model ID | `facebook/wav2vec2-xls-r-300m` |
| Model revision | `1a640f32ac3e39899438a2931f9924c02f080a54`, matching the preprocessing notebooks |
| Encoder layer | `6`; configurable integer from 1 through the checkpoint's transformer-layer count |
| Duration cap | `MAX_SECONDS = None`: full recordings |
| Optional cap policy | Explicitly enabled deterministic center crop; never repeat short clips or silently truncate |
| Extraction precision | Float32, including persisted frame features |
| Extraction batch size | `1`, configurable; support length-aware batches when increased |
| Neural batch sizes | MLP: `64`; ASP/CNN: `16`, configurable |
| DataLoader workers | `0` initially |
| Classifier parallelism | Fit classifiers sequentially; bound supported CPU thread counts to at most 2 initially |
| Neural training | Comparison: at most 20 epochs, patience 5; smoke: at most 2 epochs |
| Optimizer | AdamW, learning rate `1e-3`, weight decay `1e-4`, gradient-norm clipping `1.0` |
| Test evaluation | `RUN_TEST_EVALUATION = False` until the comparison settings are frozen |
| Local work root | `/content/model_work` in Colab; configurable local fallback outside Colab |
| Drive output root | `/content/drive/MyDrive/ds-ai-lab/model_experiments`, configurable |
| Run identity | Explicit run tag plus configuration digest; never overwrite a different or incomplete run |

Expose source paths, group membership/filters, per-group quotas, enabled classifiers, each classifier's parameters, layer, duration cap, batch sizes, and training controls in the configuration cells. Validate values before expensive work.

### Dependency policy

- Preserve Colab's installed PyTorch/CUDA stack. Do not blindly upgrade `torch`, `torchaudio`, or CUDA packages.
- Start from the existing bounded `datasets==4.4.1` and `transformers==4.57.3` APIs; select and record compatible fixed scikit-learn/XGBoost versions during implementation smoke validation, rather than installing unbounded upgrades.
- Use the existing Hugging Face/TorchCodec audio-decoding approach. Verify the TorchCodec build against the actual installed PyTorch/Python compatibility matrix and perform a real synthetic WAV decoding check; imports alone are insufficient.
- Require NumPy, scikit-learn, XGBoost, pandas, and the existing PyTorch/Hugging Face decoding dependencies. Do not add Fairseq, Lightning, FlashAttention, tuning frameworks, or augmentation packages.
- Run XGBoost on CPU using histogram trees; do not install a new GPU stack for it. A CPU-only XGBoost wheel is acceptable if needed for a minimal compatible installation.
- Install before importing; replacing an already-imported incompatible package requires a runtime restart. Record Python, package, GPU, CUDA, precision, and attention-backend information.
- Require a GPU for the actual Colab feature-extraction run, with an actionable message if unavailable. CPU synthetic checks remain supported.

The concrete new classifier defaults below are starting configurations, not tuned recommendations or guarantees of optimal performance.

## 5. Source contract, grouping, and sampling

### 5.1 Source registry

Each source entry names a saved `DatasetDict` directory and explicitly maps its audio, label, and clip-ID columns. Speaker, language, duration/sample-count, and provenance fields are optional mappings. Allow a configured constant language for a source with no per-row language field; otherwise preserve unknown language as unknown.

Require:

- The canonical three splits, with no automatic aliases, split creation, or fallback merging.
- A stable, unique original clip ID within each source/split, explicitly mapped labels, and decodable standardized mono 16 kHz audio.
- Explicit label mapping to `0 = bonafide`, `1 = spoof`; never infer labels from filenames, source order, or Hugging Face packaging conventions.
- Existing ASV/ITW completion manifests must be complete full artifacts; validate their metadata/counts and referenced files. Preserve preprocessing and split-protocol digests.
- For future sources without that manifest schema, permit an explicit source configuration declaring no required upstream manifest. Still validate the canonical input contract and fingerprint selected audio. If a completion marker is present but invalid or incomplete, reject it rather than ignoring it.

Inspect metadata without decoding full audio columns. Select original row references first; decode only chosen recordings. Do not copy entire source corpora into Colab merely to process a subset.

Build common metadata views instead of directly concatenating incompatible raw schemas. Model inputs contain only audio-derived features, never source, group, speaker, language, filename, or attack IDs.

### 5.2 Sampling groups

A group is a configured list of source members with optional explicit metadata filters, such as language. Its union must contain both labels in every required split. Examples:

- ASV group: ASVspoof folder alone.
- ITW group: In-The-Wild folder alone.
- Hindi group: a Hindi view of real-only Kathbath plus explicitly selected Hindi synthetic sources.

Do not guess pairings or languages. Filters require actual configured metadata. A folder may provide disjoint language views, but an original row may not be selected twice through overlapping groups/views.

Pairing sources supplies class coverage; it does not remove source/channel confounding. Real-only and fake-only corpora may still differ in recording conditions that a classifier can exploit. Report this limitation.

### 5.3 Quotas and selection

| Preset | Train per group | Validation per group | Test per group |
|---|---:|---:|---:|
| Smoke | 32 | 16 | 16 |
| Comparison | 400 | 100 | 100 |

- Quotas are configurable positive even totals per group/split; take exactly half from each label, without replacement.
- Validate available counts after source filters. If a quota is impossible, stop with requested/available counts and configuration guidance; do not silently reduce, duplicate, rebalance differently, or move examples across splits.
- Within a group/split/label, pool eligible member rows and sample uniformly from that pool. Equal group quotas do not guarantee equal contributions from every source inside a group; print actual source counts.
- Use stable sorted source/clip references and deterministic group/split/label-specific RNG seeds derived from seed 42 and stable hashes, not Python's randomized `hash()`.
- Make smaller quotas prefixes of the same seeded per-label ordering where the eligible population is unchanged, so smoke selections can be traced to comparison selections.
- Freeze the selected IDs, original split/row references, labels, group/source membership, and selection digest before feature extraction. Reuse exactly these clips for all classifiers.
- Defaults with ASV and ITW yield 128 total smoke clips or 1,200 total comparison clips. Each added group adds its configured quotas.

### 5.4 Leakage and audio audits

- Preserve original split membership; report source-level clip/speaker overlap using known identity semantics. Do not claim that namespacing unrelated speaker IDs proves global speaker disjointness.
- For related real/synthetic sources, use supplied identity mappings where available; do not invent cross-source identity matches. Record unresolved identity/near-duplicate limitations.
- Check all selected decoded audio for conflicting-label exact duplicates and exact-waveform duplicates crossing splits. Stop on either; report same-split, same-label duplicates without silently dropping them.
- Reject a selected original row assigned to multiple groups. Retain original source ID namespaces so equal filenames in unrelated corpora do not collide.
- Validate sample rate, mono shape, nonempty finite samples, and saved length metadata when available. Reject incompatible preprocessing rather than silently resampling, normalizing loudness again, trimming silence, or dropping rows.

## 6. Frozen feature extraction

1. Load the pinned `Wav2Vec2FeatureExtractor` and `Wav2Vec2Model`; no tokenizer or CTC head is needed.
2. Validate the checkpoint's sample rate, hidden size 1,024, transformer-layer range, frontend kernels/strides, and layer-normalized frontend contract.
3. Set all encoder parameters to `requires_grad = False`, call `eval()`, explicitly disable SpecAugment, and use `torch.no_grad()` during extraction. Do not put the encoder inside the trainable head optimizer.
4. Feed mono float waveforms at 16 kHz, with the checkpoint's per-recording normalization, `padding=True`, `truncation=False`, and an explicit waveform attention mask.
5. Request hidden states and select tuple index 6 for transformer layer 6; index 0 is the pre-transformer representation, not transformer layer 1. Do not truncate/restructure the encoder or fuse layers in this baseline.
6. Compute each recording's valid output-frame count from its unpadded input length and the checkpoint convolution configuration. Require at least one valid frame; encoder masking is off, so no training-mask minimum applies.
7. Remove padded frames before caching. Store one `T x 1024` float32 sequence per selected clip through the packed cache described below.
8. Derive a `2048`-dimensional clip vector by concatenating temporal mean and population standard deviation over valid frames only. For one-frame sequences, standard deviation is zero.
9. Keep canonical row order in cached indexes and predictions even if extraction batches are sorted by duration.
10. Release the encoder from GPU memory after verified extraction/cache publication. Head training uses cached features, not another XLS-R pass each epoch.

Full recordings remain the default. If `MAX_SECONDS` is explicitly set, center-crop only over-length clips, record original/used lengths and crop offsets, and change the cache fingerprint. Short recordings remain intact; padding is only a batch operation. An out-of-memory error must suggest reducing batch size or enabling the cap, not automatically crop or discard audio.

## 7. Feature cache and storage lifecycle

Use a content/configuration-addressed cache independent of classifier parameters. A compatible cache must allow classifier changes without loading XLS-R weights again.

### Cache identity

Include the selected clip/label membership and original split references, selected audio-content hashes, source manifest/protocol digests when available, label mappings, model ID/revision, feature-extractor configuration, layer, frontend configuration, duration/crop policy, precision, extraction mode, relevant producing versions, and pooling-definition version. Also digest the canonical selected-row reporting metadata, including effective source/group/language/speaker annotations and configured constant-language values. A metadata-only change must not reuse an index with stale annotations; this baseline invalidates the metadata-aware cache rather than rebinding its immutable index.

Changing a classifier's settings must not invalidate features. Changing layer, selected clips/audio, crop policy, checkpoint, or extraction precision must not reuse an incompatible cache. Before reuse, recompute selected-input identity and verify cache contents; do not trust only filenames or counts.

### Storage format

For each split, use:

- A packed `frames.npy` float32 array with shape `sum(valid_frame_counts) x 1024`, written/read through NumPy memory mapping.
- An ordered `index.jsonl` containing each clip's frame offset/count, IDs, labels, source/group/language metadata, original/used sample counts, crop offsets, and content identity.
- An ordered `pooled.npy` float32 matrix of shape `N x 2048` aligned with that index.

Preflight selected audio to obtain exact frame counts, then allocate the packed arrays. Stream extraction batches into their assigned slices; do not hold every clip's features or all encoder hidden states for the corpus in RAM. Sequence loaders read slices and construct fresh padded training tensors.

Estimate frame/pooled bytes plus weights, staging, temporary-space overhead, and a safety margin before allocation. Check local free space and require the user to confirm actual Drive capacity if necessary; a mounted Drive filesystem's free-space report is not a reliable account quota check. Do not claim a guaranteed runtime or memory bound for arbitrarily long full recordings.

### Publication and reuse

- Write locally, flush/close arrays, verify shapes/index consistency and file checksums, then copy to a new Drive cache directory.
- Verify copied files and publish a completion manifest last. Refuse incomplete/corrupt caches; never overwrite or delete them automatically.
- Stage a verified reusable Drive cache locally for head training instead of performing every random feature read through Drive.
- An interrupted extraction need not support partial resume in this first version. It must never masquerade as a completed cache.
- Record actual cache size, selected durations, extraction time, and peak GPU allocation as diagnostics.

## 8. Classifier registry and concrete initial configurations

All values below remain editable; run exactly one configured instance of each enabled classifier. Defaults assume the balanced training subset, so no additional class balancing or resampling is applied.

| Classifier | Initial configuration | Input handling |
|---|---|---|
| Dummy | `strategy="prior"` | Pooled vectors; constant prior score |
| Logistic Regression | `C=1.0`, `solver="lbfgs"`, `max_iter=1000` | Train-only StandardScaler |
| Linear SVM | `LinearSVC(C=1.0, dual="auto", max_iter=5000)` | Train-only StandardScaler; margin score |
| RBF SVM | `SVC(C=1.0, kernel="rbf", gamma="scale", probability=False)` | Train-only StandardScaler; margin score |
| kNN | `n_neighbors=5`, `weights="distance"`, Euclidean distance | Train-only StandardScaler |
| Random Forest | 200 trees, `max_depth=None`, `min_samples_leaf=2` | Unscaled pooled vectors |
| Extra Trees | 200 trees, `max_depth=None`, `min_samples_leaf=2` | Unscaled pooled vectors |
| Histogram Gradient Boosting | 100 iterations, learning rate 0.1, 15 leaf nodes, L2 1.0, `early_stopping=False` | Unscaled pooled vectors |
| Gaussian Naive Bayes | `var_smoothing=1e-9` | Train-only StandardScaler |
| XGBoost | 100 estimators, depth 3, learning rate 0.05, row/column sampling 0.8, L2 1.0, `objective="binary:logistic"`, `tree_method="hist"`, `device="cpu"` | Unscaled pooled vectors; no internal validation split or early stopping |
| PyTorch MLP | Described below | Train-only StandardScaler on pooled vectors |
| PyTorch ASP | Described below | Frame sequences and masks |
| PyTorch temporal CNN | Described below | Frame sequences and masks |

Set supported random-state parameters to the run seed and bounded `n_jobs` where supported. Never pass a generic parameter indiscriminately to incompatible estimators. Surface convergence warnings and include them in run status.

### 8.1 MLP

`2048 -> 256 -> 64 -> 1`, with GELU and dropout 0.2 after each hidden layer. Output a scalar spoof logit. The input is the same fixed mean/std vector used by traditional classifiers, standardized using training data only.

### 8.2 ASP-based classifier

- Frame projection: `1024 -> 128`, per-frame LayerNorm, GELU, dropout 0.1.
- Attention scorer: `128 -> 64 -> 1`, with tanh between the linear layers.
- Mask invalid frame logits before temporal softmax.
- Compute attention-weighted mean and standard deviation; concatenate to 256 dimensions. Clamp tiny negative variances from numerical error and use a documented epsilon for stability.
- Classifier: `256 -> 64 -> 1`, with GELU/dropout 0.2 in the hidden layer.

This is a single-layer, binary-classification, research-inspired ASP baseline. It does not reproduce the cited multi-layer, cosine/one-class-softmax architecture.

### 8.3 Lightweight temporal CNN

- Frame projection: `1024 -> 128`, per-frame LayerNorm and GELU.
- Two temporal Conv1d blocks with 128 channels and kernel size 3: dilation/padding 1/1, then 2/2. Preserve sequence length.
- Use per-frame LayerNorm, GELU, and dropout 0.1 rather than batch statistics spanning padded time steps.
- Zero invalid frames after projection and before/after each convolutional block so padded activations cannot bleed back into valid positions through later convolutions.
- Masked mean/std pooling gives 256 dimensions; use `256 -> 64 -> 1` with GELU/dropout 0.2.

Convolve over time, not over coordinates of an already-pooled clip vector. This is an engineering baseline from the temporal-convolution family, not an exact paper reproduction.

## 9. Training protocol

- Keep cached features unscaled. Fit each required scaler on the training pooled matrix only and persist it with its classifier. Validation/test features must never enter `fit` or `fit_transform`.
- Seed Python, NumPy, PyTorch, DataLoaders, and supported estimators. Request deterministic behavior where supported and record warnings; do not promise identical results across different hardware/library versions.
- Reset the global Python/NumPy/PyTorch RNGs to the configured run seed immediately before constructing each classifier/head, and reset them again immediately before that classifier's fitting/training phase. Do not inherit RNG state from encoder construction/extraction, cache-hit versus cache-miss paths, or earlier classifiers. Do not reset these RNGs every epoch.
- Construct fresh, independently seeded training and evaluation DataLoader generators for each neural classifier, with component seeds derived only from the run seed, stable classifier ID, and loader role. Record these seeds; do not share generator objects across classifiers or let validation iteration advance the training/dropout RNG. Classifier enablement/order must not change another classifier's initialization or shuffle sequence.
- Fit classical models once on training features. Validate classifier prerequisites, such as `k <= number of training clips`, before fitting.
- Train neural heads on cached features with `BCEWithLogitsLoss` and the defaults in Section 4. Encoder parameter training is impossible in this loop because only heads are optimized.
- Use shuffled training loaders and deterministic, non-shuffled validation loaders. Switch only the heads between train/eval modes.
- After every neural epoch, calculate overall validation EER. Keep the lowest-EER checkpoint; resolve exact ties with lower validation loss, then the earlier epoch.
- Stop after 5 epochs without validation-EER improvement, or the configured epoch limit. Smoke mode uses the 2-epoch limit and is not a quality benchmark.
- Record epoch history, best epoch, losses/EER, fit time, parameters, warnings, and trainable parameter count.
- Restore the best neural checkpoint before threshold selection, reporting, and persistence.
- Do not refit on train+validation before final test evaluation; this would change the model/scaler without a fresh calibration set.

## 10. Metrics, thresholds, and ranking

### 10.1 Continuous score contract

Every classifier returns a finite one-dimensional score with **higher = more likely spoof**:

- Use `predict_proba` for class 1 where available, explicitly finding that column through `classes_`.
- Use correctly oriented `decision_function` margins for SVMs; do not enable expensive SVM probability fitting solely to calculate EER.
- Use raw neural logits, avoiding sigmoid saturation/ties in EER calculations.
- Require the trained binary class contract `[0, 1]`; never flip scores because doing so improves validation/test performance.
- Save score type. Margins/logits and probabilities are not interchangeable calibrated confidence estimates.

### 10.2 EER calculation

- Use the complete binary ROC points (`pos_label=1`, `drop_intermediate=False`) from continuous scores.
- Define FPR as bonafide clips flagged spoof and FNR as spoof clips passed as bonafide.
- Find where `FPR - FNR` crosses zero. Use an exact crossing when present; otherwise linearly interpolate along the neighboring ROC segment. This avoids fragile interpolation requiring unique FPR coordinates.
- Constant scores must produce 50% interpolated EER, perfect ordering 0%, and perfectly reversed ordering 100%.
- Reject invalid/non-finite scores. Overall/group EER requires both labels; for a diagnostic subgroup with only one label, return an explicitly undefined value with support counts, not zero.
- Store EER internally as a fraction and display it as a percentage.

### 10.3 Operating threshold for discrete metrics

For each fitted model, choose a usable validation threshold minimizing `abs(FPR - FNR)` among attainable threshold operating points. Include the all-negative endpoint using a finite sentinel above the maximum score rather than persisting an infinite ROC threshold. Resolve ties by lower balanced error, then the higher threshold.

Classify with `score >= threshold` as spoof. Persist the threshold and actual validation FPR/FNR. Interpolated EER may not be attainable by a deterministic threshold on a finite dataset; explain that distinction rather than labeling the discrete operating point as exact EER.

Use the same frozen model threshold for all group/language reports and final test predictions. Do not retune by test set, group, or language.

Report Accuracy and binary Precision/Recall/F1 for `pos_label=1`, with a documented zero-division policy (`0` plus support/predicted-positive diagnostics). Constant-score baselines and zero-positive predictions must remain visible rather than causing a silent metric exception.

### 10.4 Selection and held-out test policy

- Primary ranking: **overall pooled validation EER**, across all selected groups. Custom quota differences change group contributions; print counts and do not silently replace this objective with mean per-group EER.
- Resolve equal validation EERs with higher validation F1, then classifier name for deterministic ordering. Group/language metrics are diagnostics, not extra selection objectives.
- Produce the validation leaderboard first. Failed classifiers have no numeric rank and no fabricated metric values.
- Gate final test scoring behind a separate explicit opt-in cell/flag. Score all successfully frozen candidate artifacts for the final comparison, keeping the validation-selected winner and ordering.
- Test EER describes its own score distributions; test Accuracy/Precision/Recall/F1 still use the frozen validation threshold. Computing descriptive test EER must not update any threshold or model decision.
- Test labels/scores must never affect training, scaling, checkpoint choice, model ranking, layer selection, or threshold selection. Fixed unsupervised feature extraction for the test split is permitted before final scoring.

## 11. Reports and persistent artifacts

### Tables

The validation leaderboard must contain classifier name, status, validation EER (%), Accuracy, Precision, Recall, F1, and support. Include threshold/score type and fit time in saved detailed results. Display the other four metrics consistently as percentages, while machine-readable results use fractions with explicit field names.

Provide separate group and available-language breakdowns, including class counts. Single-class source reports cannot meaningfully include EER and must show it as undefined. Do not invent language or attack annotations.

The optional final test table contains the same performance metrics plus validation rank/EER for context; do not re-sort to select a test winner. Save tables as CSV and JSON, with unavailable JSON metrics represented by `null`, never NaN/Infinity.

### Directory layout

```text
model_experiments/
  feature_cache/<feature_digest>/
    manifest.json
    train/{frames.npy, pooled.npy, index.jsonl}
    validation/{frames.npy, pooled.npy, index.jsonl}
    test/{frames.npy, pooled.npy, index.jsonl}
  runs/<run_identity>/
    config.json
    environment.json
    input_summary.json
    selection.jsonl
    validation_metrics.csv
    validation_metrics.json
    grouped_validation_metrics.csv
    validation_predictions.csv
    models/<classifier_id>/
      metadata.json
      model artifact and fitted preprocessing, where applicable
      training_history.json, for neural heads
    test_evaluation/, only after explicit final test evaluation
    manifest.json
```

Artifact metadata links each model to its feature digest, dimensions, source/selection identity, label mapping, threshold/score type, hyperparameters, producing versions, and best checkpoint where applicable.

- Persist scikit-learn models and their pipelines using joblib; load only explicitly selected, trusted artifacts and check producing versions. Joblib/pickle is not safe for arbitrary downloaded files.
- Save XGBoost through its native `.ubj` or `.json` model API with separate run metadata.
- Save PyTorch head `state_dict`s plus architecture JSON; use weights-only loading. Persist the MLP's training-fitted scaler as numeric parameters or its trusted scaler artifact, consistently with the reload helper.
- Save continuous scores, predictions, labels, and source/group/clip references for every scored split.
- Do not duplicate the 300M encoder weights into each run. Record its pinned identifier and feature-extractor settings; reload the encoder only when new raw-audio inference/features are required.
- Verify save/reload predictions before marking each model complete, then publish the run completion manifest last. A partially failed run must be labeled partial, with successful/failed classifier IDs listed.
- Final test evaluation is a separate immutable artifact referencing the frozen comparison run; it must not require rewriting the completed training run.

## 12. Error handling and resource safeguards

- Fail fast for invalid input/schema, split/protocol violations, inadequate group quotas, bad audio, conflicting/cross-split exact duplicates, metric-contract errors, or corrupt caches.
- Never silently drop a bad clip, alter split membership, apply a cap, replace labels, guess a source pairing, or reuse an incompatible cache.
- For a classifier-specific fitting/runtime failure, record the exception and failed status, continue other classifiers in comparison mode, and clearly report that the run is incomplete. Offline/smoke acceptance requires every enabled classifier to pass.
- Stop an unstable neural fit on non-finite loss/scores; do not persist it as a successful model. Show convergence warnings for classical estimators.
- On GPU memory failure, report the clip/batch and configured duration policy; suggest reducing batch size or explicitly enabling the cap. No automatic CPU fallback or silent truncation.
- Avoid holding source audio, every hidden layer, all frame features, and all neural models on GPU simultaneously. Fit one head at a time using locally staged memory-mapped features.
- Never delete caches, overwrite existing datasets/results, or upload anything to the Hugging Face Hub automatically.

## 13. Implementation work sequence

Implement incrementally in the single notebook source. Each unit below has its own checks; do not wait for a full Colab experiment to discover basic contract errors.

1. **Notebook skeleton and configuration:** percent-cell metadata, guarded setup/execution, dataclasses or plain configuration structures consistent with the existing style, source/group/classifier registries, path controls, versions/seeds. Verify Python parsing and Jupytext conversion.
2. **Input views and selection:** canonical metadata/label mapping, source manifest handling, ITW assignment checks, group filters, seeded balanced quotas, selected-ID freezing, and leakage/audio preflight. Verify with small synthetic single-/two-class sources before touching Drive data.
3. **Pure feature and cache helpers:** frame-length calculation, mean/std pooling, packed-array indexes, stable fingerprints, checksums, completion markers, reuse/corruption checks, and storage estimates. Verify using small synthetic feature sequences.
4. **Frozen encoder adapter:** pinned assets, explicit masks, configured hidden-state selection, no-grad float32 extraction, optional center crop, streaming cache writing, and runtime diagnostics. First validate with a tiny random Wav2Vec2 configuration, then the actual checkpoint on Colab.
5. **Traditional classifier adapters:** instantiate each configured estimator, train-only scaling, continuous-score orientation, serialization, and consistent failure/status handling. Verify all ten non-neural rows, including dummy, on synthetic pooled features.
6. **Neural heads and training:** MLP, masked ASP, masked temporal CNN, common cached-feature training/evaluation loop, early stopping, best checkpoint restore, and weights-only reload. Verify gradient flow into heads only, padding invariance, and serialization.
7. **Evaluation and orchestration:** tested EER/threshold helpers, validation leaderboard/ranking, group reports, configuration/prediction persistence, raw-audio reload inference on a validation probe, and separately gated test scoring.
8. **Colab smoke then pilot comparison:** run all 13 classifiers on the approved smoke preset, verify cache reuse/reload, then run the comparison preset. Record actual hardware, time, failures, and any verification that could not be completed locally.

Do not change `asv_notebook.py`, `itw_notebook.py`, existing generated notebooks/reports, or raw/processed datasets as part of implementation.

## 14. Verification and acceptance criteria

### Offline/synthetic checks

- Python source parses; Jupytext conversion preserves code/Markdown cell ordering and kernel metadata.
- Importing definitions does not install packages, mount Drive, download data/weights, or train.
- Group sampling supports a two-class source and a pair of single-class sources; preserves original splits; is deterministic; enforces exact quotas; detects overlapping group selections and insufficient labels.
- Source/label mapping rejects ambiguous labels, missing columns/splits, known incomplete manifests, and invalid ITW selected-row assignments.
- Selected-audio checks reject empty/non-finite/wrong-rate/wrong-channel inputs and conflicting/cross-split exact duplicates; optional cropping is deterministic and leaves short clips unchanged.
- Frame lengths match the tiny model's frontend outputs; cached frames exclude padding. Mean/std pooling ignores padding and handles one frame.
- ASP/CNN evaluation outputs are invariant within float32 tolerance to extra batch padding or batching alongside longer sequences. CNN tests specifically catch multi-block padding contamination.
- Tiny encoder parameters remain frozen/unchanged through a head update; head parameters receive finite gradients and update. Cached tensors can be used safely in head autograd.
- With fixed features/configuration, a cache-hit path and a cache-miss path produce the same neural initial state, batch order, and fitted scores within the documented deterministic-runtime tolerance. Changing the enabled-classifier list/order must not change another head's initialization or fit.
- All classical adapters and all three neural heads return finite correctly oriented scores of the expected length; no fitted scaler sees validation/test rows.
- EER fixtures cover perfect, reversed, constant, tied-score, invalid-score, and single-class cases. Discrete thresholds are finite, deterministic, and never recalibrated using test data.
- Perturbing test labels cannot change fitted state, selected checkpoint, validation rank, or threshold; it can only change descriptive test metrics.
- Cache identity changes for input/layer/crop/checkpoint or effective reporting-metadata changes, but not classifier-only edits. A constant-language or group-annotation change must not retain a stale cached index or mislabeled grouped reports. Reject missing completion markers, bad shapes/indexes, checksum corruption, and incompatible cache contents.
- All model save/reload scores match original scores within explicit numerical tolerance; a raw-audio validation probe follows the persisted layer/pooling/scaler/head/threshold contract.

### Real Colab smoke acceptance

- GPU and actual audio decoder work; preprocessing artifacts and quotas validate.
- Actual pinned XLS-R produces finite layer-6 features for all smoke clips without encoder gradients or silent cropping.
- All 13 enabled rows finish training/scoring and artifact reload checks; smoke metrics are explicitly labeled non-conclusive.
- A rerun with the same feature configuration reuses the verified cache without loading XLS-R or recomputing embeddings; changing only a classifier configuration also reuses it.
- Feature/model/result artifacts survive Drive save/reload, and incomplete/corrupt outputs are refused.
- Test evaluation remains disabled until explicitly enabled; when enabled, it uses frozen candidate artifacts and validation thresholds without changing the validation winner.

### Comparison acceptance and interpretation

- Exact configured per-group class/split counts are printed and saved; every classifier uses the same selected clips.
- The primary table is ordered by overall validation EER and contains the requested other metrics, with failures/undefined subgroup metrics clearly represented.
- Persisted configuration and prediction scores reproduce the reported tables.
- Report the single-seed, balanced-subset, fixed-configuration limitations. Do not claim statistical significance, deployment calibration, unseen-dataset/language generalization, or full-data superiority.
- Actual local, synthetic, Colab smoke, and comparison checks must be reported separately. Passing syntax checks is not evidence that GPU extraction or training has succeeded.

## 15. User runbook

1. Convert the future source file using the exact `uvx jupytext` command, upload the notebook, and select a Colab GPU runtime.
2. Set actual full-artifact Drive paths, label/metadata mappings, and explicit group membership/filters in configuration. Source paths are user-supplied runtime inputs, not inferred corpus downloads.
3. Keep smoke mode and the default full-recording policy. Run cells in order; fix input/runtime failures before proceeding.
4. Confirm all enabled classifiers, cache publication, and model reload checks pass.
5. Switch to comparison mode and edit per-group quotas or classifier settings as needed. Keep the selected data fixed when comparing configurations.
6. Inspect the validation-ranked table and group diagnostics. If settings change, record a new run; do not repeatedly consult test results while iterating.
7. Once settings and the candidate shortlist are fixed, explicitly enable the final test cell and save its separate evaluation artifact.
8. Confirm promising candidates later using larger data and repeated seeds before making a strong best-model claim.

## 16. References and evidence boundaries

The linked research note records primary papers, architecture differences, frozen-versus-fine-tuned distinctions, and benchmark-selection caveats. It is the research companion, not a claim that these simplified heads reproduce published systems or their EERs.

Technical references checked while planning:

- [Transformers 4.57.3 Wav2Vec2 documentation](https://huggingface.co/docs/transformers/v4.57.3/en/model_doc/wav2vec2): raw-waveform input, hidden states, feature extraction, and attention-mask contracts.
- [Hugging Face Datasets processing](https://huggingface.co/docs/datasets/en/process): saved datasets and compatible-feature concatenation; canonical metadata views avoid requiring identical raw schemas.
- [PyTorch autograd notes](https://docs.pytorch.org/docs/stable/notes/autograd.html): parameter freezing, eval mode, no-grad, and inference-mode differences.
- [scikit-learn ROC documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html): continuous scores, positive-label convention, thresholds, and ROC endpoints.
- [XGBoost model IO](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html): native JSON/UBJSON persistence.

No prepared dataset was loaded, no pretrained model weights were downloaded, and no experiment metrics were generated during planning. Implementation must verify the selected dependency versions and actual Colab execution rather than treating documentation review as a successful run.
