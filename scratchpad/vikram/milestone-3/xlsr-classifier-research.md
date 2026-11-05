# Frozen XLS-R: classifier research findings

**Research date:** 2026-11-05  
**Status:** research notes supporting the [approved implementation plan](model-notebook-plan.md). No classifier has been implemented, trained, or benchmarked in this session.

## 1. Project direction confirmed with the user

- Produce a Google Colab-compatible `model_notebook.py` using Jupytext percent cells, convertible with `uvx jupytext --to ipynb model_notebook.py`.
- Use `facebook/wav2vec2-xls-r-300m` as a completely frozen speech encoder. Train only downstream classification components.
- Read separately preprocessed Hugging Face dataset folders from configurable Google Drive paths.
- Preserve each source's train/validation/test membership and combine corresponding splits. Do not randomly re-split the combined corpus.
- Support additional datasets and languages through a common input contract, rather than hard-coding ASVspoof or In-The-Wild as the only sources.
- Target a free Colab GPU runtime, such as a T4; availability and exact hardware are not guaranteed.
- Provide both a pipeline smoke check and a subset-based comparison to narrow down promising classifiers.
- Keep subset sizes configurable per sampling group and split. A group can contain one two-class source or an explicitly configured combination of single-class sources, such as Hindi real speech plus Hindi synthetic speech. The following balanced defaults were accepted; half of each quota is bonafide and half spoof:

| Preset | Train per group | Validation per group | Test per group |
|---|---:|---:|---:|
| Smoke check | 32 | 16 | 16 |
| Classifier comparison | 400 | 100 | 100 |

The comparison preset totals 1,200 clips for the two initial groups, ASVspoof and In-The-Wild. This is not a new 400/100/100 split of the original data: it is sampling inside the existing partitions. Single-class folders are not independently forced to contain both labels; their configured group's union provides the class coverage.

The existing preprocessing notebooks use `0 = bonafide`, `1 = spoof`, unaugmented mono 16 kHz PCM16 WAV audio embedded in saved Hugging Face datasets. Source identity and grouping metadata still matter for leakage checks and interpretation, even when the audio contract is shared.

## 2. Main research findings

### 2.1 Frozen wav2vec2 with attentive statistics pooling

**Martín-Doñas and Álvarez, ICASSP 2022 — The Vicomtech Audio Deepfake Detection System Based on Wav2vec2 for the 2022 ADD Challenge.**

- Evaluates multilingual 300M-parameter wav2vec2 models trained on 53 and 128 languages.
- Section 3.4 explicitly freezes wav2vec2 and updates only the downstream classifier.
- The original head combines 25 hidden representations with learned layer weights, applies frame-wise feed-forward layers, and performs attentive statistics pooling (ASP).
- ASP computes an attention-weighted temporal mean and standard deviation, yielding an utterance embedding.
- The original system then uses a linear embedding projection, cosine scoring, and one-class softmax loss.

**Relevance:** direct precedent for a trainable attention-pooling classifier on a frozen multilingual speech encoder. A single-layer ASP head with a conventional binary output would be a simplified, research-inspired baseline, not an exact reproduction of this paper.

Primary source: <https://arxiv.org/html/2203.01573v1>, especially Sections 2.2 and 3.4.

### 2.2 XLS-R with an AASIST graph-attention backend

**Tak et al., Odyssey 2022 — Automatic Speaker Verification Spoofing and Deepfake Detection Using Wav2vec 2.0 and Data Augmentation.**

- Replaces the original waveform frontend with XLS-R representations and uses an AASIST spectro-temporal graph-attention backend.
- Introduces self-attentive aggregation and investigates RawBoost augmentation.
- The original system fine-tunes its pretrained speech frontend. Its published results are not results for a completely frozen XLS-R pipeline.

**Relevance:** an established, task-specific backend for future experiments. It is more involved than a small MLP, ASP head, or temporal CNN. Adapting it to frozen Hugging Face XLS-R features requires explicit interface and training changes; an unmodified raw-waveform AASIST model is not simply interchangeable with a vector classifier.

Primary source: <https://www.isca-archive.org/odyssey_2022/tak22_odyssey.html> and its linked paper.

### 2.3 Frozen wav2vec2 with mixture-of-experts fusion and AASIST

**Wang et al., 2024 — Mixture of Experts Fusion for Fake Audio Detection Using Frozen wav2vec 2.0.**

- Uses the last hidden state to gate experts that process features from other encoder layers.
- Feeds the fused representations into AASIST while keeping the pretrained encoder frozen.
- The reported frozen MoE system has approximately 25.92 million trainable parameters, far more than a small classifier head.
- Its ablations show that freezing versus fine-tuning is condition-dependent: the frozen configuration performs better on some benchmarks, but the fine-tuned configuration performs better on In-The-Wild in the reported comparison.

**Relevance:** supports frozen multi-layer feature learning, but does not establish that freezing universally maximizes detection performance. Better treated as a later extension than an initial free-Colab baseline.

Primary source: <https://arxiv.org/html/2409.11909v1>, especially Sections II and III-C2.

### 2.4 Layer-wise attention pooling and decision fusion

**Xiao and Vu, Interspeech 2025 — Layer-Wise Decision Fusion for Fake Audio Detection Using XLS-R.**

- Extracts features from a frozen XLS-R encoder.
- Applies attention-based temporal pooling separately to each hidden representation, with an optional shared bottleneck projection.
- Trains per-layer one-class classifiers and combines their scores after classification rather than immediately fusing all layer features.
- Finds that layer usefulness varies; early-middle and some later layers contribute differently. The final layer should not automatically be assumed to be the best spoofing representation.
- The experiments use particular silence-trimming, cropping, augmentation, and training settings that differ from this project's existing preprocessing.

**Relevance:** a strong frozen-XLS-R research direction, but storing and training on all layers increases complexity. Do not silently change existing preprocessing to imitate the paper.

Published source: <https://www.isca-archive.org/interspeech_2025/xiao25d_interspeech.html>.  
Full text inspected: <https://arxiv.org/html/2607.20023v1> (a later arXiv posting of the Interspeech 2025 work).

### 2.5 Single-layer pooling versus multi-head factorized attention

**Serrano et al., 2025 preprint — Improving Out-of-Domain Audio Deepfake Detection via Layer Selection and Fusion of SSL-Based Countermeasures.**

- Freezes six SSL backbones, including XLS-R-300M.
- Compares single-layer mean-pooling heads with multi-head factorized attentive pooling (MHFA), which combines layer weighting and attention.
- Reports that intermediate layers can outperform deeper layers, with results varying by encoder and evaluation corpus.

**Relevance:** motivates making the feature layer configurable and comparing pooling approaches. It does not justify selecting our layer by inspecting our held-out test results. The paper's best-single-layer analysis uses evaluation-corpus performance and must not be adopted as our model-selection protocol.

Source: <https://arxiv.org/abs/2509.12003> and its linked paper.

### 2.6 Other backend and layer-selection studies

**Kang et al., 2024 preprint — Experimental Study: Enhancing Voice Spoofing Detection Models with wav2vec 2.0.**

- Studies LCNN, Non-OFD, RawNet2, RawGAT-ST, and AASIST backends with selected/fine-tuned wav2vec2 layers.
- Reinforces that backend design and frontend adaptation interact. Its strongest results do not establish equivalent performance for our frozen 300M encoder.
- Provides context for convolutional backend families, but is not a specification for the lightweight temporal CNN requested for this notebook.

Source: <https://arxiv.org/html/2402.17127v1>.

**Beheshti et al., 2026 preprint — Probing-Guided Layer Selection from Self-Supervised Speech Models for Generalizable Audio Deepfake Detection.**

- Uses XGBoost on pooled layer statistics, then trains a compact attention-pooling and MLP classifier on selected layers while keeping XLS-R frozen.
- Illustrates that traditional classifiers can probe frozen representations and that attention heads need not approach the size of the encoder.
- Important protocol caveat: Section 3.1.2 ranks layers using several benchmark test datasets. We must not reproduce that test-guided selection procedure or interpret its resulting benchmark estimates as an untouched-test protocol.

Source: <https://arxiv.org/html/2606.30791v1>.

## 3. Classifier roster requested for the initial notebook

| Family | Classifiers | Required feature representation |
|---|---|---|
| Linear / margin | Logistic Regression, Linear SVM, RBF SVM | One fixed-size vector per clip |
| Nearest-neighbor / probabilistic | k-Nearest Neighbors, Gaussian Naive Bayes | One fixed-size vector per clip |
| Tree ensembles | Random Forest, Extra Trees, Histogram Gradient Boosting, XGBoost | One fixed-size vector per clip |
| PyTorch neural baseline | MLP | One fixed-size vector per clip |
| PyTorch attention head | ASP with trainable projection/classifier | Frame-level feature sequence and valid lengths/mask |
| PyTorch convolutional head | Lightweight temporal CNN with pooling/classifier | Frame-level feature sequence and valid lengths/mask |
| Sanity baseline | Dummy classifier | Same clip/label membership as other classifiers |

The user explicitly requested all three neural classifiers, not only MLP. The enabled classifier list should remain configurable. ASP and the lightweight temporal CNN should be identified as research-inspired baselines unless an exact paper implementation is deliberately selected later.

## 4. Representation and resource implications

- XLS-R produces a sequence of 1,024-dimensional frame representations, not automatically one vector per recording.
- Traditional classifiers and the MLP use deterministic temporal mean/std pooling to form a fixed-size clip vector. The agreed initial encoder representation is layer 6, configurable, and recordings remain full length unless an optional duration cap is explicitly enabled.
- ASP learns temporal attention weights from frame features. It cannot learn which frames matter if only an already-pooled clip vector is available.
- A temporal CNN must convolve over time in the frame sequence. Applying convolutions across the coordinates of a pooled vector is not equivalent temporal modeling.
- Cache selected frozen frame features so ASP and CNN can train repeatedly without re-running the large encoder every epoch. Derive pooled vectors from those same features for comparable inputs.
- Cache size depends strongly on duration, selected layer count, and precision. For scale only: about 200 frames x 1,024 features x 4 bytes is roughly 0.78 MiB per four-second clip per layer in float32, versus 4 KiB for one mean-pooled vector. Actual frame counts and storage overhead differ.
- Attention and pooling must exclude padding. Temporal convolutions also need valid-length handling so padded activations do not contaminate the clip representation.
- Freezing parameters is distinct from evaluation mode. A stationary cached representation should use a frozen encoder in evaluation mode, disabling its dropout and training-time SpecAugment; the neural heads remain trainable.
- PyTorch documentation distinguishes `no_grad` from `inference_mode`: outputs created directly under inference mode have restrictions when reused in autograd-recorded head training. Use an appropriate no-grad extraction path or materialize ordinary cached tensors before training.

Technical reference checked: <https://docs.pytorch.org/docs/stable/notes/autograd.html>, especially grad modes and evaluation mode.

## 5. Interpretation and evaluation cautions

- More complicated heads are not guaranteed to perform better on a small subset; overfitting and initialization noise can dominate.
- Freezing XLS-R is a useful controlled, lower-cost baseline, not a claim of globally optimal performance.
- Every classifier should see the same selected original clips and labels. Differences between fixed pooling and learned temporal heads must be recorded rather than described as identical feature vectors.
- Fit scaling, optional feature transformations, and classifiers using training data only.
- Confirmed: rank models primarily by overall validation EER, choose operating thresholds near the validation EER point, and reserve test scoring for final reporting after those choices are fixed. Report Accuracy, Precision, Recall, and F1 alongside EER.
- Keep score direction consistent with the canonical labels. Several cited systems score higher for bonafide, whereas this project's positive class is spoof. EER must use continuous scores, not hard predictions.
- Balanced per-group subsets are convenient for controlled comparison but do not reproduce full-corpus class prevalence. Accuracy, precision, recall, and F1 estimates must be labeled accordingly. Pairing real-only and fake-only corpora does not by itself remove source/channel confounding.
- Preserve source and language metadata for audits and possible grouped reports; do not feed them to the classifier as predictive features.
- Combined test-split performance after training on both source datasets is not an unseen-dataset evaluation. Likewise, multilingual training plus multilingual testing does not automatically constitute an unseen-language experiment.
- Published EERs across the cited works are not directly comparable because training data, preprocessing, cropping, augmentation, losses, backbone adaptation, and evaluation protocols differ.
- Subset results can shortlist candidates; they do not establish the best full-data model. Repeated-seed or larger-data confirmation is needed before a strong performance claim.

## 6. Final design decisions and boundaries

The user confirmed the shared design and requested the implementation plan. The agreed experiment uses one editable configuration per classifier, no automatic hyperparameter search, no waveform augmentation, frozen/eval-mode XLS-R layer-6 features, full recordings by default, persistent Drive caches/models/results, and all three PyTorch heads. Neural comparison training defaults to at most 20 epochs with validation-EER early stopping patience 5.

The [implementation plan](model-notebook-plan.md) is authoritative for the final source/group contract, feature/cache layout, classifier defaults, metric details, tests, and execution sequence. This research note does not authorize code implementation, claim completed tests, or substitute published results for a new experiment.
