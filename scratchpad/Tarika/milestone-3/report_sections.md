# Tarika: Milestone-3 Report Sections (Draft)

Sections 3, 5 and 6 of `milestones/Milestone-3/Milestone-3.md`, for Himanshu to compile. Links and image paths are
written as they'll be in the final report (relative to `milestones/Milestone-3/`). The run notebook and its outputs
are already in place (`notebooks/Milestone-3/02_pipeline_check.ipynb` and `run_outputs/`). The three figures are
still in `figures/` here and go to `assets/infographics/Milestone-3/`.

Every number below comes from the saved run in `02_pipeline_check.ipynb` (Kaggle, one T4).

---

## 3. Model Architecture

The detector is a pretrained speech encoder, **XLS-R 300M** (Babu et al., 2021), followed by a small classification head that we train. It reads the waveform directly; there are no hand-made features such as spectrograms or MFCCs. It's implemented in Section 3 of [`02_pipeline_check.ipynb`](../../notebooks/Milestone-3/02_pipeline_check.ipynb).

### 3.1 Components

| Component | What it does | Pretrained? | Trained in this milestone? |
| --- | --- | --- | --- |
| Preprocessing (notebook 01) | Converts every clip to 16 kHz mono, trims edge silence, sets loudness to −20 dBFS (peak at most 0.99), stores 16-bit WAV | — | — |
| Crop / pad | Cuts or pads each clip to exactly 4 s (64,000 samples): a random 4 s in training, the middle 4 s otherwise. Padded positions are marked in an attention mask | — | — |
| Scaling | Rescales each clip to zero mean and unit variance, as XLS-R's own feature extractor does (`do_normalize=True`) | — | — |
| XLS-R CNN feature encoder | 7 convolution layers turn the waveform into one 512-value frame every 20 ms | Yes (`facebook/wav2vec2-xls-r-300m`) | No, frozen |
| XLS-R transformer | 24 layers that model the context across the clip; one 1,024-value vector per frame | Yes | Yes, fine-tuned (learning rate 1e-5) |
| Mean pooling | Averages the frame vectors over the frames that hold audio (padding excluded) | — | — |
| Classification head | Dropout(0.1) → Linear(1024, 2) | No, new | Yes (learning rate 1e-3) |
| Softmax and threshold | Turns the two logits into `p_fake`, the probability that the clip is fake. The verdict uses the threshold where validation reaches its EER | — | Threshold from `valid` only |

The model has **315.4 M parameters**. **311.2 M** are trained; the frozen CNN feature encoder holds the other 4.2 M.

**Pretrained component.** XLS-R was pretrained by Meta, without labels, on about 436,000 hours of speech in 128 languages, including Hindi, Tamil and Telugu. We use it as a general speech encoder and fine-tune it to tell real from synthetic speech. XLS-R's built-in time masking (`mask_time_prob=0.075`) is active only while training.

### 3.2 Data-Flow Diagram

<!-- Vikram's diagram: assets/infographics/Milestone-3/architecture.png -->

### 3.3 Input and Output Shapes

The shapes below were printed by the model on one training batch (notebook 02, Section 3). `B` is the batch size: 8 in training, 16 in inference.

| Stage | Tensor | Shape | Type |
| --- | --- | --- | --- |
| Model input | Waveform, 4 s at 16 kHz | `[B, 64000]` | float32 |
| Model input | Attention mask (1 = audio, 0 = padding) | `[B, 64000]` | int64 |
| After the CNN feature encoder | One frame per 20 ms | `[B, 199, 512]` | float |
| After the transformer | Contextual frame vectors | `[B, 199, 1024]` | float |
| After mean pooling | One vector per clip | `[B, 1024]` | float |
| Head output | Logits for (real, fake) | `[B, 2]` | float |
| Final output | `p_fake` and the verdict | `[B]` | float, then real / fake |

**How the processed data matches the model's input.** XLS-R expects raw 16 kHz mono audio as a float waveform. Notebook 01 already stores every clip at 16 kHz mono, so the only steps at load time are the 4 s crop or pad, the zero-mean/unit-variance scaling, and the attention mask. Because XLS-R rescales each clip itself, the −20 dBFS loudness from preprocessing doesn't reach the model directly. It still keeps the stored clips consistent and free of clipping.

---

## 5. End-to-End Pipeline Verification

We ran the complete workflow once on a small subset: load the preprocessed clips, show a sample before and after preprocessing, train XLS-R with its head for one epoch, score every validation and test clip, and evaluate. The run is saved with all its outputs in [`02_pipeline_check.ipynb`](../../notebooks/Milestone-3/02_pipeline_check.ipynb). Its scores, loss log and results table are in [`run_outputs/`](../../notebooks/Milestone-3/run_outputs/).

### 5.1 Subset Used

The subset has **3,018 clips (4.73 h after preprocessing)**. It was chosen with seed 42 by [`01_data_preparation.ipynb`](../../notebooks/Milestone-3/01_data_preparation.ipynb) (Section 2 of this report). Every split is balanced between real and fake, and every language, generator and test condition is present:

| Split | Real | Fake | Total | Contents |
| --- | ---: | ---: | ---: | --- |
| `train` | 746 | 746 | 1,492 | Kathbath + IndicSynth (Hindi, Tamil, Telugu); ASVspoof 2019 LA train (English) |
| `valid` | 197 | 200 | 397 | The same sources, other speakers; ASVspoof dev |
| `test_indomain` | 150 | 150 | 300 | Kathbath + IndicSynth, other speakers again |
| `test_unseen_gen` | 270 | 270 | 540 | Our generated set (Indic Parler-TTS, IndicF5, MMS-TTS) + Kathbath test speakers reading the same sentences |
| `test_itw` | 97 | 96 | 193 | In-the-Wild (English, real-world) |
| `test_asvspoof` | 48 | 48 | 96 | ASVspoof 2019 LA eval (English, attacks A07–A19) |

### 5.2 A Sample Before and After Preprocessing

<div align="center">
  <img src="../../assets/infographics/Milestone-3/before_after_preprocessing.png" alt="Figure 5.1: Waveform and spectrogram of a real Kathbath clip and a fake IndicSynth XTTS-v2 clip (Hindi), before and after preprocessing." width="800">
  <br>
  <em>Figure 5.1: Waveform and spectrogram of a real Kathbath clip and a fake IndicSynth XTTS-v2 clip (Hindi), before and after preprocessing.</em>
</div>

| Clip | Stage | Sample rate | Duration | RMS | Peak |
| --- | --- | ---: | ---: | ---: | ---: |
| Real (Kathbath) | before | 16 kHz | 4.53 s | −28.0 dBFS | 0.56 |
| Real (Kathbath) | after | 16 kHz | 4.48 s | −23.0 dBFS | 0.99 |
| Fake (XTTS-v2) | before | 24 kHz | 3.80 s | −18.4 dBFS | 1.00 |
| Fake (XTTS-v2) | after | 16 kHz | 3.24 s | −20.0 dBFS | 0.77 |

Before preprocessing, the fake clip was 24 kHz and the real one 16 kHz, a difference a model could use as a shortcut. Afterwards both are 16 kHz, and the edge silence is cut (0.56 s from the fake clip). The fake clip reaches −20 dBFS. The real clip stops at −23.0 dBFS because raising it further would push its peak past 0.99.

### 5.3 How to Reproduce the Run

1. **Build the subset** (optional; it's already on Kaggle): `pip install -r requirements.txt`, accept the terms of `ai4bharat/Kathbath` on Hugging Face, then run `notebooks/Milestone-3/01_data_preparation.ipynb` from top to bottom. It writes `data/`, which is uploaded as the private Kaggle dataset `tarikasubhashhedaoo/dsai-group3-m3-subset`. Ask the team for access.
2. **Run the pipeline:** on Kaggle, open `notebooks/Milestone-3/02_pipeline_check.ipynb`, choose a **T4 GPU**, turn on internet access (XLS-R is downloaded from Hugging Face), add the dataset as input, and use *Run All*. The notebook finds the dataset under `/kaggle/input` by itself.

The notebook writes `train_log.csv`, `predictions.csv`, `results.csv`, three figures, and the checkpoint `detector_1epoch.pt` (1.2 GB, not stored in the repository).

### 5.4 Run Log

| Item | Value |
| --- | --- |
| Hardware | Kaggle, one NVIDIA Tesla T4 (16 GB) |
| Model | XLS-R 300M + head, 315.4 M parameters (311.2 M trained) |
| Training | 1 epoch, 187 steps of 8 clips, weighted cross-entropy (weights 1.0 / 1.0, as train is balanced), AdamW (1e-5 encoder, 1e-3 head), fp16, seed 42 |
| Training time | 3.0 min |
| Loss | 0.701 (mean of the first 25 steps) → 0.610 (mean of the last 25) |
| Inference | 1,526 clips (every split except `train`) in 174 s, batch 16 |

<div align="center">
  <img src="../../assets/infographics/Milestone-3/training_loss.png" alt="Figure 5.2: Weighted cross-entropy per training step over one epoch, falling from about 0.70 to about 0.61." width="800">
  <br>
  <em>Figure 5.2: Weighted cross-entropy per training step over one epoch, falling from about 0.70 to about 0.61.</em>
</div>

The loss falls steadily, so the model learns from the data, and nothing breaks between the data, model and optimiser. Training on a GPU isn't exactly repeatable even with a fixed seed: two earlier runs of the same notebook ended at a slightly different loss and EERs. The numbers in this report all come from the saved run.

### 5.5 Sample Outputs

One real and one fake clip from each test split, chosen at random (seed 42). A clip is called **fake** when `p_fake` ≥ **0.316**, the threshold where validation reaches its EER. The confidence is the model's probability for the label it predicts.

| Clip | Test split | Language | Generator | Expected | Predicted | `p_fake` | Confidence |
| --- | --- | --- | --- | --- | --- | ---: | ---: |
| `kathbath_hi_844424930731870-1148-f` | `test_indomain` | Hindi | — | real | real ✅ | 0.242 | 0.758 |
| `indicsynth_te_xtts_v2____1195-m-722` | `test_indomain` | Telugu | XTTS-v2 | fake | real ❌ | 0.309 | 0.691 |
| `kathbath_te_844424932737198-944-f` | `test_unseen_gen` | Telugu | — | real | real ✅ | 0.264 | 0.736 |
| `te_0513_mms-tts` | `test_unseen_gen` | Telugu | MMS-TTS | fake | real ❌ | 0.283 | 0.717 |
| `itw_10118` | `test_itw` | English | — | real | real ✅ | 0.266 | 0.734 |
| `itw_31576` | `test_itw` | English | unknown TTS | fake | fake ✅ | 0.382 | 0.382 |
| `asvspoof_LA_E_9034056` | `test_asvspoof` | English | — | real | fake ❌ | 0.398 | 0.398 |
| `asvspoof_LA_E_4945590` | `test_asvspoof` | English | A10 | fake | fake ✅ | 0.349 | 0.349 |

After one epoch every `p_fake` is between 0.20 and 0.65, and half are below 0.30. So a correct "fake" verdict can come with a `p_fake` below 0.5 and a low confidence. All 1,526 scores are in [`run_outputs/predictions.csv`](../../notebooks/Milestone-3/run_outputs/predictions.csv).

### 5.6 Failures, Incomplete Parts and Workarounds

| Item | What happened | Workaround or plan |
| --- | --- | --- |
| Saving the executed notebook | Kaggle's command-line tool returns a notebook's output files but not the notebook with its outputs | The notebook is run through `jupyter nbconvert --execute` in a small Kaggle script, which saves the executed copy as an output file |
| Kaggle mount path | Kaggle mounted the dataset at `/kaggle/input/datasets/<owner>/<name>/`, not the usual `/kaggle/input/<name>/` | Notebook 02 searches `/kaggle/input` for the manifest instead of using a fixed path |
| ASVspoof eval threshold | At the validation threshold, every ASVspoof eval clip is called fake (accuracy 0.500), although the scores still rank them (AUC 0.763) | After one epoch the scores aren't consistent across sources. Longer training and per-source analysis come in Milestone 4 |
| IndicF5 fakes | Only 26.7% are caught at the validation threshold | IndicF5 clones real voices; it's the hardest unseen generator. To be tracked in full training |
| Augmentation | Implemented by Vikram (noise, shift, speed, pitch, reverberation), not used in this run | Switched off on purpose for the one-epoch check; planned for Milestone 4 |
| English training data | Taken in blocks of clips, so ASVspoof train covers attacks A01–A05 and valid covers A04 and A06 | Fine for a pipeline check; full training will use every attack |
| Clips longer than 4 s | Only 4 s is used: a random 4 s in training, the middle 4 s in evaluation | Scoring several windows per clip is planned for Milestone 4 |
| Explanation view | The saliency output from the Milestone-1 design isn't built yet | Planned for Milestone 4 |

---

## 6. Loss Functions and Evaluation Metrics

### 6.1 Training Loss

The model is trained with **cross-entropy** on its two logits (real, fake). This is the standard loss for a two-class decision: it pushes `p_fake` towards 1 for fake clips and towards 0 for real ones, and it penalises confident mistakes the most. Each class is weighted by `N / (2 × N_class)`, with the counts taken from `train` only, so that a smaller class isn't ignored. Our `train` split is balanced (746 real, 746 fake), so both weights are 1.0 here. The weights matter for full training, where the classes won't be balanced.

### 6.2 Evaluation Metrics

The loss is only used for training. The model is judged by these metrics, which follow the objectives set in Milestone 1:

| Metric | What it measures | Why we use it |
| --- | --- | --- |
| **EER** (main metric) | The error rate at the threshold where the share of fakes accepted as real equals the share of real clips rejected as fake | It doesn't depend on a chosen threshold, it isn't misled by class imbalance, and it's the standard measure in ASVspoof work, so our numbers can be compared with published ones. Our Milestone-1 targets are stated as EER |
| **ROC AUC** | How well `p_fake` ranks fake clips above real ones over all thresholds (0.5 = chance, 1 = perfect) | Shows whether the scores separate the classes even when one threshold works badly |
| **Accuracy and confusion matrix** | Results at one fixed threshold: the one where `valid` reaches its EER | Shows how a deployed system with one threshold would behave. The threshold is chosen on `valid` only and then used unchanged on every test split |

Each test split answers one Milestone-1 question: in-domain (new speakers), unseen generators, real-world audio (In-the-Wild) and unseen English attacks (ASVspoof eval). The cross-language and phone-channel tests are planned for full training in Milestone 4.

### 6.3 Preliminary Results

From Section 6 of notebook 02, after one epoch. The decision threshold is `p_fake` ≥ 0.316, chosen on `valid`.

| Split | Clips | EER | ROC AUC | Accuracy | Real → real | Real → fake | Fake → real | Fake → fake |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `valid` | 397 | 24.4% | 0.863 | 0.756 | 149 | 48 | 49 | 151 |
| `test_indomain` | 300 | 17.7% | 0.924 | 0.780 | 142 | 8 | 58 | 92 |
| `test_unseen_gen` | 540 | 20.6% | 0.854 | 0.743 | 251 | 19 | 120 | 150 |
| `test_itw` | 193 | 31.6% | 0.735 | 0.658 | 89 | 8 | 58 | 38 |
| `test_asvspoof` | 96 | 33.3% | 0.763 | 0.500 | 0 | 48 | 0 | 48 |

<div align="center">
  <img src="../../assets/infographics/Milestone-3/roc_curves.png" alt="Figure 6.1: ROC curves of the validation and four test splits after one epoch." width="500">
  <br>
  <em>Figure 6.1: ROC curves of the validation and four test splits after one epoch.</em>
</div>

Share of each unseen generator's fakes that are caught at the threshold:

| Generator | Hindi | Tamil | Telugu | All |
| --- | ---: | ---: | ---: | ---: |
| Indic Parler-TTS | 1.000 | 0.767 | 0.933 | 0.900 |
| IndicF5 | 0.300 | 0.133 | 0.367 | 0.267 |
| MMS-TTS | 0.967 | 0.533 | 0.000 | 0.500 |

**What these results show.** The model already ranks fakes above real clips on every test split (AUC 0.735–0.924). It's best on new speakers with known generators (EER 17.7%), close behind on unseen Indian-language generators (20.6%), and weakest on real-world English (31.6%) and ASVspoof eval (33.3%). Among the unseen generators, IndicF5 is by far the hardest. On ASVspoof eval the validation threshold calls every clip fake. The scores still separate the classes, but the threshold doesn't carry over to that source yet.

**These numbers check the pipeline.** They come from one epoch on 1,492 training clips. They don't measure final performance and shouldn't be compared with our Milestone-1 targets (EER ≤ 2% on ASVspoof eval, ≤ 5% in-domain, < 15% on unseen generators). Those are for full training in Milestone 4.
