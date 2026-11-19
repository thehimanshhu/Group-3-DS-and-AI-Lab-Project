# Team Contribution Tracker — Milestone 4

**Project:** Deepfake Voice Detection for Indian Languages

This document tracks the work completed and responsibilities assigned for Milestone 4.

---

## 1. Himanshu Saini

### Contributions in Milestone 4

- Planned Milestone 4, divided the work among the team, and set up the way we worked together: one config and one result folder per run, a run tracker issue for all 35 runs, and a shared training notebook that runs a queue of runs in one Kaggle session.
- Moved the shared preprocessing into `src/preprocess.py` and checked that it reproduces the Milestone-3 files byte for byte.
- Built the training data (`01_data_preparation.ipynb`, about 29,000 training and 4,000 validation clips with a tuning subset) and the test sets with their Milestone-4 sample (`01b_test_data.ipynb`), with leakage checks, and uploaded them as the Kaggle dataset.
- Found that almost every real Kathbath clip starts with a recording click that gives the label away, removed it from every clip (`01c_click_fix.ipynb`) and checked the result.
- Added training support for the two baselines to the training script, chose the final config with a paired bootstrap on the validation results, and wrote the F-01 to F-03 configs.
- Trained F-01, wrote the scoring script (`src/score.py`), and scored the final models, the default model, the baselines and the cross-language runs on the test sample.
- Built and ran the final notebook (`03_training_and_evaluation.ipynb`), reviewed and merged the team's pull requests, and wrote the Milestone-4 report.

---

## 2. Tarika Subhash Hedaoo

### Contributions in Milestone 4

- Wrote the training code: the model (`src/model.py`) with mean or attentive pooling, frozen layers and checkpoint loading; the data loader (`src/dataset.py`) with cropping, training-only augmentation, the attention mask and the choice of rows; and the training script (`src/train.py`) with warmup and cosine decay, mixed precision, gradient clipping and accumulation, two GPUs, early stopping, per-epoch logs, checkpoints and resuming.
- Wrote unit tests for the training code, including a check that padding a clip does not change its score.
- Wrote the configs for the hyperparameter sweeps (T-01 to T-14), the default model on the full data (D-01) and the cross-language runs (C-hi, C-ta, C-te).
- Ran the 14 hyperparameter sweeps, D-01, the cross-language runs and the T-01 rerun on the click-free data on Kaggle, and added their results.

---

## 3. Pal Amitkumar Shish

### Contributions in Milestone 4

- Wrote the shared equal error rate function (`src/metrics.py`) and the evaluation module (`src/evaluate.py`): scoring with several 4 s windows per clip, results per test set, per generator and per language, and a comparison of shared and per-source thresholds.
- Wrote the gradient-saliency explanation view (`src/explain.py`).
- Implemented the two baselines, LFCC-LCNN and AASIST-L (`src/baselines/`), with smoke tests for training and scoring.
- Trained both baselines (A-01, A-02) and the final model with seed 43 (F-02) on Kaggle, added their results, and shared the checkpoints for scoring.

---

## 4. Vikram Singh Negi

### Contributions in Milestone 4

- Wrote the augmentation module (`src/augment.py`): Gaussian noise, gain, speed, pitch, time shift, reverberation, RawBoost and a phone channel (band-limiting, 8 kHz, G.711), with one configurable entry point and unit tests.
- Wrote the configs for the regularization, augmentation and stability runs (V-01 to V-11) and tests that check every config changes only its own setting.
- Ran V-01 to V-11 on Kaggle and wrote up their results on augmentation, regularization and stability.
- Added a version of the training notebook with startup checks for the GPU, ffmpeg and augmentation.
- Found and reported a bug in the final notebook's regularization table.
- Trained the final model with seed 44 (F-03) and added its results.
