# Team Contribution Tracker — Milestone 5

**Project:** Deepfake Voice Detection for Indian Languages

This document tracks the work completed and responsibilities assigned for Milestone 5.

---

## 1. Himanshu Saini

### Contributions in Milestone 5

- Planned Milestone 5 and divided the work among the team.
- Built the full test sets: the Indian-language sets locally and the full In-the-Wild and ASVspoof sets on Kaggle (118,295 test clips), merged the manifest, ran the leakage checks and uploaded the data (`01_full_test_sets.ipynb`).
- Extended `src/score.py` to score the full test sets from several data folders, and sped up scoring about 6.5 times (windows from many clips in one batch, fp16 on both GPUs, audio read by worker processes), checked clip by clip against the Milestone-4 predictions.
- Wrote the full-scoring scripts for every model and scored F-01 on the full test sets.
- Released the final, default and baseline weights with the data manifest on the Hugging Face Hub, with a model card.
- Wrote the error analysis, operating-point and calibration notebook and the onset check on all three final seeds (`02_errors_and_calibration.ipynb`, with the `--trim-start` scoring option).
- Reviewed and merged the team's pull requests, wrote the Milestone-5 report and the AI usage disclosure.

---

## 2. Tarika Subhash Hedaoo

### Contributions in Milestone 5

- Scored the default model (D-01) on the full test sets and the cross-language models (C-hi, C-ta, C-te) on the test clips of their held-out language.
- Scored all 26 sweep checkpoints (T-01 to T-14, T-01c, V-01 to V-11) on the validation set and the Milestone-4 test sample.
- Wrote the paired-bootstrap and Holm-correction code with tests (`src/compare.py`), applied the Holm correction to the 24 sweep comparisons, measured the seed spread of the final config and compared the default with the final config on the full test sets (`03_configurations_and_sweeps.ipynb`).

---

## 3. Pal Amitkumar Shish

### Contributions in Milestone 5

- Wrote a batched fp16 version of the scoring function.
- Scored the final model with seed 43 (F-02) and both baselines (A-01, A-02) on the full test sets.
- Added PR AUC, precision, recall and F1 to the evaluation module, and built the six-model comparison with bootstrap intervals and the per-generator, per-attack and per-language tables (`results/milestone5_*.csv`).
- Wrote the temperature-scaling check fitted on validation (`src/calibrate.py`) and the per-test-set ROC, precision-recall, DET, confusion-matrix, reliability and score-distribution figures.

---

## 4. Vikram Singh Negi

### Contributions in Milestone 5

- Scored the final model with seed 44 (F-03) on the full test sets.
- Wrote the error-analysis notebook (`error_analysis.ipynb`): threshold transfer, missed fakes and flagged real clips per test set, errors by generator, attack, duration and speaker group, and the quiet-frame analysis.
- Ran the full-set onset check on F-03 and a listening review of six failures with saliency maps, using private Kaggle scripts so that no audio is published (`kaggle/m5-onset/`, `kaggle/m5-audio/`).
- Wrote the qualitative-results and error-analysis sections of the report.
