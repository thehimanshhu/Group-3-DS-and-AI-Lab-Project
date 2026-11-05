# Team Contribution Tracker — Milestone 3

**Project:** Deepfake Voice Detection for Indian Languages

This document tracks the work completed and responsibilities assigned for Milestone 3.

---

## 1. Himanshu Saini

### Contributions in Milestone 3

- Planned Milestone 3 and divided the work among the team.
- Built the data preparation notebook (`01_data_preparation.ipynb`): rebuilt the Milestone-2 speaker-disjoint split, selected the seeded 3,018-clip subset from all five datasets, applied the shared preprocessing, and wrote the manifest, leakage checks and folder structure.
- Found and fixed a decoding bug that silently dropped Kathbath test recordings stored as m4a.
- Added ASVspoof 2019 LA train and dev audio to the subset, in response to the Milestone-2 feedback, and checked its counts against the official protocol.
- Uploaded the processed subset as a private Kaggle dataset for the training notebook.
- Drew the architecture diagram, compiled the Milestone-3 report, and made the presentation.

---

## 2. Tarika Subhash Hedaoo

### Contributions in Milestone 3

- Built the end-to-end pipeline notebook (`02_pipeline_check.ipynb`): data loading, a sample before and after preprocessing, the XLS-R 300M model with its classification head, one training epoch, inference and evaluation.
- Ran the pipeline on a Kaggle T4 GPU and saved the notebook with all its outputs, plus the predictions, results and training log.
- Evaluated the run with EER, ROC AUC, accuracy and confusion matrices per test split, and analysed the unseen generators and sample outputs.
- Wrote the report sections on the model architecture, the pipeline verification, and the loss functions and evaluation metrics.

---

## 3. Pal Amitkumar Shish

### Contributions in Milestone 3

- Wrote a download script for the Kathbath train split (Hindi, Tamil, Telugu), pinned to a fixed dataset revision, with a download log and metadata.
- Built a Kathbath preprocessing notebook: checked that every speaker has one consistent gender label, made a gender-stratified 70/15/15 speaker split, and selected a 1,500-clip sample (500 per language).
- Ran the shared preprocessing on all 1,500 clips (0 failures) and verified the output: 16 kHz mono files, no missing files and no speaker in two splits.
- Analysed the effect of preprocessing on Kathbath: raw vs processed waveforms, spectrograms, duration, RMS and peak for each language and gender.
- Audited the start of 100 recordings and found a consistent peak about 47 ms into the raw audio, which silence trimming moves to about 7 ms. Wrote up the findings with figures.

---

## 4. Vikram Singh Negi

### Contributions in Milestone 3

- Found a full copy of ASVspoof 2019 LA with the train and dev audio (`Bisher/ASVspoof_2019_LA`) and checked its counts against the official protocol (train 25,380, dev 24,844, eval 71,237, with no speaker or utterance shared between partitions). This source answers the Milestone-2 feedback and supplies the English training data.
- Built preprocessing notebooks for ASVspoof 2019 LA and In-the-Wild that standardise the full datasets (16 kHz mono, −20 dBFS, 16-bit PCM) and save them as Hugging Face datasets, with verified copies and checks after reloading.
- Created a speaker-disjoint 70/15/15 split of In-the-Wild (22,247 / 4,769 / 4,763 clips), and checked the archive for duplicate files.
- Implemented and tested training-only data augmentation: noise at about 15 dB SNR, time shift, speed and pitch changes, room reverberation, and XLS-R's built-in time and feature masking.
- Wrote a research note on classifiers for a frozen XLS-R encoder, reviewing six recent papers.
- Built a model comparison notebook that extracts frozen XLS-R features and compares 13 classifiers (from logistic regression, SVMs and tree ensembles to MLP, attentive-pooling and temporal-CNN heads) on a 1,200-clip ASVspoof and In-the-Wild subset, ranked by validation EER.
