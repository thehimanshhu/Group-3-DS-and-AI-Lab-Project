# Milestone 2 Report: Kathbath Dataset Audit
## Project: Detecting AI-Generated Voices in Indian Languages

**Scope:** Hindi, Tamil, and Telugu audio from the AI4Bharat Kathbath dataset.  

## 1. Objective

This report documents the data source, dataset structure, metadata and audio-quality checks, suitability for the project, and the recommended data-splitting strategy. Kathbath is used as a source of human speech; it does not by itself provide the real-versus-AI-generated labels required to train a deepfake detector.

## 2. Dataset Access and Licensing

- **Source:** [AI4Bharat Kathbath on Hugging Face](https://huggingface.co/datasets/ai4bharat/Kathbath)
- **Access:** Hugging Face login and acceptance of the dataset's access conditions are required to access the files.
- **Licensing:** The associated IndicSUPERB licensing statement describes the dataset packaging as CC0. The original text is sourced from IndicCorp and is not owned by AI4Bharat.
- **Project use:** The Hindi, Tamil, and Telugu subsets are used for academic research on AI-generated voice detection.

## 3. Dataset size and distribution

The notebook loaded **266,487 records** across three languages and two available splits.

| Language | Train records | Validation records | Total |
|---|---:|---:|---:|
| Hindi | 91,752 | 3,151 | 94,903 |
| Tamil | 95,762 | 2,770 | 98,532 |
| Telugu | 70,673 | 2,379 | 73,052 |
| **Total** | **258,187** | **8,300** | **266,487** |

The dataset is largest for Tamil, followed by Hindi and Telugu. Only train and validation splits are available in the loaded configurations; no separate test split was present.

## 3.1 Gender-wise Distribution

Gender-wise distribution was analysed using the gender labels available in the Kathbath metadata. Record counts and percentages were calculated separately for each language and split.

| Language | Split | Female Records | Female % | Male Records | Male % |
|---|---|---:|---:|---:|---:|
| Hindi | Train | 68,390 | 74.54% | 23,362 | 25.46% |
| Hindi | Validation | 1,824 | 57.89% | 1,327 | 42.11% |
| Tamil | Train | 55,866 | 58.34% | 39,896 | 41.66% |
| Tamil | Validation | 1,460 | 52.71% | 1,310 | 47.29% |
| Telugu | Train | 46,737 | 66.13% | 23,936 | 33.87% |
| Telugu | Validation | 1,270 | 53.38% | 1,109 | 46.62% |

**Key observations:**
- Female audio records outnumber male audio records in all three languages and both splits.
- Hindi has the largest imbalance in the training split, followed by Telugu.
- The validation splits have more balanced gender distributions than the corresponding training splits.

These counts represent audio records, not unique speakers.
## 4. Metadata quality

The metadata audit checked missing values, blank transcripts, duplicate filenames, duplicate transcript strings, and missing durations for all six language/split combinations.

| Check | Result |
|---|---|
| Missing values in selected metadata columns | 0 across all splits |
| Blank transcripts | 0 |
| Missing filenames | 0 |
| Duplicate filenames within each split | 0 |
| Missing durations | 0 |
| Language-code mismatches | 0 |
| Repeated transcript strings in training | Hindi: 18; Tamil: 62; Telugu: 1 |

Repeated transcript text is mostly shared across different speakers. Out of 81 repeated transcripts, 79 are associated with multiple speakers. This suggests that repeated text is generally due to different speakers reading the same sentence, rather than necessarily indicating duplicate audio recordings.

## 5. Duration analysis

Mean durations from the metadata are:

| Language | Train mean | Train median | Validation mean | Validation median |
|---|---:|---:|---:|---:|
| Hindi | 5.29 s | 5.04 s | 5.53 s | 5.20 s |
| Tamil | 6.46 s | 6.11 s | 6.52 s | 6.20 s |
| Telugu | 7.22 s | 6.87 s | 7.58 s | 7.24 s |

A preliminary 2–15 second screen found **582 records outside that range**: 36 below 2 seconds and 546 above 15 seconds. The majority of records are within the screen. These are review flags, not automatic exclusion rules; longer recordings may contain valid speech.

## 6. Transcript characteristics

No blank transcripts or language-code mismatches were found.

| Language | Records | Mean transcript length | Median length | Range |
|---|---:|---:|---:|---:|
| Hindi | 94,903 | 57.49 characters | 57 | 16–143 |
| Tamil | 98,532 | 95.88 characters | 93 | 12–235 |
| Telugu | 73,052 | 84.40 characters | 82 | 25–173 |


## 7. Full-dataset audio technical audit

The notebook processed all **266,487 records**. The saved audit reported zero decode errors, a 16,000 Hz sample rate, and one audio channel across all three languages and both splits.

| Language | Split | Mean duration | Mean RMS | Files with clipping flag |
|---|---|---:|---:|---:|
| Hindi | Train | 5.29 s | 0.0771 | 4,483 |
| Hindi | Validation | 5.53 s | 0.0767 | 201 |
| Tamil | Train | 6.46 s | 0.0934 | 3,995 |
| Tamil | Validation | 6.52 s | 0.1063 | 181 |
| Telugu | Train | 7.22 s | 0.0663 | 2,985 |
| Telugu | Validation | 7.58 s | 0.0707 | 210 |

The clipping flag marks recordings with samples near full scale under the notebook's chosen threshold. A flagged recording is not automatically a bad recording: the number and location of such samples, and whether speech is audibly distorted, should be checked before filtering. RMS differences indicate differences in signal energy, not necessarily differences in quality.

The full results are saved to `kathbath_audio_quality_full.csv`. 

## 8. Acoustic feature EDA with librosa

A reproducible sample of **2,000 training recordings per language** (6,000 total) was used to calculate mean spectral centroid, zero-crossing rate (ZCR), and spectral roll-off.

| Language | Sample size | Mean spectral centroid | Mean ZCR | Mean roll-off |
|---|---:|---:|---:|---:|
| Hindi | 2,000 | 1,511 Hz | 0.098 | 2,935 Hz |
| Tamil | 2,000 | 1,396 Hz | 0.088 | 2,676 Hz |
| Telugu | 2,000 | 1,407 Hz | 0.090 | 2,744 Hz |


## 9. Dataset split and leakage assessment

The existing validation speakers all overlap with the training speakers:

| Language | Train speakers | Validation speakers | Validation speakers also in train |
|---|---:|---:|---:|
| Hindi | 101 | 20 | 20 |
| Tamil | 138 | 20 | 20 |
| Telugu | 84 | 20 | 20 |

**Implication:** All validation speakers also appear in the training split, limiting evaluation on unseen speakers. Kathbath contains only real speech, so synthetic speech from the IndicSynth dataset will be used as the fake class for training and evaluating the deepfake detection model.

## 10. Suitability for the project 

**Strengths**

- Covers real speech in Hindi, Tamil, and Telugu.
- No missing values in the selected metadata fields.
- All audio files decoded successfully at 16 kHz, mono.
- Includes transcripts and speaker IDs.

## 11. Reproducibility

The notebook uses Hugging Face datasets, pandas, NumPy, Matplotlib, SoundFile, and librosa. A fixed random seed (42) is used for sampling to ensure consistent results.

**Generated outputs:**

- kathbath_audio_quality_full.csv — audio-quality audit for the full dataset.
- kathbath_librosa_acoustic_sample.csv — acoustic features for 6,000 sampled recordings.
- Notebook charts and tables summarizing dataset distribution, metadata, duration, speaker overlap, and audio quality.

## 12. References

1. AI4Bharat, *Kathbath dataset card*, Hugging Face: https://huggingface.co/datasets/ai4bharat/Kathbath
2. AI4Bharat, *IndicSUPERB repository*: https://github.com/AI4Bharat/IndicSUPERB
3. Javed et al., *IndicSUPERB: A Speech Processing Universal Performance Benchmark for Indian languages* (2022): https://arxiv.org/abs/2208.11761
