# Milestone 5: Qualitative results and error analysis (draft)

Scope: F-03 on 118,295 test clips, matched to the version 4 manifest. We use F-01 and F-02 on the same clips to check whether specific failure patterns recur. Each run uses its own validation EER threshold (`0.95927` for F-03, `0.99385` for F-01, and `0.01821` for F-02).

## Qualitative results

The table pairs one seeded success per split with a mistake selected from the most extreme wrong scores. The notebook records the clip IDs and includes both true labels where available. Scores at or above `0.95927` give a fake verdict.

| Test set | Correct example (true label, score → verdict) | Mistake (true label, score → verdict) |
| --- | --- | --- |
| In-domain | Fake, `0.99999` → fake | Real, `0.98945` → fake |
| Phone channel | Fake, `0.99999` → fake | Real, `0.99988` → fake |
| Unseen generators | Fake, `0.99991` → fake | Fake, `0.00002` → real |
| In-the-Wild | Real, `0.00005` → real | Real, `0.99998` → fake |
| ASVspoof | Fake, `0.99998` → fake | Fake, `0.00004` → real |

The unseen-generator mistake was Hindi IndicF5; the ASVspoof mistake was attack A10. The in-domain false positive and the phone false positive in the table are versions of the same real utterance. F-03 had no in-domain false negatives and no unseen-generator false positives.

### Listening review of six failures

One listener reviewed six mistakes in a private Kaggle page, in the order below. True labels and verdicts were visible. Saliency is the gradient of `p_fake` with respect to the input, averaged over scoring windows and smoothed over 20 ms (`src/explain.py`).

| Case | Test set | True → verdict | `p_fake` | Saliency peak | Listener’s observation |
| --- | --- | --- | ---: | ---: | --- |
| C01 | In-the-Wild | real → fake | 0.99998 | 0.79 s | Around 0:01, the deep “r” between “Oh” and “really?” sounded generated. |
| C02 | Phone channel | real → fake | 0.99988 | 0.42 s | Robotic, with similar intonation throughout; a pause after the first few words around 0:02. |
| C03 | ASVspoof | real → fake | 0.99817 | 1.57 s | The voice sounded rushed; no sound was heard after about 0:02. |
| C04 | Unseen generators (Hindi IndicF5) | fake → real | 0.00002 | 1.97 s | Sounded like a television news report and came across as a genuine story. |
| C05 | In-the-Wild | fake → real | 0.00002 | 1.79 s | Echo and reverberation suggested a real voice recorded in an empty room. |
| C06 | ASVspoof (A10) | fake → real | 0.00004 | 1.00 s | Unnatural pronunciation and flat delivery; the listener judged it generated after one listen. |

This is one listener’s account, not a controlled perceptual test. The page showed the labels, so the notes are hypotheses about audible cues, not verified causes of the model’s errors. C04 concerns the content sounding like a genuine news story; the detector reads audio only and does not parse the transcript. Saliency peaks ranged from 0.42 s to 1.97 s, and the share of gradient in the first second ranged from 15% to 65%, so the six plots show no common location. Six cases are too few to generalize from.

## Error analysis

### How the validation threshold transfers

| Test set | F-03 missed fakes | F-03 flagged real clips |
| --- | ---: | ---: |
| In-domain | 0 / 3,000 (0.00%) | 1 / 3,000 (0.03%) |
| Phone channel | 1 / 3,000 (0.03%) | 2 / 3,000 (0.07%) |
| Unseen generators | 4,842 / 6,479 (74.73%) | 0 / 2,160 (0.00%) |
| In-the-Wild | 6,230 / 11,244 (55.41%) | 43 / 18,822 (0.23%) |
| ASVspoof | 16,191 / 60,665 (26.69%) | 4 / 6,925 (0.06%) |

F-03 reached 44.0% accuracy on unseen generators despite a ROC AUC of 0.959. Its scores separated the classes better than the accuracy at the validation threshold suggests: the threshold missed 74.7% of the fakes (95% CI 73.7–75.8%) without flagging any real clips. The same pattern holds for F-01 and F-02, which missed 4,325 and 3,414 of the 6,479 fakes, respectively, and flagged none of the 2,160 real clips. In-the-Wild was also hard for F-03: it missed 55.4% of the fakes (54.6–56.4%) while flagging fewer than 1% of the real clips. Intervals resample fake clips within each split (2,000 draws).

### Where errors concentrate

- **Generator and language.** All three runs missed a majority of IndicF5 fakes in Hindi, Tamil, and Telugu. F-03 missed 700, 574, and 648 of 720 in those languages, respectively. It missed 718 of 720 Telugu MMS-TTS fakes (99.7%); F-01 and F-02 also missed a majority in that group.
- **ASVspoof attacks.** A10 and A11 had the highest miss rates in all three runs. F-03 missed 4,472 of 4,587 A10 clips (97.5%) and 3,786 of 4,568 A11 clips (82.9%), compared with 43 of 4,601 A09 clips (0.9%). Attack IDs describe fakes; the four flagged real clips belong in the split-level count.
- **Clip duration.** F-03 missed 1,490 of 2,158 In-the-Wild fakes under two seconds (69.0%), versus 1,626 of 3,516 at four to under eight seconds (46.2%). It flagged 35 of 5,032 real clips in the short bin and 1 of 4,502 in the longer bin. F-01 and F-02 also missed a larger share of fakes in the short bin, but unseen-generator fakes did not show a consistent direction across runs. Duration bins can mix recording conditions and generators; they do not isolate the effect of length.
- **In-the-Wild speaker groups.** In anonymized group S02, F-03 missed 125 of 126 fakes and flagged none of 80 real clips. For S52, it missed 3 of 64 fakes and flagged none of 32 real clips. F-01 and F-02 showed the same contrast in misses. Speaker groups can also differ in recording conditions and how their fakes were made.

### Quiet-frame energy

For every processed test clip, we measured the 10th percentile of 20 ms frame power relative to the clip’s mean frame power, in dB. Lower values mean the quiet frames are far below the rest of that clip. This is a proxy computed after preprocessing (which normalizes each clip to −20 dBFS): it describes pauses and signal dynamics, not original loudness or background noise, and the generator or recording conditions can confound it. Original-recording loudness was not assessed: the full test sets are distributed only as processed, loudness-normalised audio.

| Test set | True label | < −30 dB | −30 to < −20 dB | −20 to < −10 dB | ≥ −10 dB |
| --- | --- | ---: | ---: | ---: | ---: |
| ASVspoof | fake (missed) | 10.8% (17,378) | 32.4% (29,094) | 34.4% (14,080) | 38.1% (113) |
| In-the-Wild | fake (missed) | 48.5% (2,723) | 60.6% (6,086) | 49.6% (2,397) | 76.3% (38) |
| Unseen generators | fake (missed) | 74.9% (4,146) | 71.6% (1,534) | 79.9% (773) | 76.9% (26) |
| ASVspoof | real (flagged) | 0.10% (3,020) | 0.00% (2,825) | 0.00% (1,059) | 4.8% (21) |
| In-the-Wild | real (flagged) | 0.04% (2,298) | 0.15% (11,683) | 0.38% (4,749) | 7.6% (92) |

Parentheses give the number of clips in the cell; the last band has fewer than 120 clips in every row and should not be read as a stable rate. The other two splits had at most one error per cell. The relationship differs by test set: ASVspoof misses were lowest in the quietest band and rose with higher low-energy frames, In-the-Wild misses peaked in the second band, and unseen-generator misses were roughly flat. We therefore do not claim a single direction for the proxy.

### Sensitivity to removing the first second

We removed the first second of the processed audio and rescored it with the same F-03 checkpoint and scoring settings. Each row compares trimmed and intact EER on the same clips. The 95% confidence intervals for the change use 1,000 paired bootstrap draws, resampling within each class. A positive change means EER increased after trimming.

| Test set | Matched clips | Excluded real / fake | Intact EER | Trimmed EER | Change, pp (95% CI) |
| --- | ---: | ---: | ---: | ---: | ---: |
| In-domain | 6,000 | 0 / 0 | 0.000% | 0.033% | +0.033 (0.000 to 0.117) |
| Phone channel | 6,000 | 0 / 0 | 0.083% | 0.067% | −0.017 (−0.083 to 0.133) |
| Unseen generators | 8,639 | 0 / 0 | 9.800% | 10.132% | +0.332 (−0.263 to 0.864) |
| In-the-Wild | 29,867 | 126 / 73 | 9.389% | 16.025% | +6.636 (6.233 to 7.080) |
| ASVspoof | 66,809 | 104 / 677 | 8.384% | 15.497% | +7.113 (6.727 to 7.470) |

The trim increased EER most on In-the-Wild and ASVspoof. The other three intervals included zero. We excluded 980 clips because fewer than 400 samples (25 ms at 16 kHz) remained after the cut. That leaves 117,315 paired clips. The result shows sensitivity to this cut, but it does not identify an onset cue: the cut also shortens the clip and changes the placement of scoring windows.

### Remaining handoff

The private review page and saliency images stay in Kaggle. The compiled report needs these sections merged with the quantitative benchmark tables (Amit) and the configuration and seed analysis (Tarika). Threshold choice and calibration are assessed in the operating-point section.
