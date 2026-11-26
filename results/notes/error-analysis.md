# Error analysis

**Setup.** F-03 scored the same 118,295 test clips; F-01 and F-02 scored the same clips to check whether patterns recur. Each run uses its own validation EER threshold (`0.95927` for F-03, `0.99385` for F-01, `0.01821` for F-02), fitted on `valid` only. Errors are counted at that fixed threshold. Missed fakes (false negatives) and flagged real clips (false positives) are reported separately with their denominators. Tables and checks: `notebooks/Milestone-5/error_analysis.ipynb`.

## How the validation threshold transfers

| Test set | F-03 missed fakes | F-03 flagged real clips |
| --- | ---: | ---: |
| In-domain | 0 / 3,000 (0.00%) | 1 / 3,000 (0.03%) |
| Phone channel | 1 / 3,000 (0.03%) | 2 / 3,000 (0.07%) |
| Unseen generators | 4,842 / 6,479 (74.73%) | 0 / 2,160 (0.00%) |
| In-the-Wild | 6,230 / 11,244 (55.41%) | 43 / 18,822 (0.23%) |
| ASVspoof | 16,191 / 60,665 (26.69%) | 4 / 6,925 (0.06%) |

**Findings.**

Intervals below are 95% percentile intervals from 2,000 bootstrap draws of fake clips within each split.

- **The threshold does not transfer to unseen sources.** F-03 reached 44.0% accuracy on unseen generators despite a ROC AUC of 0.959. The threshold missed 74.7% of the fakes (95% CI 73.7–75.8%) without flagging any real clip in that split. The same pattern holds for F-01 and F-02, which missed 4,325 and 3,414 of the 6,479 fakes and flagged none of the 2,160 real clips. The scores rank clips reasonably well, but the threshold fitted on known generators sits above most scores from new ones.
- **In-the-Wild is also far from the operating point.** F-03 missed 55.4% of the fakes (54.6–56.4%) while flagging 0.23% of the real clips. F-01 reached 49.9% accuracy on unseen generators at 0.972 AUC, so accuracy at the validation threshold understates how well the scores separate the classes.
- **Known channels are easy.** In-domain and phone-channel error rates stay below 0.1% for both classes. ASVspoof fakes are the harder English side: 26.7% missed (26.3–27.0%), with only 0.06% of real clips flagged.

## Where errors concentrate

- **Generator and language.** All three runs missed a majority of IndicF5 fakes in Hindi, Tamil, and Telugu. F-03 missed 700, 574, and 648 of 720 in those languages, respectively. It missed 718 of 720 Telugu MMS-TTS fakes (99.7%); F-01 and F-02 also missed a majority in that group.
- **ASVspoof attacks.** A10 and A11 had the highest miss rates in all three runs. F-03 missed 4,472 of 4,587 A10 clips (97.5%) and 3,786 of 4,568 A11 clips (82.9%), compared with 43 of 4,601 A09 clips (0.9%). Attack IDs describe fakes; the four flagged real clips belong in the split-level count.
- **Clip duration.** F-03 missed 1,490 of 2,158 In-the-Wild fakes under two seconds (69.0%), versus 1,626 of 3,516 at four to under eight seconds (46.2%). It flagged 35 of 5,032 real clips in the short bin and 1 of 4,502 in the longer bin. F-01 and F-02 also missed a larger share of fakes in the short bin, but unseen-generator fakes did not show a consistent direction across runs. Duration bins can mix recording conditions and generators; they do not isolate the effect of length.
- **In-the-Wild speaker groups.** In anonymized group S02, F-03 missed 125 of 126 fakes and flagged none of 80 real clips. For S52, it missed 3 of 64 fakes and flagged none of 32 real clips. F-01 and F-02 showed the same contrast in misses. Speaker groups can also differ in recording conditions and how their fakes were made.

## Quiet-frame energy

For every processed test clip, the 10th percentile of 20 ms frame power was measured relative to the clip’s mean frame power, in dB. Lower values mean the quiet frames are far below the rest of that clip. This is a proxy computed after preprocessing (which normalizes each clip to −20 dBFS): it describes pauses and signal dynamics, not original loudness or background noise, and the generator or recording conditions can confound it. Original-recording loudness was not assessed: the full test sets are distributed only as processed, loudness-normalised audio.

| Test set | True label | < −30 dB | −30 to < −20 dB | −20 to < −10 dB | ≥ −10 dB |
| --- | --- | ---: | ---: | ---: | ---: |
| ASVspoof | fake (missed) | 10.8% (17,378) | 32.4% (29,094) | 34.4% (14,080) | 38.1% (113) |
| In-the-Wild | fake (missed) | 48.5% (2,723) | 60.6% (6,086) | 49.6% (2,397) | 76.3% (38) |
| Unseen generators | fake (missed) | 74.9% (4,146) | 71.6% (1,534) | 79.9% (773) | 76.9% (26) |
| ASVspoof | real (flagged) | 0.10% (3,020) | 0.00% (2,825) | 0.00% (1,059) | 4.8% (21) |
| In-the-Wild | real (flagged) | 0.04% (2,298) | 0.15% (11,683) | 0.38% (4,749) | 7.6% (92) |

**Findings.** Parentheses give the number of clips in the cell; the last band has fewer than 120 clips in every row and should not be read as a stable rate. The other two splits had at most one error per cell. The relationship differs by test set: ASVspoof misses were lowest in the quietest band and rose with higher low-energy frames, In-the-Wild misses peaked in the second band, and unseen-generator misses were roughly flat. No single direction holds for the proxy.

## Sensitivity to removing the first second

The first second of the processed audio was removed and rescored with the same F-03 checkpoint and scoring settings. Each row compares trimmed and intact EER on the same clips. The 95% confidence intervals for the change use 1,000 paired bootstrap draws, resampling within each class. A positive change means EER increased after trimming.

| Test set | Matched clips | Excluded real / fake | Intact EER | Trimmed EER | Change, pp (95% CI) |
| --- | ---: | ---: | ---: | ---: | ---: |
| In-domain | 6,000 | 0 / 0 | 0.000% | 0.033% | +0.033 (0.000 to 0.117) |
| Phone channel | 6,000 | 0 / 0 | 0.083% | 0.067% | −0.017 (−0.083 to 0.133) |
| Unseen generators | 8,639 | 0 / 0 | 9.800% | 10.132% | +0.332 (−0.263 to 0.864) |
| In-the-Wild | 29,867 | 126 / 73 | 9.389% | 16.025% | +6.636 (6.233 to 7.080) |
| ASVspoof | 66,809 | 104 / 677 | 8.384% | 15.497% | +7.113 (6.727 to 7.470) |

**Findings.** The trim increased EER most on In-the-Wild and ASVspoof; the other three intervals included zero. In total, 980 clips were excluded because fewer than 400 samples (25 ms at 16 kHz) remained after the cut, leaving 117,315 paired clips. The result shows sensitivity to this cut, but it does not identify an onset cue: the cut also shortens the clip and changes the placement of scoring windows.

**Caveats and implication.** The dominant failure mode is unseen sources scored as confidently real, not threshold noise: ranking is often recoverable (high AUC) while the `valid` threshold sits far from the test score distributions. The structural fix is generator diversity in training; threshold choice and calibration are assessed in the operating-point section. The quiet-frame and duration analyses are observational, not causal.
