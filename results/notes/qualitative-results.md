# Qualitative results

**Setup.** F-03 (final config, seed 44) scored the five full test sets: 118,295 test clips plus `valid` (3,952), with `python -m src.score F-03 --full`, scorer commit `1756e7e`, on Kaggle 2 × T4. The decision threshold `0.95927` is the EER point of `valid`. Predictions are in `results/F-03/predictions_full.csv.gz`; the selection, tables, and checks are in `notebooks/Milestone-5/error_analysis.ipynb`. The six listening clips and their saliency plots stay in the private Kaggle output (`viknegi/dsai-group3-m5-audio-vikram`, `audio_review/`); no audio is committed.

## Sample outputs per test set

One seeded success per split (seed 42) and the most extreme mistake (highest-scoring flagged real clip, lowest-scoring missed fake; ties broken by clip ID):

| Test set | Correct example | Mistake |
| --- | --- | --- |
| In-domain | Fake, `indicsynth_ta_freevc24____m____src_690____tgt_21___4`, p 0.99999 → fake | Real, `kathbath_te_844424933311575-259-m`, p 0.98945 → fake |
| Phone channel | Fake, `phone_indicsynth_ta_freevc24____m____src_340____tgt_489___5`, p 0.99999 → fake | Real, `phone_kathbath_te_844424933311575-259-m`, p 0.99988 → fake |
| Unseen generators | Fake, `te_0671_indic-parler-tts`, p 0.99991 → fake | Fake, `hi_0042_indicf5`, p 0.00002 → real |
| In-the-Wild | Real, `itw_24551`, p 0.00005 → real | Real, `itw_16732`, p 0.99998 → fake |
| ASVspoof | Fake, `asvspoof_LA_E_3881653` (A09), p 0.99998 → fake | Fake, `asvspoof_LA_E_7508026` (A10), p 0.00004 → real |

F-03 has no in-domain false negatives and no unseen-generator false positives. The in-domain and phone false positives are the same utterance (`…259-m`) before and after the phone channel.

## Listening review of six failures

One listener reviewed six mistakes in a private Kaggle page, in the order below. True labels and verdicts were visible. Saliency is the gradient of `p_fake` with respect to the input, averaged over scoring windows and smoothed over 20 ms (`src/explain.py`).

| Case | Test set | True → verdict | `p_fake` | Saliency peak | Listener’s observation |
| --- | --- | --- | ---: | ---: | --- |
| C01 | In-the-Wild | real → fake | 0.99998 | 0.79 s | Around 0:01, the deep “r” between “Oh” and “really?” sounded generated. |
| C02 | Phone channel | real → fake | 0.99988 | 0.42 s | Robotic, with similar intonation throughout; a pause after the first few words around 0:02. |
| C03 | ASVspoof | real → fake | 0.99817 | 1.57 s | The voice sounded rushed; no sound was heard after about 0:02. |
| C04 | Unseen generators (Hindi IndicF5) | fake → real | 0.00002 | 1.97 s | Sounded like a television news report and came across as a genuine story. |
| C05 | In-the-Wild | fake → real | 0.00002 | 1.79 s | Echo and reverberation suggested a real voice recorded in an empty room. |
| C06 | ASVspoof (A10) | fake → real | 0.00004 | 1.00 s | Unnatural pronunciation and flat delivery; the listener judged it generated after one listen. |

**Caveats.** This is one listener’s account, not a controlled perceptual test. The page showed the labels, so the notes are hypotheses about audible cues, not verified causes of the model’s errors. C04 concerns the content sounding like a genuine news story; the detector reads audio only and does not parse the transcript. Saliency peaks ranged from 0.42 s to 1.97 s, and the share of gradient in the first second ranged from 15% to 65%, so the six plots show no common location. Six cases are too few to generalize from.

**Implication.** The salient failures are confident mistakes, not borderline scores: the model can call a real clip fake at p ≈ 1.0 and a fake real at p ≈ 0. The listening notes suggest plausible audible cues (synthetic-sounding segments, robotic prosody, room reverberation, flat TTS delivery), but the report should present them as observations that motivate a larger listening study, not as diagnoses.
