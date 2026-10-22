# Why We Need Our Own Fake Speech Dataset

**Status:** Rough draft for team discussion
**Related milestone:** Milestone 2 (Dataset Preparation), due 29 Oct

---

## 1. The Question

Public datasets of real and fake speech already exist. Why should we generate our own fake clips?

**Short answer:** Existing data is enough to *train* a model, but not enough to *test it properly*. We need a small set of our own fakes to run the key experiments in this project.

---

## 2. What Existing Datasets Give Us

| Dataset | Real speech | Fake speech | Indian languages | Generator known per clip? |
| --- | --- | --- | --- | --- |
| ASVspoof 2019 LA | Yes | Yes | No (English) | Yes (attack IDs) |
| In-the-Wild | Yes | Yes | No (English, test only) | No |
| Kathbath | Yes | **No** | Yes | — |
| IndicSynth | — | Yes | Yes | Not stated on its paper page |

So for Hindi, Tamil and Telugu we can get:

- **Real speech** from Kathbath
- **Fake speech** from IndicSynth

That is enough to train a model. But it leaves four problems.

---

## 3. Why We Still Need Our Own Fakes

### 3.1 Unseen-generator test

One of our main objectives is to check whether the model catches fakes from a **voice tool it has never seen**.

- To do this, we train on fakes from some generators and test on a generator held out from training.
- This only works if we **know which model made every clip**.
- When we generate the clips ourselves, every clip is labelled with its generator, and we choose which one to hold out.

### 3.2 Avoiding a shortcut

If all real clips come from Kathbath and all fake clips come from IndicSynth, the model may learn:

> "Kathbath recording style = real, IndicSynth recording style = fake"

instead of:

> "Human voice = real, AI voice = fake"

It would score well on our test set but fail on new audio.

**Fix:** Generate fakes from **Kathbath's own sentences** (and, where allowed, matching speakers). Then real and fake clips differ only in whether the voice is human or AI.

### 3.3 Newer voice tools

Recent Indic models such as **Indic Parler-TTS** and **IndicF5** are not covered by the detection benchmarks we reviewed. Testing against current tools makes our results relevant to today's voice-cloning risk.

### 3.4 Novelty and course requirements

- Building this dataset is our **main novelty claim** (5 marks for novelty/complexity).
- It is the **Generative AI** part of our "Deep Learning + Generative AI" project.
- Without it, the project mostly re-runs existing English benchmarks.

---

## 4. What We Will Build

| Item | Plan |
| --- | --- |
| Languages | Hindi, Tamil, Telugu |
| Size | At least **2,000 fake clips per language** (~6,000 total) |
| Generators | Indic Parler-TTS (Apache 2.0), IndicF5 (MIT, gated), MMS-TTS (CC-BY-NC 4.0) |
| Text | Sentences taken from Kathbath |
| Labels | Language, generator, speaker / voice, source sentence |
| Splits | Speaker-disjoint train / validation / test |
| Format | 16 kHz mono WAV, silence trimmed |
| Release | Hugging Face Hub with a dataset card (non-commercial terms stated) |

---

## 5. Cost and Effort

| Resource | Cost |
| --- | --- |
| Models | Free (open source) |
| GPU | Free (Kaggle weekly quota, Colab free tier) |
| Storage | Free (under 1 GB) |
| GPU time | Roughly 5–10 hours total, split across team accounts |
| People time | About 1–2 days for one person (notebook, quality checks, labels) |

**Total money cost: zero.**

---

## 6. Risks and Open Questions

- [ ] **Voice-cloning consent:** IndicF5's terms say to clone only voices you have permission to use. Can we clone Kathbath speakers, or should IndicF5 use only its own reference voices? To confirm before generating.
- [ ] **Quality check:** Listen to a random sample from each generator and drop broken or silent clips.
- [ ] **Kaggle session limit (~12 h):** Save clips in batches and make the notebook resumable.
- [ ] **Balance:** Keep the number of clips per generator and per language roughly equal.

---

## 7. Summary

| Without our own dataset | With our own dataset |
| --- | --- |
| Can train a model | Can train a model |
| Cannot run the unseen-generator test | Unseen-generator test possible |
| Risk of learning dataset differences | Real and fake differ only in the voice |
| No newest Indic voice tools | Covers Indic Parler-TTS, IndicF5, MMS-TTS |
| Weak novelty | Clear novel contribution |

---

## 8. References

- ASVspoof 2019: <https://datashare.ed.ac.uk/handle/10283/3336>
- Kathbath / IndicSUPERB: <https://arxiv.org/abs/2208.11761>
- IndicSynth (ACL 2025): <https://aclanthology.org/2025.acl-long.1070>
- In-the-Wild: <https://arxiv.org/abs/2203.16263>
- Indic Parler-TTS: <https://huggingface.co/ai4bharat/indic-parler-tts>
- IndicF5: <https://huggingface.co/ai4bharat/IndicF5>
- MMS-TTS: <https://huggingface.co/facebook/mms-tts-tel>
