# AI Usage Disclosure

**Project:** Deepfake Voice Detection for Indian Languages · Group-3

This document states how an AI assistant was used in this project and how the team checked its work. It covers Milestones 4 and 5.

## Tool

Claude (Anthropic), used as a coding assistant in the terminal with access to the project repository.

## What it was used for

| Area | Milestone 4 | Milestone 5 |
| --- | --- | --- |
| Data | Data-preparation and test-set notebooks, the leakage checks, the investigation and fix of the start-of-clip click | The build of the full In-the-Wild and ASVspoof test sets on Kaggle, the merged manifest |
| Code | Training, model, dataset and scoring code (`src/train.py`, `src/model.py`, `src/dataset.py`, `src/score.py`), run configs, unit tests | Faster scoring (window batching, fp16, worker processes), scoring from several data folders, the onset option, the paired-bootstrap and Holm helpers (`src/compare.py`), Kaggle scoring scripts, unit tests |
| Analysis | Bootstrap comparisons of the sweeps, the final evaluation notebook, figures | Error analysis, operating points and calibration, the Holm correction and seed spread, figures |
| Writing | Drafts and editing of report sections, the worklog, the presentation | Drafts and editing of the report, the model card of the Hugging Face release, commit messages and pull-request descriptions |
| Release | | Packaging the weights and the manifest for the Hugging Face Hub |

## What it was not used for

- Choosing the research question, the scope, the datasets or the model; these were decided by the team in Milestones 1 to 3.
- Producing results: every number in the reports comes from code run on our data, and the outputs are saved in the notebooks and in `results/`.
- Generating the fake speech in the dataset; that was done with the TTS models named in the reports.

## How the team checked the work

- **Tests.** New code comes with unit tests, for example: padding a clip must not change its score; batched scoring must match clip-by-clip scoring within 1e-5; scoring a clip with its first second cut must match scoring the cut audio directly; the Holm correction must match hand-computed values.
- **Comparison with earlier results.** The sped-up scorer was compared clip by clip with the Milestone-4 predictions before it was used; every checkpoint downloaded for the release was checked against stored scores.
- **Checks on the data.** Leakage checks (speakers, duplicate clips, ASVspoof partitions) are part of the data notebooks and are all 0. The click was found by testing whether a model could tell real from fake from the first 20 ms alone.
- **Numbers in the text.** The figures quoted in the reports were checked against the notebook outputs and result files.
- **Review.** Design decisions, interpretations and the final text were reviewed and approved by the team; assistant output that was wrong was corrected or discarded.
