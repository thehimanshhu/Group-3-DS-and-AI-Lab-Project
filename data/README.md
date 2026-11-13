# data/

Nothing in this folder is committed to git except this file. `notebooks/Milestone-3/01_data_preparation.ipynb`
creates it:

```
data/
├── raw/<dataset>/<language>/...                 # the selected clips, unchanged from the source
├── processed/<dataset>/<language>/<clip_id>.wav # 16 kHz mono 16-bit WAV, trimmed, -20 dBFS
└── manifests/
    ├── manifest_subset.csv    # one row per clip: label, generator, speaker, split, sha1, ...
    ├── speaker_split.csv      # train / valid / test_indomain split of the Kathbath train speakers
    ├── leakage_report.csv     # results of the leakage checks
    └── dropped.csv            # clips removed by preprocessing (less than 1 s left)
```

The same folder (without the two large source files `raw/kathbath/testunk_audio.tar` and
`raw/indicsynth/indicsynth_metadata_raw.parquet`) is the private Kaggle dataset
`tarikasubhashhedaoo/dsai-group3-m3-subset`, which `02_pipeline_check.ipynb` reads. Ask in the group to be added to it.
Academic, non-commercial use only.

## Milestone 4

`notebooks/Milestone-4/01_data_preparation.ipynb` creates `data/m4/` in the same layout, with more clips:

```
data/m4/
├── raw/<dataset>/<language>/...                 # a few original files per source, for the before/after figures
├── processed/<dataset>/<language>/<clip_id>.wav # every clip, 16 kHz mono 16-bit WAV
└── manifests/
    ├── manifest_m4.csv        # one row per clip; also marks tune_subset, train_eval and m4_eval
    ├── speaker_split.csv
    ├── leakage_report.csv
    └── dropped.csv
```

It is uploaded as the private Kaggle dataset `dsai-group3-m4-data`, which `02_training.ipynb` and
`03_training_and_evaluation.ipynb` read.
