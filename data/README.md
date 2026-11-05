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
