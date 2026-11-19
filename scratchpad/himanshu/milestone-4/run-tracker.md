**Title:** M4 run tracker

**Body:**

One checkbox per training run. Tick it when the run is **done and its `results/<run_id>/` PR is open**. While it's running, edit the line to add `(running, <your name>)`. Don't start a run that isn't listed here; add a new line first.

Each line says which Kaggle account runs it; that's not always the person who wrote the config.

Run IDs: the letter is the owner (T = Tarika, V = Vikram, A = Amit, F = final, D = default on full data, C = cross-language). Every sweep run is the default config (`configs/default.yaml`) with **one** setting changed.

### Tarika: hyperparameter sweeps (tuning subset, 3 epochs, ~0.8 h each) · runs on Tarika's account, 2 queued sessions
- [ ] T-01 default config
- [ ] T-02 `train.lr_encoder` = 3e-6
- [ ] T-03 `train.lr_encoder` = 3e-5
- [ ] T-04 `train.lr_encoder` = 1e-4
- [ ] T-05 `train.lr_head` = 1e-4
- [ ] T-06 `train.lr_head` = 1e-2
- [ ] T-07 `data.crop_seconds` = 2
- [ ] T-08 `data.crop_seconds` = 6
- [ ] T-09 `model.freeze_transformer_layers` = 6
- [ ] T-10 `model.freeze_transformer_layers` = 12
- [ ] T-11 `model.freeze_transformer_layers` = 18
- [ ] T-12 `train.batch_size` = 16 (via accumulation)
- [ ] T-13 `train.batch_size` = 32 (via accumulation)
- [ ] T-14 `model.pooling` = attentive statistics

### Tarika: full-data runs (Tarika writes the configs)
- [ ] D-01 default config on the full `train`, up to 5 epochs (~5 h) · Tarika's account
- [ ] C-hi cross-language: train without Hindi, test on Hindi (~2.5 h) · Himanshu's account
- [ ] C-ta cross-language: without Tamil (~2.5 h) · Himanshu's account
- [ ] C-te cross-language: without Telugu (~2.5 h) · Himanshu's account

### Vikram: regularization, augmentation, stability (tuning subset, 3 epochs) · Vikram's account
- [ ] V-01 `model.dropout` = 0
- [ ] V-02 `model.dropout` = 0.3
- [ ] V-03 `model.dropout` = 0.5
- [ ] V-04 `train.weight_decay` = 0
- [ ] V-05 `train.weight_decay` = 0.1
- [ ] V-06 augmentation: noise + gain
- [ ] V-07 augmentation: speed + pitch + reverberation
- [ ] V-08 augmentation: RawBoost
- [ ] V-09 augmentation: codec / phone channel
- [ ] V-10 augmentation: all together
- [ ] V-11 stability: no warmup, no gradient clipping

### Amit: baselines (full data) · Amit's account
- [ ] A-01 LFCC-LCNN (~2 h)
- [ ] A-02 AASIST-L (~4 h)

### Himanshu: final model (after Mon 16 noon, full data, T4 ×2, up to 8 epochs)
- [ ] F-01 final config, seed 42 (+ scoring after each epoch) · Himanshu's account
- [ ] F-02 final config, seed 43 · Amit's account (starts with F-01)
- [ ] F-03 final config, seed 44 · Vikram's account (starts with F-01)

### Final notebook
- [ ] Notebook 3 run on Kaggle with every output attached (Tue 17 evening)
