import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

from src.metrics import eer


def evaluate_splits(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize each split using a shared validation threshold."""
    valid = predictions.loc[predictions["split"] == "valid"]
    _, threshold = eer(valid["label"].eq("fake"), valid["p_fake"])

    results = []
    for split, group in predictions.groupby("split", sort=False):
        is_fake = group["label"].eq("fake")
        scores = group["p_fake"]
        predicted_fake = scores.ge(threshold)
        split_eer, _ = eer(is_fake, scores)
        tn, fp, fn, tp = confusion_matrix(
            is_fake, predicted_fake, labels=[False, True]
        ).ravel()

        results.append({
            "split": split,
            "clips": len(group),
            "eer_pct": 100 * split_eer,
            "roc_auc": roc_auc_score(is_fake, scores),
            "accuracy": accuracy_score(is_fake, predicted_fake),
            "real_as_real": tn,
            "real_as_fake": fp,
            "fake_as_real": fn,
            "fake_as_fake": tp,
            "threshold": threshold,
        })

    return pd.DataFrame(results)


def compare_source_thresholds(
    predictions: pd.DataFrame, dataset: str, test_split: str
) -> pd.DataFrame:
    """Fit thresholds on validation and compare them on a source's test clips."""
    valid = predictions.loc[predictions["split"].eq("valid")]
    source_valid = valid.loc[valid["dataset"].eq(dataset)]
    test = predictions.loc[
        predictions["split"].eq(test_split) & predictions["dataset"].eq(dataset)
    ]

    if test_split == "valid" or test.empty:
        raise ValueError("Provide a separate test split with clips for this dataset.")

    _, shared_threshold = eer(valid["label"].eq("fake"), valid["p_fake"])
    _, source_threshold = eer(
        source_valid["label"].eq("fake"), source_valid["p_fake"]
    )
    is_fake = test["label"].eq("fake")

    results = []
    for name, threshold in [("shared", shared_threshold), ("source", source_threshold)]:
        predicted_fake = test["p_fake"].ge(threshold)
        tn, fp, fn, tp = confusion_matrix(
            is_fake, predicted_fake, labels=[False, True]
        ).ravel()
        results.append({
            "threshold_type": name,
            "threshold": threshold,
            "accuracy": (tn + tp) / len(test),
            "real_as_fake": fp,
            "fake_as_real": fn,
        })

    return pd.DataFrame(results)


def evaluate_generators(
    predictions: pd.DataFrame, test_split: str = "test_unseen_gen"
) -> pd.DataFrame:
    """Measure fake detection by generator and language on a test split."""
    valid = predictions.loc[predictions["split"].eq("valid")]
    _, threshold = eer(valid["label"].eq("fake"), valid["p_fake"])

    fake_clips = predictions.loc[
        predictions["split"].eq(test_split) & predictions["label"].eq("fake")
    ].copy()
    fake_clips["detected"] = fake_clips["p_fake"].ge(threshold)

    summary = (
        fake_clips.groupby(["generator", "language"], as_index=False)
        .agg(fake_clips=("detected", "size"), detected=("detected", "sum"))
    )
    summary["missed"] = summary["fake_clips"] - summary["detected"]
    summary["detection_pct"] = 100 * summary["detected"] / summary["fake_clips"]
    return summary


def evaluate_languages(
    predictions: pd.DataFrame, test_split: str | None = None
) -> pd.DataFrame:
    """Report EER and ROC AUC for each language within each test split."""
    splits = predictions["split"]
    if test_split is None:
        selected = splits.str.startswith("test_", na=False)
    else:
        selected = splits.eq(test_split)

    results = []
    for (split, language), group in predictions.loc[selected].groupby(
        ["split", "language"]
    ):
        is_fake = group["label"].eq("fake")
        scores = group["p_fake"]
        fake_clips = int(is_fake.sum())
        real_clips = len(group) - fake_clips

        eer_pct = roc_auc = float("nan")
        if real_clips and fake_clips:
            error, _ = eer(is_fake, scores)
            eer_pct = 100 * error
            roc_auc = roc_auc_score(is_fake, scores)

        results.append({
            "split": split,
            "language": language,
            "real_clips": real_clips,
            "fake_clips": fake_clips,
            "eer_pct": eer_pct,
            "roc_auc": roc_auc,
        })

    return pd.DataFrame(results)


def plot_split_roc(predictions: pd.DataFrame):
    """Plot one ROC curve for each split with both real and fake clips."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay

    _, ax = plt.subplots(figsize=(7, 6))

    for split, group in predictions.groupby("split", sort=False):
        is_fake = group["label"].eq("fake")
        if is_fake.nunique() < 2:
            continue

        RocCurveDisplay.from_predictions(
            is_fake, group["p_fake"], name=split, ax=ax
        )

    ax.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    ax.set_title("Deepfake detection ROC by split")
    ax.legend(loc="lower right")
    return ax


def _window_starts(num_samples: int, window_size: int, hop_size: int) -> list[int]:
    """Return the start sample of each scoring window."""
    last_start = max(0, num_samples - window_size)
    starts = list(range(0, last_start + 1, hop_size))

    # Include the clip's tail when it falls between regular starts.
    if starts[-1] != last_start:
        starts.append(last_start)

    return starts

def _prepare_window(
    audio: np.ndarray, start: int, window_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize a mono audio window, then pad it and build its mask."""
    samples = np.asarray(audio[start : start + window_size], dtype=np.float32)
    if samples.size == 0:
        raise ValueError("Cannot prepare an empty audio window.")

    normalized = (samples - samples.mean()) / (samples.std() + 1e-7)

    wave = np.zeros(window_size, dtype=np.float32)
    mask = np.zeros(window_size, dtype=np.int64)
    wave[:samples.size] = normalized
    mask[:samples.size] = 1
    return wave, mask


def _score_clip(model, audio: np.ndarray, sample_rate: int, device, crop_seconds=4.0):
    """Return (mean sliding-window fake probability, middle-window fake probability)."""
    import torch

    window_size = round(crop_seconds * sample_rate)
    hop_size = 2 * sample_rate
    starts = _window_starts(len(audio), window_size, hop_size)
    num_sliding_windows = len(starts)

    middle_start = max(0, (len(audio) - window_size) // 2)
    if middle_start not in starts:
        starts.append(middle_start)
    middle_index = starts.index(middle_start)

    waves, masks = zip(*(
        _prepare_window(audio, start, window_size) for start in starts
    ))
    wave_batch = torch.from_numpy(np.stack(waves)).to(device)
    mask_batch = torch.from_numpy(np.stack(masks)).to(device)

    with torch.inference_mode():
        logits = model(wave_batch, mask_batch)
        fake_probs = logits.softmax(dim=1)[:, 1]

    return (
        fake_probs[:num_sliding_windows].mean().item(),
        fake_probs[middle_index].item(),
    )


def _clip_windows(audio: np.ndarray, window_size: int, hop_size: int):
    """The sliding windows of one clip plus its middle window: (waves, masks, sliding count, middle index)."""
    starts = _window_starts(len(audio), window_size, hop_size)
    num_sliding_windows = len(starts)
    middle_start = max(0, (len(audio) - window_size) // 2)
    if middle_start not in starts:
        starts.append(middle_start)
    waves, masks = zip(*(_prepare_window(audio, start, window_size) for start in starts))
    return waves, masks, num_sliding_windows, starts.index(middle_start)


class _ClipWindows:
    """Dataset of clips -> their windows, for a DataLoader whose worker processes read and cut the clips."""

    def __init__(self, paths, sample_rate, window_size, hop_size):
        self.paths, self.sample_rate, self.window_size, self.hop_size = paths, sample_rate, window_size, hop_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        import soundfile as sf

        audio, actual_rate = sf.read(self.paths[index], dtype="float32")
        if actual_rate != self.sample_rate or audio.ndim != 1:
            raise ValueError(f"Expected mono audio at {self.sample_rate} Hz: {self.paths[index]}")
        waves, masks, num_sliding, middle = _clip_windows(audio, self.window_size, self.hop_size)
        return np.stack(waves), np.stack(masks), num_sliding, middle


def score(model, rows: pd.DataFrame, data_dir, data_cfg: dict, device,
          batch_size: int = 64, fp16: bool = True, workers: int = 4, log_every: float = 120) -> pd.DataFrame:
    """Score each processed WAV listed in the manifest.

    p_fake is the mean fake probability over 4 s windows every 2 s, p_fake_middle the middle
    window's. Windows from many clips go through the model together (batch_size windows), in
    fp16 on a GPU, on every visible GPU, while worker processes read the next clips (processes,
    not threads, so reading never holds up the thread that feeds the GPU). The scores match
    scoring clip by clip in fp32 (_score_clip) up to fp16 rounding. Prints progress every
    log_every seconds, with the time spent waiting for audio and in the model.
    """
    import time
    from pathlib import Path

    import torch

    sample_rate = data_cfg.get("sample_rate", 16_000)
    window_size = round(data_cfg.get("crop_seconds", 4.0) * sample_rate)
    device = torch.device(device)
    model.eval()
    net = model
    if device.type == "cuda" and fp16:
        class Half(torch.nn.Module):
            """fp16 autocast inside forward, so it also holds in DataParallel's worker threads."""

            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, wave, mask):
                with torch.autocast("cuda", dtype=torch.float16):
                    return self.inner(wave, mask).float()

        net = Half(net)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)

    paths = [str(Path(data_dir) / p) for p in rows["path"]]
    clips = torch.utils.data.DataLoader(
        _ClipWindows(paths, sample_rate, window_size, 2 * sample_rate), batch_size=None,
        num_workers=workers, prefetch_factor=32 if workers else None, persistent_workers=False)

    probs = []  # fake probability of every window, in order
    layout = []  # per clip: (first window, sliding windows, middle index)
    waves, masks = [], []
    pending = 0  # windows waiting in waves / masks
    total = 0
    waited = in_model = 0.0
    start = last_log = time.time()

    def run_batch():
        nonlocal waves, masks, pending, in_model
        t = time.time()
        wave, mask = np.concatenate(waves), np.concatenate(masks)
        waves, masks = [wave[batch_size:]], [mask[batch_size:]]
        pending = len(wave) - batch_size if len(wave) > batch_size else 0
        with torch.inference_mode():
            logits = net(torch.from_numpy(wave[:batch_size]).to(device), torch.from_numpy(mask[:batch_size]).to(device))
            probs.append(logits.float().softmax(dim=1)[:, 1].cpu().numpy())
        in_model += time.time() - t

    t = time.time()
    for clip_waves, clip_masks, num_sliding, middle in clips:
        waited += time.time() - t
        layout.append((total, int(num_sliding), int(middle)))
        total += len(clip_waves)
        waves.append(np.asarray(clip_waves))
        masks.append(np.asarray(clip_masks))
        pending += len(clip_waves)
        while pending >= batch_size:
            run_batch()
        if log_every and time.time() - last_log > log_every:
            last_log = time.time()
            elapsed = last_log - start
            print(f"  {len(layout):,}/{len(paths):,} clips, {len(layout) / elapsed:.1f} clips/s, "
                  f"waiting for audio {waited:.0f} s, model {in_model:.0f} s", flush=True)
        t = time.time()
    while pending:
        run_batch()

    window_probs = np.concatenate(probs) if probs else np.zeros(0)
    window_scores = [float(window_probs[first : first + n].mean()) for first, n, _ in layout]
    middle_scores = [float(window_probs[first + middle]) for first, _, middle in layout]
    return rows.assign(p_fake=window_scores, p_fake_middle=middle_scores)
