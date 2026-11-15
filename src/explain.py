import numpy as np
import torch
from torch.nn.functional import pad

from src.evaluate import _window_starts


def saliency(
    model,
    audio: np.ndarray,
    device,
    sample_rate: int = 16_000,
    crop_seconds: float = 4.0,
) -> np.ndarray:
    """Return the smoothed gradient magnitude of the mean fake probability."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("Expected a nonempty mono waveform.")

    window_size = round(crop_seconds * sample_rate)
    starts = _window_starts(audio.size, window_size, 2 * sample_rate)
    gradient = np.zeros_like(audio)
    model.eval()

    # Process one window at a time to limit memory use during backpropagation.
    with torch.enable_grad():
        for start in starts:
            stop = min(start + window_size, audio.size)
            samples = torch.from_numpy(audio[start:stop]).to(device).requires_grad_()
            normalized = (samples - samples.mean()) / (
                samples.std(unbiased=False) + 1e-7
            )

            wave = pad(normalized, (0, window_size - samples.numel()))
            mask = torch.arange(window_size, device=device) < samples.numel()
            logits = model(wave.unsqueeze(0), mask.long().unsqueeze(0))
            fake_probability = logits.softmax(dim=1)[0, 1]

            (window_gradient,) = torch.autograd.grad(fake_probability, samples)
            gradient[start:stop] += window_gradient.detach().cpu().numpy()

    magnitude = np.abs(gradient / len(starts))
    smooth_samples = max(1, min(audio.size, round(0.02 * sample_rate)))
    kernel = np.ones(smooth_samples, dtype=np.float32) / smooth_samples
    return np.convolve(magnitude, kernel, mode="same")