import torch
from torch import nn
from torch.nn import functional as F
from torchaudio.transforms import LFCC


class MaxFeatureMap(nn.Module):
    def forward(self, features):
        first, second = features.chunk(2, dim=1)
        return torch.maximum(first, second)


class LFCCLCNN(nn.Module):
    def __init__(self, sample_rate=16_000):
        super().__init__()
        self.hop_length = round(0.010 * sample_rate)
        self.lfcc = LFCC(
            sample_rate=sample_rate,
            n_lfcc=20,
            log_lf=True,
            speckwargs={
                "n_fft": 512,
                "win_length": round(0.020 * sample_rate),
                "hop_length": self.hop_length,
                "center": True,
            },
        )
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            MaxFeatureMap(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 64, kernel_size=3, padding=1),
            MaxFeatureMap(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 128, kernel_size=3, padding=1),
            MaxFeatureMap(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(64, 2))

    def forward(self, wave, mask):
        # Keep LFCC extraction in float32 during mixed-precision training.
        with torch.autocast(device_type=wave.device.type, enabled=False):
            features = self.lfcc(wave.float())

        frame_centers = torch.arange(features.size(-1), device=wave.device)
        frame_centers = frame_centers * self.hop_length
        valid_frames = frame_centers[None, :] < mask.sum(dim=1, keepdim=True)

        features = features.masked_fill(~valid_frames[:, None, :], 0.0)
        encoded = self.encoder(features.unsqueeze(1)).mean(dim=2)

        # Three time-pooling layers reduce the frame count by a factor of eight.
        pooled_mask = F.max_pool1d(
            valid_frames[:, None, :].float(), kernel_size=8, stride=8
        ).to(encoded.dtype)
        pooled = (encoded * pooled_mask).sum(dim=-1)
        pooled = pooled / pooled_mask.sum(dim=-1).clamp_min(1)

        return self.classifier(pooled)