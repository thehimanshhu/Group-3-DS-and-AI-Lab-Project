from torch import nn

from src.baselines._aasist_upstream import Model


class AASISTL(nn.Module):
    """AASIST-L with the project's [real, fake] output order."""

    def __init__(self):
        super().__init__()
        self.network = Model({
            "first_conv": 128,
            "filts": [70, [1, 32], [32, 32], [32, 24], [24, 24]],
            "gat_dims": [24, 32],
            "pool_ratios": [0.4, 0.5, 0.7, 0.5],
            "temperatures": [2.0, 2.0, 100.0, 100.0],
        })

    def forward(self, wave, mask):
        wave = wave.masked_fill(~mask.bool(), 0.0)
        _, logits = self.network(wave)
        return logits.flip(dims=(1,))