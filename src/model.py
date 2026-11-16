"""XLS-R 300M with a small classification head (Milestone 3 design, configurable for Milestone 4).

    model = build_model(cfg["model"])
    logits = model(wave, mask)        # wave [B, samples] float, mask [B, samples] 1 = audio, 0 = padding
    p_fake = logits.softmax(dim=1)[:, 1]
"""
import torch
from torch import nn
from transformers import Wav2Vec2Config, Wav2Vec2Model
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()


class AttentiveStatsPooling(nn.Module):
    """Attention-weighted mean and standard deviation over the frames that hold audio."""

    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(dim, 128), nn.Tanh(), nn.Linear(128, 1))

    def forward(self, frames, frame_mask):
        scores = self.attention(frames).squeeze(-1).masked_fill(frame_mask == 0, -1e4)
        weights = scores.softmax(dim=1).unsqueeze(-1)
        mean = (weights * frames).sum(1)
        std = ((weights * (frames - mean.unsqueeze(1)) ** 2).sum(1) + 1e-6).sqrt()
        return torch.cat([mean, std], dim=1)


class Detector(nn.Module):
    """XLS-R encoder → pooling over frames with audio → Dropout → Linear(·, 2)."""

    def __init__(self, name="facebook/wav2vec2-xls-r-300m", freeze_feature_encoder=True,
                 freeze_transformer_layers=0, pooling="mean", dropout=0.1):
        super().__init__()
        if name == "tiny":  # a small random encoder with XLS-R's layout, for tests on a CPU; no download
            self.encoder = Wav2Vec2Model(Wav2Vec2Config(
                hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64, conv_dim=(16,) * 7,
                feat_extract_norm="layer", do_stable_layer_norm=True, conv_bias=True))  # as in XLS-R
        else:
            self.encoder = Wav2Vec2Model.from_pretrained(name)
        if freeze_feature_encoder:
            self.encoder.freeze_feature_encoder()
        for layer in self.encoder.encoder.layers[:freeze_transformer_layers]:
            for p in layer.parameters():
                p.requires_grad = False
        dim = self.encoder.config.hidden_size
        if pooling == "mean":
            self.pool, out_dim = None, dim
        elif pooling == "attentive":
            self.pool, out_dim = AttentiveStatsPooling(dim), 2 * dim
        else:
            raise ValueError(f"Unknown pooling {pooling!r}: use 'mean' or 'attentive'")
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(out_dim, 2))

    def forward(self, wave, mask):
        frames = self.encoder(wave, attention_mask=mask).last_hidden_state
        # Which frames hold audio and which are padding (padding must not change the score)
        frame_mask = self.encoder._get_feature_vector_attention_mask(frames.shape[1], mask).to(frames.dtype)
        if self.pool is None:
            pooled = (frames * frame_mask[..., None]).sum(1) / frame_mask.sum(1, keepdim=True)
        else:
            pooled = self.pool(frames, frame_mask)
        return self.head(pooled)

    def parameter_groups(self):
        """(XLS-R parameters, new parameters), for the two learning rates."""
        encoder = [p for p in self.encoder.parameters() if p.requires_grad]
        new = [p for n, p in self.named_parameters() if not n.startswith("encoder.") and p.requires_grad]
        return encoder, new


def build_model(model_cfg):
    return Detector(name=model_cfg.get("name", "facebook/wav2vec2-xls-r-300m"),
                    freeze_feature_encoder=model_cfg.get("freeze_feature_encoder", True),
                    freeze_transformer_layers=model_cfg.get("freeze_transformer_layers", 0),
                    pooling=model_cfg.get("pooling", "mean"),
                    dropout=model_cfg.get("dropout", 0.1))


def load_model(checkpoint, cfg, device):
    """A trained model from a best.pt written by src.train, ready for scoring."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model(cfg["model"])
    model.load_state_dict(state["model"] if "model" in state else state)
    return model.to(device).eval()
