"""Draw the architecture diagram of the Milestone-3 report (assets/infographics/Milestone-3/architecture.png).

Every shape and number comes from the saved run in notebooks/Milestone-3/02_pipeline_check.ipynb
(shapes printed in Section 3, parameter counts, threshold from Section 6) and from
notebooks/Milestone-3/01_data_preparation.ipynb (preprocessing settings).

Usage: python make_architecture.py <output.png>
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, MUTED, SURFACE, LINE = "#0b0b0b", "#52514e", "#fcfcfb", "#cfcdc6"
DATA, PRETRAINED, NEW, OUTPUT = "#8a8984", "#2a78d6", "#eb6834", "#1baf7a"
OUT = sys.argv[1] if len(sys.argv) > 1 else "architecture.png"

fig, ax = plt.subplots(figsize=(14.2, 8.0), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ax.set_xlim(0, 14.2)
ax.set_ylim(0, 8.0)
ax.axis("off")


def box(x, y, w, h, title, lines, accent, shape=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor="#ffffff", edgecolor=LINE, linewidth=1.2))
    ax.add_patch(FancyBboxPatch((x, y), 0.09, h, boxstyle="round,pad=0,rounding_size=0.04",
                                facecolor=accent, edgecolor="none"))
    ax.text(x + 0.22, y + h - 0.17, title, fontsize=10.5, color=INK, va="top", weight="bold")
    for i, line in enumerate(lines):
        ax.text(x + 0.22, y + h - 0.47 - 0.25 * i, line, fontsize=8.6, color=MUTED, va="top")
    if shape:
        ax.text(x + w - 0.12, y + 0.12, shape, fontsize=9, color=INK, ha="right", va="bottom",
                family="monospace", bbox=dict(boxstyle="round,pad=0.25", facecolor="#f1f0ec", edgecolor="none"))


def arrow(x1, y1, x2, y2, label=None):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                                 color=MUTED, linewidth=1.3, shrinkA=2, shrinkB=2))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.1, label, fontsize=8, color=MUTED, ha="center", va="bottom")


ax.text(0.3, 7.75, "Milestone 3: the detector from raw audio to verdict", fontsize=14, color=INK, weight="bold", va="top")
ax.text(0.3, 7.38, "Shapes are for a batch of B clips, as printed by the model in 02_pipeline_check.ipynb (Section 3).",
        fontsize=9.5, color=MUTED, va="top")

# Row 1: data preparation
W, H, Y1 = 3.2, 1.95, 4.75
box(0.3, Y1, W, H, "Raw audio", ["Any source: Kathbath, IndicSynth,", "our generated set, ASVspoof,",
                                  "In-the-Wild", "16 / 24 / 44.1 kHz, flac, wav, m4a"], DATA)
box(3.85, Y1, W, H, "Preprocessing (notebook 01)", ["Mono, resample to 16 kHz", "Trim edge silence (40 dB)",
                                                    "RMS −20 dBFS, peak ≤ 0.99", "16-bit WAV + SHA-1"], DATA)
box(7.4, Y1, W, H, "Load a clip (notebook 02)", ["Crop or zero-pad to 4 s", "(random 4 s in training,",
                                                 "middle 4 s otherwise)", "Zero mean, unit variance", "Attention mask for padding"],
    DATA, "[B, 64000]")
box(10.95, Y1, W, H, "XLS-R CNN encoder", ["7 convolution layers", "One frame every 20 ms",
                                              "Pretrained, frozen", "4.2 M parameters"], PRETRAINED, "[B, 199, 512]")
arrow(0.3 + W, Y1 + H / 2, 3.85, Y1 + H / 2)
arrow(3.85 + W, Y1 + H / 2, 7.4, Y1 + H / 2)
arrow(7.4 + W, Y1 + H / 2, 10.95, Y1 + H / 2)

# Row 2: model and output (right to left)
Y2 = 1.75
box(10.95, Y2, W, H, "XLS-R transformer", ["24 layers, 1,024 values", "per frame", "Pretrained, fine-tuned",
                                              "learning rate 1e-5"], PRETRAINED, "[B, 199, 1024]")
box(7.4, Y2, W, H, "Mean pooling", ["Average over the frames", "that hold audio", "(padding excluded)", "No parameters"], NEW, "[B, 1024]")
box(3.85, Y2, W, H, "Classification head", ["Dropout(0.1) → Linear(1024, 2)", "New, learning rate 1e-3",
                                            "Trained with weighted", "cross-entropy"], NEW, "[B, 2]")
box(0.3, Y2, W, H, "Output", ["Softmax → p_fake", "Fake if p_fake ≥ threshold", "(0.316, chosen on valid only)",
                              "Verdict + confidence"], OUTPUT, "[B]")
arrow(10.95 + W / 2, Y1, 10.95 + W / 2, Y2 + H)
arrow(10.95, Y2 + H / 2, 7.4 + W, Y2 + H / 2)
arrow(7.4, Y2 + H / 2, 3.85 + W, Y2 + H / 2)
arrow(3.85, Y2 + H / 2, 0.3 + W, Y2 + H / 2)

# Legend
legend = [(DATA, "Data preparation"), (PRETRAINED, "Pretrained XLS-R 300M (facebook/wav2vec2-xls-r-300m)"),
          (NEW, "Added by us (head trained from scratch)"), (OUTPUT, "Output")]
for (color, text), x in zip(legend, [0.3, 2.75, 8.0, 12.4]):
    ax.add_patch(FancyBboxPatch((x, 0.62), 0.22, 0.22, boxstyle="round,pad=0,rounding_size=0.04",
                                facecolor=color, edgecolor="none"))
    ax.text(x + 0.32, 0.73, text, fontsize=9, color=INK, va="center")
ax.text(0.3, 0.3, "Model: 315.4 M parameters, 311.2 M trained. Training and evaluation details: Sections 5 and 6 of the report.",
        fontsize=9, color=MUTED, va="center")

fig.savefig(OUT, dpi=150, facecolor=SURFACE, bbox_inches="tight")
print("Wrote", OUT)
