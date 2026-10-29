"""Draw the data-flow diagram of the Milestone-2 report (assets/infographics/Milestone-2/data_flow.png).

The diagram is a schematic. Its numbers come from the EDA notebooks:
Kathbath 266,487 train/valid clips (kathbath_eda.ipynb); Kathbath test split 1,929 + 1,642 + 1,492
recordings from 20 speakers per language and IndicSynth 658,146 clips (IndicSynth_notebook.ipynb,
Tables 3.1 and 1.1); generated set 6,480 clips (GeneratedSet_EDA_notebook.ipynb); ASVspoof 2019 LA
13 attacks (ASVspoof_notebook.ipynb); In-the-Wild 31,779 clips (InTheWild_notebook.ipynb).
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
REAL, FAKE, BOTH = "#2a78d6", "#eb6834", "#1baf7a"
OUT = sys.argv[1] if len(sys.argv) > 1 else "data_flow.png"

fig, ax = plt.subplots(figsize=(13.2, 8.2), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ax.set_xlim(0, 13.2)
ax.set_ylim(0.0, 7.95)
ax.axis("off")


def box(x, y, w, h, title, lines=(), accent=None, fill="#ffffff"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=fill, edgecolor="#cfcdc6", linewidth=1.2))
    if accent:
        ax.add_patch(FancyBboxPatch((x, y), 0.09, h, boxstyle="round,pad=0,rounding_size=0.04",
                                    facecolor=accent, edgecolor="none"))
    ax.text(x + 0.22, y + h - 0.17, title, fontsize=10.5, color=INK, va="top", weight="bold")
    for i, line in enumerate(lines):
        ax.text(x + 0.22, y + h - 0.47 - 0.26 * i, line, fontsize=8.6, color=MUTED, va="top")


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                 color="#8a887f", linewidth=1.3))


for x, title in [(0.2, "Sources"), (4.75, "Split"), (9.55, "Preparation")]:
    ax.text(x, 7.82, title, fontsize=11.5, color=INK, weight="bold", va="top")

H = 0.95
sources = [
    (6.3, "Kathbath · train speakers", ["Real · Hindi, Tamil, Telugu", "266,487 clips, 84–138 speakers per language"], REAL),
    (5.2, "IndicSynth", ["Fake · XTTS-v2, FreeVC24", "658,146 clips of Kathbath train speakers"], FAKE),
    (4.1, "Kathbath · test speakers", ["Real · 20 separate speakers per language", "5,063 recordings"], REAL),
    (3.0, "Our generated set", ["Fake · Indic Parler-TTS, IndicF5, MMS-TTS", "6,480 clips of the test speakers' sentences"], FAKE),
    (1.9, "ASVspoof 2019 LA", ["Real + fake · English", "13 attack systems (A07–A19)"], BOTH),
    (0.8, "In-the-Wild", ["Real + fake · English", "31,779 clips from online videos"], BOTH),
]
for y, title, lines, accent in sources:
    box(0.2, y, 4.0, H, title, lines, accent)

splits = [  # y, height, title, lines, rows of sources that feed it
    (5.2, 2.05, "Speaker-disjoint 70 / 15 / 15", ["Kathbath train speakers, split by speaker;", "fakes follow their speakers",
                                                  "→ train / validation / test"], [0, 1]),
    (3.0, 2.05, "Unseen-generator test", ["Real and fake speech of the same", "sentences, from generators not", "used in training"], [2, 3]),
    (1.9, H, "Official protocol", ["train / dev / eval"], [4]),
    (0.8, H, "Real-world test", ["test only"], [5]),
]
for y, h, title, lines, feeds in splits:
    box(4.75, y, 4.2, h, title, lines, fill="#f4f3ef")
    for i in feeds:
        sy = sources[i][0] + H / 2
        arrow(4.2, sy, 4.75, min(max(sy, y + 0.2), y + h - 0.2))
    arrow(8.95, y + h / 2, 9.55, min(max(y + h / 2, 4.25), 7.0))

box(9.55, 4.0, 3.45, 3.25, "Shared preprocessing", [
    "mono, 16 kHz, 16-bit PCM", "trim leading / trailing silence", "loudness → −20 dBFS",
    "drop < 1 s and flagged clips", "SHA-1 hash → remove duplicates", "fixed-length crops + augmentation"])
box(9.55, 2.4, 3.45, 1.15, "One manifest (CSV)", ["clip, dataset, language, label,", "generator, speaker, split, sha1"], fill="#f4f3ef")
box(9.55, 0.8, 3.45, 1.1, "XLS-R 300M detector", ["real vs AI-generated", "EER on every test view"], fill="#f4f3ef")
arrow(11.27, 4.0, 11.27, 3.55)
arrow(11.27, 2.4, 11.27, 1.9)

for x, color, text in [(0.25, REAL, "real"), (1.05, FAKE, "fake"), (1.85, BOTH, "real + fake")]:
    ax.add_patch(FancyBboxPatch((x, 0.22), 0.16, 0.16, boxstyle="round,pad=0,rounding_size=0.03",
                                facecolor=color, edgecolor="none"))
    ax.text(x + 0.24, 0.3, text, fontsize=8.6, color=MUTED, va="center")

fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"wrote {OUT}")
