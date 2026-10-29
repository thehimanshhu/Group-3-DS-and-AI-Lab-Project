"""Select the Kathbath sentences used as text input for our generated (fake) speech.

The generated set is a test-only set (unseen-generator test). Its text therefore comes
only from Kathbath's official test split ("test_unknown"), whose 20 speakers per language
never appear in Kathbath train. Sentences that also occur in train, valid or test_known are
dropped, so no generated sentence was ever read by a training speaker.

Each selected sentence is synthesised by all three generators (a parallel design), so
differences between generators are not caused by differences in text.

Usage:
    python select_sentences.py --kathbath-dir <dir containing kb_data_clean_m4a/>
or, to download the 209 MB transcript archive first:
    python select_sentences.py --download
"""

import argparse
import csv
import random
import tarfile
import urllib.request
from pathlib import Path

TRANSCRIPTS_URL = "https://objectstore.e2enetworks.net/indic-superb/kathbath/clean/transcripts_n2w.tar"
LANGS = {"hi": "hindi", "ta": "tamil", "te": "telugu"}
GENERATORS = ["indic-parler-tts", "indicf5", "mms-tts"]
SENTENCES_PER_LANG = 720  # x 3 generators = 2,160 clips per language (>= 2,000 after quality drops)
MIN_WORDS, MAX_WORDS = 6, 18
SEED = 42
HERE = Path(__file__).resolve().parent


def read_split(root, lang_name, kb_split):
    path = root / "kb_data_clean_m4a" / lang_name / kb_split / "transcription_n2w.txt"
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            kb_file, text = line.rstrip("\n").split("\t")
            _, speaker, gender = kb_file.removesuffix(".m4a").split("-")
            rows.append({"kb_file": kb_file, "kb_speaker": speaker, "kb_gender": gender, "text": text.strip()})
    return rows


def select(root, out_path):
    rng = random.Random(SEED)
    out = []
    for lang, lang_name in LANGS.items():
        other = {r["text"] for s in ["train", "valid", "test_known"] for r in read_split(root, lang_name, s)}
        unique = {}
        for r in read_split(root, lang_name, "test"):
            if r["text"] not in other and MIN_WORDS <= len(r["text"].split()) <= MAX_WORDS:
                unique.setdefault(r["text"], r)  # keep one recording per sentence text
        pool = sorted(unique.values(), key=lambda r: r["kb_file"])
        # Keep sampling order: raising SENTENCES_PER_LANG later only appends sentences, existing IDs stay the same.
        picked = rng.sample(pool, SENTENCES_PER_LANG)
        for i, r in enumerate(picked, 1):
            sent_id = f"{lang}_{i:04d}"
            for gen in GENERATORS:
                out.append({"clip_id": f"{sent_id}_{gen}", "sent_id": sent_id, "lang": lang, "generator": gen,
                            "split": "test", **r, "n_words": len(r["text"].split())})

    fields = ["clip_id", "sent_id", "lang", "generator", "split", "kb_file", "kb_speaker", "kb_gender", "n_words", "text"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in out:
            w.writerow({k: r[k] for k in fields})
    print(f"wrote {len(out)} clips ({len(out) // len(GENERATORS)} sentences) to {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kathbath-dir", type=Path, default=HERE / "kathbath")
    p.add_argument("--download", action="store_true", help="download and extract the transcript archive first")
    p.add_argument("--out", type=Path, default=HERE / "sentences.csv")
    args = p.parse_args()

    if args.download:
        args.kathbath_dir.mkdir(parents=True, exist_ok=True)
        tar_path = args.kathbath_dir / "transcripts_n2w.tar"
        if not tar_path.exists():
            urllib.request.urlretrieve(TRANSCRIPTS_URL, tar_path)
        with tarfile.open(tar_path) as tar:
            members = [m for m in tar.getmembers() if any(f"/{n}/" in m.name for n in LANGS.values())]
            tar.extractall(args.kathbath_dir, members=members)

    select(args.kathbath_dir, args.out)


if __name__ == "__main__":
    main()
