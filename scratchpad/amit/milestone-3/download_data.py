
"""
Download Kathbath Hindi, Tamil and Telugu training datasets.

Dataset: ai4bharat/Kathbath
Purpose: Milestone 3 - Deepfake Voice Detection

Downloads only the original train split.
Does not modify the original data or create model splits.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATASET_ID = "ai4bharat/Kathbath"

# Initially resolve "main" to a commit SHA.
# After the first successful run, replace "main"
# with the recorded SHA for reproducible future runs.
DATASET_REVISION = "5b9e92849222026d9141acba4e8434fe816396bf"

LANGUAGES = ["hindi", "tamil", "telugu"]
SPLITS = ["train"]

PROJECT_ROOT = Path(__file__).resolve().parent

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "kathbath"
LOG_DIR = PROJECT_ROOT / "logs"

LOG_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Logging
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_DIR / "kathbath_download.log",
            encoding="utf-8"
        ),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# Dataset revision verification
# --------------------------------------------------

def get_verified_revision():
    """Resolve the selected dataset revision to a commit SHA."""

    api = HfApi()

    info = api.dataset_info(
        repo_id=DATASET_ID,
        revision=DATASET_REVISION
    )

    revision = info.sha

    logger.info("Dataset: %s", DATASET_ID)
    logger.info("Verified commit SHA: %s", revision)

    return revision


# --------------------------------------------------
# Download function
# --------------------------------------------------

def download_language(language, revision):
    """Download and save the original training split."""

    language_dir = OUTPUT_DIR / language
    language_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:

        output_path = language_dir / split

        # Skip an existing download
        if output_path.exists():
            logger.info(
                "Already exists, skipping: %s | %s",
                language,
                split
            )
            continue

        logger.info("Downloading %s | %s", language, split)

        dataset = load_dataset(
            DATASET_ID,
            language,
            split=split,
            revision=revision
        )

        dataset.save_to_disk(str(output_path))

        logger.info(
            "Saved %s | %s | %d records",
            language,
            split,
            len(dataset)
        )

        del dataset


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    revision = get_verified_revision()

    for language in LANGUAGES:
        download_language(language, revision)

    metadata = {
        "dataset": DATASET_ID,
        "revision": revision,
        "languages": LANGUAGES,
        "splits": SPLITS,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat()
    }

    metadata_path = OUTPUT_DIR / "download_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    logger.info("Kathbath download completed.")
    logger.info("Metadata saved to %s", metadata_path)


if __name__ == "__main__":
    main()
