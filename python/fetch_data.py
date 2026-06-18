#!/usr/bin/env python3
"""Fetch the IEEE-CIS Fraud Detection competition data from Kaggle.

Thin, reproducible fetch layer for an otherwise R-centric project. The raw
data (~1.3 GB) is gitignored and reconstructed from this script — it is not
stored in Git.

Prerequisites (one-time, manual):
  - Kaggle account + API token at ~/.kaggle/kaggle.json
  - Accept the competition rules once (otherwise the API returns 403):
    https://www.kaggle.com/competitions/ieee-fraud-detection/rules

Usage:
    python python/fetch_data.py

Idempotent: if the CSVs are already present in data/raw/, it does nothing.
"""
from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

COMPETITION = "ieee-fraud-detection"
EXPECTED_FILES = [
    "train_transaction.csv",
    "train_identity.csv",
    "test_transaction.csv",
    "test_identity.csv",
    "sample_submission.csv",
]

# repo_root/python/fetch_data.py -> repo_root
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"


def already_downloaded() -> bool:
    """True if every expected CSV is already extracted."""
    return all((RAW_DIR / name).exists() for name in EXPECTED_FILES)


def download() -> Path:
    """Download the competition archive via the canonical Kaggle CLI command."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading '{COMPETITION}' into {RAW_DIR} ...")
    subprocess.run(
        ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(RAW_DIR)],
        check=True,
    )
    zip_path = RAW_DIR / f"{COMPETITION}.zip"
    if not zip_path.exists():
        sys.exit(f"Expected archive not found: {zip_path}")
    return zip_path


def unzip(zip_path: Path) -> None:
    """Extract the archive into data/raw/ and drop the archive."""
    print(f"Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW_DIR)
    zip_path.unlink()
    print("Removed archive after extraction.")


def main() -> None:
    if already_downloaded():
        print(f"Data already present in {RAW_DIR}; nothing to do.")
        return

    zip_path = download()
    unzip(zip_path)

    missing = [name for name in EXPECTED_FILES if not (RAW_DIR / name).exists()]
    if missing:
        sys.exit(f"Fetch incomplete, missing: {missing}")

    print("Done. Files in data/raw/:")
    for name in EXPECTED_FILES:
        size_mb = (RAW_DIR / name).stat().st_size / 1e6
        print(f"  {name:24s} {size_mb:8.1f} MB")


if __name__ == "__main__":
    main()
