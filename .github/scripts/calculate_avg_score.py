#!/usr/bin/env python3
"""Calculate average scores for all leaderboard entries and update HuggingFace.

Replaces the IRT-based TCI with a simple arithmetic mean of benchmark scores.

This script is called by the GitHub Action after syncing parquet files to
HuggingFace. It:
1. Fetches the current dataset from GSMA/leaderboard
2. Computes mean(benchmark_scores) for entries with all 5 benchmarks
3. Stores as [avg_score, 0, 0] array format (matching benchmark columns)
4. Drops legacy 'tci' column if present
5. Pushes the updated dataset back to HuggingFace
"""

from __future__ import annotations

import os
import sys

import pandas as pd
from datasets import Dataset, load_dataset

DATASET_REPO = "GSMA/leaderboard"
BENCHMARK_COLUMNS = ["teleqna", "telelogs", "telemath", "3gpp_tsg", "teletables"]


def extract_score(value) -> float | None:
    """Extract the primary score from [score, stderr, n_samples] format."""
    if value is None:
        return None
    try:
        if len(value) < 1:
            return None
        return float(value[0])
    except (TypeError, ValueError, IndexError):
        return None


def compute_avg_score(row: pd.Series) -> list[float] | None:
    """Compute average score across all benchmarks for a single row.

    Returns [avg, 0, 0] to match the benchmark column array format,
    or None if any benchmark is missing.
    """
    scores = [extract_score(row.get(col)) for col in BENCHMARK_COLUMNS]
    if any(s is None for s in scores):
        return None
    avg = sum(scores) / len(scores)
    return [avg, 0, 0]


def load_leaderboard(token: str) -> pd.DataFrame:
    """Load existing leaderboard from HuggingFace."""
    ds = load_dataset(DATASET_REPO, split="train", token=token)
    df = ds.to_pandas()
    if "__index_level_0__" in df.columns:
        df = df.drop(columns=["__index_level_0__"])
    return df


def main() -> None:
    """Calculate average scores and push to HuggingFace."""
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN environment variable required")
        sys.exit(1)

    print(f"Loading dataset from {DATASET_REPO}...")
    df = load_leaderboard(hf_token)
    print(f"Loaded {len(df)} entries")

    if df.empty:
        print("Dataset is empty, skipping average score calculation")
        sys.exit(0)

    # Drop legacy TCI column if it exists
    if "tci" in df.columns:
        print("Dropping legacy 'tci' column")
        df = df.drop(columns=["tci"])

    print("Computing average scores...")
    df["avg_score"] = df.apply(compute_avg_score, axis=1)

    scored = df["avg_score"].notna().sum()
    print(f"Computed average score for {scored}/{len(df)} models")

    # Clean up index columns before upload
    index_cols = [c for c in df.columns if c.startswith("__index_level")]
    if index_cols:
        df = df.drop(columns=index_cols)
    df = df.reset_index(drop=True)

    print("Uploading to HuggingFace...")
    dataset = Dataset.from_pandas(df)
    dataset.push_to_hub(DATASET_REPO, token=hf_token)
    print(f"Successfully updated {DATASET_REPO} with avg_score column")


if __name__ == "__main__":
    main()
