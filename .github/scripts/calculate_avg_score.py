#!/usr/bin/env python3
"""Calculate average scores for all leaderboard entries and update HuggingFace.

Replaces the IRT-based TCI with a simple arithmetic mean of benchmark scores.
Called by the approval-sync workflow after syncing parquet files.
"""

from __future__ import annotations

import os
import sys

import pandas as pd
from datasets import Dataset, load_dataset

from registry import get_benchmark_columns

DATASET_REPO = "GSMA/leaderboard"
BENCHMARK_COLUMNS = get_benchmark_columns()


def extract_score(value: object) -> float | None:
    """Extract the primary score from [score, stderr, n_samples] format."""
    if value is None:
        return None
    try:
        return float(value[0])
    except (TypeError, ValueError, IndexError):
        return None


def compute_avg_score(row: pd.Series) -> list[float] | None:
    """Compute average score across available benchmarks for a single row.

    Returns [avg, 0, n_benchmarks] to match benchmark column format, where
    n_benchmarks indicates how many scores contributed to the average.
    Returns None only if zero valid scores exist.
    """
    scores = [extract_score(row.get(col)) for col in BENCHMARK_COLUMNS]
    valid = [(col, s) for col, s in zip(BENCHMARK_COLUMNS, scores) if s is not None]
    missing = [col for col, s in zip(BENCHMARK_COLUMNS, scores) if s is None]
    if missing:
        model = row.get("model", "unknown")
        print(f"Info: {model} missing benchmarks: {', '.join(missing)} "
              f"(averaging {len(valid)}/{len(BENCHMARK_COLUMNS)})")
    if not valid:
        return None
    avg = sum(s for _, s in valid) / len(valid)
    return [avg, 0, len(valid)]


def load_leaderboard(token: str) -> pd.DataFrame:
    """Load the leaderboard dataset from HuggingFace, exiting on failure."""
    try:
        ds = load_dataset(DATASET_REPO, split="train", token=token)
        return ds.to_pandas()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)


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

    # Archive legacy TCI column if it exists (preserve historical data)
    if "tci" in df.columns:
        print("Archiving legacy 'tci' column as 'tci_legacy'")
        df = df.rename(columns={"tci": "tci_legacy"})

    # Backfill missing benchmark columns so compute_avg_score sees them
    for col in BENCHMARK_COLUMNS:
        if col not in df.columns:
            print(f"Adding missing benchmark column: {col}")
            df[col] = None

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
