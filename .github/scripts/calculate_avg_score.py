#!/usr/bin/env python3
"""Calculate derived columns for all leaderboard entries and update HuggingFace.

Computes `average`, `benchmarks_completed`, and `rank` from benchmark scores.
Called by the approval-sync workflow after syncing parquet files.
"""

from __future__ import annotations

import ast
import os
import sys

import pandas as pd
from datasets import Dataset, load_dataset

from registry import get_benchmark_columns

DATASET_REPO = "GSMA/leaderboard"
BENCHMARK_COLUMNS = get_benchmark_columns()


def extract_score(value: object) -> float | None:
    """Extract the primary score from a string-encoded or list score value.

    Handles both new format (string "[score, stderr]") and legacy
    format (list/array [score, stderr, ...]).
    """
    if value is None:
        return None
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list) and len(parsed) >= 1:
                return float(parsed[0])
        except (ValueError, SyntaxError):
            return None
    try:
        return float(value[0])
    except (TypeError, ValueError, IndexError):
        return None


def compute_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute average, benchmarks_completed, and rank for all rows.

    - average: arithmetic mean of available benchmark scores (0-1 scale)
    - benchmarks_completed: count of non-null benchmark columns
    - rank: 1-based rank by average (descending), ties get same rank
    """
    averages = []
    completed_counts = []

    for _, row in df.iterrows():
        scores = []
        n_completed = 0
        for col in BENCHMARK_COLUMNS:
            score = extract_score(row.get(col))
            if score is not None:
                scores.append(score)
                n_completed += 1

        model = row.get("model", "unknown")
        missing = [col for col in BENCHMARK_COLUMNS if extract_score(row.get(col)) is None]
        if missing:
            print(f"Info: {model} missing benchmarks: {', '.join(missing)} "
                  f"(averaging {n_completed}/{len(BENCHMARK_COLUMNS)})")

        if scores:
            averages.append(sum(scores) / len(scores))
        else:
            averages.append(None)
        completed_counts.append(n_completed)

    df["average"] = averages
    df["benchmarks_completed"] = completed_counts

    # Rank by average descending (None values get no rank)
    df["rank"] = (
        df["average"]
        .rank(ascending=False, method="min", na_option="bottom")
        .where(df["average"].notna())
        .astype("Int64")
    )

    return df


def load_leaderboard(token: str) -> pd.DataFrame:
    """Load the leaderboard dataset from HuggingFace, exiting on failure."""
    try:
        ds = load_dataset(DATASET_REPO, split="train", token=token)
        return ds.to_pandas()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)


def main() -> None:
    """Calculate derived columns and push to HuggingFace."""
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN environment variable required")
        sys.exit(1)

    print(f"Loading dataset from {DATASET_REPO}...")
    df = load_leaderboard(hf_token)
    print(f"Loaded {len(df)} entries")

    if df.empty:
        print("Dataset is empty, skipping derived column calculation")
        sys.exit(0)

    # Archive legacy columns if they exist (preserve historical data)
    if "tci" in df.columns:
        print("Archiving legacy 'tci' column as 'tci_legacy'")
        df = df.rename(columns={"tci": "tci_legacy"})
    if "avg_score" in df.columns:
        print("Archiving legacy 'avg_score' column as 'avg_score_legacy'")
        df = df.rename(columns={"avg_score": "avg_score_legacy"})

    # Backfill missing benchmark columns so compute sees them
    for col in BENCHMARK_COLUMNS:
        if col not in df.columns:
            print(f"Adding missing benchmark column: {col}")
            df[col] = None

    print("Computing derived columns (average, benchmarks_completed, rank)...")
    df = compute_derived_columns(df)

    scored = df["average"].notna().sum()
    print(f"Computed scores for {scored}/{len(df)} models")

    # Clean up index columns before upload
    index_cols = [c for c in df.columns if c.startswith("__index_level")]
    if index_cols:
        df = df.drop(columns=index_cols)
    df = df.reset_index(drop=True)

    print("Uploading to HuggingFace...")
    dataset = Dataset.from_pandas(df)
    dataset.push_to_hub(DATASET_REPO, token=hf_token)
    print(f"Successfully updated {DATASET_REPO} with derived columns")


if __name__ == "__main__":
    main()
