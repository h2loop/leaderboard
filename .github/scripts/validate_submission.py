#!/usr/bin/env python3
"""Validate leaderboard submission files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["model", "teleqna", "telelogs", "telemath", "3gpp_tsg", "date"]

RECOGNIZED_PROVIDERS = [
    "Openai",
    "Anthropic",
    "Google",
    "Mistral",
    "Deepseek",
    "Meta",
    "Cohere",
    "Together",
    "Openrouter",
    "Groq",
    "Fireworks",
]


def validate_parquet(parquet_path: Path) -> tuple[dict[str, bool], list[str]]:
    """Validate parquet file structure."""
    checks = {
        "parquet_exists": False,
        "parquet_schema": False,
        "model_format": False,
        "provider_recognized": False,
    }
    errors = []

    if not parquet_path.exists():
        errors.append(f"Parquet file not found: {parquet_path}")
        return checks, errors

    checks["parquet_exists"] = True

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        errors.append(f"Failed to read parquet: {e}")
        return checks, errors

    # Check required columns
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        errors.append(f"Missing columns: {missing}")
    else:
        checks["parquet_schema"] = True

    # Validate model format: "model_name (Provider)"
    model_format_ok = True
    provider_recognized = True

    for model in df["model"].unique():
        if " (" not in model or not model.endswith(")"):
            errors.append(f"Invalid model format: {model}")
            model_format_ok = False
        else:
            provider = model.split("(")[-1].rstrip(")")
            if provider not in RECOGNIZED_PROVIDERS:
                errors.append(f"Unrecognized provider: {provider}")
                provider_recognized = False

    checks["model_format"] = model_format_ok
    checks["provider_recognized"] = provider_recognized

    return checks, errors


def validate_trajectory_json(json_path: Path) -> tuple[dict[str, bool], list[str]]:
    """Validate trajectory JSON file."""
    checks = {
        "json_valid": False,
        "inspect_eval": False,
        "no_errors": False,
    }
    errors = []

    try:
        with open(json_path) as f:
            data = json.load(f)
        checks["json_valid"] = True
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON in {json_path.name}: {e}")
        return checks, errors
    except Exception as e:
        errors.append(f"Failed to read {json_path.name}: {e}")
        return checks, errors

    # Check for Inspect eval signature
    if "eval" in data:
        checks["inspect_eval"] = True
    elif "model" in data and "results" in data:
        checks["inspect_eval"] = True
    else:
        errors.append(f"{json_path.name}: Missing eval data - may not be from Inspect")

    # Check for errors in trajectory
    status = data.get("status", "")
    error_field = data.get("error")

    if status == "error" or error_field:
        error_msg = error_field or "unknown error"
        errors.append(f"{json_path.name}: Trajectory has error: {error_msg}")
    else:
        checks["no_errors"] = True

    return checks, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate leaderboard submission")
    parser.add_argument("--files", required=True, help="Space-separated list of changed files")
    args = parser.parse_args()

    files = args.files.strip().split()

    result = {
        "passed": True,
        "errors": [],
        "checks": {
            "parquet_exists": True,
            "parquet_schema": True,
            "json_valid": True,
            "model_format": True,
            "provider_recognized": True,
            "inspect_eval": True,
            "no_errors": True,
        },
    }

    parquet_found = False
    json_found = False

    for file_path in files:
        if not file_path:
            continue

        path = Path(file_path)

        if path.suffix == ".parquet":
            parquet_found = True
            checks, errors = validate_parquet(path)
            for key, value in checks.items():
                if not value:
                    result["checks"][key] = False
            result["errors"].extend(errors)

        elif path.suffix == ".json":
            json_found = True
            checks, errors = validate_trajectory_json(path)
            for key, value in checks.items():
                if not value:
                    result["checks"][key] = False
            result["errors"].extend(errors)

    if not parquet_found:
        result["checks"]["parquet_exists"] = False
        result["errors"].append("No parquet file found in submission")

    # Trajectories are optional - don't fail if missing
    if not json_found:
        # Mark as N/A rather than failed
        result["checks"]["json_valid"] = True
        result["checks"]["inspect_eval"] = True
        result["checks"]["no_errors"] = True

    result["passed"] = len(result["errors"]) == 0

    with open("validation_result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"Validation {'passed' if result['passed'] else 'failed'}")
    if result["errors"]:
        print("Errors:")
        for error in result["errors"]:
            print(f"  - {error}")

    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
