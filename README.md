# Open Telco Leaderboard Submissions

This repository stores evaluation submissions for the [Open Telco Leaderboard](https://huggingface.co/datasets/GSMA/leaderboard).

## Structure

- `model_cards/` - Model score summaries (parquet)
- `trajectories/` - Evaluation traces (JSON)

## Submission

Use the Open Telco CLI to submit your evaluation results:

1. Run evaluations: `uv run open-telco` → select `run-evals`
2. Submit results: select `submit` from the main menu
3. A PR will be created automatically for review
