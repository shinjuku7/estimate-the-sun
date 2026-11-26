# Estimate the sun

Submission 6 (public MAE 129.997789 MW) averages the two pipelines by ID and sets fully dark intervals to zero. Submission 7 (public MAE 133.14 MW) is the second pipeline alone.

## Reproduce the submissions

Place the original competition files at `data/train.csv`, `data/test.csv` and `data/sample_submission.csv`. The verified combination is A on Windows x86-64 (Python 3.12.10) and B plus blending on Ubuntu x86-64 (Python 3.12.3) under WSL. B trained with the Windows wheels gave different predictions despite matching package versions.

From the repository root in PowerShell, use Windows Python 3.12 as `python` and a WSL distribution named Ubuntu with Python 3.12 and venv installed:

```powershell
python -m venv .venv2
.\.venv2\Scripts\python -m pip install -r regression_neural/requirements.txt
wsl.exe -d Ubuntu --exec python3 -m venv .venv3-linux
wsl.exe -d Ubuntu --exec ./.venv3-linux/bin/python -m pip install -r boosted_trees/requirements.txt
python reproduce.py
```

The script writes A's standalone predictions (submission 2, public MAE 133.29 MW) to `regression_neural/submissions/submission_round2.csv`, submission 7 to `boosted_trees/submissions/submission_persist.csv` and submission 6 to `submissions/final_blend.csv`. After normalizing CSV line endings, it checks all three SHA-256 digests against the recorded values. To check files already generated, run `python reproduce.py --verify-only`.

## Backtests

For A, run `.\.venv2\Scripts\python regression_neural/src/round2_models.py blend_family --folds dev follow` from the repository root. This writes per-row validation predictions for each year under `regression_neural/experiments/round2/`.

B's two validation years can be run separately. In the B environment, use `python boosted_trees/run.py --data data --validate --folds F2` for July 2023 to June 2024 and `--folds F1` for July 2024 to June 2025. The held-level model scored 109.50 and 112.92 MW respectively.
