"""Train both pipelines, generate submissions 2, 7 and 6, and verify their SHA-256 digests."""
import argparse
import hashlib
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
A = "regression_neural/submissions/submission_round2.csv"
B = "boosted_trees/submissions/submission_persist.csv"
BLEND = "submissions/final_blend.csv"
REFERENCE = {
    A: "9bf4d6167baf8eecc476c4e9783ff8cd3ab8b6a29092ca3ef027726dd332208b",
    B: "7de0557bf73625c0af7e2db817713bbee87d484659420e1fbea25ee5bf81d13e",
    BLEND: "a90784b7dbf733b25461bc6ae9f504ac0cffc782840e548799b5d00f0b525049",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Check existing outputs without retraining.")
    args = parser.parse_args()
    if not args.verify_only:
        python_a = ROOT / ".venv2/Scripts/python.exe"
        python_b = ["wsl.exe", "-d", "Ubuntu", "--exec", "./.venv3-linux/bin/python"]
        subprocess.run([python_a, "regression_neural/src/round2_submit.py", "--data-dir", "data"], cwd=ROOT, check=True)
        subprocess.run([*python_b, "boosted_trees/run.py", "--data", "data", "--out-dir", "boosted_trees/submissions"],
                       cwd=ROOT, check=True)
        subprocess.run([*python_b, "boosted_trees/blend.py", "--data", "data", "--mine", B, "--other", A,
                        "--out", BLEND], cwd=ROOT, check=True)

    for path, expected in REFERENCE.items():
        digest = hashlib.sha256((ROOT / path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if digest != expected:
            raise SystemExit(f"{path}: expected {expected}, got {digest}")
        print(f"{digest}  {path}")


if __name__ == "__main__":
    main()
