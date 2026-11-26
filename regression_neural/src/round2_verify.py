"""Verify the delivered bytes independently of the training and serialization code."""
import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from utils import DEFAULT_DATA_DIR, EXP_DIR, SUB_DIR, resolve_data_file


ROUND1_HASH = "06896f7f3a9a930545c6985971af64cdde8531855218062b76fb2944b1b1f846"
ROUND2_HASH = "9bf4d6167baf8eecc476c4e9783ff8cd3ab8b6a29092ca3ef027726dd332208b"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(require_replay=False, *, data_dir=DEFAULT_DATA_DIR, check_round1=False,
           check_reference=False):
    directory = Path(SUB_DIR)
    final = directory/"submission_round2.csv"
    data = {s: Path(resolve_data_file(data_dir, f"{s}.csv", f"solar-britain-{s}.csv"))
            for s in ("train", "test", "sample_submission")}
    out = pd.read_csv(final)
    test = pd.read_csv(data["test"])
    sample = pd.read_csv(data["sample_submission"])
    assert final.read_bytes().splitlines()[0] == b"id,generation_mw"
    assert list(out.columns) == ["id","generation_mw"] and len(out) == len(test) == 17520
    assert out.id.notna().all() and out.id.is_unique and test.id.is_unique and set(out.id) == set(test.id)
    assert np.array_equal(out.id, sample.id)
    assert np.isfinite(out.generation_mw).all() and out.generation_mw.ge(0).all()
    assert final.stat().st_size < 10_000_000
    digest = sha(final)
    if check_reference and digest != ROUND2_HASH:
        raise ValueError(f"Round 2 reference SHA-256 mismatch: expected {ROUND2_HASH}, got {digest}")
    audit = {
        "regenerate_from_repository_root": "python src/round2_submit.py",
        "round2_sha256":digest, "reference_sha256_matches":digest == ROUND2_HASH,
        "rows":len(out), "unique_test_ids":int(out.id.nunique()),
        "header":list(out), "min_mw":out.generation_mw.min(), "max_mw":out.generation_mw.max(),
        "bytes":final.stat().st_size,
        "shipped_sha256": {path.name:sha(path) for path in data.values()},
        "python":platform.python_version(), "packages":{name:version(name) for name in
            ("numpy","pandas","scipy","scikit-learn","lightgbm","torch","threadpoolctl")},
    }
    if check_round1:
        old = {name: sha(directory/name) for name in ("submission.csv", "sub_m2_no_coordinates.csv")}
        if any(digest != ROUND1_HASH for digest in old.values()):
            raise ValueError(f"Round 1 originals differ from their reference SHA-256: {old}")
        audit["round1_sha256"] = old
    if require_replay:
        replay = Path(EXP_DIR)/"round2"/"round2_replay.csv"
        assert final.read_bytes() == replay.read_bytes(), "Independent full fits differ"
        audit["independent_replay_sha256"] = sha(replay)
    source = Path(__file__).resolve().parent
    audit["source_sha256"] = {p.name:sha(p) for p in sorted(source.glob("*.py"))
                              if p.name.startswith("round2_") or p.stem in ("utils","modeling","features","run_pipeline")}
    text = json.dumps(audit, indent=2)
    print(text)
    audit_path = Path(EXP_DIR)/"round2"/"audit.json"
    audit_path.parent.mkdir(exist_ok=True,parents=True)
    audit_path.write_text(text+"\n",encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--require-replay", action="store_true")
    parser.add_argument("--check-reference", action="store_true",
                        help="Require the original round-2 SHA-256.")
    parser.add_argument("--check-round1", action="store_true",
                        help="Also check the two local round-1 originals.")
    args = parser.parse_args()
    verify(args.require_replay, data_dir=args.data_dir, check_round1=args.check_round1,
           check_reference=args.check_reference)
