#!/usr/bin/env python3
"""
features_engineering_test.py

Produces test features (per user-movie row) analogous to train features so model can be validated.
Usage:
    python features_engineering_test.py /data/output/test.csv /data/output/test_features.parquet /data/ml-latest
"""
import argparse
import sys
from pathlib import Path

# Implementation intentionally mirrors features_engineering_train.py
# to ensure same columns / schema. For brevity we import logic from train script if present.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ratings_csv", help="test CSV (ratings rows) e.g. /data/output/test.csv")
    parser.add_argument("out_parquet", help="Where to write test features parquet")
    parser.add_argument("ml_dir", help="Path to ml-latest (for movies.csv)")
    args = parser.parse_args()

    # reuse logic by importing train script if available
    try:
        # attempt to import functions from train module if present in same folder
        from features_engineering_train import spark_pipeline
        movies_csv = str(Path(args.ml_dir) / "movies.csv")
        spark_pipeline(args.ratings_csv, movies_csv, args.out_parquet)
        
    except Exception:
        # fallback: copy-paste minimal implementation (safe fallback)
        # We'll call the train script as subprocess to avoid duplication if possible
        import subprocess
        cmd = ["python", str(Path(__file__).with_name("features_engineering_train.py")), args.ratings_csv, args.out_parquet, args.ml_dir]
        print("[fallback] calling:", " ".join(cmd))
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print("[error] features_engineering_test failed (fallback)", file=sys.stderr)
            sys.exit(res.returncode)

if __name__ == "__main__":
    main()
