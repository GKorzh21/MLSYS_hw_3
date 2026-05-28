#!/usr/bin/env python3
# mlflow/train_model.py
"""
Train a simple classifier and log results to MLflow.

Improvements and changes:
 - timing for major steps (read, preprocess, fit, validate, mlflow logging)
 - optional sampling (--sample-frac or --sample-n)
 - faster solver & parallelism by default for large datasets (solver='saga', n_jobs=-1)
 - safe Redis validation using SCAN and pipeline (no r.keys())
 - option to skip Redis validation (--no-redis-validation)
 - robust metric computation (handle single-class case for ROC AUC)
 - clear exit codes and helpful logging
 - writes run_id to /mlflow/run_id.txt as before
Usage:
    python train_model.py /data/output/train_features.parquet \
        --mlflow-uri http://mlflow:5000 \
        --sample-frac 0.1 \
        --max-iter 200 \
        --no-redis-validation
"""
from __future__ import annotations
import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Tuple, Optional

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

EXIT_OK = 0
EXIT_BAD_ARGS = 2
EXIT_IO = 3
EXIT_REDIS = 4
EXIT_TRAIN = 5
EXIT_OTHER = 6

logger = logging.getLogger("train_model")


def parse_args():
    p = argparse.ArgumentParser(description="Train model and log to MLflow.")
    p.add_argument("train_parquet", help="Path to train features parquet (e.g. /data/output/train_features.parquet)")
    p.add_argument("--mlflow-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
                   help="MLflow tracking server URI (default from MLFLOW_TRACKING_URI or http://mlflow:5000)")
    p.add_argument("--sample-frac", type=float, default=None,
                   help="If provided, random sample this fraction of training rows (0-1).")
    p.add_argument("--sample-n", type=int, default=None,
                   help="If provided, random sample this many rows from training data (overrides --sample-frac if given).")
    p.add_argument("--max-iter", type=int, default=200, help="Max iterations for solver (default 200).")
    p.add_argument("--solver", choices=["lbfgs", "saga", "liblinear"], default="saga",
                   help="Solver for LogisticRegression (default 'saga' — good for large data).")
    p.add_argument("--use-sgd", action="store_true", help="Use SGDClassifier (log loss) instead of LogisticRegression.")
    p.add_argument("--no-redis-validation", action="store_true", help="Skip validation step that reads features from Redis.")
    p.add_argument("--redis-host", default="redis", help="Redis host for validation (default 'redis').")
    p.add_argument("--redis-port", type=int, default=6379, help="Redis port (default 6379).")
    p.add_argument("--redis-db", type=int, default=0, help="Redis DB index (default 0).")
    p.add_argument("--redis-scan-count", type=int, default=1000,
                   help="SCAN batch size when reading from Redis (default 1000).")
    p.add_argument("--random-seed", type=int, default=42, help="Random seed for sampling (default 42).")
    p.add_argument("--verbose", action="store_true", help="Verbose logging.")
    return p.parse_args()


def configure_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


def read_parquet(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    # pandas will generally be fine for moderate-size parquet; sampling options exist below
    df = pd.read_parquet(str(p))
    return df


def features_and_label(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # Expected columns in df (as in original):
    feats = ['user_avg_rating', 'user_num_movies', 'movie_avg_rating', 'movie_popularity', 'movie_year']
    missing = [c for c in feats + ['label'] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataframe: {missing}")
    X = df[feats].fillna(0).astype(float)
    y = df['label'].astype(int)
    return X, y


def read_redis_features(host: str = "redis", port: int = 6379, db: int = 0, scan_count: int = 1000, max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Efficiently read JSON-serialized feature rows from Redis using SCAN + pipeline.
    Each key is expected to be "userId:movieId" and value is JSON with required fields.
    Returns DataFrame (possibly empty) with rows.
    max_rows: optional cap to avoid reading extremely large DB during validation.
    """
    try:
        import redis
    except Exception as e:
        logger.exception("Redis python client not installed.")
        raise RuntimeError("redis lib not installed") from e

    r = redis.Redis(host=host, port=port, db=db, decode_responses=True)
    try:
        r.ping()
    except Exception as e:
        logger.exception("Cannot connect to Redis %s:%s db=%s", host, port, db)
        raise

    rows = []
    cursor = "0"
    total = 0
    # use scan_iter convenience wrapper (which uses SCAN under the hood)
    try:
        for key in r.scan_iter(match="*", count=scan_count):
            # pipeline a small batch of GETs for performance
            # We'll accumulate keys into a batch then execute pipeline
            # But redis-py scan_iter yields one key at a time; we'll buffer them here
            batch = [key]
            # try to get more keys immediately from iterator without blocking; simple approach:
            # (we'll use next() on iterator up to scan_count-1 times)
            # Note: scan_iter is itself using count, but buffering helps pipeline efficiency.
            for _ in range(scan_count - 1):
                try:
                    k = next(r.scan_iter(match="*", count=scan_count))
                    batch.append(k)
                except StopIteration:
                    break
                except Exception:
                    break

            pipe = r.pipeline(transaction=False)
            for k in batch:
                pipe.get(k)
            try:
                results = pipe.execute()
            except Exception:
                # fallback to per-key get if pipeline fails
                results = []
                for k in batch:
                    try:
                        results.append(r.get(k))
                    except Exception:
                        results.append(None)

            for raw in results:
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                    rows.append(obj)
                    total += 1
                except Exception:
                    # skip bad payloads
                    continue

            if max_rows is not None and total >= max_rows:
                logger.info("Reached max_rows=%d while reading Redis", max_rows)
                break

    except Exception as e:
        # As a fallback, try a simple SCAN loop
        logger.debug("read_redis_features: scan_iter approach raised: %s", e)
        # try classic scan
        try:
            cursor = 0
            while True:
                cursor, keys = r.scan(cursor=cursor, match="*", count=scan_count)
                if not keys:
                    if cursor == 0:
                        break
                    continue
                pipe = r.pipeline(transaction=False)
                for k in keys:
                    pipe.get(k)
                res = pipe.execute()
                for raw in res:
                    if not raw:
                        continue
                    try:
                        obj = json.loads(raw)
                        rows.append(obj)
                        total += 1
                    except Exception:
                        continue
                if max_rows is not None and total >= max_rows:
                    break
                if cursor == 0:
                    break
        except Exception as e2:
            logger.exception("Failed to read Redis via scan fallback: %s", e2)
            raise

    if not rows:
        return pd.DataFrame()
    # convert to DataFrame; let pandas handle types
    df = pd.DataFrame(rows)
    return df


def safe_metrics(y_true, preds, probs=None):
    """Compute metrics robustly; returns dict"""
    out = {}
    try:
        out["accuracy"] = float(accuracy_score(y_true, preds))
    except Exception:
        out["accuracy"] = None
    try:
        if probs is not None and len(set(y_true)) > 1:
            out["roc_auc"] = float(roc_auc_score(y_true, probs))
        else:
            out["roc_auc"] = None
    except Exception:
        out["roc_auc"] = None
    try:
        out["precision"] = float(precision_score(y_true, preds, zero_division=0))
    except Exception:
        out["precision"] = None
    try:
        out["recall"] = float(recall_score(y_true, preds, zero_division=0))
    except Exception:
        out["recall"] = None
    try:
        out["f1"] = float(f1_score(y_true, preds, zero_division=0))
    except Exception:
        out["f1"] = None
    return out


def write_run_id(run_id: str):
    try:
        Path('/mlflow').mkdir(parents=True, exist_ok=True)
        with open('/mlflow/run_id.txt', 'w') as f:
            f.write(run_id)
        logger.info("Wrote run_id to /mlflow/run_id.txt: %s", run_id)
    except Exception as e:
        logger.exception("Failed to write run_id to /mlflow/run_id.txt: %s", e)


def main():
    args = parse_args()
    configure_logging(args.verbose)
    logger.info("train_model.py starting with args: %s", args)

    # configure mlflow
    mlflow.set_tracking_uri(args.mlflow_uri)
    logger.info("MLflow tracking URI set to %s", args.mlflow_uri)

    t_start = time.time()
    # 1) read parquet
    try:
        t0 = time.time()
        df_train = read_parquet(args.train_parquet)
        t1 = time.time()
        logger.info("Read parquet %s rows=%d elapsed=%.2f sec", args.train_parquet, len(df_train), t1 - t0)
    except FileNotFoundError as e:
        logger.exception("Train parquet not found: %s", e)
        sys.exit(EXIT_IO)
    except Exception as e:
        logger.exception("Failed to read parquet: %s", e)
        sys.exit(EXIT_IO)

    # 1.a) optional sampling
    if args.sample_n is not None or args.sample_frac is not None:
        try:
            if args.sample_n is not None:
                n = int(args.sample_n)
                if n <= 0:
                    raise ValueError("sample-n must be > 0")
                n = min(n, len(df_train))
                df_train = df_train.sample(n=n, random_state=args.random_seed)
                logger.info("Sampled n=%d rows from train (random_state=%d)", n, args.random_seed)
            else:
                frac = float(args.sample_frac)
                if not (0.0 < frac <= 1.0):
                    raise ValueError("sample-frac must be in (0,1]")
                df_train = df_train.sample(frac=frac, random_state=args.random_seed)
                logger.info("Sampled frac=%.4f of train (result rows=%d)", frac, len(df_train))
        except Exception as e:
            logger.exception("Sampling failed: %s", e)
            sys.exit(EXIT_BAD_ARGS)

    # 2) prepare features and labels
    try:
        t0 = time.time()
        X_train, y_train = features_and_label(df_train)
        t1 = time.time()
        logger.info("Prepared X/y shapes X=%s y=%s elapsed=%.2f sec", X_train.shape, y_train.shape, t1 - t0)
    except Exception as e:
        logger.exception("Failed to prepare features and label: %s", e)
        sys.exit(EXIT_IO)

    # 3) fit model
    try:
        t0 = time.time()
        if args.use_sgd:
            logger.info("Using SGDClassifier (log loss) for training")
            clf = SGDClassifier(loss="log", max_iter=args.max_iter, random_state=args.random_seed, n_jobs=-1)
            clf.fit(X_train, y_train)
            # predict_proba not available for some SGDClassifier depending on config; we will handle gracefully
            has_proba = hasattr(clf, "predict_proba")
        else:
            logger.info("Using LogisticRegression solver=%s max_iter=%d n_jobs=-1", args.solver, args.max_iter)
            clf = LogisticRegression(solver=args.solver, max_iter=args.max_iter, n_jobs=-1, random_state=args.random_seed)
            clf.fit(X_train, y_train)
            has_proba = True
        t1 = time.time()
        logger.info("Model training finished elapsed=%.2f sec", t1 - t0)
    except Exception as e:
        logger.exception("Training failed: %s", e)
        sys.exit(EXIT_TRAIN)

    # 4) train metrics
    try:
        t0 = time.time()
        preds = clf.predict(X_train)
        probs = None
        if has_proba:
            try:
                probs = clf.predict_proba(X_train)[:, 1]
            except Exception:
                probs = None
        train_metrics = safe_metrics(y_train, preds, probs)
        # prefix metric keys
        prefixed = {f"train/{k}": v for k, v in train_metrics.items() if v is not None}
        t1 = time.time()
        logger.info("Computed train metrics elapsed=%.2f sec metrics=%s", t1 - t0, train_metrics)
    except Exception as e:
        logger.exception("Failed to compute train metrics: %s", e)
        prefixed = {}

    # 5) Start MLflow run and log model + metrics
    run_id = None
    try:
        t0 = time.time()
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            logger.info("Started MLflow run %s", run_id)

            # log metrics (non-None)
            for k, v in prefixed.items():
                try:
                    mlflow.log_metric(k, v)
                except Exception:
                    logger.exception("Failed to log metric %s=%s", k, v)

            # save model artifact
            try:
                mlflow.sklearn.log_model(sk_model=clf, artifact_path="model")
                logger.info("Logged model to MLflow (artifact_path=model)")
            except Exception:
                logger.exception("Failed to log model to MLflow")

            # 6) validation using Redis (optional)
            if not args.no_redis_validation:
                try:
                    t_r0 = time.time()
                    df_redis = read_redis_features(host=args.redis_host, port=args.redis_port, db=args.redis_db, scan_count=args.redis_scan_count)
                    t_r1 = time.time()
                    logger.info("Read %d rows from Redis for validation elapsed=%.2f sec", len(df_redis), t_r1 - t_r0)
                    if not df_redis.empty:
                        # ensure feature columns exist
                        try:
                            X_test, y_test = features_and_label(df_redis)
                            preds_test = clf.predict(X_test)
                            probs_test = None
                            if has_proba:
                                try:
                                    probs_test = clf.predict_proba(X_test)[:, 1]
                                except Exception:
                                    probs_test = None
                            val_metrics = safe_metrics(y_test, preds_test, probs_test)
                            for k, v in val_metrics.items():
                                if v is not None:
                                    mlflow.log_metric(f"val/{k}", v)
                            logger.info("Validation metrics: %s", val_metrics)
                        except Exception:
                            logger.exception("Failed to compute/ log validation metrics from Redis data")
                    else:
                        logger.warning("No test rows found in Redis for validation (or Redis empty)")
                except Exception:
                    logger.exception("Redis validation failed; continuing without validation")
            else:
                logger.info("Skipping Redis validation as requested (--no-redis-validation)")

            # write run_id for downstream serving
            write_run_id(run.info.run_id)

        t1 = time.time()
        logger.info("MLflow run finished elapsed=%.2f sec", t1 - t0)
    except Exception as e:
        logger.exception("MLflow operations failed: %s", e)
        # still try to write run_id if available
        if run_id:
            write_run_id(run_id)
        sys.exit(EXIT_OTHER)

    total_elapsed = time.time() - t_start
    logger.info("Finished training total_elapsed=%.2f sec run_id=%s", total_elapsed, run_id)
    print("Finished training. run_id=", run_id)
    sys.exit(EXIT_OK)


if __name__ == "__main__":
    main()
