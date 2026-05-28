#!/usr/bin/env python3
# mlflow/train_model.py

import os
import argparse
import json
from pathlib import Path
import mlflow
import mlflow.sklearn
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

def read_parquet(path):
    return pd.read_parquet(path)

def features_and_label(df):
    # ожидается, что df содержит нужные колонны
    # пример: user_avg_rating, user_num_movies, movie_avg_rating, movie_popularity, label
    feats = ['user_avg_rating','user_num_movies','movie_avg_rating','movie_popularity','movie_year']
    X = df[feats].fillna(0).astype(float)
    y = df['label'].astype(int)
    return X, y

def read_redis_features(host='redis', port=6379, db=0):
    try:
        import redis
    except ImportError:
        raise RuntimeError("redis lib not installed")
    r = redis.Redis(host=host, port=port, db=db, decode_responses=True)
    keys = r.keys()
    rows = []
    for k in keys:
        try:
            obj = json.loads(r.get(k))
            rows.append(obj)
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_parquet", help="/data/output/train_features.parquet")
    args = parser.parse_args()

    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(mlflow_uri)

    df_train = read_parquet(args.train_parquet)
    X_train, y_train = features_and_label(df_train)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)

        preds = clf.predict(X_train)
        probs = clf.predict_proba(X_train)[:,1]

        metrics = {
            "train/accuracy": float(accuracy_score(y_train, preds)),
            "train/roc_auc": float(roc_auc_score(y_train, probs)),
            "train/precision": float(precision_score(y_train, preds)),
            "train/recall": float(recall_score(y_train, preds)),
            "train/f1": float(f1_score(y_train, preds))
        }
        for k,v in metrics.items():
            mlflow.log_metric(k, v)

        # Save model artifact (mlflow will track artifact)
        mlflow.sklearn.log_model(sk_model=clf, artifact_path="model")

        # validate using features from Redis (если есть)
        df_redis = read_redis_features()
        if not df_redis.empty:
            # ensure same features
            X_test = df_redis[['user_avg_rating','user_num_movies','movie_avg_rating','movie_popularity','movie_year']].fillna(0).astype(float)
            y_test = df_redis['label'].astype(int)
            preds_test = clf.predict(X_test)
            probs_test = clf.predict_proba(X_test)[:,1]
            val_metrics = {
                "val/accuracy": float(accuracy_score(y_test, preds_test)),
                "val/roc_auc": float(roc_auc_score(y_test, probs_test)) if len(set(y_test))>1 else 0.0,
                "val/precision": float(precision_score(y_test, preds_test)),
                "val/recall": float(recall_score(y_test, preds_test)),
                "val/f1": float(f1_score(y_test, preds_test))
            }
            for k,v in val_metrics.items():
                mlflow.log_metric(k, v)
            print("[ok] validation metrics:", val_metrics)
        else:
            print("[warn] No test rows found in Redis for validation")

        # write run_id to shared folder so run.sh can start model serving
        try:
            Path('/mlflow').mkdir(parents=True, exist_ok=True)
            with open('/mlflow/run_id.txt', 'w') as f:
                f.write(run_id)
            print("[ok] wrote run_id to /mlflow/run_id.txt:", run_id)
        except Exception as e:
            print("[warn] Failed to write run_id:", e)

    print("Finished training. run_id=", run_id)

if __name__ == "__main__":
    main()
