#!/usr/bin/env python3
"""
load_to_redis_ratings.py

Загружает фичи из Parquet (файл или директорию) в Redis.
Каждая строка сохраняется под ключом "userId:movieId".

Usage:
    python load_to_redis_ratings.py /data/output/test_features.parquet redis 6379
"""

import argparse
import json
import sys
import time
from pathlib import Path


def log(msg: str, level: str = "INFO"):
    """Аккуратный вывод логов с временем."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts} {level} {msg}", flush=True)


def read_parquet_in_batches(path: str, batch_size: int = 50000):
    """Читает Parquet-файл или директорию по батчам, совместимо с любой версией pyarrow."""
    import pyarrow.parquet as pq

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {p}")

    # Если это директория parquet (как Spark output)
    if p.is_dir():
        log(f"Detected parquet directory: {p}")
        dataset = pq.ParquetDataset(str(p))  # старый, но стабильный API
        table = dataset.read()
        total_rows = table.num_rows
        log(f"Loaded parquet directory into memory: {total_rows} rows")

        for start in range(0, total_rows, batch_size):
            yield table.slice(start, batch_size).to_pylist()
    else:
        # Обычный одиночный parquet-файл
        log(f"Detected single parquet file: {p}")
        pf = pq.ParquetFile(str(p))
        for batch in pf.iter_batches(batch_size=batch_size):
            yield batch.to_pylist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet_path")
    parser.add_argument("redis_host")
    parser.add_argument("redis_port", type=int)
    parser.add_argument("--redis-db", type=int, default=0)
    args = parser.parse_args()

    log("Starting load_to_redis_ratings.py")

    try:
        import redis
    except ImportError:
        log("Redis library not installed. Run: pip install redis", "ERROR")
        sys.exit(4)

    try:
        r = redis.Redis(
            host=args.redis_host,
            port=args.redis_port,
            db=args.redis_db,
            decode_responses=True,
        )
        r.ping()
    except Exception as e:
        log(f"Failed to connect to Redis: {e}", "ERROR")
        sys.exit(5)

    total_written = 0
    try:
        for rows in read_parquet_in_batches(args.parquet_path, batch_size=50000):
            pipe = r.pipeline(transaction=False)
            for rec in rows:
                user = rec.get("userId") or rec.get("user_id") or rec.get("user")
                movie = rec.get("movieId") or rec.get("movie_id") or rec.get("movie")
                if user is None or movie is None:
                    continue

                key = f"{int(user)}:{int(movie)}"
                payload = {
                    "userId": int(user),
                    "movieId": int(movie),
                    "user_avg_rating": float(rec.get("user_avg_rating") or 0),
                    "user_num_movies": int(rec.get("user_num_movies") or 0),
                    "movie_avg_rating": float(rec.get("movie_avg_rating") or 0),
                    "movie_popularity": int(rec.get("movie_popularity") or 0),
                    "movie_year": int(rec.get("movie_year"))
                    if rec.get("movie_year") not in (None, "")
                    else None,
                    "genres": rec.get("genres"),
                    "timestamp": int(rec.get("timestamp") or 0),
                    "label": int(rec.get("label") if rec.get("label") is not None else 0),
                }
                pipe.set(key, json.dumps(payload, ensure_ascii=False))

            pipe.execute()
            total_written += len(rows)
            log(f"Written {total_written} entries so far...")

    except Exception as e:
        log(f"Unexpected error during processing: {e}", "ERROR")
        sys.exit(6)

    log(f"✅ Done. Wrote {total_written} entries to Redis at {args.redis_host}:{args.redis_port}")


if __name__ == "__main__":
    main()
