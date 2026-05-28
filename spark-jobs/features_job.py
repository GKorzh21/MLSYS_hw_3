#!/usr/bin/env python3
"""
features_job.py

Usage from DAG:
python /opt/airflow/spark-jobs/features_job.py /data /data/output/features.parquet

Args:
    data_dir (e.g. /data)
    output_parquet_path (file or directory; e.g. /data/output/features.parquet)

This script:
 - reads ratings.csv and movies.csv from data_dir/ml-latest-small (or ml-latest)
 - computes per-user features:
     avg_rating, num_movies, genre_profile (normalized), last_interaction_ts, movie_ids
 - writes result as parquet to output_parquet_path
"""
import argparse
import json
import os
import sys
import math
from pathlib import Path

def spark_pipeline(ratings_path, movies_path, out_path):
    from pyspark.sql import SparkSession, functions as F, types as T
    print("Running Spark pipeline.")
    spark = SparkSession.builder.appName("hw3_features_job").master("local[*]").getOrCreate()

    # read CSVs
    ratings = spark.read.option("header", True).option("inferSchema", True).csv(ratings_path)
    movies = spark.read.option("header", True).option("inferSchema", True).csv(movies_path)

    # Ensure column names
    for df in (ratings, movies):
        for c in df.columns:
            pass

    # Basic aggregates
    agg = ratings.groupBy(F.col("userId").alias("user_id")) \
        .agg(
            F.round(F.avg("rating"), 4).alias("avg_rating"),
            F.countDistinct("movieId").alias("num_movies"),
            F.max("timestamp").alias("last_interaction_ts"),
            F.collect_set("movieId").alias("movie_ids")
        )

    # Prepare genre counts: join ratings->movies -> explode genres
    # Replace null genres with empty string
    joined = ratings.join(movies, on="movieId", how="left").select("userId", "movieId", "genres")
    # handle missing
    joined = joined.withColumn("genres", F.coalesce(F.col("genres"), F.lit("")))
    split_col = F.split(F.col("genres"), "\\|")
    exploded = joined.withColumn("genre", F.explode(split_col)).filter(F.col("genre") != "")

    # Count genres per user
        # Count genres per user
    genre_counts = exploded.groupBy(
        F.col("userId").alias("user_id"),
        F.col("genre")
    ).agg(F.count("*").alias("cnt"))

    # Нормализуем жанры в Map через Spark UDF
    # Превращаем (user_id, genre, cnt) → (user_id, map(genre->freq))
    total_counts = genre_counts.groupBy("user_id").agg(F.sum("cnt").alias("total"))
    joined_counts = genre_counts.join(total_counts, on="user_id", how="left")

    genre_profile = joined_counts.withColumn(
        "freq",
        F.col("cnt") / F.col("total")
    ).groupBy("user_id").agg(
        F.map_from_entries(
            F.collect_list(F.struct(F.col("genre"), F.col("freq")))
        ).alias("genre_profile")
    )

    # Соединяем с основными агрегатами
    # Соединяем с основными агрегатами
    merged = agg.join(genre_profile, on="user_id", how="left")

    # Заполняем пустые значения корректно
    merged = merged.withColumn(
        "genre_profile",
        F.when(F.col("genre_profile").isNotNull(), F.col("genre_profile"))
        .otherwise(F.create_map())
    )

    # Приводим movie_ids к отсортированным спискам
    @F.udf(returnType=T.ArrayType(T.IntegerType()))
    def sort_ids(ids):
        if not ids:
            return []
        return sorted([int(x) for x in ids])

    merged = merged.withColumn("movie_ids", sort_ids("movie_ids"))

    # Записываем результат в parquet
    merged.write.mode("overwrite").parquet(out_path)
    print(f"[ok] Wrote parquet to {out_path}")
    spark.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Base data dir (e.g. /data)")
    parser.add_argument("output_parquet", help="Output parquet file path (e.g. /data/output/features.parquet)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    # prefer ml-latest-small if exists
    candidate_small = data_dir / "ml-latest-small"
    candidate_latest = data_dir / "ml-latest"
    if candidate_small.exists():
        ds_dir = candidate_small
    elif candidate_latest.exists():
        ds_dir = candidate_latest
    else:
        # attempt both names in case user downloaded differently
        # fallback to data_dir itself if ratings.csv is inside
        if (data_dir / "ratings.csv").exists():
            ds_dir = data_dir
        else:
            print(f"[error] Could not find ml-latest-small or ml-latest in {data_dir}", file=sys.stderr)
            sys.exit(2)

    ratings_path = ds_dir / "ratings.csv"
    movies_path = ds_dir / "movies.csv"
    if not ratings_path.exists() or not movies_path.exists():
        print(f"[error] ratings.csv or movies.csv not found in {ds_dir}", file=sys.stderr)
        sys.exit(3)

    spark_pipeline(str(ratings_path), str(movies_path), args.output_parquet)

if __name__ == "__main__":
    main()
