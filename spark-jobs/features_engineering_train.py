#!/usr/bin/env python3
"""
features_engineering_train.py

Reads train CSV (ratings-level rows) and movies.csv, produces per-(user,movie) features + label,
and writes parquet suitable for model training.

Usage:
    python features_engineering_train.py /data/output/train.csv /data/output/train_features.parquet /data/ml-latest
"""
import argparse
import sys
from pathlib import Path


def spark_pipeline(ratings_csv, movies_csv, out_path):
    from pyspark.sql import SparkSession, functions as F, types as T
    from pyspark.sql.functions import broadcast
    import os

    print("[spark] start")

    # Параметры — подбери под своё окружение (если контейнер маленький, уменьши память)
    driver_mem = "4g"          # <-- попробуй 4g, если не хватает — увеличить
    executor_mem = "4g"
    # количество shuffle partitions — увеличиваем, чтобы уменьшить размер данных на партицию
    shuffle_partitions = 400

    # явная временная директория внутри контейнера (должна быть доступна для записи)
    tmpdir = "/tmp/spark_tmp"
    os.makedirs(tmpdir, exist_ok=True)

    spark = SparkSession.builder \
        .appName("features_engineering_train") \
        .master("local[*]") \
        .config("spark.driver.memory", driver_mem) \
        .config("spark.executor.memory", executor_mem) \
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions)) \
        .config("spark.local.dir", tmpdir) \
        .config("spark.driver.extraJavaOptions", f"-Djava.io.tmpdir={tmpdir}") \
        .config("spark.executor.extraJavaOptions", f"-Djava.io.tmpdir={tmpdir}") \
        .getOrCreate()

    sc = spark.sparkContext
    # адаптивно под defaultParallelism — меньше шансов залить одну задачу
    default_parallelism = sc.defaultParallelism or 4
    repart_n = max(default_parallelism * 4, shuffle_partitions)

    # чтение
    ratings = spark.read.option("header", True).option("inferSchema", True).csv(ratings_csv)
    movies = spark.read.option("header", True).option("inferSchema", True).csv(movies_csv)

    # broadcast маленькой таблицы movies (обычно small) — уменьшит shuffle
    movies = broadcast(movies)

    # Репартиционируем ratings по большему числу партиций чтобы уменьшить нагрузку на отдельный task
    ratings = ratings.repartition(repart_n)

    # user aggregates (обычный groupBy) — теперь партиционирование помогает
    user_agg = ratings.groupBy("userId").agg(
        F.avg("rating").alias("user_avg_rating"),
        F.countDistinct("movieId").alias("user_num_movies")
    )

    # movie aggregates
    movie_agg = ratings.groupBy("movieId").agg(
        F.avg("rating").alias("movie_avg_rating"),
        F.count("userId").alias("movie_popularity")
    )

    # extract year from title (исправленный regex — без лишних экранирований)
    movie_with_year = movies.withColumn(
        "movie_year",
        F.regexp_extract(F.col("title"), r"\((\d{4})\)\s*$", 1).cast("int")
    )

    # join everything to ratings (per-rating rows) — movies уже broadcast
    df = ratings.join(user_agg, on="userId", how="left") \
                .join(movie_agg, on="movieId", how="left") \
                .join(movie_with_year.select("movieId", "movie_year", "genres"), on="movieId", how="left")

    # label
    df = df.withColumn("label", F.when(F.col("rating") >= 4.0, F.lit(1)).otherwise(F.lit(0)).cast("int"))

    out = df.select(
        F.col("userId"),
        F.col("movieId"),
        F.col("timestamp"),
        F.col("user_avg_rating"),
        F.col("user_num_movies"),
        F.col("movie_avg_rating"),
        F.col("movie_popularity"),
        F.col("movie_year"),
        F.col("genres"),
        F.col("rating"),
        F.col("label")
    )

    print("[spark] writing parquet ...")
    out.write.mode("overwrite").parquet(out_path)
    print("[spark] done.")
    spark.stop()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ratings_csv", help="train CSV (ratings rows), e.g. /data/output/train.csv")
    parser.add_argument("out_parquet", help="Where to write train features parquet")
    parser.add_argument("ml_dir", help="Path to ml-latest (for movies.csv) e.g. /data/ml-latest")
    args = parser.parse_args()

    movies_csv = str(Path(args.ml_dir) / "movies.csv")
    if not Path(movies_csv).exists():
        print("[error] movies.csv not found:", movies_csv, file=sys.stderr)
        sys.exit(2)

    spark_pipeline(args.ratings_csv, movies_csv, args.out_parquet)

if __name__ == "__main__":
    main()
