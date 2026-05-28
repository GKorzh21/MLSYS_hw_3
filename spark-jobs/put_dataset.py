#!/usr/bin/env python3
"""
put_dataset.py

Uploads dataset folder to MinIO (S3-compatible).
Usage:
    python put_dataset.py /data/ml-latest-small

Environment variables (defaults match your docker-compose):
    MINIO_ENDPOINT (default: http://minio:9000)
    MINIO_ACCESS_KEY (default: minioadmin)
    MINIO_SECRET_KEY (default: minioadmin)
    MINIO_BUCKET (default: movielens)
"""
import argparse
import os
import sys
from pathlib import Path

def upload_all(local_dir, bucket, prefix="", s3_client=None):
    """
    Walk local_dir and upload every file preserving relative paths under prefix/
    """
    local_dir = Path(local_dir)
    if not local_dir.is_dir():
        raise RuntimeError(f"{local_dir} is not a directory")

    uploaded = []
    for root, _, files in os.walk(local_dir):
        for fname in files:
            full = Path(root) / fname
            rel = full.relative_to(local_dir)
            key = f"{prefix.rstrip('/')}/{rel}".lstrip('/')
            # upload
            s3_client.upload_file(str(full), bucket, key)
            uploaded.append(key)
            print(f"Uploaded {full} -> s3://{bucket}/{key}")
    return uploaded

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ml_dir", help="Path to ml-latest(-small) directory to upload")
    args = parser.parse_args()

    ml_dir = args.ml_dir
    if not os.path.isdir(ml_dir):
        print(f"[error] Provided path is not a directory: {ml_dir}", file=sys.stderr)
        return 2

    # Lazy import boto3 to fail with informative error
    try:
        import boto3
        from botocore.exceptions import ClientError
        from botocore.client import Config
    except Exception as e:
        print("[error] boto3 is required for uploading to MinIO. Install it (pip install boto3).", file=sys.stderr)
        print("Exception:", e, file=sys.stderr)
        return 3

    endpoint = os.environ.get("MINIO_ENDPOINT", "http://minio:9000")
    access_key = os.environ.get("MINIO_ACCESS_KEY", os.environ.get("MINIO_ROOT_USER", "minioadmin"))
    secret_key = os.environ.get("MINIO_SECRET_KEY", os.environ.get("MINIO_ROOT_PASSWORD", "minioadmin"))
    bucket = os.environ.get("MINIO_BUCKET", "movielens")

    print(f"Using MinIO endpoint={endpoint}, bucket={bucket}")

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1"
    )

    # Create bucket if not exists
    try:
        s3.head_bucket(Bucket=bucket)
        print(f"Bucket '{bucket}' already exists")
    except ClientError:
        try:
            # create bucket
            s3.create_bucket(Bucket=bucket)
            print(f"Created bucket '{bucket}'")
        except Exception as e:
            print(f"[error] Failed creating bucket {bucket}: {e}", file=sys.stderr)
            return 4

    # upload
    prefix = os.path.basename(os.path.normpath(ml_dir))
    try:
        uploaded = upload_all(ml_dir, bucket=bucket, prefix=prefix, s3_client=s3)
    except Exception as e:
        print(f"[error] Upload failed: {e}", file=sys.stderr)
        return 5

    print(f"[ok] Uploaded {len(uploaded)} files to bucket {bucket} under prefix '{prefix}/'")
    return 0

if __name__ == "__main__":
    sys.exit(main())
