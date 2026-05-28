#!/usr/bin/env python3
"""
get_dataset.py

Usage:
    python get_dataset.py /data [--version small|latest]

Downloads MovieLens dataset (by default ml-latest-small) into the provided data directory.
If the target folder already exists and contains files, it will skip download.

Saves into: <data_dir>/ml-latest-small  (or ml-latest)
"""
import argparse
import os
import sys
import shutil
import zipfile
import tempfile
import time

try:
    import requests
except Exception:
    requests = None
    import urllib.request

def download(url, dest_path):
    print(f"Downloading {url} -> {dest_path}")
    if requests:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    else:
        urllib.request.urlretrieve(url, dest_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Base data directory (e.g. /data)")
    parser.add_argument("--version", choices=["small", "latest"], default="small",
                        help="Which MovieLens to download: small (default) or latest (larger)")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)

    version = args.version
    folder_name = "ml-latest-small" if version == "small" else "ml-latest"
    target_dir = os.path.join(data_dir, folder_name)

    # If already exists and non-empty, skip
    if os.path.isdir(target_dir) and any(os.scandir(target_dir)):
        print(f"[skip] Target dataset directory already exists and is non-empty: {target_dir}")
        return 0

    base_url = "https://files.grouplens.org/datasets/movielens"
    zip_name = f"{folder_name}.zip"
    url = f"{base_url}/{zip_name}"

    tmpzip = os.path.join(tempfile.gettempdir(), f"movielens_{int(time.time())}.zip")
    try:
        download(url, tmpzip)
    except Exception as e:
        print(f"[error] Failed downloading {url}: {e}", file=sys.stderr)
        return 2

    try:
        with zipfile.ZipFile(tmpzip, "r") as z:
            # Extract into data_dir (so path becomes data_dir/ml-latest-small/...)
            print(f"Extracting to {data_dir} ...")
            z.extractall(path=data_dir)
    except Exception as e:
        print(f"[error] Failed extracting zip {tmpzip}: {e}", file=sys.stderr)
        return 3
    finally:
        try:
            os.remove(tmpzip)
        except Exception:
            pass

    # Verify
    if not os.path.isdir(target_dir):
        print(f"[error] Expected folder {target_dir} not found after extraction", file=sys.stderr)
        return 4

    print(f"[ok] Dataset ready at {target_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
