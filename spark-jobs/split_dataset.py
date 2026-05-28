import csv
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ml_dir")
    parser.add_argument("out_dir")
    args = parser.parse_args()

    ratings_file = Path(args.ml_dir) / "ratings.csv"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(ratings_file, newline='', encoding='utf-8') as f:
        rdr = csv.reader(f)
        header = next(rdr)
        rows = list(rdr)
        split_idx = int(len(rows) * 0.8)

    with open(out_dir / "train.csv", "w", newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows[:split_idx])

    with open(out_dir / "test.csv", "w", newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows[split_idx:])

    print(f"Wrote train/test to {out_dir}")

if __name__ == "__main__":
    main()
