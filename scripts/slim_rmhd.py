#!/usr/bin/env python3
"""Shrink the Reddit Mental Health Dataset by dropping unused feature columns.

The Zenodo CSVs carry 350 columns, of which 346 are precomputed readability and
tf-idf features that SymptoGraph-LLM does not use (it learns its own
representations). Keeping only subreddit/author/date/post and gzipping reduces
the collection from 3.14 GB to 0.41 GB (108 files, 1,107,302 posts), which
matters when the data lives in Google Drive.

Usage:
    python scripts/slim_rmhd.py --src data/reddit_mental_health \
                                --dst data/reddit_mental_health_slim [--delete-src]
"""
import argparse
import csv
import glob
import gzip
import os
import sys

csv.field_size_limit(sys.maxsize)
KEEP = ["subreddit", "author", "date", "post"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/reddit_mental_health")
    ap.add_argument("--dst", default="data/reddit_mental_health_slim")
    ap.add_argument("--delete-src", action="store_true",
                    help="remove each source CSV after it converts successfully")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.src, "*.csv")))
    if not files:
        sys.exit(f"no CSV files found in {args.src}")
    os.makedirs(args.dst, exist_ok=True)

    before = after = rows_total = 0
    for i, src in enumerate(files, 1):
        name = os.path.basename(src).replace("_features_tfidf_256.csv", ".csv.gz")
        dst = os.path.join(args.dst, name)
        if os.path.exists(dst):
            print(f"[{i}/{len(files)}] skip {name}")
            after += os.path.getsize(dst)
            continue
        tmp = dst + ".part"
        rows = 0
        with open(src, newline="", encoding="utf-8", errors="replace") as fh, \
             gzip.open(tmp, "wt", newline="", encoding="utf-8") as out:
            reader = csv.DictReader(fh)
            missing = [c for c in KEEP if c not in (reader.fieldnames or [])]
            if missing:
                os.remove(tmp)
                sys.exit(f"{src} is missing expected columns: {missing}")
            writer = csv.DictWriter(out, fieldnames=KEEP)
            writer.writeheader()
            for row in reader:
                writer.writerow({k: row.get(k, "") for k in KEEP})
                rows += 1
        os.replace(tmp, dst)
        a, b = os.path.getsize(src), os.path.getsize(dst)
        before += a
        after += b
        rows_total += rows
        print(f"[{i}/{len(files)}] {name}  {a / 1e6:.1f} MB -> {b / 1e6:.1f} MB  ({rows} rows)")
        if args.delete_src:
            os.remove(src)

    print(f"\n{len(files)} files, {rows_total} posts")
    if before:
        print(f"{before / 1e9:.2f} GB -> {after / 1e9:.2f} GB "
              f"({100 * (1 - after / before):.0f}% smaller)")
    print(f"Output: {args.dst}")
    print('Read one with: pandas.read_csv(path, compression="gzip")')


if __name__ == "__main__":
    main()
