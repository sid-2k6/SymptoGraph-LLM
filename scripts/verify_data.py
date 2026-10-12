#!/usr/bin/env python3
"""Verify a SymptoGraph-LLM data folder against the expected contents.

Checks that every dataset is present, that row counts match the published
figures, and that no partial downloads were left behind. Use it after copying
the data to Google Drive, where folder sizes are not shown.

Usage:
    python scripts/verify_data.py                         # checks ./data
    python scripts/verify_data.py --data /content/drive/MyDrive/SymptoGraph-LLM/data
    python scripts/verify_data.py --data ... --deep        # also sums all Reddit MH rows
"""
import argparse
import csv
import glob
import gzip
import os
import sys

csv.field_size_limit(sys.maxsize)

# (label, [relative paths], expected rows) - verified after download, 12 October 2026.
ROW_CHECKS = [
    ("DepressionEmo train", ["depressionemo/Dataset/train.json"], 4225),
    ("DepressionEmo val", ["depressionemo/Dataset/val.json"], 906),
    ("DepressionEmo test", ["depressionemo/Dataset/test.json"], 906),
    ("DepSeverity", ["depseverity/Reddit_depression_dataset.csv"], 3553),
    ("CSSRS-Reddit-500", ["cssrs_reddit_500/anonymized_dataset/500_Reddit_users_posts_labels.csv"], 500),
    ("IRF (train+val+test)", ["irf/train_data.csv", "irf/val_data.csv", "irf/test_data.csv"], 3522),
    ("LoST", ["lost/LoSTdataset.csv"], 3251),
]
RMHD_FILES, RMHD_POSTS = 108, 1107302
IMHI_TRAIN_FILES, IMHI_TEST_FILES = 20, 18
DREADDIT_ROWS = 3553
SLIM_COLUMNS = ["subreddit", "author", "date", "post"]


def count_rows(path):
    """Row count excluding any header, for csv / json-lines / gzipped csv."""
    if path.endswith(".json"):
        with open(path, encoding="utf-8", errors="replace") as fh:
            return sum(1 for line in fh if line.strip())
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", newline="", encoding="utf-8", errors="replace") as fh:
        return max(sum(1 for _ in csv.reader(fh)) - 1, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--deep", action="store_true",
                    help="read every Reddit MH file and sum the rows (slower)")
    args = ap.parse_args()
    D = args.data.rstrip("/")
    if not os.path.isdir(D):
        sys.exit(f"no such folder: {D}")

    rows, failures = [], []

    def check(label, found, want, note=""):
        ok = found == want
        rows.append((label, found, want, ok, note))
        if not ok:
            failures.append(label)
        return ok

    for label, paths, want in ROW_CHECKS:
        try:
            check(label, sum(count_rows(os.path.join(D, p)) for p in paths), want)
        except FileNotFoundError as e:
            rows.append((label, "MISSING", want, False, os.path.basename(e.filename)))
            failures.append(label)

    # Dreaddit (parquet)
    pq = sorted(glob.glob(f"{D}/dreaddit/*.parquet"))
    if not pq:
        rows.append(("Dreaddit", "MISSING", DREADDIT_ROWS, False, ""))
        failures.append("Dreaddit")
    else:
        try:
            import pandas as pd
            check("Dreaddit", sum(len(pd.read_parquet(f)) for f in pq), DREADDIT_ROWS)
        except ImportError:
            rows.append(("Dreaddit", f"{len(pq)} files", DREADDIT_ROWS, True, "pandas missing"))

    check("IMHI train files", len(glob.glob(f"{D}/imhi/train_data/**/*.*", recursive=True)),
          IMHI_TRAIN_FILES)
    check("IMHI test files", len(glob.glob(f"{D}/imhi/test_data/**/*.*", recursive=True)),
          IMHI_TEST_FILES)

    # Reddit Mental Health: slim preferred, raw accepted
    slim = sorted(glob.glob(f"{D}/reddit_mental_health_slim/*.csv.gz"))
    raw = sorted(glob.glob(f"{D}/reddit_mental_health/*.csv"))
    if slim:
        check("Reddit MH slim files", len(slim), RMHD_FILES)
        try:
            with gzip.open(slim[0], "rt", encoding="utf-8") as fh:
                cols = next(csv.reader(fh))
            check("Reddit MH columns", cols, SLIM_COLUMNS,
                  note=os.path.basename(slim[0]))
        except Exception as e:
            rows.append(("Reddit MH columns", f"unreadable", "readable", False, str(e)[:40]))
            failures.append("Reddit MH columns")
        if args.deep:
            check("Reddit MH posts", sum(count_rows(f) for f in slim), RMHD_POSTS)
    elif raw:
        check("Reddit MH files (raw)", len(raw), RMHD_FILES)
    else:
        rows.append(("Reddit MH", "MISSING", f"{RMHD_FILES} files", False, ""))
        failures.append("Reddit MH")

    check("Incomplete .part files", len(glob.glob(f"{D}/**/*.part", recursive=True)), 0)

    def show(v):
        if isinstance(v, int):
            return f"{v:,}"
        if isinstance(v, list):
            return f"{len(v)} cols"
        return str(v)

    w = max(len(r[0]) for r in rows)
    print(f"Checking {D}\n")
    print(f"{'check'.ljust(w)}  {'found':>10}  {'expected':>10}  status")
    print("-" * (w + 34))
    for label, found, want, ok, note in rows:
        line = f"{label.ljust(w)}  {show(found):>10}  {show(want):>10}  {'ok' if ok else 'FAIL'}"
        if note and not ok:
            line += f"  ({note})"
        print(line)

    if failures:
        print(f"\n{len(failures)} check(s) failed: {', '.join(failures)}")
        print("Re-run scripts/download_data.sh (it resumes), then copy to Drive again.")
        sys.exit(1)

    print("\nAll checks passed. The dataset is complete.")
    if slim and not args.deep:
        print(f"Add --deep to also count all {RMHD_POSTS:,} Reddit MH posts.")


if __name__ == "__main__":
    main()
