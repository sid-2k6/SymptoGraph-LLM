#!/usr/bin/env bash
# Download every open dataset used by SymptoGraph-LLM.
#
# Usage:
#   bash scripts/download_data.sh                       # -> ./data
#   DATA_DIR=/content/drive/MyDrive/SymptoGraph-LLM/data bash scripts/download_data.sh
#
# Environment variables:
#   DATA_DIR     target folder            (default: <repo>/data)
#   JOBS         parallel downloads        (default: 6; use 3-4 when writing to Google Drive)
#   SKIP         comma-separated keys to skip, e.g. SKIP=reddit_mental_health,imhi
#   RMHD_FILTER  regex: only download matching Reddit Mental Health files,
#                e.g. RMHD_FILTER='depression|anxiety|bipolar|suicidewatch'
#
# The script is resumable: anything already downloaded is skipped, and every
# Reddit Mental Health file is verified against its Zenodo checksum.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${DATA_DIR:-$ROOT/data}"
JOBS="${JOBS:-6}"
SKIP="${SKIP:-}"
mkdir -p "$DATA"
cd "$DATA"
echo "Target: $DATA"

skipped() { case ",$SKIP," in *",$1,"*) return 0;; *) return 1;; esac; }

clone() {  # clone <key> <url> <dir>
  if skipped "$1"; then echo "[skip] $3 (SKIP)"; return; fi
  if [ -n "$(ls -A "$3" 2>/dev/null)" ]; then
    echo "[skip] $3 already present"
  else
    echo "[get ] $3"
    rm -rf "$3"
    git clone --depth 1 -q "$2" "$3"
    rm -rf "$3/.git"
  fi
}

clone depressionemo      https://github.com/abuBakarSiddiqurRahman/DepressionEmo.git depressionemo
clone depseverity        https://github.com/usmaann/Depression_Severity_Dataset.git depseverity
clone cssrs              https://github.com/manasgaur/Knowledge-aware-Assessment-of-Severity-of-Suicide-Risk-for-Early-Intervention.git cssrs_reddit_500
clone imhi               https://github.com/SteveKGYang/MentalLLaMA.git imhi
clone irf                https://github.com/drmuskangarg/Irf.git irf
clone lost               https://github.com/drmuskangarg/LoST.git lost

# Dreaddit (Hugging Face parquet, no login required)
if skipped dreaddit; then echo "[skip] dreaddit (SKIP)"; else
mkdir -p dreaddit
python3 - <<'EOF'
import json, os, urllib.request
urls = json.load(urllib.request.urlopen(
    "https://huggingface.co/api/datasets/andreagasparini/dreaddit/parquet"))
for cfg, splits in urls.items():
    for split, files in splits.items():
        for i, u in enumerate(files):
            out = f"dreaddit/{split}-{i}.parquet"
            if os.path.exists(out):
                print("[skip] dreaddit", split); continue
            print("[get ] dreaddit", split)
            urllib.request.urlretrieve(u, out)
EOF
fi

# Reddit Mental Health Dataset (Low et al., 2020), Zenodo 3941387, ~3.1 GB
if skipped reddit_mental_health; then echo "[skip] reddit_mental_health (SKIP)"; else
mkdir -p reddit_mental_health
JOBS="$JOBS" RMHD_FILTER="${RMHD_FILTER:-}" python3 - <<'EOF'
import hashlib, json, os, re, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

jobs = int(os.environ.get("JOBS", "6"))
pat = os.environ.get("RMHD_FILTER") or None
rec = json.load(urllib.request.urlopen("https://zenodo.org/api/records/3941387"))
files = rec["files"]
if pat:
    files = [f for f in files if re.search(pat, f["key"])]
    print(f"RMHD filter {pat!r}: {len(files)}/{len(rec['files'])} files")

def digest(path, algo):
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def get(f):
    out = os.path.join("reddit_mental_health", f["key"])
    algo, want = f["checksum"].split(":")
    if os.path.exists(out) and digest(out, algo) == want:
        return f"[skip] {f['key']}"
    last = None
    for _ in range(4):
        try:
            tmp = out + ".part"
            urllib.request.urlretrieve(f["links"]["self"], tmp)
            if digest(tmp, algo) == want:
                os.replace(tmp, out)
                return f"[ok  ] {f['key']}"
            last = "checksum mismatch"
        except Exception as e:
            last = e
    raise RuntimeError(f"failed {f['key']}: {last}")

with ThreadPoolExecutor(jobs) as ex:
    for msg in ex.map(get, files):
        print(msg, flush=True)
EOF
fi

echo
echo "Done. Data is in $DATA"
du -sh "$DATA"/* 2>/dev/null | sort -h
