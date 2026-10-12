#!/usr/bin/env bash
# Download every open dataset used by SymptoGraph-LLM into ./data
# Usage: bash scripts/download_data.sh   (run from the repo root)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/data"
mkdir -p "$DATA"
cd "$DATA"

clone() {  # clone <url> <dir>
  if [ -d "$2/.git" ] || [ -d "$2" ] && [ -n "$(ls -A "$2" 2>/dev/null)" ]; then
    echo "[skip] $2 already present"
  else
    echo "[get ] $2"
    git clone --depth 1 -q "$1" "$2"
    rm -rf "$2/.git"
  fi
}

# 1. Gold symptom labels
clone https://github.com/abuBakarSiddiqurRahman/DepressionEmo.git depressionemo
# 2. Depression severity (4 levels)
clone https://github.com/usmaann/Depression_Severity_Dataset.git depseverity
# 3. Suicide-risk severity (C-SSRS, 500 users)
clone https://github.com/manasgaur/Knowledge-aware-Assessment-of-Severity-of-Suicide-Risk-for-Early-Intervention.git cssrs_reddit_500
# 4. IMHI explanations (MentaLLaMA)
clone https://github.com/SteveKGYang/MentalLLaMA.git imhi
# 5-6. Optional concept datasets with evidence spans
clone https://github.com/drmuskangarg/Irf.git irf
clone https://github.com/drmuskangarg/LoST.git lost

# 7. Dreaddit (Hugging Face parquet, no login needed)
mkdir -p dreaddit
python3 - <<'EOF'
import json, os, urllib.request
api = "https://huggingface.co/api/datasets/andreagasparini/dreaddit/parquet"
urls = json.load(urllib.request.urlopen(api))
for cfg, splits in urls.items():
    for split, files in splits.items():
        for i, u in enumerate(files):
            out = f"dreaddit/{split}-{i}.parquet"
            if not os.path.exists(out):
                print("[get ] dreaddit", split); urllib.request.urlretrieve(u, out)
EOF

# 8. Reddit Mental Health Dataset (Low et al., 2020), Zenodo 3941387, ~3.1 GB
mkdir -p reddit_mental_health
python3 - <<'EOF'
import json, os, hashlib, urllib.request
from concurrent.futures import ThreadPoolExecutor
rec = json.load(urllib.request.urlopen("https://zenodo.org/api/records/3941387"))
def get(f):
    out = os.path.join("reddit_mental_health", f["key"])
    algo, digest = f["checksum"].split(":")
    if os.path.exists(out) and hashlib.new(algo, open(out, "rb").read()).hexdigest() == digest:
        return f"[skip] {f['key']}"
    for attempt in range(4):
        try:
            urllib.request.urlretrieve(f["links"]["self"], out)
            if hashlib.new(algo, open(out, "rb").read()).hexdigest() == digest:
                return f"[ok  ] {f['key']}"
        except Exception as e:
            err = e
    raise RuntimeError(f"failed {f['key']}")
with ThreadPoolExecutor(6) as ex:
    for msg in ex.map(get, rec["files"]):
        print(msg)
EOF

echo "Done. Data is in $DATA"
du -sh "$DATA"/* | sort -h
