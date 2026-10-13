"""Atomic JSON writes, and a reader that survives a non-truncating overwrite.

Google Drive's FUSE mount does not reliably truncate a file that is reopened in
``"w"`` mode. Rewriting a slightly shorter file leaves the tail of the previous
one in place, and the result fails to parse::

    JSONDecodeError: Extra data: line 1281 column 4 (char 35063)

Every run here writes its JSON straight to Drive and rewrites it on resume, so
this hit ``performance_metrics.json``. ``write_json`` avoids it by writing a
temporary file and renaming over the target, so the visible file is only ever
complete. ``read_json`` tolerates files already damaged this way, which means
existing results can be recovered without retraining.
"""
from __future__ import annotations

import json
import os


def write_json(path: str, obj, indent: int = 2) -> str:
    """Write ``obj`` to ``path`` atomically."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, default=str)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass                      # some mounts do not support fsync
    try:
        os.replace(tmp, path)
    except OSError:
        # Fall back to an explicit truncate-and-write if rename is unsupported.
        with open(path, "w", encoding="utf-8") as fh:
            fh.truncate(0)
            json.dump(obj, fh, indent=indent, default=str)
            fh.flush()
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def read_json(path: str):
    """Read JSON, ignoring any stale tail left by a partial overwrite."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        obj, _ = json.JSONDecoder().raw_decode(text.lstrip())
        return obj


def is_damaged(path: str) -> bool:
    try:
        with open(path, encoding="utf-8") as fh:
            json.load(fh)
        return False
    except json.JSONDecodeError:
        return True
    except OSError:
        return False


def repair_json_tree(root: str, verbose: bool = True) -> dict:
    """Rewrite every damaged JSON file under ``root``.

    Recovers the valid leading object and writes it back atomically, so results
    produced before the atomic-write fix stay usable.
    """
    repaired, failed, checked = [], [], 0
    for base, _, files in os.walk(root):
        for name in files:
            if not name.endswith(".json"):
                continue
            path = os.path.join(base, name)
            checked += 1
            if not is_damaged(path):
                continue
            try:
                write_json(path, read_json(path))
                repaired.append(path)
            except Exception as exc:
                failed.append((path, f"{type(exc).__name__}: {exc}"))
    if verbose:
        print(f"checked {checked} JSON files")
        for p in repaired:
            print("  repaired:", p)
        for p, why in failed:
            print("  UNRECOVERABLE:", p, why)
        if not repaired and not failed:
            print("  nothing damaged")
    return {"checked": checked, "repaired": repaired, "failed": failed}
