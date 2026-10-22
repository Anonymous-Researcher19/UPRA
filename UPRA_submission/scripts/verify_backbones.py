"""Verify external backbone files against the publication manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hash", action="store_true", help="also compute SHA-256 (slow for XL-SFT)")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "backbone_manifest.json").read_text(encoding="utf-8"))
    failures = 0
    for name, checkpoint in manifest["checkpoints"].items():
        base = (ROOT / checkpoint["local_path"]).resolve()
        for relative, expected in checkpoint["files"].items():
            path = base / relative
            ok = path.is_file() and path.stat().st_size == int(expected["bytes"])
            if ok and args.hash:
                ok = digest(path) == expected["sha256"]
            print(f"{'OK' if ok else 'MISSING/INVALID'} {name}: {path}")
            failures += int(not ok)
    if failures:
        raise SystemExit(f"backbone verification failed: {failures} file(s)")
    print("backbone verification: PASS")


if __name__ == "__main__":
    main()
