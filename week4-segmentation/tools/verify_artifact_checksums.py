"""Verify SHA-256 hashes and byte sizes listed in reports/artifact_checksums.json."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("reports/artifact_checksums.json"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    manifest_path = args.manifest
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    failures = []
    for item in manifest["files"]:
        path = root / item["path"]
        if not path.is_file():
            failures.append(f"MISSING {item['path']}")
            continue
        actual_size = path.stat().st_size
        actual_hash = sha256(path)
        expected_size = int(item["bytes"])
        expected_hash = str(item["sha256"]).upper()
        if actual_size != expected_size:
            failures.append(
                f"SIZE {item['path']}: expected {expected_size}, got {actual_size}"
            )
        if actual_hash != expected_hash:
            failures.append(
                f"SHA256 {item['path']}: expected {expected_hash}, got {actual_hash}"
            )
        if actual_size == expected_size and actual_hash == expected_hash:
            print(f"OK  {item['path']}  {actual_size} bytes  {actual_hash}")

    if failures:
        print("\nVerification failed:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print(f"\nAll {len(manifest['files'])} artifacts verified.")


if __name__ == "__main__":
    main()
