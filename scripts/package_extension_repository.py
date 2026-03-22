#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gzip
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package DuckDB extension CI artifacts into a static extension repository layout."
    )
    parser.add_argument("--artifacts-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--extension-name", required=True)
    parser.add_argument("--duckdb-version", required=True)
    return parser.parse_args()


def iter_extension_binaries(artifacts_dir: Path, extension_name: str) -> list[tuple[str, Path]]:
    prefix = f"{extension_name}-"
    marker = "-extension-"
    results: list[tuple[str, Path]] = []

    for artifact_dir in sorted(path for path in artifacts_dir.iterdir() if path.is_dir()):
        if not artifact_dir.name.startswith(prefix) or marker not in artifact_dir.name:
            continue
        arch = artifact_dir.name.split(marker, 1)[1]
        binary = artifact_dir / f"{extension_name}.duckdb_extension"
        if binary.exists():
            results.append((arch, binary))

    return results


def write_gzip(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as fsrc, dest.open("wb") as raw_out:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_out, compresslevel=9, mtime=0) as gz_out:
            shutil.copyfileobj(fsrc, gz_out)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    binaries = iter_extension_binaries(args.artifacts_dir, args.extension_name)
    if not binaries:
        raise SystemExit("No extension binaries found in downloaded artifacts")

    for arch, binary in binaries:
        output_path = (
            args.out_dir
            / args.duckdb_version
            / arch
            / f"{args.extension_name}.duckdb_extension.gz"
        )
        write_gzip(binary, output_path)


if __name__ == "__main__":
    main()
