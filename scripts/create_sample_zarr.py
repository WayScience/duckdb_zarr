#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "test" / "data" / "simple_v2.zarr"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")


def write_chunk(path: Path, size: int, fill: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes([fill]) * size)


def main() -> None:
    write_json(STORE / ".zgroup", {"zarr_format": 2})

    write_json(
        STORE / "temperature" / ".zarray",
        {
            "chunks": [2, 2],
            "compressor": {"id": "gzip", "level": 1},
            "dtype": "<f8",
            "fill_value": None,
            "filters": None,
            "order": "C",
            "shape": [4, 3],
            "zarr_format": 2,
            "dimension_separator": ".",
        },
    )
    write_chunk(STORE / "temperature" / "0.0", 8, 1)
    write_chunk(STORE / "temperature" / "0.1", 4, 2)
    write_chunk(STORE / "temperature" / "1.0", 8, 3)
    write_chunk(STORE / "temperature" / "1.1", 4, 4)

    write_json(STORE / "stations" / ".zgroup", {"zarr_format": 2})
    write_json(
        STORE / "stations" / "elevation" / ".zarray",
        {
            "chunks": [2],
            "compressor": None,
            "dtype": "<i4",
            "fill_value": None,
            "filters": None,
            "order": "C",
            "shape": [2],
            "zarr_format": 2,
            "dimension_separator": "/",
        },
    )
    write_chunk(STORE / "stations" / "elevation" / "0", 8, 5)


if __name__ == "__main__":
    main()
