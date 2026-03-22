#!/usr/bin/env python3

from __future__ import annotations

import json
import gzip
import shutil
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "test" / "data" / "simple_v2.zarr"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")


def write_chunk(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def main() -> None:
    shutil.rmtree(STORE, ignore_errors=True)
    root_group = {"zarr_format": 2}
    write_json(STORE / ".zgroup", root_group)

    temperature_array = {
        "chunks": [2, 2],
        "compressor": {"id": "gzip", "level": 1},
        "dtype": "<f8",
        "fill_value": None,
        "filters": None,
        "order": "C",
        "shape": [4, 3],
        "zarr_format": 2,
        "dimension_separator": ".",
    }
    write_json(
        STORE / "temperature" / ".zarray",
        temperature_array,
    )
    write_chunk(STORE / "temperature" / "0.0", gzip.compress(struct.pack("<4d", 1.5, 2.5, 4.5, 5.5)))
    write_chunk(STORE / "temperature" / "0.1", gzip.compress(struct.pack("<4d", 3.5, 0.0, 6.5, 0.0)))
    write_chunk(STORE / "temperature" / "1.0", gzip.compress(struct.pack("<4d", 7.5, 8.5, 10.5, 11.5)))
    write_chunk(STORE / "temperature" / "1.1", gzip.compress(struct.pack("<4d", 9.5, 0.0, 12.5, 0.0)))

    stations_group = {"zarr_format": 2}
    write_json(STORE / "stations" / ".zgroup", stations_group)
    elevation_array = {
        "chunks": [2],
        "compressor": None,
        "dtype": "<i4",
        "fill_value": None,
        "filters": None,
        "order": "C",
        "shape": [2],
        "zarr_format": 2,
        "dimension_separator": "/",
    }
    write_json(
        STORE / "stations" / "elevation" / ".zarray",
        elevation_array,
    )
    write_chunk(STORE / "stations" / "elevation" / "0", struct.pack("<2i", 100, 200))

    write_json(
        STORE / ".zmetadata",
        {
            "zarr_consolidated_format": 1,
            "metadata": {
                ".zgroup": root_group,
                "temperature/.zarray": temperature_array,
                "stations/.zgroup": stations_group,
                "stations/elevation/.zarray": elevation_array,
            },
        },
    )


if __name__ == "__main__":
    main()
