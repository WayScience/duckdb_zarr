#!/usr/bin/env python3

from __future__ import annotations

import json
import gzip
import shutil
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "test" / "data" / "simple_v2.zarr"
OME_STORE = ROOT / "test" / "data" / "ome_example.ome.zarr"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")


def write_chunk(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def gzip_chunk(payload: bytes) -> bytes:
    return gzip.compress(payload, compresslevel=1, mtime=0)


def create_simple_store() -> None:
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
    write_chunk(STORE / "temperature" / "0.0", gzip_chunk(struct.pack("<4d", 1.5, 2.5, 4.5, 5.5)))
    write_chunk(STORE / "temperature" / "0.1", gzip_chunk(struct.pack("<4d", 3.5, 0.0, 6.5, 0.0)))
    write_chunk(STORE / "temperature" / "1.0", gzip_chunk(struct.pack("<4d", 7.5, 8.5, 10.5, 11.5)))
    write_chunk(STORE / "temperature" / "1.1", gzip_chunk(struct.pack("<4d", 9.5, 0.0, 12.5, 0.0)))

    mask_array = {
        "chunks": [2, 2],
        "compressor": None,
        "dtype": "|b1",
        "fill_value": False,
        "filters": None,
        "order": "C",
        "shape": [2, 3],
        "zarr_format": 2,
        "dimension_separator": ".",
    }
    write_json(STORE / "mask" / ".zarray", mask_array)
    write_chunk(STORE / "mask" / "0.0", bytes([1, 0, 1, 1]))
    write_chunk(STORE / "mask" / "0.1", bytes([0, 0, 1, 0]))

    half_array = {
        "chunks": [2, 2],
        "compressor": None,
        "dtype": "<f2",
        "fill_value": None,
        "filters": None,
        "order": "C",
        "shape": [2, 2],
        "zarr_format": 2,
        "dimension_separator": ".",
    }
    write_json(STORE / "half_precision" / ".zarray", half_array)
    write_chunk(STORE / "half_precision" / "0.0", struct.pack("<4e", 1.5, -2.0, 0.5, 4.0))

    sparse_fill_array = {
        "chunks": [2, 2],
        "compressor": None,
        "dtype": "<i4",
        "fill_value": -1,
        "filters": None,
        "order": "C",
        "shape": [2, 3],
        "zarr_format": 2,
        "dimension_separator": ".",
    }
    write_json(STORE / "sparse_fill" / ".zarray", sparse_fill_array)
    write_chunk(STORE / "sparse_fill" / "0.0", struct.pack("<4i", 1, 2, 4, 5))

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
                "half_precision/.zarray": half_array,
                "mask/.zarray": mask_array,
                "sparse_fill/.zarray": sparse_fill_array,
                "temperature/.zarray": temperature_array,
                "stations/.zgroup": stations_group,
                "stations/elevation/.zarray": elevation_array,
            },
        },
    )


def create_ome_store() -> None:
    shutil.rmtree(OME_STORE, ignore_errors=True)
    root_group = {"zarr_format": 2}
    root_attrs = {
        "multiscales": [
            {
                "version": "0.4",
                "name": "example",
                "axes": [
                    {"name": "c", "type": "channel"},
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ],
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [{"type": "scale", "scale": [1.0, 0.65, 0.65]}],
                    }
                ],
            }
        ],
        "omero": {
            "name": "example",
            "channels": [
                {"label": "DNA", "color": "FF0000"},
                {"label": "RNA", "color": "00FF00"},
            ],
        },
    }
    level0_array = {
        "chunks": [1, 2, 2],
        "compressor": {"id": "gzip", "level": 1},
        "dtype": "<u2",
        "fill_value": 0,
        "filters": None,
        "order": "C",
        "shape": [2, 2, 3],
        "zarr_format": 2,
        "dimension_separator": "/",
    }
    level0_attrs = {"_ARRAY_DIMENSIONS": ["c", "y", "x"]}

    write_json(OME_STORE / ".zgroup", root_group)
    write_json(OME_STORE / ".zattrs", root_attrs)
    write_json(OME_STORE / "0" / ".zarray", level0_array)
    write_json(OME_STORE / "0" / ".zattrs", level0_attrs)

    write_chunk(OME_STORE / "0" / "0" / "0" / "0", gzip_chunk(struct.pack("<4H", 1, 2, 4, 5)))
    write_chunk(OME_STORE / "0" / "0" / "0" / "1", gzip_chunk(struct.pack("<4H", 3, 0, 6, 0)))
    write_chunk(OME_STORE / "0" / "1" / "0" / "0", gzip_chunk(struct.pack("<4H", 7, 8, 10, 11)))
    write_chunk(OME_STORE / "0" / "1" / "0" / "1", gzip_chunk(struct.pack("<4H", 9, 0, 12, 0)))

    write_json(
        OME_STORE / ".zmetadata",
        {
            "zarr_consolidated_format": 1,
            "metadata": {
                ".zgroup": root_group,
                ".zattrs": root_attrs,
                "0/.zarray": level0_array,
                "0/.zattrs": level0_attrs,
            },
        },
    )


def main() -> None:
    create_simple_store()
    create_ome_store()


if __name__ == "__main__":
    main()
