import gzip
import shutil
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "test" / "data" / "simple_v2.zarr"
LOCAL_BUILD_EXTENSION = ROOT / "build" / "release" / "extension" / "duckdb_zarr" / "duckdb_zarr.duckdb_extension"
RELEASE_ASSET = ROOT / "duckdb_zarr-v1.5.0-osx_arm64.duckdb_extension.gz"
DECOMPRESSED_EXTENSION = ROOT / "duckdb_zarr.duckdb_extension"


def resolve_extension_path() -> Path:
    if LOCAL_BUILD_EXTENSION.exists():
        return LOCAL_BUILD_EXTENSION
    if RELEASE_ASSET.exists():
        with gzip.open(RELEASE_ASSET, "rb") as src, DECOMPRESSED_EXTENSION.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        return DECOMPRESSED_EXTENSION
    raise FileNotFoundError(
        "No extension artifact found. Build the repo with `make` or download "
        "`duckdb_zarr-v1.5.0-osx_arm64.duckdb_extension.gz` into the repo root."
    )


def main() -> None:
    if not FIXTURE_PATH.exists():
        raise FileNotFoundError("Fixture store missing. Run `make fixture` from the repo root first.")

    extension_path = resolve_extension_path()

    con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
    print("DuckDB version:", con.execute("select version()").fetchone()[0])

    con.execute(f"LOAD '{extension_path.as_posix()}'")

    overview = con.execute(
        """
        SELECT array_path, rank, dtype
        FROM zarr(?)
        ORDER BY array_path
        """,
        [FIXTURE_PATH.as_posix()],
    ).fetchdf()
    print("\nArray overview:")
    print(overview)

    temperature = con.execute(
        """
        SELECT *
        FROM zarr(?, 'temperature')
        ORDER BY dim_0, dim_1
        """,
        [FIXTURE_PATH.as_posix()],
    ).fetchdf()
    print("\nTemperature cells:")
    print(temperature)

    values = temperature["value"].to_numpy(dtype=np.float64)
    print("\nTemperature summary:")
    print(pd.Series(values).describe())

    assert len(temperature) == 12
    assert np.isclose(values.sum(), 78.0)
    assert overview["array_path"].tolist() == [
        "half_precision",
        "mask",
        "sparse_fill",
        "stations/elevation",
        "temperature",
    ]

    print("\nPython smoke test passed.")


if __name__ == "__main__":
    main()
