import gzip
import os
import platform
import shutil
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "test" / "data" / "simple_v2.zarr"
FIXTURE_V3_PATH = ROOT / "test" / "data" / "simple_v3.zarr"
IDR_FIXTURE_PATH = ROOT / "test" / "data" / "idr0062A" / "6001240_labels.zarr"
LOCAL_BUILD_EXTENSION = ROOT / "build" / "release" / "extension" / "duckdb_zarr" / "duckdb_zarr.duckdb_extension"
DECOMPRESSED_EXTENSION = ROOT / "duckdb_zarr.duckdb_extension"


def expected_release_asset() -> Path:
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "darwin":
        platform_name = "osx_arm64" if machine in {"arm64", "aarch64"} else "osx_amd64"
    elif system == "linux":
        platform_name = "linux_arm64" if machine in {"arm64", "aarch64"} else "linux_amd64"
    elif system == "windows":
        platform_name = "windows_amd64"
    else:
        raise RuntimeError(f"Unsupported platform for release asset lookup: {platform.system()} / {platform.machine()}")

    return ROOT / f"duckdb_zarr-v1.5.0-{platform_name}.duckdb_extension.gz"


def resolve_extension_path() -> Path:
    override = os.environ.get("DUCKDB_ZARR_EXTENSION")
    if override:
        return Path(override).resolve()
    if LOCAL_BUILD_EXTENSION.exists():
        return LOCAL_BUILD_EXTENSION
    release_asset = expected_release_asset()
    if release_asset.exists():
        with gzip.open(release_asset, "rb") as src, DECOMPRESSED_EXTENSION.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        return DECOMPRESSED_EXTENSION
    raise FileNotFoundError(
        "No extension artifact found. Build the repo with `make`, set DUCKDB_ZARR_EXTENSION "
        "to an explicit extension path, or download the platform-matching release asset "
        f"`{release_asset.name}` into the repo root."
    )


def should_allow_unsigned_extensions(extension_path: Path) -> bool:
    if os.environ.get("DUCKDB_ZARR_ALLOW_UNSIGNED", "").lower() in {"1", "true", "yes"}:
        return True
    return extension_path.is_file() and extension_path.resolve().is_relative_to(ROOT.resolve())


def main() -> None:
    if not FIXTURE_PATH.exists() or not FIXTURE_V3_PATH.exists():
        raise FileNotFoundError("Fixture stores missing. Run `make fixture` from the repo root first.")

    extension_path = resolve_extension_path()
    config = {}
    if should_allow_unsigned_extensions(extension_path):
        # Local build artifacts and manually downloaded release files are unsigned in this repo.
        config["allow_unsigned_extensions"] = "true"

    con = duckdb.connect(config=config)
    print("DuckDB version:", con.execute("select version()").fetchone()[0])

    try:
        con.execute(f"LOAD '{extension_path.as_posix()}'")
    except duckdb.Error as exc:
        raise RuntimeError(
            "Failed to load the extension. If you are loading a local unsigned artifact, set "
            "DUCKDB_ZARR_ALLOW_UNSIGNED=1."
        ) from exc

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

    v3_overview = con.execute(
        """
        SELECT array_path, rank, zarr_format
        FROM zarr(?)
        ORDER BY array_path
        """,
        [FIXTURE_V3_PATH.as_posix()],
    ).fetchdf()
    print("\nV3 array overview:")
    print(v3_overview)

    v3_temperature = con.execute(
        """
        SELECT *
        FROM zarr(?, 'temperature_v3', 'v3')
        ORDER BY dim_0, dim_1
        """,
        [FIXTURE_V3_PATH.as_posix()],
    ).fetchdf()
    print("\nV3 temperature cells:")
    print(v3_temperature)

    if IDR_FIXTURE_PATH.exists():
        idr_overview = con.execute(
            """
            SELECT array_path, rank
            FROM zarr(?)
            ORDER BY array_path
            """,
            [IDR_FIXTURE_PATH.as_posix()],
        ).fetchdf()
        print("\nIDR OME-Zarr overview:")
        print(idr_overview)

        idr_head = con.execute(
            """
            SELECT *
            FROM zarr(?, '0')
            LIMIT 5
            """,
            [IDR_FIXTURE_PATH.as_posix()],
        ).fetchdf()
        print("\nIDR OME-Zarr first cells:")
        print(idr_head)

        idr_label_sum = con.execute(
            """
            SELECT SUM(value) AS label_sum
            FROM zarr(?, 'labels/0/0')
            WHERE dim_0 = 0 AND dim_1 = 0 AND dim_2 < 4 AND dim_3 < 4
            """,
            [IDR_FIXTURE_PATH.as_posix()],
        ).fetchone()[0]
        print("\nIDR label block sum:", idr_label_sum)

        assert idr_overview["array_path"].tolist() == [
            "0",
            "1",
            "2",
            "labels/0/0",
            "labels/0/1",
            "labels/0/2",
            "labels/0/3",
        ]
        assert idr_head["value"].tolist() == [8, 9, 8, 10, 8]
        assert idr_label_sum == 0

    assert len(temperature) == 12
    assert np.isclose(values.sum(), 78.0)
    assert overview["array_path"].tolist() == [
        "half_precision",
        "mask",
        "sparse_fill",
        "stations/elevation",
        "temperature",
    ]
    assert v3_overview["array_path"].tolist() == ["fortran_v3", "mask_v3", "temperature_v3"]
    assert v3_overview["zarr_format"].tolist() == [3, 3, 3]
    assert np.isclose(v3_temperature["value"].to_numpy(dtype=np.float64).sum(), 21.0)

    print("\nPython smoke test passed.")


if __name__ == "__main__":
    main()
