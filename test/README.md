# Testing this extension

This directory contains SQLLogicTests for the `duckdb_zarr` extension.

The current suite covers the metadata and `zarr_cells` slices:

- `zarr_groups(path)`
- `zarr_arrays(path)`
- `zarr_chunks(path)`
- `zarr_cells(path, array_path)`
- `test/python_smoke.py` for a Python-side smoke example using DuckDB 1.5.0
- real OME-Zarr v3 sharded/Blosc cell scans against `test/data/idr0062A/6001240_labels.zarr`

The checked-in fixtures used by `test/sql/duckdb_zarr.test` are:

- `test/data/simple_v2.zarr`
- `test/data/simple_v3.zarr`
- `test/data/ome_example.ome.zarr`
- `test/data/idr0062A/6001240_labels.zarr`

Both fixtures can be regenerated with:

```bash
make fixture
```

The root makefile contains targets to build and run these tests:

```bash
make test
```

or

```bash
make test_debug
```

The Python smoke example can be run with:

```bash
uvx --with duckdb==1.5.0 --with numpy --with pandas python test/python_smoke.py
```
