# Testing this extension

This directory contains SQLLogicTests for the `duckdb_zarr` extension.

The current suite covers the metadata and `zarr_cells` slices:

- `zarr_groups(path)`
- `zarr_arrays(path)`
- `zarr_chunks(path)`
- `zarr_cells(path, array_path)`

The checked-in fixtures used by `test/sql/duckdb_zarr.test` are:

- `test/data/simple_v2.zarr`
- `test/data/ome_example.ome.zarr`

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
