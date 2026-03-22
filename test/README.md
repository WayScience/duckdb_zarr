# Testing this extension

This directory contains SQLLogicTests for the `duckdb_zarr` extension.

The current suite focuses on the Phase 1 metadata slice:

- `zarr_groups(path)`
- `zarr_arrays(path)`
- `zarr_chunks(path)`
- `zarr_cells(path, array_path)`

The sample store is checked in under `test/data/simple_v2.zarr` and can be regenerated with:

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
