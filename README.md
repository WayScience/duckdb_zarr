# duckdb_zarr

`duckdb_zarr` is a DuckDB extension for exploring Zarr stores with SQL through a relational projection.

The current implementation follows the project documents conservatively:

- `SPEC.md`: start with relational metadata and chunk discovery
- `ARCHITECTURE.md`: build the store adapter, metadata parser, and DuckDB bridge first
- `ROADMAP.md`: advance incrementally from metadata discovery to relational cell scans and planner-aware execution

Today’s extension provides local-store metadata table functions:

- `zarr_groups(path)`
- `zarr_arrays(path)`
- `zarr_chunks(path)`
- `zarr_cells(path, array_path)`

This gives a usable SQL entrypoint for understanding a Zarr store and projecting dense numeric arrays into relational rows.

## What Works

The MVP currently supports:

- Local filesystem Zarr v2 discovery
- Group enumeration from `.zgroup`
- Array enumeration from `.zarray`
- Chunk enumeration for both `.` and `/` dimension separators
- `zarr_cells(path, array_path)` for dense numeric arrays
- Uncompressed and gzip-compressed chunk payloads
- Dynamic `(dim_0, ..., value)` projection based on array rank and dtype
- Projection-aware `zarr_cells()` scans
- Filter pushdown on dimension and value columns
- Chunk pruning from dimension filters before chunk decode
- Chunk-streamed execution with bounded scan-time memory instead of init-time row buffering
- Developer fixture generation and SQLLogic tests

The MVP does not yet support:

- Blosc chunk decode
- Missing-chunk fill-value materialization
- Remote stores
- Zarr v3 metadata
- Arrow materialization

## Quick Start

Build the extension:

```sh
make
```

Create the checked-in sample fixture again if needed:

```sh
make fixture
```

Run the extension tests:

```sh
make test_metadata
```

Open the DuckDB shell with the extension linked in:

```sh
./build/release/duckdb
```

Then query a sample store:

```sql
SELECT * FROM zarr_groups('test/data/simple_v2.zarr');
SELECT * FROM zarr_arrays('test/data/simple_v2.zarr');
SELECT * FROM zarr_chunks('test/data/simple_v2.zarr');
SELECT * FROM zarr_cells('test/data/simple_v2.zarr', 'temperature');
```

## Developer Workflow

The repository is still the standard DuckDB extension template, so the main commands are unchanged:

- `make`: build DuckDB and the extension
- `make test`: run SQLLogic tests
- `make fixture`: regenerate the sample Zarr store
- `make test_metadata`: regenerate the fixture and run tests

Useful build outputs:

- `./build/release/duckdb`
- `./build/release/test/unittest`
- `./build/release/extension/duckdb_zarr/duckdb_zarr.duckdb_extension`

## Architecture Mapping

The current code maps directly to the architecture document:

- Store Adapter: local filesystem traversal through DuckDB’s `FileSystem`
- Metadata Parser: `.zgroup` and `.zarray` parsing via `yyjson`
- DuckDB Bridge: table functions registered from the extension entrypoint
- Relational Cell Scan: `zarr_cells()` with pushed filters, projection-aware output, and dimension-based chunk pruning

The current `zarr_cells()` path now streams chunk-by-chunk at execution time, but there is still room to improve execution internals. The next major implementation step is to tighten the scan loop further with:

1. chunk-file reads
2. codec decode
3. typed value materialization
4. row projection `(dim_0, ..., value)`
5. Arrow-oriented batching and broader codec support

## Code Layout

- [`src/duckdb_zarr_extension.cpp`](/Users/buntend/Documents/work/duckdb_zarr/src/duckdb_zarr_extension.cpp): extension entrypoint and function registration
- [`src/zarr_metadata.cpp`](/Users/buntend/Documents/work/duckdb_zarr/src/zarr_metadata.cpp): metadata discovery and table function implementations
- [`scripts/create_sample_zarr.py`](/Users/buntend/Documents/work/duckdb_zarr/scripts/create_sample_zarr.py): reproducible fixture generation
- [`test/sql/duckdb_zarr.test`](/Users/buntend/Documents/work/duckdb_zarr/test/sql/duckdb_zarr.test): SQLLogic coverage for metadata, cell projection, and pushed-filter behavior

## Dependency Notes

No new third-party runtime dependencies were needed for the current implementation beyond what is already vendored in DuckDB.

For the next phase, useful dependency choices will likely be:

- a stable Zarr metadata/codec implementation strategy in C++
- blosc decode support for chunk payloads
- Arrow integration once `zarr_cells()` starts producing typed batches

If you want me to take on the next phase, the highest-value dependency discussion is around broader codec support and remote-store access. In practice that likely means deciding on Blosc support first, then choosing how much of the execution path should stay purely native C++ versus moving closer to an Arrow-oriented bridge for larger scans.
