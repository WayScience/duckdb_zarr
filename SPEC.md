# duckdb_zarr: Relational Projection of Zarr for DuckDB

## Status
Draft MVP specification.

## Summary
duckdb_zarr is a DuckDB extension that exposes a relational projection over Zarr for SQL analytics in DuckDB.

## Core Idea
Zarr (chunked N-D arrays) -> relational projection -> DuckDB (SQL)

Arrow remains a planned bridge for future execution improvements, but the current MVP decodes directly into DuckDB scan output.

## Goals
- SQL over Zarr
- Chunk-aware filtering
- Read-focused analytics
- Ergonomic relational access to array data

## Non-Goals
- Full xarray semantics
- Writes (MVP)
- Arbitrary codec support

## Core Tables
- zarr_groups()
- zarr_arrays()
- zarr_chunks()
- zarr_cells()

## Projection Model
Each array → rows:
(dim_0, dim_1, ..., value)

## Pushdown
- dimension filters → chunk pruning

## Execution Flow
Current MVP:
Zarr -> decode -> DuckDB table-function scan

Planned evolution:
Zarr -> decode -> Arrow/DuckDB batch bridge -> DuckDB

## MVP Scope
- Dense arrays
- Numeric types
- Basic codecs (gzip; blosc planned / not yet implemented)
- Local + remote (later)

## Future
- sparse arrays
- write support
- convention adapters
