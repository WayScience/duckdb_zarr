# duckdb_zarr: Relational Projection of Zarr for DuckDB via Arrow

## Status
Draft MVP specification.

## Summary
duckdb_zarr is a DuckDB extension that exposes a relational projection over Zarr using Apache Arrow as the bridge.

## Core Idea
Zarr (chunked N-D arrays) → Arrow (columnar batches) → DuckDB (SQL)

## Goals
- SQL over Zarr
- Arrow as bridge
- Chunk-aware filtering
- Read-focused analytics

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
Zarr → decode → Arrow → DuckDB

## MVP Scope
- Dense arrays
- Numeric types
- Basic codecs (gzip, blosc)
- Local + remote (later)

## Future
- sparse arrays
- write support
- convention adapters
