# duckdb_zarr Architecture

## Components

1. Store Adapter
- Local FS
- HTTP/S
- Object storage

2. Metadata Parser
- Zarr v2/v3
- Array descriptors

3. Codec Pipeline
- Decode chunks
- Apply codec chain

4. Arrow Materializer
- Convert to Arrow batches

5. DuckDB Bridge
- Table functions
- Scan integration

6. Planner
- Predicate pushdown
- Chunk pruning

## Current MVP

- Implemented now: local Zarr v2 and v3 metadata discovery, Zarr v2 consolidated `.zmetadata` support, gzip/uncompressed numeric chunk decode, and DuckDB table functions (`zarr_groups`, `zarr_arrays`, `zarr_chunks`, `zarr_cells`, `zarr`) with projection-aware scans and chunk pruning.
- Partially implemented: the Store Adapter includes practical remote support for consolidated HTTP/S3-style Zarr v2 stores via DuckDB `httpfs`, the Codec Pipeline now handles the current v3 subset of `bytes`, optional `gzip`, constrained `transpose`, and `sharding_indexed` with inner Blosc (`zstd`) for common OME-Zarr image and label data, and a thin `ome_arrow(...)` adapter exposes v3 arrays with dimension-aware relational columns.
- Planned: the Arrow Materializer, broader codec chains beyond the current Blosc path, richer convention adapters beyond the current `ome_arrow(...)` layer, and the fuller target flow described below.

## Flow

User SQL
  ↓
DuckDB planner
  ↓
zarr_cells()
  ↓
Chunk selection
  ↓
Decode chunks
  ↓
Arrow batches
  ↓
DuckDB execution
