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

- Implemented now: local Zarr v2 metadata discovery, consolidated `.zmetadata` support, gzip/uncompressed numeric chunk decode, and DuckDB table functions (`zarr_groups`, `zarr_arrays`, `zarr_chunks`, `zarr_cells`) with projection-aware scans and chunk pruning.
- Partially implemented: the Store Adapter includes practical remote support for consolidated HTTP/S3-style stores via DuckDB `httpfs`, and the Planner path performs predicate pushdown for the current `zarr_cells()` scan.
- Planned: the Arrow Materializer, broader codec chains such as Blosc, richer convention adapters, and the fuller target flow described below.

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
