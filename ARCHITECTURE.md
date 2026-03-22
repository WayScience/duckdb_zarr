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
