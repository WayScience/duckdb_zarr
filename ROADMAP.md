# duckdb_zarr Roadmap

## [x] Phase 1: Metadata
- [x] zarr_groups
- [x] zarr_arrays
- [x] local store discovery
- [x] local Zarr v3 metadata discovery

## [x] Phase 2: Cell Scan
- [x] zarr_cells
- [x] numeric types
- [x] gzip
- [x] automatic v2/v3 detection
- [x] explicit version override support
- [x] constrained Zarr v3 codec support (`bytes`, optional `gzip`, constrained `transpose`)
- [x] Zarr v3 `sharding_indexed` with Blosc (`zstd`) for local OME-Zarr-style arrays
- [~] broader blosc coverage

## [x] Phase 3: Optimization
- [x] chunk pruning
- [x] predicate pushdown
- [x] chunk-streamed execution

## [~] Phase 4: Remote
- [~] S3 / HTTP for consolidated stores
- [ ] non-consolidated Zarr v3 remote discovery
- [ ] caching

## [ ] Phase 5: Advanced
- [ ] chunk_values
- [~] convention adapters (`ome_arrow(...)` compatibility layer added for Zarr v3)
- [ ] parallel decode
