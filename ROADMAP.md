# duckdb_zarr Roadmap

## [x] Phase 1: Metadata
- [x] zarr_groups
- [x] zarr_arrays
- [x] local store discovery

## [x] Phase 2: Cell Scan
- [x] zarr_cells
- [x] numeric types
- [~] gzip
- [ ] blosc

## [x] Phase 3: Optimization
- [x] chunk pruning
- [x] predicate pushdown
- [x] chunk-streamed execution

## [~] Phase 4: Remote
- [~] S3 / HTTP for consolidated stores
- [ ] caching

## [ ] Phase 5: Advanced
- [ ] chunk_values
- [ ] convention adapters
- [ ] parallel decode
