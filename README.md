# duckdb_zarr

`duckdb_zarr` is a DuckDB extension for exploring Zarr stores with SQL through a relational projection.

The current implementation follows the project documents conservatively:

- `SPEC.md`: start with relational metadata and chunk discovery
- `ARCHITECTURE.md`: build the store adapter, metadata parser, and DuckDB bridge first
- `ROADMAP.md`: advance incrementally from metadata discovery to relational cell scans and planner-aware execution

Today’s extension provides metadata and relational scan table functions for local stores and a constrained remote read path:

- `zarr(path)` for a simple array overview
- `zarr(path, array_path)` as a convenience alias for `zarr_cells(path, array_path)`
- `zarr_groups(path)`
- `zarr_arrays(path)`
- `zarr_chunks(path)`
- `zarr_cells(path, array_path)` for Zarr dtypes `|b1`, `i1`, `i2`, `i4`, `i8`, `u1`, `u2`, `u4`, `u8`, `f2`, `f4`, and `f8`

This gives a usable SQL entrypoint for understanding a Zarr store and projecting dense numeric arrays into relational rows. Other dtypes such as complex values and other unsupported or non-standard Zarr dtypes are currently rejected by `zarr_cells(path, array_path)`.

## What Works

The MVP currently supports:

- Local filesystem Zarr v2 discovery
- Zarr v2 consolidated metadata discovery from `.zmetadata`
- Remote `http://`, `https://`, and `s3://` stores when DuckDB `httpfs` is available and the store is consolidated
- OME-Zarr-style multiscales groups built on Zarr v2, including level arrays such as `0`
- Group enumeration from `.zgroup`
- Array enumeration from `.zarray`
- Chunk enumeration for both `.` and `/` dimension separators
- `zarr_cells(path, array_path)` for dense arrays with dtypes `|b1`, `i1`, `i2`, `i4`, `i8`, `u1`, `u2`, `u4`, `u8`, `f2`, `f4`, and `f8`
- Uncompressed and gzip-compressed chunk payloads
- Missing-chunk fill-value materialization for supported `zarr_cells()` dtypes when `fill_value` is present
- Dynamic `(dim_0, ..., value)` projection based on array rank and dtype
- Projection-aware `zarr_cells()` scans
- Filter pushdown on dimension and value columns
- Chunk pruning from dimension filters before chunk decode
- Chunk-streamed execution with bounded scan-time memory instead of init-time row buffering
- Developer fixture generation and SQLLogic tests

The MVP does not yet support:

- `zarr_cells()` for dtypes outside `|b1`, `i1`, `i2`, `i4`, `i8`, `u1`, `u2`, `u4`, `u8`, `f2`, `f4`, and `f8`
- Blosc chunk decode
- Non-consolidated remote store discovery
- Zarr v3 metadata
- Arrow materialization

## Quick Start

Build the extension:

```sh
make
```

Install developer tooling for formatting and hooks:

```sh
python3 -m pip install pre-commit clang-format==11.0.1
pre-commit install
```

That single `pre-commit install` command now installs both:
- a `pre-commit` hook that auto-fixes DuckDB formatting
- a `pre-push` hook that runs the same formatter in `--check` mode before CI sees the push

Create the checked-in sample fixture again if needed:

```sh
make fixture
```

Run the extension tests:

```sh
make test_metadata
```

Run the Python smoke example with the same environment shape many Python users will use:

```sh
uvx --with duckdb==1.5.0 --with numpy --with pandas python test/python_smoke.py
```

There is also a matching make target:

```sh
make test_python_example
```

Package a static extension repository layout from downloaded CI artifacts:

```sh
python3 scripts/package_extension_repository.py \
  --artifacts-dir build/distribution-artifacts \
  --out-dir build/extension-repository \
  --extension-name duckdb_zarr \
  --duckdb-version v1.5.0
```

Run the formatter checks directly:

```sh
make format-check
```

Auto-fix formatting:

```sh
make format-fix
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
SELECT * FROM zarr('test/data/simple_v2.zarr');
SELECT * FROM zarr('test/data/ome_example.ome.zarr', '0');
```

For remote stores in this phase, the practical contract is:

- the store must expose Zarr v2 consolidated metadata at `.zmetadata`
- DuckDB must have `httpfs` available for the URI scheme you use
- chunk data are still read directly from remote chunk object paths

## Install Like An Extension

This repo now includes a release-asset publishing workflow in
[`PublishExtensionRepository.yml`](./.github/workflows/PublishExtensionRepository.yml)
that builds extension binaries on version tags and attaches platform-specific
`.duckdb_extension.gz` assets to the matching GitHub Release.

To make this work for your repository:

- push a version tag such as `v0.1.0`
- create the matching GitHub Release, or let the workflow create/update it from the tag

Then users can install directly from a release asset URL:

```sql
INSTALL 'https://github.com/d33bs/duckdb_zarr/releases/download/v0.1.0/duckdb_zarr-v1.5.0-osx_arm64.duckdb_extension.gz';
LOAD duckdb_zarr;
```

Each release publishes assets named like:

```text
duckdb_zarr-v1.5.0-osx_arm64.duckdb_extension.gz
duckdb_zarr-v1.5.0-linux_amd64.duckdb_extension.gz
```

## Release Checklist

To publish a new installable extension release from this repository:

1. Ensure the release workflow in
   [`PublishExtensionRepository.yml`](./.github/workflows/PublishExtensionRepository.yml)
   is enabled and passing on the default branch.
2. Create and push a version tag such as `v0.1.0`.
3. Wait for the publish workflow to:
   - build the extension binaries
   - package platform-specific `.duckdb_extension.gz` assets
   - upload those assets to the matching GitHub Release
4. Verify that a published artifact path exists, for example:
   `https://github.com/d33bs/duckdb_zarr/releases/download/v0.1.0/duckdb_zarr-v1.5.0-osx_arm64.duckdb_extension.gz`
5. Optionally add human-readable release notes. The installable path is the GitHub Release asset itself.

After that, users can install the extension with:

```sql
INSTALL 'https://github.com/d33bs/duckdb_zarr/releases/download/v0.1.0/duckdb_zarr-v1.5.0-osx_arm64.duckdb_extension.gz';
LOAD duckdb_zarr;
```

For remote Zarr stores, users may also need:

```sql
INSTALL httpfs;
LOAD httpfs;
```

## Developer Workflow

The repository is still the standard DuckDB extension template, so the main commands are unchanged:

- `make`: build DuckDB and the extension
- `make test`: run SQLLogic tests
- `make fixture`: regenerate the sample Zarr store
- `make test_metadata`: regenerate the fixture and run tests
- `make format-check`: verify DuckDB formatting for `src/` and `test/`
- `make format-fix`: apply DuckDB formatting for `src/` and `test/`

This repo also includes a pre-commit configuration in
[`\.pre-commit-config.yaml`](./.pre-commit-config.yaml)
that runs the DuckDB formatter automatically before commits. The formatter wrapper depends on `clang-format`,
and the current DuckDB script expects `clang-format==11.0.1`.

Useful build outputs:

- `./build/release/duckdb`
- `./build/release/test/unittest`
- `./build/release/extension/duckdb_zarr/duckdb_zarr.duckdb_extension`
- `./build/extension-repository/<duckdb-version>/<platform>/duckdb_zarr.duckdb_extension.gz`

## Architecture Mapping

The current code maps directly to the architecture document:

- Store Adapter: local filesystem traversal through DuckDB’s `FileSystem`
- Store Adapter: consolidated remote discovery for `http://`, `https://`, and `s3://` paths through DuckDB filesystem routing
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

- [`src/duckdb_zarr_extension.cpp`](./src/duckdb_zarr_extension.cpp): extension entrypoint and function registration
- [`src/zarr_metadata.cpp`](./src/zarr_metadata.cpp): metadata discovery and table function implementations
- [`scripts/create_sample_zarr.py`](./scripts/create_sample_zarr.py): reproducible fixture generation
- [`test/sql/duckdb_zarr.test`](./test/sql/duckdb_zarr.test): SQLLogic coverage for metadata, cell projection, and pushed-filter behavior

## Dependency Notes

No new third-party runtime dependencies were added in this repo. For remote access, `duckdb_zarr` relies on DuckDB’s `httpfs` extension being available to the runtime.

For the next phase, useful dependency choices will likely be:

- a stable Zarr metadata/codec implementation strategy in C++
- blosc decode support for chunk payloads
- Arrow integration once `zarr_cells()` starts producing typed batches

If you want me to take on the next phase, the highest-value dependency discussion is now around broader codec support, especially Blosc, and whether remote execution should grow from the current consolidated-metadata path into fuller object-store discovery plus caching.
