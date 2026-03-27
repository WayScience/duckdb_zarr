PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
.DEFAULT_GOAL := all

# Configuration of extension
EXT_NAME=duckdb_zarr
EXT_CONFIG=${PROJ_DIR}extension_config.cmake

.PHONY: fixture test_metadata test_python_example

fixture:
	python3 scripts/create_sample_zarr.py

test_metadata: fixture
	${MAKE} test

test_python_example: fixture
	uvx --with duckdb==1.5.0 --with numpy --with pandas python test/python_smoke.py

# Include the Makefile from extension-ci-tools
include extension-ci-tools/makefiles/duckdb_extension.Makefile
