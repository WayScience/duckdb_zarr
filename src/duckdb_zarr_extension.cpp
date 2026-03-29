#define DUCKDB_EXTENSION_MAIN

#include "duckdb_zarr_extension.hpp"
#include "zarr_metadata.hpp"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {

static void LoadInternal(ExtensionLoader &loader) {
	loader.RegisterFunction(ZarrMetadata::GetGroupsFunction());
	loader.RegisterFunction(ZarrMetadata::GetArraysFunction());
	loader.RegisterFunction(ZarrMetadata::GetChunksFunction());
	loader.RegisterFunction(ZarrMetadata::GetCellsFunction());
	loader.RegisterFunction(ZarrMetadata::GetZarrFunction());
	loader.RegisterFunction(ZarrMetadata::GetZarrCellsAliasFunction());
	loader.RegisterFunction(ZarrMetadata::GetOmeArrowFunction());
}

void DuckdbZarrExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}

std::string DuckdbZarrExtension::Name() {
	return "duckdb_zarr";
}

std::string DuckdbZarrExtension::Version() const {
#ifdef EXT_VERSION_DUCKDB_ZARR
	return EXT_VERSION_DUCKDB_ZARR;
#else
	return "";
#endif
}

} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(duckdb_zarr, loader) {
	duckdb::LoadInternal(loader);
}
}
