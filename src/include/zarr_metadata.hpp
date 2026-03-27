#pragma once

#include "duckdb/function/table_function.hpp"

namespace duckdb {

class ZarrMetadata {
public:
	static TableFunction GetGroupsFunction();
	static TableFunction GetArraysFunction();
	static TableFunction GetChunksFunction();
	static TableFunction GetCellsFunction();
	static TableFunction GetZarrFunction();
	static TableFunction GetZarrCellsAliasFunction();
};

} // namespace duckdb
