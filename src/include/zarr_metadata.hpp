#pragma once

#include "duckdb/function/function_set.hpp"
#include "duckdb/function/table_function.hpp"

namespace duckdb {

class ZarrMetadata {
public:
	static TableFunctionSet GetGroupsFunction();
	static TableFunctionSet GetArraysFunction();
	static TableFunctionSet GetChunksFunction();
	static TableFunctionSet GetCellsFunction();
	static TableFunctionSet GetZarrFunction();
	static TableFunctionSet GetZarrCellsAliasFunction();
	static TableFunctionSet GetOmeArrowFunction();
};

} // namespace duckdb
