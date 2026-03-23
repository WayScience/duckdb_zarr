#include "zarr_metadata.hpp"

#include "duckdb/common/helper.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/extension_helper.hpp"
#include "duckdb/planner/filter/conjunction_filter.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "duckdb/planner/filter/in_filter.hpp"
#include "duckdb/planner/filter/null_filter.hpp"
#include "miniz_wrapper.hpp"
#include "yyjson.hpp"

#include <algorithm>
#include <charconv>
#include <cstring>
#include <tuple>

namespace duckdb {

using namespace duckdb_yyjson; // NOLINT

struct ZarrGroupEntry {
	string store_path;
	string group_path;
	int64_t zarr_format;
	string metadata_path;
};

struct ZarrArrayEntry {
	string store_path;
	string array_path;
	int64_t zarr_format;
	int64_t rank;
	vector<int64_t> shape;
	vector<int64_t> chunks;
	string dtype;
	string order;
	string compressor;
	string dimension_separator;
	string metadata_path;
};

struct ZarrChunkEntry {
	string store_path;
	string array_path;
	string chunk_key;
	vector<int64_t> chunk_coords;
	string file_path;
	int64_t file_size_bytes;
};

struct ZarrNumericType {
	LogicalType logical_type;
	idx_t element_size;
	bool is_float;
	bool is_signed;
	bool is_unsigned;
	bool little_endian;
};

template <class ENTRY>
struct ZarrBindData : public TableFunctionData {
	explicit ZarrBindData(vector<ENTRY> entries_p) : entries(std::move(entries_p)) {
	}

	vector<ENTRY> entries;
};

struct ZarrGlobalState : public GlobalTableFunctionState {
	idx_t offset = 0;
};

struct ZarrCellsBindData : public TableFunctionData {
	ZarrCellsBindData(ZarrArrayEntry array_p, vector<ZarrChunkEntry> chunks_p)
	    : array(std::move(array_p)), chunks(std::move(chunks_p)) {
	}
	ZarrArrayEntry array;
	vector<ZarrChunkEntry> chunks;
};

struct ZarrCellsGlobalState : public GlobalTableFunctionState {
	vector<column_t> column_ids;
	vector<idx_t> projection_ids;
	unique_ptr<TableFilterSet> filters;
	vector<idx_t> chunk_indexes;
	ZarrNumericType dtype;
	idx_t chunk_element_count = 0;
	idx_t expected_chunk_bytes = 0;
	idx_t next_chunk_offset = 0;
	idx_t current_chunk_index = DConstants::INVALID_INDEX;
	idx_t current_linear_index = 0;
	vector<char> decoded_chunk;
};

static ZarrGroupEntry ParseGroupMetadata(yyjson_val *root, const string &store_path, const string &relative_path,
                                         const string &metadata_path);
static ZarrArrayEntry ParseArrayMetadataObject(yyjson_val *root, const string &store_path, const string &relative_path,
                                               const string &metadata_path);

static string JoinNodePath(const string &base, const string &name) {
	if (base.empty()) {
		return name;
	}
	return base + "/" + name;
}

static string FormatGroupPath(const string &relative_path) {
	return relative_path.empty() ? "/" : relative_path;
}

static string ReadTextFile(FileSystem &fs, const string &path) {
	auto handle = fs.OpenFile(path, FileFlags::FILE_FLAGS_READ);
	auto size = handle->GetFileSize();
	if (size == 0) {
		return "";
	}
	auto buffer = make_unsafe_uniq_array<char>(size);
	fs.Read(*handle, buffer.get(), NumericCast<int64_t>(size));
	return string(buffer.get(), size);
}

static vector<char> ReadBinaryFile(FileSystem &fs, const string &path) {
	auto handle = fs.OpenFile(path, FileFlags::FILE_FLAGS_READ);
	auto size = handle->GetFileSize();
	vector<char> buffer(size);
	if (size > 0) {
		fs.Read(*handle, buffer.data(), NumericCast<int64_t>(size));
	}
	return buffer;
}

static yyjson_val *RequireObjectKey(yyjson_val *obj, const char *key) {
	auto value = yyjson_obj_get(obj, key);
	if (!value) {
		throw InvalidInputException("Required Zarr metadata key \"%s\" was not found", key);
	}
	return value;
}

static int64_t RequireInt64(yyjson_val *obj, const char *key) {
	auto value = RequireObjectKey(obj, key);
	if (yyjson_is_uint(value)) {
		return NumericCast<int64_t>(unsafe_yyjson_get_uint(value));
	}
	if (yyjson_is_sint(value)) {
		return unsafe_yyjson_get_sint(value);
	}
	throw InvalidInputException("Zarr metadata key \"%s\" is not an integer", key);
}

static string OptionalString(yyjson_val *obj, const char *key, const string &default_value = "") {
	auto value = yyjson_obj_get(obj, key);
	if (!value || yyjson_is_null(value)) {
		return default_value;
	}
	if (!yyjson_is_str(value)) {
		throw InvalidInputException("Zarr metadata key \"%s\" is not a string", key);
	}
	return string(unsafe_yyjson_get_str(value), unsafe_yyjson_get_len(value));
}

static vector<int64_t> RequireInt64Array(yyjson_val *obj, const char *key) {
	auto value = RequireObjectKey(obj, key);
	if (!yyjson_is_arr(value)) {
		throw InvalidInputException("Zarr metadata key \"%s\" is not an array", key);
	}
	vector<int64_t> result;
	result.reserve(unsafe_yyjson_get_len(value));
	yyjson_val *element;
	yyjson_arr_iter iter = yyjson_arr_iter_with(value);
	while ((element = yyjson_arr_iter_next(&iter))) {
		if (yyjson_is_uint(element)) {
			result.push_back(NumericCast<int64_t>(unsafe_yyjson_get_uint(element)));
		} else if (yyjson_is_sint(element)) {
			result.push_back(unsafe_yyjson_get_sint(element));
		} else {
			throw InvalidInputException("Zarr metadata key \"%s\" contains a non-integer array element", key);
		}
	}
	return result;
}

static string JsonToString(yyjson_val *value) {
	if (!value || yyjson_is_null(value)) {
		return "";
	}
	size_t len = 0;
	auto rendered = yyjson_val_write(value, YYJSON_WRITE_NOFLAG, &len);
	if (!rendered) {
		throw InvalidInputException("Failed to render JSON metadata");
	}
	string result(rendered, len);
	free(rendered);
	return result;
}

static string NormalizeArrayPath(const string &path) {
	idx_t start = 0;
	idx_t end = path.size();
	while (start < end && path[start] == '/') {
		start++;
	}
	while (end > start && path[end - 1] == '/') {
		end--;
	}
	return path.substr(start, end - start);
}

static string NormalizeStorePath(FileSystem &fs, const string &path) {
	if (FileSystem::IsRemoteFile(path)) {
		return path;
	}
	return fs.ExpandPath(path);
}

static idx_t Product(const vector<int64_t> &values) {
	idx_t result = 1;
	for (auto value : values) {
		if (value < 0) {
			throw InvalidInputException("Negative Zarr dimensions are not supported");
		}
		result *= NumericCast<idx_t>(value);
	}
	return result;
}

static string ParseCompressorId(const string &compressor) {
	if (compressor.empty()) {
		return "";
	}
	unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)> doc(yyjson_read(compressor.c_str(), compressor.size(), 0),
	                                                       yyjson_doc_free);
	if (!doc) {
		throw InvalidInputException("Failed to parse Zarr compressor metadata");
	}
	auto root = yyjson_doc_get_root(doc.get());
	if (!yyjson_is_obj(root)) {
		throw InvalidInputException("Zarr compressor metadata is not an object");
	}
	return OptionalString(root, "id");
}

static bool HostIsLittleEndian() {
	uint16_t value = 1;
	return *reinterpret_cast<unsigned char *>(&value) == 1;
}

static ZarrNumericType ParseNumericDType(const string &dtype) {
	if (dtype.size() < 3) {
		throw InvalidInputException("Unsupported Zarr dtype: %s", dtype);
	}
	ZarrNumericType result;
	auto endian = dtype[0];
	auto kind = dtype[1];
	auto width = std::stoll(dtype.substr(2));
	result.element_size = NumericCast<idx_t>(width);
	result.is_float = false;
	result.is_signed = false;
	result.is_unsigned = false;
	switch (endian) {
	case '<':
		result.little_endian = true;
		break;
	case '>':
		result.little_endian = false;
		break;
	case '=':
		result.little_endian = HostIsLittleEndian();
		break;
	case '|':
		if (width != 1) {
			throw InvalidInputException("Endian-agnostic dtype is only supported for single-byte values: %s", dtype);
		}
		result.little_endian = HostIsLittleEndian();
		break;
	default:
		throw InvalidInputException("Unsupported Zarr dtype byte order marker in dtype: %s", dtype);
	}

	if (kind == 'i') {
		result.is_signed = true;
		switch (width) {
		case 1:
			result.logical_type = LogicalType::TINYINT;
			break;
		case 2:
			result.logical_type = LogicalType::SMALLINT;
			break;
		case 4:
			result.logical_type = LogicalType::INTEGER;
			break;
		case 8:
			result.logical_type = LogicalType::BIGINT;
			break;
		default:
			throw InvalidInputException("Unsupported signed integer dtype: %s", dtype);
		}
		return result;
	}
	if (kind == 'u') {
		result.is_unsigned = true;
		switch (width) {
		case 1:
			result.logical_type = LogicalType::UTINYINT;
			break;
		case 2:
			result.logical_type = LogicalType::USMALLINT;
			break;
		case 4:
			result.logical_type = LogicalType::UINTEGER;
			break;
		case 8:
			result.logical_type = LogicalType::UBIGINT;
			break;
		default:
			throw InvalidInputException("Unsupported unsigned integer dtype: %s", dtype);
		}
		return result;
	}
	if (kind == 'f') {
		result.is_float = true;
		switch (width) {
		case 4:
			result.logical_type = LogicalType::FLOAT;
			break;
		case 8:
			result.logical_type = LogicalType::DOUBLE;
			break;
		default:
			throw InvalidInputException("Unsupported floating-point dtype: %s", dtype);
		}
		return result;
	}
	throw InvalidInputException("Unsupported Zarr dtype: %s", dtype);
}

static uint64_t ReadUnsignedInteger(const char *ptr, idx_t bytes, bool little_endian) {
	uint64_t result = 0;
	if (little_endian) {
		for (idx_t i = 0; i < bytes; i++) {
			result |= static_cast<uint64_t>(static_cast<unsigned char>(ptr[i])) << (8 * i);
		}
	} else {
		for (idx_t i = 0; i < bytes; i++) {
			result = (result << 8) | static_cast<uint64_t>(static_cast<unsigned char>(ptr[i]));
		}
	}
	return result;
}

static int64_t ReadSignedInteger(const char *ptr, idx_t bytes, bool little_endian) {
	auto unsigned_value = ReadUnsignedInteger(ptr, bytes, little_endian);
	if (bytes == sizeof(int64_t)) {
		return static_cast<int64_t>(unsigned_value);
	}
	auto bit_width = bytes * 8;
	auto sign_bit = uint64_t(1) << (bit_width - 1);
	if (unsigned_value & sign_bit) {
		auto mask = ~((uint64_t(1) << bit_width) - 1);
		unsigned_value |= mask;
	}
	return static_cast<int64_t>(unsigned_value);
}

static Value DecodeNumericValue(const char *ptr, const ZarrNumericType &dtype) {
	if (dtype.is_float) {
		if (dtype.element_size == sizeof(float)) {
			auto bits = NumericCast<uint32_t>(ReadUnsignedInteger(ptr, dtype.element_size, dtype.little_endian));
			float value;
			std::memcpy(&value, &bits, sizeof(value));
			return Value::FLOAT(value);
		}
		if (dtype.element_size == sizeof(double)) {
			auto bits = ReadUnsignedInteger(ptr, dtype.element_size, dtype.little_endian);
			double value;
			std::memcpy(&value, &bits, sizeof(value));
			return Value::DOUBLE(value);
		}
	}
	if (dtype.is_signed) {
		auto value = ReadSignedInteger(ptr, dtype.element_size, dtype.little_endian);
		switch (dtype.element_size) {
		case 1:
			return Value::TINYINT(NumericCast<int8_t>(value));
		case 2:
			return Value::SMALLINT(NumericCast<int16_t>(value));
		case 4:
			return Value::INTEGER(NumericCast<int32_t>(value));
		case 8:
			return Value::BIGINT(value);
		default:
			break;
		}
	}
	if (dtype.is_unsigned) {
		auto value = ReadUnsignedInteger(ptr, dtype.element_size, dtype.little_endian);
		switch (dtype.element_size) {
		case 1:
			return Value::UTINYINT(NumericCast<uint8_t>(value));
		case 2:
			return Value::USMALLINT(NumericCast<uint16_t>(value));
		case 4:
			return Value::UINTEGER(NumericCast<uint32_t>(value));
		case 8:
			return Value::UBIGINT(value);
		default:
			break;
		}
	}
	throw InvalidInputException("Failed to decode Zarr numeric dtype");
}

static vector<int64_t> LinearToCoords(idx_t linear_index, const vector<int64_t> &shape, const string &order) {
	vector<int64_t> coordinates(shape.size(), 0);
	if (shape.empty()) {
		return coordinates;
	}
	if (order == "F") {
		for (idx_t i = 0; i < shape.size(); i++) {
			auto dim = NumericCast<idx_t>(shape[i]);
			coordinates[i] = NumericCast<int64_t>(linear_index % dim);
			linear_index /= dim;
		}
		return coordinates;
	}
	for (idx_t offset = 0; offset < shape.size(); offset++) {
		auto index = shape.size() - offset - 1;
		auto dim = NumericCast<idx_t>(shape[index]);
		coordinates[index] = NumericCast<int64_t>(linear_index % dim);
		linear_index /= dim;
	}
	return coordinates;
}

static vector<char> DecompressChunk(const vector<char> &compressed_data, const string &compressor,
                                    idx_t expected_size) {
	auto compressor_id = ParseCompressorId(compressor);
	if (compressor_id.empty()) {
		if (compressed_data.size() != expected_size) {
			throw InvalidInputException("Uncompressed Zarr chunk size mismatch: expected %llu bytes, got %llu bytes",
			                            expected_size, compressed_data.size());
		}
		return compressed_data;
	}
	if (compressor_id != "gzip") {
		throw InvalidInputException("Unsupported Zarr compressor for zarr_cells: %s", compressor_id);
	}
	vector<char> decompressed(expected_size);
	MiniZStream stream;
	stream.Decompress(compressed_data.data(), compressed_data.size(), decompressed.data(), decompressed.size());
	return decompressed;
}

static const ZarrArrayEntry &FindArrayEntry(const vector<ZarrArrayEntry> &arrays, const string &array_path) {
	for (idx_t i = 0; i < arrays.size(); i++) {
		if (arrays[i].array_path == array_path) {
			return arrays[i];
		}
	}
	throw InvalidInputException("Zarr array \"%s\" was not found in the store", array_path);
}

static bool MatchesFilter(const TableFilter &filter, const Value &value) {
	switch (filter.filter_type) {
	case TableFilterType::CONSTANT_COMPARISON:
		return filter.Cast<ConstantFilter>().Compare(value);
	case TableFilterType::IS_NULL:
		return value.IsNull();
	case TableFilterType::IS_NOT_NULL:
		return !value.IsNull();
	case TableFilterType::IN_FILTER: {
		auto &in_filter = filter.Cast<InFilter>();
		for (idx_t i = 0; i < in_filter.values.size(); i++) {
			if (Value::NotDistinctFrom(value, in_filter.values[i])) {
				return true;
			}
		}
		return false;
	}
	case TableFilterType::CONJUNCTION_AND: {
		auto &and_filter = filter.Cast<ConjunctionAndFilter>();
		for (idx_t i = 0; i < and_filter.child_filters.size(); i++) {
			if (!MatchesFilter(*and_filter.child_filters[i], value)) {
				return false;
			}
		}
		return true;
	}
	case TableFilterType::CONJUNCTION_OR: {
		auto &or_filter = filter.Cast<ConjunctionOrFilter>();
		for (idx_t i = 0; i < or_filter.child_filters.size(); i++) {
			if (MatchesFilter(*or_filter.child_filters[i], value)) {
				return true;
			}
		}
		return false;
	}
	default:
		throw InvalidInputException("Unsupported pushed filter type for zarr_cells");
	}
}

static bool ChunkMayMatchDimensionFilter(const TableFilter &filter, int64_t min_value, int64_t max_value) {
	switch (filter.filter_type) {
	case TableFilterType::CONSTANT_COMPARISON: {
		auto &constant_filter = filter.Cast<ConstantFilter>();
		auto constant = constant_filter.constant;
		if (constant.IsNull()) {
			return false;
		}
		auto constant_value = constant.DefaultCastAs(LogicalType::BIGINT).GetValue<int64_t>();
		switch (constant_filter.comparison_type) {
		case ExpressionType::COMPARE_EQUAL:
			return min_value <= constant_value && constant_value <= max_value;
		case ExpressionType::COMPARE_GREATERTHAN:
			return max_value > constant_value;
		case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
			return max_value >= constant_value;
		case ExpressionType::COMPARE_LESSTHAN:
			return min_value < constant_value;
		case ExpressionType::COMPARE_LESSTHANOREQUALTO:
			return min_value <= constant_value;
		case ExpressionType::COMPARE_NOTEQUAL:
			return min_value != max_value || min_value != constant_value;
		default:
			return true;
		}
	}
	case TableFilterType::IS_NULL:
		return false;
	case TableFilterType::IS_NOT_NULL:
		return true;
	case TableFilterType::IN_FILTER: {
		auto &in_filter = filter.Cast<InFilter>();
		for (idx_t i = 0; i < in_filter.values.size(); i++) {
			if (in_filter.values[i].IsNull()) {
				continue;
			}
			auto constant_value = in_filter.values[i].DefaultCastAs(LogicalType::BIGINT).GetValue<int64_t>();
			if (min_value <= constant_value && constant_value <= max_value) {
				return true;
			}
		}
		return false;
	}
	case TableFilterType::CONJUNCTION_AND: {
		auto &and_filter = filter.Cast<ConjunctionAndFilter>();
		for (idx_t i = 0; i < and_filter.child_filters.size(); i++) {
			if (!ChunkMayMatchDimensionFilter(*and_filter.child_filters[i], min_value, max_value)) {
				return false;
			}
		}
		return true;
	}
	case TableFilterType::CONJUNCTION_OR: {
		auto &or_filter = filter.Cast<ConjunctionOrFilter>();
		for (idx_t i = 0; i < or_filter.child_filters.size(); i++) {
			if (ChunkMayMatchDimensionFilter(*or_filter.child_filters[i], min_value, max_value)) {
				return true;
			}
		}
		return false;
	}
	default:
		return true;
	}
}

static column_t GetActualColumnIndex(const vector<column_t> &column_ids, idx_t filter_column_index) {
	if (filter_column_index < column_ids.size()) {
		return column_ids[filter_column_index];
	}
	return filter_column_index;
}

static bool RowMatchesFilters(optional_ptr<TableFilterSet> filters, const vector<column_t> &column_ids,
                              const vector<int64_t> &coordinates, const Value &value, idx_t rank) {
	if (!filters) {
		return true;
	}
	for (auto it = filters->filters.begin(); it != filters->filters.end(); it++) {
		auto column_index = GetActualColumnIndex(column_ids, it->first);
		if (column_index < rank) {
			if (!MatchesFilter(*it->second, Value::BIGINT(coordinates[column_index]))) {
				return false;
			}
		} else if (column_index == rank) {
			if (!MatchesFilter(*it->second, value)) {
				return false;
			}
		}
	}
	return true;
}

static bool ChunkMatchesFilters(optional_ptr<TableFilterSet> filters, const vector<column_t> &column_ids,
                                const ZarrArrayEntry &array, const ZarrChunkEntry &chunk) {
	if (!filters) {
		return true;
	}
	for (auto it = filters->filters.begin(); it != filters->filters.end(); it++) {
		auto column_index = GetActualColumnIndex(column_ids, it->first);
		if (column_index >= array.rank) {
			continue;
		}
		auto min_value = chunk.chunk_coords[column_index] * array.chunks[column_index];
		auto max_value = min_value + array.chunks[column_index] - 1;
		auto array_max = array.shape[column_index] - 1;
		if (max_value > array_max) {
			max_value = array_max;
		}
		if (!ChunkMayMatchDimensionFilter(*it->second, min_value, max_value)) {
			return false;
		}
	}
	return true;
}

static vector<idx_t> SelectFilteredChunks(const ZarrArrayEntry &array, const vector<ZarrChunkEntry> &all_chunks,
                                          optional_ptr<TableFilterSet> filters, const vector<column_t> &column_ids) {
	vector<idx_t> chunk_indexes;
	for (idx_t chunk_idx = 0; chunk_idx < all_chunks.size(); chunk_idx++) {
		auto &chunk = all_chunks[chunk_idx];
		if (chunk.array_path != array.array_path) {
			continue;
		}
		if (!ChunkMatchesFilters(filters, column_ids, array, chunk)) {
			continue;
		}
		chunk_indexes.push_back(chunk_idx);
	}
	return chunk_indexes;
}

static bool ComputeGlobalCoords(idx_t linear_index, const ZarrArrayEntry &array, const ZarrChunkEntry &chunk,
                                vector<int64_t> &global_coords) {
	auto local_coords = LinearToCoords(linear_index, array.chunks, array.order);
	global_coords.assign(local_coords.size(), 0);
	for (idx_t dim = 0; dim < local_coords.size(); dim++) {
		global_coords[dim] = chunk.chunk_coords[dim] * array.chunks[dim] + local_coords[dim];
		if (global_coords[dim] >= array.shape[dim]) {
			return false;
		}
	}
	return true;
}

static idx_t GetOutputColumnCount(const ZarrCellsGlobalState &state) {
	return state.projection_ids.empty() ? state.column_ids.size() : state.projection_ids.size();
}

static void WriteCellToOutput(const ZarrCellsBindData &bind_data, const ZarrCellsGlobalState &state,
                              const vector<int64_t> &coordinates, const Value &value, DataChunk &output,
                              idx_t row_idx) {
	auto output_column_count = GetOutputColumnCount(state);
	for (idx_t col = 0; col < output_column_count; col++) {
		auto base_col = state.projection_ids.empty() ? col : state.projection_ids[col];
		auto actual_col = state.column_ids[base_col];
		if (actual_col < bind_data.array.rank) {
			output.SetValue(col, row_idx, Value::BIGINT(coordinates[actual_col]));
		} else if (actual_col == bind_data.array.rank) {
			output.SetValue(col, row_idx, value);
		} else {
			throw InternalException("Unexpected zarr_cells column index");
		}
	}
}

static bool LoadNextChunk(FileSystem &fs, const ZarrCellsBindData &bind_data, ZarrCellsGlobalState &state) {
	while (state.next_chunk_offset < state.chunk_indexes.size()) {
		state.current_chunk_index = state.chunk_indexes[state.next_chunk_offset++];
		state.current_linear_index = 0;
		auto &chunk = bind_data.chunks[state.current_chunk_index];
		auto raw_data = ReadBinaryFile(fs, chunk.file_path);
		state.decoded_chunk = DecompressChunk(raw_data, bind_data.array.compressor, state.expected_chunk_bytes);
		return true;
	}
	state.current_chunk_index = DConstants::INVALID_INDEX;
	state.decoded_chunk.clear();
	return false;
}

static bool TryParseInteger(const string &text, int64_t &value) {
	if (text.empty()) {
		return false;
	}
	for (auto ch : text) {
		if (!StringUtil::CharacterIsDigit(ch)) {
			return false;
		}
	}
	int64_t parsed_value;
	auto result = std::from_chars(text.data(), text.data() + text.size(), parsed_value);
	if (result.ec != std::errc() || result.ptr != text.data() + text.size()) {
		return false;
	}
	value = parsed_value;
	return true;
}

static bool TryParseChunkCoords(const string &chunk_key, char separator, vector<int64_t> &coords) {
	auto parts = StringUtil::Split(chunk_key, separator);
	if (parts.empty()) {
		return false;
	}
	coords.clear();
	coords.reserve(parts.size());
	for (auto &part : parts) {
		int64_t coord;
		if (!TryParseInteger(part, coord)) {
			return false;
		}
		coords.push_back(coord);
	}
	return true;
}

static void CollectSlashChunks(FileSystem &fs, const ZarrArrayEntry &array, const string &dir_path,
                               const string &relative_key, vector<ZarrChunkEntry> &entries) {
	fs.ListFiles(dir_path, [&](const string &child_name, bool is_dir) {
		if (StringUtil::StartsWith(child_name, ".")) {
			return;
		}
		auto child_path = fs.JoinPath(dir_path, child_name);
		auto next_key = relative_key.empty() ? child_name : relative_key + "/" + child_name;
		if (is_dir) {
			CollectSlashChunks(fs, array, child_path, next_key, entries);
			return;
		}
		vector<int64_t> coords;
		if (!TryParseChunkCoords(next_key, '/', coords)) {
			return;
		}
		if (coords.size() != NumericCast<idx_t>(array.rank)) {
			return;
		}
		auto handle = fs.OpenFile(child_path, FileFlags::FILE_FLAGS_READ);
		entries.push_back({array.store_path, array.array_path, next_key, std::move(coords), child_path,
		                   NumericCast<int64_t>(handle->GetFileSize())});
	});
}

static void CollectDotChunks(FileSystem &fs, const ZarrArrayEntry &array, const string &dir_path,
                             vector<ZarrChunkEntry> &entries) {
	fs.ListFiles(dir_path, [&](const string &child_name, bool is_dir) {
		if (is_dir) {
			return;
		}
		if (StringUtil::StartsWith(child_name, ".")) {
			return;
		}
		auto child_path = fs.JoinPath(dir_path, child_name);
		vector<int64_t> coords;
		if (!TryParseChunkCoords(child_name, '.', coords)) {
			return;
		}
		if (coords.size() != NumericCast<idx_t>(array.rank)) {
			return;
		}
		auto handle = fs.OpenFile(child_path, FileFlags::FILE_FLAGS_READ);
		entries.push_back({array.store_path, array.array_path, child_name, std::move(coords), child_path,
		                   NumericCast<int64_t>(handle->GetFileSize())});
	});
}

static ZarrArrayEntry ParseArrayMetadata(FileSystem &fs, const string &store_path, const string &relative_path,
                                         const string &dir_path) {
	auto metadata_path = fs.JoinPath(dir_path, ".zarray");
	auto metadata_text = ReadTextFile(fs, metadata_path);
	unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)> doc(yyjson_read(metadata_text.c_str(), metadata_text.size(), 0),
	                                                       yyjson_doc_free);
	if (!doc) {
		throw InvalidInputException("Failed to parse %s", metadata_path);
	}
	return ParseArrayMetadataObject(yyjson_doc_get_root(doc.get()), store_path, relative_path, metadata_path);
}

static string ArrayDirectoryPath(FileSystem &fs, const string &store_path, const string &array_path) {
	if (array_path.empty()) {
		return store_path;
	}
	return fs.JoinPath(store_path, array_path);
}

static void CollectArrayChunks(FileSystem &fs, const ZarrArrayEntry &array, vector<ZarrChunkEntry> &chunks) {
	auto dir_path = ArrayDirectoryPath(fs, array.store_path, array.array_path);
	if (array.dimension_separator == "/") {
		CollectSlashChunks(fs, array, dir_path, "", chunks);
	} else {
		CollectDotChunks(fs, array, dir_path, chunks);
	}
}

static void TraverseStore(FileSystem &fs, const string &store_path, const string &dir_path, const string &relative_path,
                          bool collect_chunks, vector<ZarrGroupEntry> &groups, vector<ZarrArrayEntry> &arrays,
                          vector<ZarrChunkEntry> &chunks) {
	auto group_metadata = fs.JoinPath(dir_path, ".zgroup");
	auto array_metadata = fs.JoinPath(dir_path, ".zarray");

	if (fs.FileExists(group_metadata)) {
		auto metadata_text = ReadTextFile(fs, group_metadata);
		unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)> doc(
		    yyjson_read(metadata_text.c_str(), metadata_text.size(), 0), yyjson_doc_free);
		if (!doc) {
			throw InvalidInputException("Failed to parse %s", group_metadata);
		}
		groups.push_back(ParseGroupMetadata(yyjson_doc_get_root(doc.get()), store_path, relative_path, group_metadata));
	}

	if (fs.FileExists(array_metadata)) {
		auto entry = ParseArrayMetadata(fs, store_path, relative_path, dir_path);
		if (collect_chunks) {
			CollectArrayChunks(fs, entry, chunks);
		}
		arrays.push_back(std::move(entry));
		return;
	}

	fs.ListFiles(dir_path, [&](const string &child_name, bool is_dir) {
		if (!is_dir) {
			return;
		}
		if (StringUtil::StartsWith(child_name, ".")) {
			return;
		}
		auto child_path = fs.JoinPath(dir_path, child_name);
		TraverseStore(fs, store_path, child_path, JoinNodePath(relative_path, child_name), collect_chunks, groups,
		              arrays, chunks);
	});
}

static vector<Value> ToBigIntValues(const vector<int64_t> &values) {
	vector<Value> result;
	result.reserve(values.size());
	for (auto value : values) {
		result.emplace_back(Value::BIGINT(value));
	}
	return result;
}

static ZarrGroupEntry ParseGroupMetadata(yyjson_val *root, const string &store_path, const string &relative_path,
                                         const string &metadata_path) {
	if (!yyjson_is_obj(root)) {
		throw InvalidInputException("%s does not contain a JSON object", metadata_path);
	}
	return {store_path, FormatGroupPath(relative_path), RequireInt64(root, "zarr_format"), metadata_path};
}

static ZarrArrayEntry ParseArrayMetadataObject(yyjson_val *root, const string &store_path, const string &relative_path,
                                               const string &metadata_path) {
	if (!yyjson_is_obj(root)) {
		throw InvalidInputException("%s does not contain a JSON object", metadata_path);
	}
	auto compressor = JsonToString(yyjson_obj_get(root, "compressor"));
	auto dimension_separator = OptionalString(root, "dimension_separator", ".");
	auto shape = RequireInt64Array(root, "shape");
	auto chunks = RequireInt64Array(root, "chunks");
	if (shape.size() != chunks.size()) {
		throw InvalidInputException("%s has mismatched shape/chunks rank", metadata_path);
	}
	for (idx_t i = 0; i < shape.size(); i++) {
		if (shape[i] < 0) {
			throw InvalidInputException("%s contains a negative shape dimension", metadata_path);
		}
		if (chunks[i] <= 0) {
			throw InvalidInputException("%s contains a non-positive chunk dimension", metadata_path);
		}
	}

	return {store_path,
	        relative_path,
	        RequireInt64(root, "zarr_format"),
	        NumericCast<int64_t>(shape.size()),
	        std::move(shape),
	        std::move(chunks),
	        OptionalString(root, "dtype"),
	        OptionalString(root, "order"),
	        std::move(compressor),
	        std::move(dimension_separator),
	        metadata_path};
}

static string MetadataPathForKey(const string &store_path, const string &key) {
	return store_path + "/.zmetadata#" + key;
}

static string RelativePathFromMetadataKey(const string &key, const string &suffix) {
	if (key == suffix) {
		return "";
	}
	auto key_suffix = "/" + suffix;
	if (!StringUtil::EndsWith(key, key_suffix)) {
		throw InvalidInputException("Unexpected consolidated metadata key: %s", key);
	}
	return key.substr(0, key.size() - key_suffix.size());
}

static string ChunkKeyFromCoords(const vector<int64_t> &coords, const string &dimension_separator) {
	string separator = dimension_separator == "/" ? "/" : ".";
	string result;
	for (idx_t i = 0; i < coords.size(); i++) {
		if (i > 0) {
			result += separator;
		}
		result += std::to_string(coords[i]);
	}
	return result;
}

static string ChunkPath(const string &store_path, const string &array_path, const string &chunk_key) {
	auto path = store_path;
	if (!array_path.empty()) {
		path += "/" + array_path;
	}
	if (!chunk_key.empty()) {
		path += "/" + chunk_key;
	}
	return path;
}

static void GenerateChunkEntriesRecursive(FileSystem &fs, const string &store_path, const ZarrArrayEntry &array,
                                          vector<int64_t> &coords, idx_t dim, vector<ZarrChunkEntry> &chunks) {
	if (dim == coords.size()) {
		auto chunk_key = ChunkKeyFromCoords(coords, array.dimension_separator);
		auto file_path = ChunkPath(store_path, array.array_path, chunk_key);
		if (!fs.FileExists(file_path)) {
			return;
		}
		auto handle = fs.OpenFile(file_path, FileFlags::FILE_FLAGS_READ);
		chunks.push_back(
		    {store_path, array.array_path, chunk_key, coords, file_path, NumericCast<int64_t>(handle->GetFileSize())});
		return;
	}
	auto chunk_count = (array.shape[dim] + array.chunks[dim] - 1) / array.chunks[dim];
	for (int64_t chunk_coord = 0; chunk_coord < chunk_count; chunk_coord++) {
		coords[dim] = chunk_coord;
		GenerateChunkEntriesRecursive(fs, store_path, array, coords, dim + 1, chunks);
	}
}

static vector<ZarrChunkEntry> GenerateChunkEntries(FileSystem &fs, const string &store_path,
                                                   const ZarrArrayEntry &array) {
	vector<ZarrChunkEntry> chunks;
	vector<int64_t> coords(NumericCast<idx_t>(array.rank), 0);
	GenerateChunkEntriesRecursive(fs, store_path, array, coords, 0, chunks);
	return chunks;
}

static bool DiscoverConsolidatedStore(FileSystem &fs, const string &store_path, vector<ZarrGroupEntry> &groups,
                                      vector<ZarrArrayEntry> &arrays, vector<ZarrChunkEntry> &chunks,
                                      bool collect_chunks) {
	auto metadata_path = store_path + "/.zmetadata";
	if (!fs.FileExists(metadata_path)) {
		return false;
	}
	auto metadata_text = ReadTextFile(fs, metadata_path);
	unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)> doc(yyjson_read(metadata_text.c_str(), metadata_text.size(), 0),
	                                                       yyjson_doc_free);
	if (!doc) {
		throw InvalidInputException("Failed to parse %s", metadata_path);
	}
	auto root = yyjson_doc_get_root(doc.get());
	if (!yyjson_is_obj(root)) {
		throw InvalidInputException("%s does not contain a JSON object", metadata_path);
	}
	auto metadata_obj = RequireObjectKey(root, "metadata");
	if (!yyjson_is_obj(metadata_obj)) {
		throw InvalidInputException("%s metadata key is not an object", metadata_path);
	}
	yyjson_obj_iter iter = yyjson_obj_iter_with(metadata_obj);
	yyjson_val *entry_key;
	while ((entry_key = yyjson_obj_iter_next(&iter))) {
		auto entry_value = yyjson_obj_iter_get_val(entry_key);
		string key(unsafe_yyjson_get_str(entry_key), unsafe_yyjson_get_len(entry_key));
		if (StringUtil::EndsWith(key, ".zgroup")) {
			auto relative_path = RelativePathFromMetadataKey(key, ".zgroup");
			groups.push_back(
			    ParseGroupMetadata(entry_value, store_path, relative_path, MetadataPathForKey(store_path, key)));
		} else if (StringUtil::EndsWith(key, ".zarray")) {
			auto relative_path = RelativePathFromMetadataKey(key, ".zarray");
			auto array =
			    ParseArrayMetadataObject(entry_value, store_path, relative_path, MetadataPathForKey(store_path, key));
			if (collect_chunks) {
				auto array_chunks = GenerateChunkEntries(fs, store_path, array);
				chunks.insert(chunks.end(), array_chunks.begin(), array_chunks.end());
			}
			arrays.push_back(std::move(array));
		}
	}
	return true;
}

static void EnsureRemoteFilesystemSupport(ClientContext &context, const string &path) {
	if (!FileSystem::IsRemoteFile(path)) {
		return;
	}
	if (!ExtensionHelper::TryAutoLoadExtension(context, "httpfs")) {
		throw InvalidInputException(
		    "Remote Zarr access requires DuckDB's httpfs extension to be available for path: %s", path);
	}
}

static vector<ZarrGroupEntry> DiscoverGroups(ClientContext &context, const string &path) {
	FileSystem &fs = FileSystem::GetFileSystem(context);
	EnsureRemoteFilesystemSupport(context, path);
	auto store_path = NormalizeStorePath(fs, path);
	vector<ZarrGroupEntry> groups;
	vector<ZarrArrayEntry> arrays;
	vector<ZarrChunkEntry> chunks;
	if (!DiscoverConsolidatedStore(fs, store_path, groups, arrays, chunks, false)) {
		if (!fs.DirectoryExists(store_path)) {
			if (FileSystem::IsRemoteFile(path)) {
				throw InvalidInputException(
				    "Remote Zarr stores currently require consolidated metadata (.zmetadata): %s", store_path);
			}
			throw InvalidInputException("Zarr store path does not exist or is not a directory: %s", store_path);
		}
		TraverseStore(fs, store_path, store_path, "", false, groups, arrays, chunks);
	}
	std::sort(groups.begin(), groups.end(), [](const ZarrGroupEntry &lhs, const ZarrGroupEntry &rhs) {
		return std::tie(lhs.group_path, lhs.metadata_path) < std::tie(rhs.group_path, rhs.metadata_path);
	});
	return groups;
}

static void DiscoverStore(ClientContext &context, const string &path, vector<ZarrGroupEntry> &groups,
                          vector<ZarrArrayEntry> &arrays, vector<ZarrChunkEntry> &chunks, bool collect_chunks) {
	FileSystem &fs = FileSystem::GetFileSystem(context);
	EnsureRemoteFilesystemSupport(context, path);
	auto store_path = NormalizeStorePath(fs, path);
	if (!DiscoverConsolidatedStore(fs, store_path, groups, arrays, chunks, collect_chunks)) {
		if (!fs.DirectoryExists(store_path)) {
			if (FileSystem::IsRemoteFile(path)) {
				throw InvalidInputException(
				    "Remote Zarr stores currently require consolidated metadata (.zmetadata): %s", store_path);
			}
			throw InvalidInputException("Zarr store path does not exist or is not a directory: %s", store_path);
		}
		TraverseStore(fs, store_path, store_path, "", collect_chunks, groups, arrays, chunks);
	}
	std::sort(groups.begin(), groups.end(), [](const ZarrGroupEntry &lhs, const ZarrGroupEntry &rhs) {
		return std::tie(lhs.group_path, lhs.metadata_path) < std::tie(rhs.group_path, rhs.metadata_path);
	});
	std::sort(arrays.begin(), arrays.end(), [](const ZarrArrayEntry &lhs, const ZarrArrayEntry &rhs) {
		return std::tie(lhs.array_path, lhs.metadata_path) < std::tie(rhs.array_path, rhs.metadata_path);
	});
	std::sort(chunks.begin(), chunks.end(), [](const ZarrChunkEntry &lhs, const ZarrChunkEntry &rhs) {
		return std::tie(lhs.array_path, lhs.chunk_key, lhs.file_path) <
		       std::tie(rhs.array_path, rhs.chunk_key, rhs.file_path);
	});
}

static unique_ptr<GlobalTableFunctionState> ZarrInit(ClientContext &, TableFunctionInitInput &) {
	return make_uniq<ZarrGlobalState>();
}

static unique_ptr<FunctionData> BindGroups(ClientContext &context, TableFunctionBindInput &input,
                                           vector<LogicalType> &return_types, vector<string> &names) {
	if (input.inputs.empty() || input.inputs[0].IsNull()) {
		throw BinderException("zarr_groups requires a non-NULL store path");
	}
	names = {"store_path", "group_path", "zarr_format", "metadata_path"};
	return_types = {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::BIGINT, LogicalType::VARCHAR};
	return make_uniq<ZarrBindData<ZarrGroupEntry>>(DiscoverGroups(context, StringValue::Get(input.inputs[0])));
}

static unique_ptr<FunctionData> BindArrays(ClientContext &context, TableFunctionBindInput &input,
                                           vector<LogicalType> &return_types, vector<string> &names) {
	if (input.inputs.empty() || input.inputs[0].IsNull()) {
		throw BinderException("zarr_arrays requires a non-NULL store path");
	}
	vector<ZarrGroupEntry> groups;
	vector<ZarrArrayEntry> arrays;
	vector<ZarrChunkEntry> chunks;
	DiscoverStore(context, StringValue::Get(input.inputs[0]), groups, arrays, chunks, false);

	names = {"store_path", "array_path", "zarr_format",         "rank",         "shape", "chunk_shape", "dtype",
	         "order",      "compressor", "dimension_separator", "metadata_path"};
	return_types = {LogicalType::VARCHAR,
	                LogicalType::VARCHAR,
	                LogicalType::BIGINT,
	                LogicalType::BIGINT,
	                LogicalType::LIST(LogicalType::BIGINT),
	                LogicalType::LIST(LogicalType::BIGINT),
	                LogicalType::VARCHAR,
	                LogicalType::VARCHAR,
	                LogicalType::VARCHAR,
	                LogicalType::VARCHAR,
	                LogicalType::VARCHAR};
	return make_uniq<ZarrBindData<ZarrArrayEntry>>(std::move(arrays));
}

static unique_ptr<FunctionData> BindChunks(ClientContext &context, TableFunctionBindInput &input,
                                           vector<LogicalType> &return_types, vector<string> &names) {
	if (input.inputs.empty() || input.inputs[0].IsNull()) {
		throw BinderException("zarr_chunks requires a non-NULL store path");
	}
	vector<ZarrGroupEntry> groups;
	vector<ZarrArrayEntry> arrays;
	vector<ZarrChunkEntry> chunks;
	DiscoverStore(context, StringValue::Get(input.inputs[0]), groups, arrays, chunks, true);

	names = {"store_path", "array_path", "chunk_key", "chunk_coords", "file_path", "file_size_bytes"};
	return_types = {LogicalType::VARCHAR, LogicalType::VARCHAR,
	                LogicalType::VARCHAR, LogicalType::LIST(LogicalType::BIGINT),
	                LogicalType::VARCHAR, LogicalType::BIGINT};
	return make_uniq<ZarrBindData<ZarrChunkEntry>>(std::move(chunks));
}

static unique_ptr<FunctionData> BindCells(ClientContext &context, TableFunctionBindInput &input,
                                          vector<LogicalType> &return_types, vector<string> &names) {
	if (input.inputs.size() < 2 || input.inputs[0].IsNull() || input.inputs[1].IsNull()) {
		throw BinderException("zarr_cells requires a non-NULL store path and array path");
	}

	auto store_path = StringValue::Get(input.inputs[0]);
	auto array_path = NormalizeArrayPath(StringValue::Get(input.inputs[1]));

	vector<ZarrGroupEntry> groups;
	vector<ZarrArrayEntry> arrays;
	vector<ZarrChunkEntry> chunks;
	DiscoverStore(context, store_path, groups, arrays, chunks, false);
	auto &array = FindArrayEntry(arrays, array_path);
	auto dtype = ParseNumericDType(array.dtype);
	chunks = GenerateChunkEntries(FileSystem::GetFileSystem(context), array.store_path, array);

	names.reserve(array.rank + 1);
	return_types.reserve(array.rank + 1);
	for (idx_t dim = 0; dim < NumericCast<idx_t>(array.rank); dim++) {
		names.push_back("dim_" + std::to_string(dim));
		return_types.push_back(LogicalType::BIGINT);
	}
	names.push_back("value");
	return_types.push_back(dtype.logical_type);

	return make_uniq<ZarrCellsBindData>(array, std::move(chunks));
}

static void ScanGroups(ClientContext &, TableFunctionInput &data_p, DataChunk &output) {
	auto &state = data_p.global_state->Cast<ZarrGlobalState>();
	auto &entries = data_p.bind_data->Cast<ZarrBindData<ZarrGroupEntry>>().entries;
	idx_t count = 0;
	while (state.offset < entries.size() && count < STANDARD_VECTOR_SIZE) {
		auto &entry = entries[state.offset++];
		output.SetValue(0, count, Value(entry.store_path));
		output.SetValue(1, count, Value(entry.group_path));
		output.SetValue(2, count, Value::BIGINT(entry.zarr_format));
		output.SetValue(3, count, Value(entry.metadata_path));
		count++;
	}
	output.SetCardinality(count);
}

static void ScanArrays(ClientContext &, TableFunctionInput &data_p, DataChunk &output) {
	auto &state = data_p.global_state->Cast<ZarrGlobalState>();
	auto &entries = data_p.bind_data->Cast<ZarrBindData<ZarrArrayEntry>>().entries;
	idx_t count = 0;
	while (state.offset < entries.size() && count < STANDARD_VECTOR_SIZE) {
		auto &entry = entries[state.offset++];
		output.SetValue(0, count, Value(entry.store_path));
		output.SetValue(1, count, Value(entry.array_path));
		output.SetValue(2, count, Value::BIGINT(entry.zarr_format));
		output.SetValue(3, count, Value::BIGINT(entry.rank));
		output.SetValue(4, count, Value::LIST(LogicalType::BIGINT, ToBigIntValues(entry.shape)));
		output.SetValue(5, count, Value::LIST(LogicalType::BIGINT, ToBigIntValues(entry.chunks)));
		output.SetValue(6, count, Value(entry.dtype));
		output.SetValue(7, count, Value(entry.order));
		output.SetValue(8, count, Value(entry.compressor));
		output.SetValue(9, count, Value(entry.dimension_separator));
		output.SetValue(10, count, Value(entry.metadata_path));
		count++;
	}
	output.SetCardinality(count);
}

static void ScanChunks(ClientContext &, TableFunctionInput &data_p, DataChunk &output) {
	auto &state = data_p.global_state->Cast<ZarrGlobalState>();
	auto &entries = data_p.bind_data->Cast<ZarrBindData<ZarrChunkEntry>>().entries;
	idx_t count = 0;
	while (state.offset < entries.size() && count < STANDARD_VECTOR_SIZE) {
		auto &entry = entries[state.offset++];
		output.SetValue(0, count, Value(entry.store_path));
		output.SetValue(1, count, Value(entry.array_path));
		output.SetValue(2, count, Value(entry.chunk_key));
		output.SetValue(3, count, Value::LIST(LogicalType::BIGINT, ToBigIntValues(entry.chunk_coords)));
		output.SetValue(4, count, Value(entry.file_path));
		output.SetValue(5, count, Value::BIGINT(entry.file_size_bytes));
		count++;
	}
	output.SetCardinality(count);
}

static void ScanCells(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &state = data_p.global_state->Cast<ZarrCellsGlobalState>();
	auto &bind_data = data_p.bind_data->Cast<ZarrCellsBindData>();
	FileSystem &fs = FileSystem::GetFileSystem(context);
	idx_t count = 0;
	vector<int64_t> global_coords;
	while (count < STANDARD_VECTOR_SIZE) {
		if (state.current_chunk_index == DConstants::INVALID_INDEX ||
		    state.current_linear_index >= state.chunk_element_count) {
			if (!LoadNextChunk(fs, bind_data, state)) {
				break;
			}
		}
		auto &chunk = bind_data.chunks[state.current_chunk_index];
		while (state.current_linear_index < state.chunk_element_count && count < STANDARD_VECTOR_SIZE) {
			auto linear_index = state.current_linear_index++;
			if (!ComputeGlobalCoords(linear_index, bind_data.array, chunk, global_coords)) {
				continue;
			}
			auto value_ptr = state.decoded_chunk.data() + (linear_index * state.dtype.element_size);
			auto cell_value = DecodeNumericValue(value_ptr, state.dtype);
			if (!RowMatchesFilters(state.filters.get(), state.column_ids, global_coords, cell_value,
			                       NumericCast<idx_t>(bind_data.array.rank))) {
				continue;
			}
			WriteCellToOutput(bind_data, state, global_coords, cell_value, output, count);
			count++;
		}
	}
	output.SetCardinality(count);
}

static unique_ptr<GlobalTableFunctionState> InitCells(ClientContext &context, TableFunctionInitInput &input) {
	auto &bind_data = input.bind_data->Cast<ZarrCellsBindData>();
	auto result = make_uniq<ZarrCellsGlobalState>();
	result->column_ids = input.column_ids;
	result->projection_ids = input.projection_ids;
	result->filters = input.filters ? input.filters->Copy() : nullptr;
	result->chunk_indexes = SelectFilteredChunks(bind_data.array, bind_data.chunks, input.filters, input.column_ids);
	result->dtype = ParseNumericDType(bind_data.array.dtype);
	result->chunk_element_count = Product(bind_data.array.chunks);
	result->expected_chunk_bytes = result->chunk_element_count * result->dtype.element_size;
	return std::move(result);
}

TableFunction ZarrMetadata::GetGroupsFunction() {
	return TableFunction("zarr_groups", {LogicalType::VARCHAR}, ScanGroups, BindGroups, ZarrInit);
}

TableFunction ZarrMetadata::GetArraysFunction() {
	return TableFunction("zarr_arrays", {LogicalType::VARCHAR}, ScanArrays, BindArrays, ZarrInit);
}

TableFunction ZarrMetadata::GetChunksFunction() {
	return TableFunction("zarr_chunks", {LogicalType::VARCHAR}, ScanChunks, BindChunks, ZarrInit);
}

TableFunction ZarrMetadata::GetCellsFunction() {
	TableFunction function("zarr_cells", {LogicalType::VARCHAR, LogicalType::VARCHAR}, ScanCells, BindCells, InitCells);
	function.projection_pushdown = true;
	function.filter_pushdown = true;
	function.filter_prune = true;
	return function;
}

} // namespace duckdb
