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
#include "zstd.h"

#include <algorithm>
#include <cstring>
#include <limits>
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
	ZarrArrayEntry() {
	}
	ZarrArrayEntry(string store_path_p, string array_path_p, int64_t zarr_format_p, int64_t rank_p,
	               vector<int64_t> shape_p, vector<int64_t> chunks_p, string dtype_p, string order_p,
	               string compressor_p, string fill_value_p, string chunk_key_encoding_p,
	               string dimension_separator_p, string metadata_path_p, bool supports_cells_p,
	               string cells_error_p, bool is_sharding_indexed_p, vector<int64_t> storage_chunks_p,
	               vector<int64_t> inner_chunks_p,
	               string inner_compressor_p, string index_codecs_p, string index_location_p)
	    : store_path(std::move(store_path_p)), array_path(std::move(array_path_p)), zarr_format(zarr_format_p),
	      rank(rank_p), shape(std::move(shape_p)), chunks(std::move(chunks_p)), dtype(std::move(dtype_p)),
	      order(std::move(order_p)), compressor(std::move(compressor_p)), fill_value(std::move(fill_value_p)),
	      chunk_key_encoding(std::move(chunk_key_encoding_p)),
	      dimension_separator(std::move(dimension_separator_p)), metadata_path(std::move(metadata_path_p)),
	      supports_cells(supports_cells_p), cells_error(std::move(cells_error_p)),
	      is_sharding_indexed(is_sharding_indexed_p), storage_chunks(std::move(storage_chunks_p)),
	      inner_chunks(std::move(inner_chunks_p)),
	      inner_compressor(std::move(inner_compressor_p)), index_codecs(std::move(index_codecs_p)),
	      index_location(std::move(index_location_p)) {
	}
	string store_path;
	string array_path;
	int64_t zarr_format;
	int64_t rank;
	vector<int64_t> shape;
	vector<int64_t> chunks;
	string dtype;
	string order;
	string compressor;
	string fill_value;
	string chunk_key_encoding;
	string dimension_separator;
	string metadata_path;
	bool supports_cells;
	string cells_error;
	bool is_sharding_indexed = false;
	vector<int64_t> storage_chunks;
	vector<int64_t> inner_chunks;
	string inner_compressor;
	string index_codecs;
	string index_location;
};

struct ZarrChunkEntry {
	ZarrChunkEntry() {
	}
	ZarrChunkEntry(string store_path_p, string array_path_p, string chunk_key_p, vector<int64_t> chunk_coords_p,
	               string file_path_p, int64_t file_size_bytes_p, bool present_p, bool is_virtual_inner_chunk_p,
	               vector<int64_t> storage_chunk_coords_p, vector<int64_t> inner_chunk_coords_p)
	    : store_path(std::move(store_path_p)), array_path(std::move(array_path_p)), chunk_key(std::move(chunk_key_p)),
	      chunk_coords(std::move(chunk_coords_p)), file_path(std::move(file_path_p)),
	      file_size_bytes(file_size_bytes_p), present(present_p),
	      is_virtual_inner_chunk(is_virtual_inner_chunk_p),
	      storage_chunk_coords(std::move(storage_chunk_coords_p)),
	      inner_chunk_coords(std::move(inner_chunk_coords_p)) {
	}
	string store_path;
	string array_path;
	string chunk_key;
	vector<int64_t> chunk_coords;
	string file_path;
	int64_t file_size_bytes;
	bool present;
	bool is_virtual_inner_chunk = false;
	vector<int64_t> storage_chunk_coords;
	vector<int64_t> inner_chunk_coords;
};

struct ZarrNumericType {
	LogicalType logical_type;
	idx_t element_size;
	bool is_float;
	bool is_signed;
	bool is_unsigned;
	bool is_boolean;
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
	Value fill_value;
	bool has_fill_value = false;
	idx_t chunk_element_count = 0;
	idx_t expected_chunk_bytes = 0;
	idx_t next_chunk_offset = 0;
	idx_t current_chunk_index = DConstants::INVALID_INDEX;
	idx_t current_linear_index = 0;
	vector<char> decoded_chunk;
};

enum class ZarrVersionOverride : uint8_t { AUTO, V2, V3 };

static ZarrGroupEntry ParseGroupMetadata(yyjson_val *root, const string &store_path, const string &relative_path,
                                         const string &metadata_path);
static ZarrGroupEntry ParseGroupMetadataV3(yyjson_val *root, const string &store_path, const string &relative_path,
                                           const string &metadata_path);
static ZarrArrayEntry ParseArrayMetadataObject(yyjson_val *root, const string &store_path, const string &relative_path,
                                               const string &metadata_path);
static ZarrArrayEntry ParseArrayMetadataObjectV3(yyjson_val *root, const string &store_path,
                                                 const string &relative_path, const string &metadata_path);
static vector<ZarrChunkEntry> GenerateChunkEntries(FileSystem &fs, const string &store_path, const ZarrArrayEntry &array,
                                                   bool include_missing);
static idx_t FlattenCoordsCOrder(const vector<int64_t> &coords, const vector<int64_t> &shape);
static vector<int64_t> ShardInnerChunksPerShard(const ZarrArrayEntry &array);
static string RequireNodeType(yyjson_val *root, const string &metadata_path);

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

static yyjson_val *RequireObject(yyjson_val *obj, const char *key) {
	auto value = RequireObjectKey(obj, key);
	if (!yyjson_is_obj(value)) {
		throw InvalidInputException("Zarr metadata key \"%s\" is not an object", key);
	}
	return value;
}

static yyjson_val *RequireArray(yyjson_val *obj, const char *key) {
	auto value = RequireObjectKey(obj, key);
	if (!yyjson_is_arr(value)) {
		throw InvalidInputException("Zarr metadata key \"%s\" is not an array", key);
	}
	return value;
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

static ZarrVersionOverride ParseZarrVersionOverride(const Value &value, const string &function_name) {
	if (value.IsNull()) {
		return ZarrVersionOverride::AUTO;
	}
	auto text = StringUtil::Lower(StringValue::Get(value));
	if (text.empty() || text == "auto") {
		return ZarrVersionOverride::AUTO;
	}
	if (text == "2" || text == "v2") {
		return ZarrVersionOverride::V2;
	}
	if (text == "3" || text == "v3") {
		return ZarrVersionOverride::V3;
	}
	throw BinderException("%s version override must be one of auto, v2, or v3", function_name);
}

static idx_t Product(const vector<int64_t> &values) {
	idx_t result = 1;
	for (auto value : values) {
		if (value < 0) {
			throw InvalidInputException("Negative Zarr dimensions are not supported");
		}
		auto cast_value = NumericCast<idx_t>(value);
		if (cast_value != 0 && result > std::numeric_limits<idx_t>::max() / cast_value) {
			throw InvalidInputException("Zarr dimension product overflow");
		}
		result *= cast_value;
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
	return OptionalString(root, "id", OptionalString(root, "name"));
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
	int64_t width;
	try {
		width = std::stoll(dtype.substr(2));
	} catch (const std::exception &) {
		throw InvalidInputException("Unsupported Zarr dtype: %s", dtype);
	}
	if (width <= 0) {
		throw InvalidInputException("Unsupported Zarr dtype: %s", dtype);
	}
	result.element_size = NumericCast<idx_t>(width);
	result.is_float = false;
	result.is_signed = false;
	result.is_unsigned = false;
	result.is_boolean = false;
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
	if (kind == 'b') {
		result.is_boolean = true;
		if (width != 1) {
			throw InvalidInputException("Unsupported boolean dtype: %s", dtype);
		}
		result.logical_type = LogicalType::BOOLEAN;
		return result;
	}
	if (kind == 'f') {
		result.is_float = true;
		switch (width) {
		case 2:
			result.logical_type = LogicalType::FLOAT;
			break;
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

static string GetExtensionName(yyjson_val *extension, const string &metadata_path, const string &field_name) {
	if (yyjson_is_str(extension)) {
		return string(unsafe_yyjson_get_str(extension), unsafe_yyjson_get_len(extension));
	}
	if (yyjson_is_obj(extension)) {
		auto name = OptionalString(extension, "name");
		if (name.empty()) {
			throw InvalidInputException("%s contains an extension without a name in %s", field_name, metadata_path);
		}
		return name;
	}
	throw InvalidInputException("%s contains an invalid extension value in %s", field_name, metadata_path);
}

static yyjson_val *GetExtensionConfiguration(yyjson_val *extension, const string &metadata_path,
                                             const string &field_name) {
	if (!yyjson_is_obj(extension)) {
		return nullptr;
	}
	auto configuration = yyjson_obj_get(extension, "configuration");
	if (!configuration) {
		return nullptr;
	}
	if (!yyjson_is_obj(configuration)) {
		throw InvalidInputException("%s configuration is not an object in %s", field_name, metadata_path);
	}
	return configuration;
}

static string ParseV3DataType(yyjson_val *root, const string &metadata_path, bool little_endian) {
	auto data_type = RequireObjectKey(root, "data_type");
	if (!yyjson_is_str(data_type)) {
		throw InvalidInputException("Only string Zarr v3 data_type values are currently supported: %s", metadata_path);
	}
	string type_name(unsafe_yyjson_get_str(data_type), unsafe_yyjson_get_len(data_type));
	if (type_name == "bool") {
		return "|b1";
	}
	if (type_name == "int8") {
		return "|i1";
	}
	if (type_name == "uint8") {
		return "|u1";
	}
	string endian_prefix = little_endian ? "<" : ">";
	if (type_name == "int16") {
		return endian_prefix + "i2";
	}
	if (type_name == "int32") {
		return endian_prefix + "i4";
	}
	if (type_name == "int64") {
		return endian_prefix + "i8";
	}
	if (type_name == "uint16") {
		return endian_prefix + "u2";
	}
	if (type_name == "uint32") {
		return endian_prefix + "u4";
	}
	if (type_name == "uint64") {
		return endian_prefix + "u8";
	}
	if (type_name == "float16") {
		return endian_prefix + "f2";
	}
	if (type_name == "float32") {
		return endian_prefix + "f4";
	}
	if (type_name == "float64") {
		return endian_prefix + "f8";
	}
	throw InvalidInputException("Unsupported Zarr v3 data_type in %s: %s", metadata_path, type_name);
}

static bool IsIdentityPermutation(const vector<int64_t> &order) {
	for (idx_t i = 0; i < order.size(); i++) {
		if (order[i] != NumericCast<int64_t>(i)) {
			return false;
		}
	}
	return true;
}

static bool IsReversePermutation(const vector<int64_t> &order) {
	for (idx_t i = 0; i < order.size(); i++) {
		auto expected = NumericCast<int64_t>(order.size() - i - 1);
		if (order[i] != expected) {
			return false;
		}
	}
	return true;
}

static string JsonObjectWithId(yyjson_val *codec_value, const string &codec_name) {
	auto object = JsonToString(codec_value);
	if (!object.empty()) {
		return object;
	}
	return "{\"id\":\"" + codec_name + "\"}";
}

static bool IsSupportedBloscShuffle(const string &shuffle, idx_t typesize) {
	if (shuffle.empty() || shuffle == "noshuffle") {
		return true;
	}
	if (shuffle == "shuffle") {
		return true;
	}
	if (shuffle == "bitshuffle") {
		return typesize == 1;
	}
	return false;
}

static float DecodeFloat16(uint16_t half_bits) {
	auto sign = (half_bits >> 15) & 0x1;
	auto exponent = (half_bits >> 10) & 0x1f;
	auto mantissa = half_bits & 0x3ff;

	uint32_t float_bits;
	if (exponent == 0) {
		if (mantissa == 0) {
			float_bits = NumericCast<uint32_t>(sign) << 31;
		} else {
			exponent = 1;
			while ((mantissa & 0x400) == 0) {
				mantissa <<= 1;
				exponent--;
			}
			mantissa &= 0x3ff;
			auto float_exponent = NumericCast<uint32_t>(exponent + (127 - 15));
			float_bits =
			    (NumericCast<uint32_t>(sign) << 31) | (float_exponent << 23) | (NumericCast<uint32_t>(mantissa) << 13);
		}
	} else if (exponent == 0x1f) {
		float_bits = (NumericCast<uint32_t>(sign) << 31) | 0x7f800000U | (NumericCast<uint32_t>(mantissa) << 13);
	} else {
		auto float_exponent = NumericCast<uint32_t>(exponent + (127 - 15));
		float_bits =
		    (NumericCast<uint32_t>(sign) << 31) | (float_exponent << 23) | (NumericCast<uint32_t>(mantissa) << 13);
	}
	float value;
	std::memcpy(&value, &float_bits, sizeof(value));
	return value;
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
	if (dtype.is_boolean) {
		return Value::BOOLEAN(static_cast<unsigned char>(ptr[0]) != 0);
	}
	if (dtype.is_float) {
		if (dtype.element_size == 2) {
			auto bits = NumericCast<uint16_t>(ReadUnsignedInteger(ptr, dtype.element_size, dtype.little_endian));
			return Value::FLOAT(DecodeFloat16(bits));
		}
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

static void ByteUnshuffle(const char *input, idx_t input_size, idx_t typesize, vector<char> &output) {
	if (typesize <= 1) {
		output.assign(input, input + input_size);
		return;
	}
	auto element_count = input_size / typesize;
	output.resize(input_size);
	for (idx_t element_idx = 0; element_idx < element_count; element_idx++) {
		for (idx_t byte_idx = 0; byte_idx < typesize; byte_idx++) {
			output[element_idx * typesize + byte_idx] = input[byte_idx * element_count + element_idx];
		}
	}
}

static void BitUnshuffleTypesizeOne(const char *input, idx_t input_size, vector<char> &output) {
	if (input_size % 8 != 0) {
		throw InvalidInputException("Unsupported Blosc bitshuffle payload size: %llu", input_size);
	}
	auto plane_size = input_size / 8;
	output.assign(input_size, 0);
	for (idx_t i = 0; i < input_size; i++) {
		auto byte_index = i / 8;
		auto bit_index = i % 8;
		uint8_t value = 0;
		for (idx_t bit = 0; bit < 8; bit++) {
			auto bit_value = (static_cast<uint8_t>(input[bit * plane_size + byte_index]) >> bit_index) & 0x1;
			value |= NumericCast<uint8_t>(bit_value << bit);
		}
		output[i] = NumericCast<char>(value);
	}
}

static vector<char> DecompressZstd(const char *compressed_data, idx_t compressed_size, idx_t expected_size) {
	vector<char> result(expected_size);
	auto decompressed_size = duckdb_zstd::ZSTD_decompress(result.data(), result.size(), compressed_data, compressed_size);
	if (duckdb_zstd::ZSTD_isError(decompressed_size)) {
		throw InvalidInputException("Failed to decompress Zstd payload inside Blosc chunk");
	}
	if (decompressed_size != expected_size) {
		throw InvalidInputException("Unexpected Zstd decompressed size: expected %llu bytes, got %llu bytes",
		                            expected_size, decompressed_size);
	}
	return result;
}

static vector<char> DecompressBloscChunk(const vector<char> &compressed_data, const string &compressor,
                                         idx_t expected_size) {
	if (compressed_data.size() < 20) {
		throw InvalidInputException("Blosc chunk is too small");
	}
	unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)> doc(
	    yyjson_read(compressor.c_str(), compressor.size(), 0), yyjson_doc_free);
	if (!doc) {
		throw InvalidInputException("Failed to parse Blosc codec metadata");
	}
	auto root = yyjson_doc_get_root(doc.get());
	auto configuration = yyjson_obj_get(root, "configuration");
	if (configuration) {
		if (!yyjson_is_obj(configuration)) {
			throw InvalidInputException("Blosc codec configuration is not an object");
		}
		root = configuration;
	}
	auto cname = OptionalString(root, "cname");
	auto shuffle = OptionalString(root, "shuffle", "noshuffle");
	auto typesize_value = yyjson_obj_get(root, "typesize");
	if (!typesize_value || (!yyjson_is_uint(typesize_value) && !yyjson_is_sint(typesize_value))) {
		throw InvalidInputException("Blosc codec metadata is missing integer typesize");
	}
	auto typesize = yyjson_is_uint(typesize_value) ? NumericCast<idx_t>(unsafe_yyjson_get_uint(typesize_value))
	                                               : NumericCast<idx_t>(unsafe_yyjson_get_sint(typesize_value));
	if (cname != "zstd") {
		throw InvalidInputException("Only Blosc with cname=zstd is currently supported");
	}

	auto nbytes = NumericCast<idx_t>(Load<uint32_t>(reinterpret_cast<const_data_ptr_t>(compressed_data.data() + 4)));
	auto blocksize = NumericCast<idx_t>(Load<uint32_t>(reinterpret_cast<const_data_ptr_t>(compressed_data.data() + 8)));
	auto cbytes = NumericCast<idx_t>(Load<uint32_t>(reinterpret_cast<const_data_ptr_t>(compressed_data.data() + 12)));
	if (nbytes != expected_size) {
		throw InvalidInputException("Blosc chunk size mismatch: expected %llu bytes, got %llu bytes", expected_size,
		                            nbytes);
	}
	if (cbytes != compressed_data.size()) {
		throw InvalidInputException("Unexpected Blosc compressed byte count");
	}
	auto nblocks = (nbytes + blocksize - 1) / blocksize;
	auto offsets_table_size = nblocks * sizeof(uint32_t);
	if (compressed_data.size() < 16 + offsets_table_size) {
		throw InvalidInputException("Blosc chunk is missing the block offsets table");
	}

	vector<char> result;
	result.reserve(expected_size);
	for (idx_t block_idx = 0; block_idx < nblocks; block_idx++) {
		auto block_offset_ptr = compressed_data.data() + 16 + block_idx * sizeof(uint32_t);
		auto block_offset = NumericCast<idx_t>(Load<uint32_t>(reinterpret_cast<const_data_ptr_t>(block_offset_ptr)));
		if (block_offset + sizeof(uint32_t) > compressed_data.size()) {
			throw InvalidInputException("Invalid Blosc block offset");
		}
		auto block_cbytes =
		    NumericCast<idx_t>(Load<uint32_t>(reinterpret_cast<const_data_ptr_t>(compressed_data.data() + block_offset)));
		auto block_payload_offset = block_offset + sizeof(uint32_t);
		if (block_payload_offset + block_cbytes > compressed_data.size()) {
			throw InvalidInputException("Invalid Blosc block payload length");
		}
		auto block_expected_size = MinValue<idx_t>(blocksize, nbytes - block_idx * blocksize);
		auto raw_block =
		    DecompressZstd(compressed_data.data() + block_payload_offset, block_cbytes, block_expected_size);
		vector<char> decoded_block;
		if (shuffle == "shuffle") {
			ByteUnshuffle(raw_block.data(), raw_block.size(), typesize, decoded_block);
		} else if (shuffle == "bitshuffle") {
			if (typesize != 1) {
				throw InvalidInputException("Blosc bitshuffle is currently only supported for typesize=1");
			}
			BitUnshuffleTypesizeOne(raw_block.data(), raw_block.size(), decoded_block);
		} else {
			decoded_block = std::move(raw_block);
		}
		result.insert(result.end(), decoded_block.begin(), decoded_block.end());
	}
	return result;
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
		if (compressor_id == "blosc") {
			return DecompressBloscChunk(compressed_data, compressor, expected_size);
		}
		throw InvalidInputException("Unsupported Zarr compressor for zarr_cells: %s", compressor_id);
	}
	vector<char> decompressed(expected_size);
	MiniZStream stream;
	stream.Decompress(compressed_data.data(), compressed_data.size(), decompressed.data(), decompressed.size());
	return decompressed;
}

static bool HasMaterializedFillValue(const ZarrArrayEntry &array) {
	return !array.fill_value.empty() && array.fill_value != "null";
}

static Value ParseFillValue(const ZarrArrayEntry &array, const ZarrNumericType &dtype) {
	if (!HasMaterializedFillValue(array)) {
		return Value();
	}
	unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)> doc(
	    yyjson_read(array.fill_value.c_str(), array.fill_value.size(), 0), yyjson_doc_free);
	if (!doc) {
		throw InvalidInputException("Failed to parse Zarr fill_value for array: %s", array.array_path);
	}
	auto root = yyjson_doc_get_root(doc.get());
	if (yyjson_is_null(root)) {
		return Value();
	}
	if (dtype.is_boolean) {
		if (yyjson_is_bool(root)) {
			return Value::BOOLEAN(yyjson_get_bool(root));
		}
		if (yyjson_is_uint(root)) {
			return Value::BOOLEAN(unsafe_yyjson_get_uint(root) != 0);
		}
		if (yyjson_is_sint(root)) {
			return Value::BOOLEAN(unsafe_yyjson_get_sint(root) != 0);
		}
		throw InvalidInputException("Unsupported boolean fill_value for array: %s", array.array_path);
	}
	if (dtype.is_float) {
		double fill_value;
		if (yyjson_is_real(root)) {
			fill_value = unsafe_yyjson_get_real(root);
		} else if (yyjson_is_uint(root)) {
			fill_value = static_cast<double>(unsafe_yyjson_get_uint(root));
		} else if (yyjson_is_sint(root)) {
			fill_value = static_cast<double>(unsafe_yyjson_get_sint(root));
		} else {
			throw InvalidInputException("Unsupported floating-point fill_value for array: %s", array.array_path);
		}
		return dtype.element_size == 8 ? Value::DOUBLE(fill_value) : Value::FLOAT(static_cast<float>(fill_value));
	}
	if (dtype.is_signed) {
		int64_t fill_value;
		if (yyjson_is_sint(root)) {
			fill_value = unsafe_yyjson_get_sint(root);
		} else if (yyjson_is_uint(root)) {
			fill_value = NumericCast<int64_t>(unsafe_yyjson_get_uint(root));
		} else {
			throw InvalidInputException("Unsupported signed integer fill_value for array: %s", array.array_path);
		}
		switch (dtype.element_size) {
		case 1:
			return Value::TINYINT(NumericCast<int8_t>(fill_value));
		case 2:
			return Value::SMALLINT(NumericCast<int16_t>(fill_value));
		case 4:
			return Value::INTEGER(NumericCast<int32_t>(fill_value));
		case 8:
			return Value::BIGINT(fill_value);
		default:
			break;
		}
	}
	if (dtype.is_unsigned) {
		uint64_t fill_value;
		if (yyjson_is_uint(root)) {
			fill_value = unsafe_yyjson_get_uint(root);
		} else if (yyjson_is_sint(root)) {
			auto signed_value = unsafe_yyjson_get_sint(root);
			if (signed_value < 0) {
				throw InvalidInputException("Unsigned integer fill_value cannot be negative for array: %s",
				                            array.array_path);
			}
			fill_value = NumericCast<uint64_t>(signed_value);
		} else {
			throw InvalidInputException("Unsupported unsigned integer fill_value for array: %s", array.array_path);
		}
		switch (dtype.element_size) {
		case 1:
			return Value::UTINYINT(NumericCast<uint8_t>(fill_value));
		case 2:
			return Value::USMALLINT(NumericCast<uint16_t>(fill_value));
		case 4:
			return Value::UINTEGER(NumericCast<uint32_t>(fill_value));
		case 8:
			return Value::UBIGINT(fill_value);
		default:
			break;
		}
	}
	throw InvalidInputException("Unsupported fill_value type for array: %s", array.array_path);
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

static idx_t ShardingIndexEncodedSize(const ZarrArrayEntry &array) {
	auto chunk_count = Product(ShardInnerChunksPerShard(array));
	idx_t index_size = chunk_count * 2 * sizeof(uint64_t);
	if (array.index_codecs == "bytes+crc32c") {
		index_size += sizeof(uint32_t);
	}
	return index_size;
}

static vector<char> DecodeShardedInnerChunk(const vector<char> &shard_data, const ZarrArrayEntry &array,
                                            const ZarrChunkEntry &chunk, idx_t expected_size) {
	auto index_size = ShardingIndexEncodedSize(array);
	if (shard_data.size() < index_size) {
		throw InvalidInputException("Shard file is smaller than its sharding index: %s", chunk.file_path);
	}
	idx_t index_offset = 0;
	if (array.index_location == "end") {
		index_offset = shard_data.size() - index_size;
	}
	auto crc_size = array.index_codecs == "bytes+crc32c" ? sizeof(uint32_t) : 0;
	auto index_bytes = reinterpret_cast<const unsigned char *>(shard_data.data() + index_offset);
	auto index_payload_size = index_size - crc_size;
	auto chunks_per_shard = ShardInnerChunksPerShard(array);
	auto flat_index = FlattenCoordsCOrder(chunk.inner_chunk_coords, chunks_per_shard);
	auto entry_offset = flat_index * 2 * sizeof(uint64_t);
	if (entry_offset + 2 * sizeof(uint64_t) > index_payload_size) {
		throw InvalidInputException("Invalid sharding index entry offset for %s", chunk.file_path);
	}
	auto offset = ReadUnsignedInteger(reinterpret_cast<const char *>(index_bytes + entry_offset), sizeof(uint64_t), true);
	auto nbytes = ReadUnsignedInteger(reinterpret_cast<const char *>(index_bytes + entry_offset + sizeof(uint64_t)),
	                                  sizeof(uint64_t), true);
	if (offset == std::numeric_limits<uint64_t>::max() && nbytes == std::numeric_limits<uint64_t>::max()) {
		return {};
	}
	if (offset + nbytes > shard_data.size()) {
		throw InvalidInputException("Invalid sharding index offset/length for %s", chunk.file_path);
	}
	vector<char> encoded_inner_chunk;
	encoded_inner_chunk.insert(encoded_inner_chunk.end(), shard_data.begin() + NumericCast<idx_t>(offset),
	                           shard_data.begin() + NumericCast<idx_t>(offset + nbytes));
	return DecompressChunk(encoded_inner_chunk, array.inner_compressor, expected_size);
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
		if (chunk.present) {
			auto raw_data = ReadBinaryFile(fs, chunk.file_path);
			if (bind_data.array.is_sharding_indexed || ParseCompressorId(bind_data.array.compressor) == "sharding_indexed") {
				state.decoded_chunk = DecodeShardedInnerChunk(raw_data, bind_data.array, chunk, state.expected_chunk_bytes);
			} else {
				state.decoded_chunk = DecompressChunk(raw_data, bind_data.array.compressor, state.expected_chunk_bytes);
			}
		} else {
			state.decoded_chunk.clear();
		}
		return true;
	}
	state.current_chunk_index = DConstants::INVALID_INDEX;
	state.decoded_chunk.clear();
	return false;
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

static ZarrArrayEntry ParseArrayMetadataV3(FileSystem &fs, const string &store_path, const string &relative_path,
                                           const string &dir_path) {
	auto metadata_path = fs.JoinPath(dir_path, "zarr.json");
	auto metadata_text = ReadTextFile(fs, metadata_path);
	unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)> doc(yyjson_read(metadata_text.c_str(), metadata_text.size(), 0),
	                                                       yyjson_doc_free);
	if (!doc) {
		throw InvalidInputException("Failed to parse %s", metadata_path);
	}
	return ParseArrayMetadataObjectV3(yyjson_doc_get_root(doc.get()), store_path, relative_path, metadata_path);
}

static void TraverseStoreV2(FileSystem &fs, const string &store_path, const string &dir_path,
                            const string &relative_path, bool collect_chunks, vector<ZarrGroupEntry> &groups,
                            vector<ZarrArrayEntry> &arrays, vector<ZarrChunkEntry> &chunks) {
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
			auto array_chunks = GenerateChunkEntries(fs, store_path, entry, false);
			chunks.insert(chunks.end(), array_chunks.begin(), array_chunks.end());
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
		TraverseStoreV2(fs, store_path, child_path, JoinNodePath(relative_path, child_name), collect_chunks, groups,
		                arrays, chunks);
	});
}

static void TraverseStoreV3(FileSystem &fs, const string &store_path, const string &dir_path,
                            const string &relative_path, bool collect_chunks, vector<ZarrGroupEntry> &groups,
                            vector<ZarrArrayEntry> &arrays, vector<ZarrChunkEntry> &chunks) {
	auto metadata_path = fs.JoinPath(dir_path, "zarr.json");
	if (fs.FileExists(metadata_path)) {
		auto metadata_text = ReadTextFile(fs, metadata_path);
		unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)> doc(
		    yyjson_read(metadata_text.c_str(), metadata_text.size(), 0), yyjson_doc_free);
		if (!doc) {
			throw InvalidInputException("Failed to parse %s", metadata_path);
		}
		auto root = yyjson_doc_get_root(doc.get());
		if (!yyjson_is_obj(root)) {
			throw InvalidInputException("%s does not contain a JSON object", metadata_path);
		}
		auto node_type = RequireNodeType(root, metadata_path);
		if (node_type == "group") {
			groups.push_back(ParseGroupMetadataV3(root, store_path, relative_path, metadata_path));
		} else if (node_type == "array") {
			auto entry = ParseArrayMetadataObjectV3(root, store_path, relative_path, metadata_path);
			if (collect_chunks) {
				auto array_chunks = GenerateChunkEntries(fs, store_path, entry, false);
				chunks.insert(chunks.end(), array_chunks.begin(), array_chunks.end());
			}
			arrays.push_back(std::move(entry));
			return;
		} else {
			throw InvalidInputException("Unsupported Zarr v3 node_type in %s: %s", metadata_path, node_type);
		}
	}

	fs.ListFiles(dir_path, [&](const string &child_name, bool is_dir) {
		if (!is_dir) {
			return;
		}
		if (StringUtil::StartsWith(child_name, ".")) {
			return;
		}
		auto child_path = fs.JoinPath(dir_path, child_name);
		TraverseStoreV3(fs, store_path, child_path, JoinNodePath(relative_path, child_name), collect_chunks, groups,
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

static int64_t RequireZarrFormatV2(yyjson_val *root, const string &metadata_path) {
	auto zarr_format = RequireInt64(root, "zarr_format");
	if (zarr_format != 2) {
		throw InvalidInputException("Only Zarr v2 metadata is currently supported: %s", metadata_path);
	}
	return zarr_format;
}

static int64_t RequireZarrFormatV3(yyjson_val *root, const string &metadata_path) {
	auto zarr_format = RequireInt64(root, "zarr_format");
	if (zarr_format != 3) {
		throw InvalidInputException("Only Zarr v3 metadata is currently supported: %s", metadata_path);
	}
	return zarr_format;
}

static string RequireNodeType(yyjson_val *root, const string &metadata_path) {
	auto node_type = OptionalString(root, "node_type");
	if (node_type.empty()) {
		throw InvalidInputException("Zarr v3 metadata is missing node_type: %s", metadata_path);
	}
	return node_type;
}

static ZarrGroupEntry ParseGroupMetadata(yyjson_val *root, const string &store_path, const string &relative_path,
                                         const string &metadata_path) {
	if (!yyjson_is_obj(root)) {
		throw InvalidInputException("%s does not contain a JSON object", metadata_path);
	}
	return {store_path, FormatGroupPath(relative_path), RequireZarrFormatV2(root, metadata_path), metadata_path};
}

static ZarrArrayEntry ParseArrayMetadataObject(yyjson_val *root, const string &store_path, const string &relative_path,
                                               const string &metadata_path) {
	if (!yyjson_is_obj(root)) {
		throw InvalidInputException("%s does not contain a JSON object", metadata_path);
	}
	auto compressor = JsonToString(yyjson_obj_get(root, "compressor"));
	auto fill_value = JsonToString(yyjson_obj_get(root, "fill_value"));
	auto dimension_separator = OptionalString(root, "dimension_separator", ".");
	if (dimension_separator != "." && dimension_separator != "/") {
		throw InvalidInputException("Unsupported Zarr dimension_separator in %s: %s", metadata_path,
		                            dimension_separator);
	}
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

	auto order = OptionalString(root, "order");
	if (order != "C" && order != "F") {
		throw InvalidInputException("Unsupported Zarr order in %s: %s", metadata_path, order);
	}

	return {store_path,
	        relative_path,
	        RequireZarrFormatV2(root, metadata_path),
	        NumericCast<int64_t>(shape.size()),
	        std::move(shape),
	        std::move(chunks),
	        OptionalString(root, "dtype"),
	        std::move(order),
	        std::move(compressor),
	        std::move(fill_value),
	        "v2",
	        std::move(dimension_separator),
	        metadata_path,
	        true,
	        "",
	        false,
	        {},
	        {},
	        "",
	        "",
	        ""};
}

static ZarrGroupEntry ParseGroupMetadataV3(yyjson_val *root, const string &store_path, const string &relative_path,
                                           const string &metadata_path) {
	if (!yyjson_is_obj(root)) {
		throw InvalidInputException("%s does not contain a JSON object", metadata_path);
	}
	auto zarr_format = RequireZarrFormatV3(root, metadata_path);
	auto node_type = RequireNodeType(root, metadata_path);
	if (node_type != "group") {
		throw InvalidInputException("Expected a Zarr v3 group metadata document at %s", metadata_path);
	}
	return {store_path, FormatGroupPath(relative_path), zarr_format, metadata_path};
}

static void ParseV3ShardingCodec(yyjson_val *codec_value, const string &metadata_path, vector<int64_t> &outer_chunks,
                                 string &compressor, bool &supports_cells, string &cells_error,
                                 vector<int64_t> &inner_chunks, string &inner_compressor, string &index_codecs,
                                 string &index_location, bool &little_endian) {
	auto codec_config = GetExtensionConfiguration(codec_value, metadata_path, "codecs");
	if (!codec_config) {
		throw InvalidInputException("Zarr v3 sharding_indexed codec is missing configuration in %s", metadata_path);
	}
	inner_chunks = RequireInt64Array(codec_config, "chunk_shape");
	if (inner_chunks.size() != outer_chunks.size()) {
		throw InvalidInputException("Zarr v3 sharding_indexed inner chunk rank mismatch in %s", metadata_path);
	}
	for (idx_t i = 0; i < inner_chunks.size(); i++) {
		if (inner_chunks[i] <= 0 || outer_chunks[i] <= 0 || outer_chunks[i] % inner_chunks[i] != 0) {
			throw InvalidInputException(
			    "Zarr v3 sharding_indexed requires inner chunk_shape to evenly divide the outer chunk shape in %s",
			    metadata_path);
		}
	}

	auto inner_codecs = RequireArray(codec_config, "codecs");
	bool saw_bytes = false;
	little_endian = HostIsLittleEndian();
	yyjson_val *inner_codec_value;
	yyjson_arr_iter inner_codec_iter = yyjson_arr_iter_with(inner_codecs);
	while ((inner_codec_value = yyjson_arr_iter_next(&inner_codec_iter))) {
		auto inner_codec_name = GetExtensionName(inner_codec_value, metadata_path, "codecs");
		auto inner_codec_config = GetExtensionConfiguration(inner_codec_value, metadata_path, "codecs");
		if (inner_codec_name == "bytes") {
			if (saw_bytes) {
				throw InvalidInputException("Zarr v3 sharding_indexed metadata contains multiple inner bytes codecs: %s",
				                            metadata_path);
			}
			saw_bytes = true;
			if (inner_codec_config) {
				auto endian = OptionalString(inner_codec_config, "endian", "");
				if (endian == "little") {
					little_endian = true;
				} else if (endian == "big") {
					little_endian = false;
				} else if (!endian.empty()) {
					supports_cells = false;
					cells_error = "zarr_cells does not yet support Zarr v3 sharding_indexed bytes endian " + endian;
				}
			}
			continue;
		}
		if (inner_codec_name == "blosc") {
			inner_compressor = JsonObjectWithId(inner_codec_value, "blosc");
			compressor = JsonObjectWithId(codec_value, "sharding_indexed");
			if (!inner_codec_config) {
				supports_cells = false;
				cells_error = "zarr_cells requires Blosc configuration metadata for sharding_indexed arrays";
				continue;
			}
			auto cname = OptionalString(inner_codec_config, "cname", "");
			auto shuffle = OptionalString(inner_codec_config, "shuffle", "noshuffle");
			auto typesize = NumericCast<idx_t>(RequireInt64(inner_codec_config, "typesize"));
			if (cname != "zstd") {
				supports_cells = false;
				cells_error = "zarr_cells currently supports Blosc only with cname=zstd";
			} else if (!IsSupportedBloscShuffle(shuffle, typesize)) {
				supports_cells = false;
				cells_error =
				    "zarr_cells currently supports Blosc shuffle, noshuffle, and bitshuffle only for typesize=1";
			}
			continue;
		}
		if (inner_codec_name == "gzip") {
			inner_compressor = "{\"id\":\"gzip\"}";
			compressor = JsonObjectWithId(codec_value, "sharding_indexed");
			if (inner_codec_config) {
				auto level_value = yyjson_obj_get(inner_codec_config, "level");
				if (level_value && (yyjson_is_uint(level_value) || yyjson_is_sint(level_value))) {
					auto level = yyjson_is_uint(level_value) ? NumericCast<int64_t>(unsafe_yyjson_get_uint(level_value))
					                                        : unsafe_yyjson_get_sint(level_value);
					inner_compressor = "{\"id\":\"gzip\",\"level\":" + std::to_string(level) + "}";
				}
			}
			continue;
		}
		supports_cells = false;
		cells_error = "zarr_cells does not yet support Zarr v3 sharding_indexed inner codec: " + inner_codec_name;
	}
	if (!saw_bytes) {
		supports_cells = false;
		if (cells_error.empty()) {
			cells_error = "zarr_cells requires an inner bytes codec for Zarr v3 sharding_indexed arrays";
		}
	}
	auto index_codec_array = RequireArray(codec_config, "index_codecs");
	index_location = OptionalString(codec_config, "index_location", "end");
	if (index_location != "start" && index_location != "end") {
		supports_cells = false;
		if (cells_error.empty()) {
			cells_error = "zarr_cells only supports Zarr v3 sharding_indexed with index_location start or end";
		}
	}
	yyjson_val *index_codec_value;
	yyjson_arr_iter index_codec_iter = yyjson_arr_iter_with(index_codec_array);
	bool saw_index_bytes = false;
	bool saw_crc32c = false;
	while ((index_codec_value = yyjson_arr_iter_next(&index_codec_iter))) {
		auto index_codec_name = GetExtensionName(index_codec_value, metadata_path, "index_codecs");
		auto index_codec_config = GetExtensionConfiguration(index_codec_value, metadata_path, "index_codecs");
		if (index_codec_name == "bytes") {
			saw_index_bytes = true;
			if (index_codec_config) {
				auto endian = OptionalString(index_codec_config, "endian", "little");
				if (endian != "little") {
					supports_cells = false;
					cells_error = "zarr_cells currently requires little-endian sharding index bytes codec";
				}
			}
			continue;
		}
		if (index_codec_name == "crc32c") {
			saw_crc32c = true;
			continue;
		}
		supports_cells = false;
		cells_error = "zarr_cells does not yet support Zarr v3 sharding index codec: " + index_codec_name;
	}
	if (!saw_index_bytes) {
		supports_cells = false;
		cells_error = "zarr_cells requires a bytes codec for Zarr v3 sharding indexes";
	}
	index_codecs = saw_crc32c ? "bytes+crc32c" : "bytes";
}

static ZarrArrayEntry ParseArrayMetadataObjectV3(yyjson_val *root, const string &store_path,
                                                 const string &relative_path, const string &metadata_path) {
	if (!yyjson_is_obj(root)) {
		throw InvalidInputException("%s does not contain a JSON object", metadata_path);
	}
	auto zarr_format = RequireZarrFormatV3(root, metadata_path);
	auto node_type = RequireNodeType(root, metadata_path);
	if (node_type != "array") {
		throw InvalidInputException("Expected a Zarr v3 array metadata document at %s", metadata_path);
	}
	auto shape = RequireInt64Array(root, "shape");
	auto chunk_grid = RequireObject(root, "chunk_grid");
	auto chunk_grid_name = GetExtensionName(chunk_grid, metadata_path, "chunk_grid");
	if (chunk_grid_name != "regular") {
		throw InvalidInputException("Unsupported Zarr v3 chunk_grid in %s: %s", metadata_path, chunk_grid_name);
	}
	auto chunk_grid_config = GetExtensionConfiguration(chunk_grid, metadata_path, "chunk_grid");
	if (!chunk_grid_config) {
		throw InvalidInputException("Zarr v3 chunk_grid is missing configuration in %s", metadata_path);
	}
	auto chunks = RequireInt64Array(chunk_grid_config, "chunk_shape");
	if (shape.size() != chunks.size()) {
		throw InvalidInputException("%s has mismatched shape/chunk_grid rank", metadata_path);
	}
	for (idx_t i = 0; i < shape.size(); i++) {
		if (shape[i] < 0) {
			throw InvalidInputException("%s contains a negative shape dimension", metadata_path);
		}
		if (chunks[i] <= 0) {
			throw InvalidInputException("%s contains a non-positive chunk dimension", metadata_path);
		}
	}

	auto chunk_key_encoding = RequireObjectKey(root, "chunk_key_encoding");
	auto chunk_key_encoding_name = GetExtensionName(chunk_key_encoding, metadata_path, "chunk_key_encoding");
	if (chunk_key_encoding_name != "default" && chunk_key_encoding_name != "v2") {
		throw InvalidInputException("Unsupported Zarr v3 chunk_key_encoding in %s: %s", metadata_path,
		                            chunk_key_encoding_name);
	}
	string dimension_separator = chunk_key_encoding_name == "default" ? "/" : ".";
	auto chunk_key_config = GetExtensionConfiguration(chunk_key_encoding, metadata_path, "chunk_key_encoding");
	if (chunk_key_config) {
		dimension_separator = OptionalString(chunk_key_config, "separator", dimension_separator);
	}
	if (dimension_separator != "." && dimension_separator != "/") {
		throw InvalidInputException("Unsupported Zarr v3 chunk key separator in %s: %s", metadata_path,
		                            dimension_separator);
	}

	auto codecs = RequireArray(root, "codecs");
	auto codec_count = unsafe_yyjson_get_len(codecs);
	if (codec_count == 1) {
		auto first_codec = yyjson_arr_get_first(codecs);
		auto first_codec_name = GetExtensionName(first_codec, metadata_path, "codecs");
		if (first_codec_name == "sharding_indexed") {
			string compressor = JsonObjectWithId(first_codec, "sharding_indexed");
			string inner_compressor;
			string index_codecs;
			string index_location = "end";
			bool supports_cells = true;
			string cells_error;
			vector<int64_t> inner_chunks;
			bool little_endian = HostIsLittleEndian();
			ParseV3ShardingCodec(first_codec, metadata_path, chunks, compressor, supports_cells, cells_error,
			                     inner_chunks, inner_compressor, index_codecs, index_location, little_endian);
			return {store_path,
			        relative_path,
			        zarr_format,
			        NumericCast<int64_t>(shape.size()),
			        std::move(shape),
			        std::move(chunks),
			        ParseV3DataType(root, metadata_path, little_endian),
			        "C",
			        std::move(compressor),
			        JsonToString(yyjson_obj_get(root, "fill_value")),
			        std::move(chunk_key_encoding_name),
			        std::move(dimension_separator),
			        metadata_path,
			        supports_cells,
			        std::move(cells_error),
			        true,
			        {},
			        std::move(inner_chunks),
			        std::move(inner_compressor),
			        std::move(index_codecs),
			        std::move(index_location)};
		}
	}
	bool saw_bytes = false;
	string order = "C";
	string compressor;
	bool little_endian = HostIsLittleEndian();
	bool supports_cells = true;
	string cells_error;
	yyjson_val *codec_value;
	yyjson_arr_iter codec_iter = yyjson_arr_iter_with(codecs);
	while ((codec_value = yyjson_arr_iter_next(&codec_iter))) {
		auto codec_name = GetExtensionName(codec_value, metadata_path, "codecs");
		auto codec_config = GetExtensionConfiguration(codec_value, metadata_path, "codecs");
		if (codec_name == "transpose") {
			if (saw_bytes) {
				throw InvalidInputException("Zarr v3 transpose codec must appear before bytes in %s", metadata_path);
			}
			if (!codec_config) {
				throw InvalidInputException("Zarr v3 transpose codec is missing configuration in %s", metadata_path);
			}
			auto transpose_order = RequireInt64Array(codec_config, "order");
			if (transpose_order.size() != shape.size()) {
				throw InvalidInputException("Zarr v3 transpose order rank mismatch in %s", metadata_path);
			}
			if (IsIdentityPermutation(transpose_order)) {
				order = "C";
			} else if (IsReversePermutation(transpose_order)) {
				order = "F";
			} else {
				throw InvalidInputException(
				    "Only identity and full-reverse Zarr v3 transpose orders are currently supported: %s",
				    metadata_path);
			}
			continue;
		}
		if (codec_name == "bytes") {
			if (saw_bytes) {
				throw InvalidInputException("Zarr v3 metadata contains multiple bytes codecs: %s", metadata_path);
			}
			saw_bytes = true;
			if (codec_config) {
				auto endian = OptionalString(codec_config, "endian", "");
				if (endian == "little") {
					little_endian = true;
				} else if (endian == "big") {
					little_endian = false;
				} else if (!endian.empty()) {
					throw InvalidInputException("Unsupported Zarr v3 bytes codec endian in %s: %s", metadata_path,
					                            endian);
				}
			}
			continue;
		}
		if (codec_name == "sharding_indexed") {
			supports_cells = false;
			compressor = JsonToString(codec_value);
			cells_error =
			    "zarr_cells does not yet support Zarr v3 sharding_indexed codec pipelines (including inner Blosc)";
			continue;
		}
		if (codec_name == "gzip") {
			if (!saw_bytes) {
				throw InvalidInputException("Zarr v3 gzip codec must appear after bytes in %s", metadata_path);
			}
			if (!compressor.empty()) {
				throw InvalidInputException("Only one Zarr v3 compression codec is currently supported: %s",
				                            metadata_path);
			}
			if (!codec_config) {
				compressor = "{\"id\":\"gzip\"}";
				continue;
			}
			auto level_value = yyjson_obj_get(codec_config, "level");
			if (!level_value) {
				compressor = "{\"id\":\"gzip\"}";
				continue;
			}
			if (!yyjson_is_uint(level_value) && !yyjson_is_sint(level_value)) {
				throw InvalidInputException("Zarr v3 gzip level must be an integer in %s", metadata_path);
			}
			auto level = yyjson_is_uint(level_value) ? NumericCast<int64_t>(unsafe_yyjson_get_uint(level_value))
			                                        : unsafe_yyjson_get_sint(level_value);
			compressor = "{\"id\":\"gzip\",\"level\":" + std::to_string(level) + "}";
			continue;
		}
		supports_cells = false;
		compressor = JsonToString(codec_value);
		cells_error = "zarr_cells does not yet support Zarr v3 codec: " + codec_name;
	}
	if (!saw_bytes && supports_cells) {
		throw InvalidInputException("Zarr v3 arrays currently require a bytes codec in %s", metadata_path);
	}
	if (!supports_cells && cells_error.empty()) {
		cells_error = "zarr_cells does not yet support the Zarr v3 codec pipeline for this array";
	}

	return {store_path,
	        relative_path,
	        zarr_format,
	        NumericCast<int64_t>(shape.size()),
	        std::move(shape),
	        std::move(chunks),
	        ParseV3DataType(root, metadata_path, little_endian),
	        std::move(order),
	        std::move(compressor),
	        JsonToString(yyjson_obj_get(root, "fill_value")),
	        std::move(chunk_key_encoding_name),
	        std::move(dimension_separator),
	        metadata_path,
	        supports_cells,
	        std::move(cells_error),
	        false,
	        {},
	        {},
	        "",
	        "",
	        ""};
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

static string ChunkKeyFromCoords(const vector<int64_t> &coords, const string &chunk_key_encoding,
                                 const string &dimension_separator) {
	string separator = dimension_separator == "/" ? "/" : ".";
	if (chunk_key_encoding == "default") {
		if (coords.empty()) {
			return "c";
		}
		string result = "c";
		for (idx_t i = 0; i < coords.size(); i++) {
			result += separator;
			result += std::to_string(coords[i]);
		}
		return result;
	}
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

static idx_t FlattenCoordsCOrder(const vector<int64_t> &coords, const vector<int64_t> &shape) {
	if (coords.size() != shape.size()) {
		throw InternalException("Coordinate rank mismatch");
	}
	idx_t linear_index = 0;
	for (idx_t i = 0; i < coords.size(); i++) {
		auto dim = NumericCast<idx_t>(shape[i]);
		auto coord = NumericCast<idx_t>(coords[i]);
		if (coord >= dim) {
			throw InternalException("Coordinate out of bounds");
		}
		linear_index = linear_index * dim + coord;
	}
	return linear_index;
}

static vector<int64_t> ShardInnerChunksPerShard(const ZarrArrayEntry &array) {
	auto &storage_chunks = array.storage_chunks.empty() ? array.chunks : array.storage_chunks;
	vector<int64_t> chunks_per_shard;
	chunks_per_shard.reserve(array.rank);
	for (idx_t dim = 0; dim < NumericCast<idx_t>(array.rank); dim++) {
		chunks_per_shard.push_back(storage_chunks[dim] / array.inner_chunks[dim]);
	}
	return chunks_per_shard;
}

static void GenerateChunkEntriesRecursive(FileSystem &fs, const string &store_path, const ZarrArrayEntry &array,
                                          vector<int64_t> &coords, idx_t dim, bool include_missing,
                                          vector<ZarrChunkEntry> &chunks) {
	if (dim == coords.size()) {
		auto chunk_key = ChunkKeyFromCoords(coords, array.chunk_key_encoding, array.dimension_separator);
		auto file_path = ChunkPath(store_path, array.array_path, chunk_key);
		auto file_exists = fs.FileExists(file_path);
		if (!file_exists && !include_missing) {
			return;
		}
		int64_t file_size = 0;
		if (file_exists) {
			auto handle = fs.OpenFile(file_path, FileFlags::FILE_FLAGS_READ);
			file_size = NumericCast<int64_t>(handle->GetFileSize());
		}
		chunks.push_back(
		    {store_path, array.array_path, chunk_key, coords, file_path, file_size, file_exists, false, {}, {}});
		return;
	}
	auto chunk_count = (array.shape[dim] + array.chunks[dim] - 1) / array.chunks[dim];
	for (int64_t chunk_coord = 0; chunk_coord < chunk_count; chunk_coord++) {
		coords[dim] = chunk_coord;
		GenerateChunkEntriesRecursive(fs, store_path, array, coords, dim + 1, include_missing, chunks);
	}
}

static vector<ZarrChunkEntry> GenerateChunkEntries(FileSystem &fs, const string &store_path,
                                                   const ZarrArrayEntry &array, bool include_missing) {
	vector<ZarrChunkEntry> chunks;
	vector<int64_t> coords(NumericCast<idx_t>(array.rank), 0);
	GenerateChunkEntriesRecursive(fs, store_path, array, coords, 0, include_missing, chunks);
	return chunks;
}

static void GenerateCellChunkEntriesRecursive(FileSystem &fs, const string &store_path, const ZarrArrayEntry &array,
                                              const vector<int64_t> &chunks_per_shard, vector<int64_t> &logical_coords,
                                              idx_t dim, bool include_missing, vector<ZarrChunkEntry> &chunks) {
	if (dim == logical_coords.size()) {
		vector<int64_t> shard_coords(logical_coords.size(), 0);
		vector<int64_t> inner_coords(logical_coords.size(), 0);
		for (idx_t i = 0; i < logical_coords.size(); i++) {
			shard_coords[i] = logical_coords[i] / chunks_per_shard[i];
			inner_coords[i] = logical_coords[i] % chunks_per_shard[i];
		}
		auto storage_chunk_key = ChunkKeyFromCoords(shard_coords, array.chunk_key_encoding, array.dimension_separator);
		auto file_path = ChunkPath(store_path, array.array_path, storage_chunk_key);
		auto file_exists = fs.FileExists(file_path);
		if (!file_exists && !include_missing) {
			return;
		}
		int64_t file_size = 0;
		if (file_exists) {
			auto handle = fs.OpenFile(file_path, FileFlags::FILE_FLAGS_READ);
			file_size = NumericCast<int64_t>(handle->GetFileSize());
		}
		chunks.push_back({store_path, array.array_path, storage_chunk_key, logical_coords, file_path, file_size, file_exists,
		                  true, std::move(shard_coords), std::move(inner_coords)});
		return;
	}
	auto chunk_count = (array.shape[dim] + array.inner_chunks[dim] - 1) / array.inner_chunks[dim];
	for (int64_t chunk_coord = 0; chunk_coord < chunk_count; chunk_coord++) {
		logical_coords[dim] = chunk_coord;
		GenerateCellChunkEntriesRecursive(fs, store_path, array, chunks_per_shard, logical_coords, dim + 1,
		                                  include_missing, chunks);
	}
}

static vector<ZarrChunkEntry> GenerateCellChunkEntries(FileSystem &fs, const string &store_path,
                                                       const ZarrArrayEntry &array, bool include_missing) {
	auto chunks_per_shard = ShardInnerChunksPerShard(array);
	vector<ZarrChunkEntry> chunks;
	vector<int64_t> logical_coords(NumericCast<idx_t>(array.rank), 0);
	GenerateCellChunkEntriesRecursive(fs, store_path, array, chunks_per_shard, logical_coords, 0, include_missing,
	                                  chunks);
	return chunks;
}

static bool DiscoverConsolidatedStoreV2(FileSystem &fs, const string &store_path, vector<ZarrGroupEntry> &groups,
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
				auto array_chunks = GenerateChunkEntries(fs, store_path, array, false);
				chunks.insert(chunks.end(), array_chunks.begin(), array_chunks.end());
			}
			arrays.push_back(std::move(array));
		}
	}
	return true;
}

static bool HasV2RootMarker(FileSystem &fs, const string &store_path) {
	return fs.FileExists(store_path + "/.zmetadata") || fs.FileExists(store_path + "/.zgroup") ||
	       fs.FileExists(store_path + "/.zarray");
}

static bool HasV3RootMarker(FileSystem &fs, const string &store_path) {
	return fs.FileExists(store_path + "/zarr.json");
}

static bool DiscoverStoreV2(FileSystem &fs, const string &store_path, bool is_remote, vector<ZarrGroupEntry> &groups,
                            vector<ZarrArrayEntry> &arrays, vector<ZarrChunkEntry> &chunks, bool collect_chunks) {
	if (DiscoverConsolidatedStoreV2(fs, store_path, groups, arrays, chunks, collect_chunks)) {
		return true;
	}
	if (is_remote) {
		if (HasV2RootMarker(fs, store_path)) {
			throw InvalidInputException("Remote Zarr v2 stores currently require consolidated metadata (.zmetadata): %s",
			                            store_path);
		}
		return false;
	}
	if (!fs.DirectoryExists(store_path)) {
		return false;
	}
	TraverseStoreV2(fs, store_path, store_path, "", collect_chunks, groups, arrays, chunks);
	return !groups.empty() || !arrays.empty();
}

static bool DiscoverStoreV3(FileSystem &fs, const string &store_path, bool is_remote, vector<ZarrGroupEntry> &groups,
                            vector<ZarrArrayEntry> &arrays, vector<ZarrChunkEntry> &chunks, bool collect_chunks) {
	if (!HasV3RootMarker(fs, store_path)) {
		return false;
	}
	if (is_remote) {
		auto metadata_path = store_path + "/zarr.json";
		auto metadata_text = ReadTextFile(fs, metadata_path);
		unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)> doc(
		    yyjson_read(metadata_text.c_str(), metadata_text.size(), 0), yyjson_doc_free);
		if (!doc) {
			throw InvalidInputException("Failed to parse %s", metadata_path);
		}
		auto root = yyjson_doc_get_root(doc.get());
		auto node_type = RequireNodeType(root, metadata_path);
		if (node_type == "array") {
			auto entry = ParseArrayMetadataObjectV3(root, store_path, "", metadata_path);
			if (collect_chunks) {
				auto array_chunks = GenerateChunkEntries(fs, store_path, entry, false);
				chunks.insert(chunks.end(), array_chunks.begin(), array_chunks.end());
			}
			arrays.push_back(std::move(entry));
			return true;
		}
		throw InvalidInputException(
		    "Remote non-consolidated Zarr v3 group discovery is not yet supported; query a specific array path or use a local store: %s",
		    store_path);
	}
	TraverseStoreV3(fs, store_path, store_path, "", collect_chunks, groups, arrays, chunks);
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

static void DiscoverStore(ClientContext &context, const string &path, vector<ZarrGroupEntry> &groups,
                          vector<ZarrArrayEntry> &arrays, vector<ZarrChunkEntry> &chunks, bool collect_chunks,
                          ZarrVersionOverride version_override) {
	FileSystem &fs = FileSystem::GetFileSystem(context);
	EnsureRemoteFilesystemSupport(context, path);
	auto store_path = NormalizeStorePath(fs, path);
	auto is_remote = FileSystem::IsRemoteFile(path);
	auto has_v2_root = HasV2RootMarker(fs, store_path);
	auto has_v3_root = HasV3RootMarker(fs, store_path);
	bool discovered = false;
	switch (version_override) {
	case ZarrVersionOverride::AUTO:
		if (has_v2_root && has_v3_root) {
			throw InvalidInputException(
			    "Zarr store contains both v2 and v3 root metadata markers; use an explicit version override: %s",
			    store_path);
		}
		discovered = DiscoverStoreV3(fs, store_path, is_remote, groups, arrays, chunks, collect_chunks) ||
		             DiscoverStoreV2(fs, store_path, is_remote, groups, arrays, chunks, collect_chunks);
		break;
	case ZarrVersionOverride::V2:
		discovered = DiscoverStoreV2(fs, store_path, is_remote, groups, arrays, chunks, collect_chunks);
		break;
	case ZarrVersionOverride::V3:
		discovered = DiscoverStoreV3(fs, store_path, is_remote, groups, arrays, chunks, collect_chunks);
		break;
	default:
		throw InternalException("Unexpected zarr version override");
	}
	if (!discovered) {
		auto version_label = version_override == ZarrVersionOverride::V2 ? "v2"
		                   : version_override == ZarrVersionOverride::V3 ? "v3"
		                                                               : "v2 or v3";
		throw InvalidInputException("Zarr store path does not contain recognizable %s metadata: %s", version_label,
		                            store_path);
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

static unique_ptr<FunctionData> BindGroupsInternal(ClientContext &context, const string &store_path,
                                                   ZarrVersionOverride version_override,
                                                   vector<LogicalType> &return_types, vector<string> &names) {
	vector<ZarrGroupEntry> groups;
	vector<ZarrArrayEntry> arrays;
	vector<ZarrChunkEntry> chunks;
	DiscoverStore(context, store_path, groups, arrays, chunks, false, version_override);
	names = {"store_path", "group_path", "zarr_format", "metadata_path"};
	return_types = {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::BIGINT, LogicalType::VARCHAR};
	return make_uniq<ZarrBindData<ZarrGroupEntry>>(std::move(groups));
}

static unique_ptr<FunctionData> BindGroups(ClientContext &context, TableFunctionBindInput &input,
                                           vector<LogicalType> &return_types, vector<string> &names) {
	if (input.inputs.empty() || input.inputs[0].IsNull()) {
		throw BinderException("zarr_groups requires a non-NULL store path");
	}
	auto version_override = input.inputs.size() > 1 ? ParseZarrVersionOverride(input.inputs[1], "zarr_groups")
	                                                : ZarrVersionOverride::AUTO;
	return BindGroupsInternal(context, StringValue::Get(input.inputs[0]), version_override, return_types, names);
}

static unique_ptr<FunctionData> BindArraysInternal(ClientContext &context, const string &store_path,
                                                   ZarrVersionOverride version_override,
                                                   vector<LogicalType> &return_types, vector<string> &names) {
	vector<ZarrGroupEntry> groups;
	vector<ZarrArrayEntry> arrays;
	vector<ZarrChunkEntry> chunks;
	DiscoverStore(context, store_path, groups, arrays, chunks, false, version_override);

	names = {"store_path", "array_path", "zarr_format",         "rank",         "shape", "chunk_shape", "dtype",
	         "order",      "compressor", "chunk_key_encoding",  "dimension_separator", "metadata_path"};
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
	                LogicalType::VARCHAR,
	                LogicalType::VARCHAR};
	return make_uniq<ZarrBindData<ZarrArrayEntry>>(std::move(arrays));
}

static unique_ptr<FunctionData> BindArrays(ClientContext &context, TableFunctionBindInput &input,
                                           vector<LogicalType> &return_types, vector<string> &names) {
	if (input.inputs.empty() || input.inputs[0].IsNull()) {
		throw BinderException("zarr_arrays requires a non-NULL store path");
	}
	auto version_override = input.inputs.size() > 1 ? ParseZarrVersionOverride(input.inputs[1], "zarr_arrays")
	                                                : ZarrVersionOverride::AUTO;
	return BindArraysInternal(context, StringValue::Get(input.inputs[0]), version_override, return_types, names);
}

static unique_ptr<FunctionData> BindChunksInternal(ClientContext &context, const string &store_path,
                                                   ZarrVersionOverride version_override,
                                                   vector<LogicalType> &return_types, vector<string> &names) {
	vector<ZarrGroupEntry> groups;
	vector<ZarrArrayEntry> arrays;
	vector<ZarrChunkEntry> chunks;
	DiscoverStore(context, store_path, groups, arrays, chunks, true, version_override);

	names = {"store_path", "array_path", "chunk_key", "chunk_coords", "file_path", "file_size_bytes"};
	return_types = {LogicalType::VARCHAR, LogicalType::VARCHAR,
	                LogicalType::VARCHAR, LogicalType::LIST(LogicalType::BIGINT),
	                LogicalType::VARCHAR, LogicalType::BIGINT};
	return make_uniq<ZarrBindData<ZarrChunkEntry>>(std::move(chunks));
}

static unique_ptr<FunctionData> BindChunks(ClientContext &context, TableFunctionBindInput &input,
                                           vector<LogicalType> &return_types, vector<string> &names) {
	if (input.inputs.empty() || input.inputs[0].IsNull()) {
		throw BinderException("zarr_chunks requires a non-NULL store path");
	}
	auto version_override = input.inputs.size() > 1 ? ParseZarrVersionOverride(input.inputs[1], "zarr_chunks")
	                                                : ZarrVersionOverride::AUTO;
	return BindChunksInternal(context, StringValue::Get(input.inputs[0]), version_override, return_types, names);
}

static unique_ptr<FunctionData> BindCellsInternal(ClientContext &context, const string &store_path,
                                                  const string &array_path, ZarrVersionOverride version_override,
                                                  vector<LogicalType> &return_types, vector<string> &names) {
	vector<ZarrGroupEntry> groups;
	vector<ZarrArrayEntry> arrays;
	vector<ZarrChunkEntry> chunks;
	DiscoverStore(context, store_path, groups, arrays, chunks, false, version_override);
	auto &array = FindArrayEntry(arrays, array_path);
	if (!array.supports_cells) {
		throw InvalidInputException("%s: %s", array.cells_error, array.array_path);
	}
	auto cell_array = array;
	auto dtype = ParseNumericDType(array.dtype);
	chunks = cell_array.is_sharding_indexed
	             ? GenerateCellChunkEntries(FileSystem::GetFileSystem(context), array.store_path, array,
	                                        HasMaterializedFillValue(array))
	             : GenerateChunkEntries(FileSystem::GetFileSystem(context), cell_array.store_path, cell_array,
	                                    HasMaterializedFillValue(cell_array));
	if (cell_array.is_sharding_indexed) {
		cell_array.storage_chunks = cell_array.chunks;
		cell_array.chunks = cell_array.inner_chunks;
	}

	names.reserve(cell_array.rank + 1);
	return_types.reserve(cell_array.rank + 1);
	for (idx_t dim = 0; dim < NumericCast<idx_t>(cell_array.rank); dim++) {
		names.push_back("dim_" + std::to_string(dim));
		return_types.push_back(LogicalType::BIGINT);
	}
	names.push_back("value");
	return_types.push_back(dtype.logical_type);

	return make_uniq<ZarrCellsBindData>(cell_array, std::move(chunks));
}

static unique_ptr<FunctionData> BindCells(ClientContext &context, TableFunctionBindInput &input,
                                          vector<LogicalType> &return_types, vector<string> &names) {
	if (input.inputs.size() < 2 || input.inputs[0].IsNull() || input.inputs[1].IsNull()) {
		throw BinderException("zarr_cells requires a non-NULL store path and array path");
	}
	auto version_override = input.inputs.size() > 2 ? ParseZarrVersionOverride(input.inputs[2], "zarr_cells")
	                                                : ZarrVersionOverride::AUTO;
	return BindCellsInternal(context, StringValue::Get(input.inputs[0]),
	                         NormalizeArrayPath(StringValue::Get(input.inputs[1])), version_override, return_types,
	                         names);
}

static unique_ptr<FunctionData> BindZarr(ClientContext &context, TableFunctionBindInput &input,
                                         vector<LogicalType> &return_types, vector<string> &names) {
	if (input.inputs.empty() || input.inputs[0].IsNull()) {
		throw BinderException("zarr requires a non-NULL store path");
	}
	return BindArraysInternal(context, StringValue::Get(input.inputs[0]), ZarrVersionOverride::AUTO, return_types,
	                          names);
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
		output.SetValue(9, count, Value(entry.chunk_key_encoding));
		output.SetValue(10, count, Value(entry.dimension_separator));
		output.SetValue(11, count, Value(entry.metadata_path));
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
			Value cell_value;
			if (chunk.present) {
				auto value_ptr = state.decoded_chunk.data() + (linear_index * state.dtype.element_size);
				cell_value = DecodeNumericValue(value_ptr, state.dtype);
			} else {
				cell_value = state.fill_value;
			}
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
	result->has_fill_value = HasMaterializedFillValue(bind_data.array);
	if (result->has_fill_value) {
		result->fill_value = ParseFillValue(bind_data.array, result->dtype);
	}
	result->chunk_element_count = Product(bind_data.array.chunks);
	if (result->dtype.element_size != 0 &&
	    result->chunk_element_count > std::numeric_limits<idx_t>::max() / result->dtype.element_size) {
		throw InvalidInputException("Zarr chunk byte size overflow");
	}
	result->expected_chunk_bytes = result->chunk_element_count * result->dtype.element_size;
	return std::move(result);
}

TableFunctionSet ZarrMetadata::GetGroupsFunction() {
	TableFunctionSet set("zarr_groups");
	set.AddFunction(TableFunction("zarr_groups", {LogicalType::VARCHAR}, ScanGroups, BindGroups, ZarrInit));
	set.AddFunction(
	    TableFunction("zarr_groups", {LogicalType::VARCHAR, LogicalType::VARCHAR}, ScanGroups, BindGroups, ZarrInit));
	return set;
}

TableFunctionSet ZarrMetadata::GetArraysFunction() {
	TableFunctionSet set("zarr_arrays");
	set.AddFunction(TableFunction("zarr_arrays", {LogicalType::VARCHAR}, ScanArrays, BindArrays, ZarrInit));
	set.AddFunction(
	    TableFunction("zarr_arrays", {LogicalType::VARCHAR, LogicalType::VARCHAR}, ScanArrays, BindArrays, ZarrInit));
	return set;
}

TableFunctionSet ZarrMetadata::GetChunksFunction() {
	TableFunctionSet set("zarr_chunks");
	set.AddFunction(TableFunction("zarr_chunks", {LogicalType::VARCHAR}, ScanChunks, BindChunks, ZarrInit));
	set.AddFunction(
	    TableFunction("zarr_chunks", {LogicalType::VARCHAR, LogicalType::VARCHAR}, ScanChunks, BindChunks, ZarrInit));
	return set;
}

TableFunctionSet ZarrMetadata::GetCellsFunction() {
	TableFunctionSet set("zarr_cells");
	TableFunction base("zarr_cells", {LogicalType::VARCHAR, LogicalType::VARCHAR}, ScanCells, BindCells, InitCells);
	base.projection_pushdown = true;
	base.filter_pushdown = true;
	base.filter_prune = true;
	set.AddFunction(base);
	TableFunction with_override("zarr_cells",
	                            {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR}, ScanCells,
	                            BindCells, InitCells);
	with_override.projection_pushdown = true;
	with_override.filter_pushdown = true;
	with_override.filter_prune = true;
	set.AddFunction(with_override);
	return set;
}

TableFunctionSet ZarrMetadata::GetZarrFunction() {
	return TableFunctionSet(TableFunction("zarr", {LogicalType::VARCHAR}, ScanArrays, BindZarr, ZarrInit));
}

TableFunctionSet ZarrMetadata::GetZarrCellsAliasFunction() {
	TableFunctionSet set("zarr");
	TableFunction base("zarr", {LogicalType::VARCHAR, LogicalType::VARCHAR}, ScanCells, BindCells, InitCells);
	base.projection_pushdown = true;
	base.filter_pushdown = true;
	base.filter_prune = true;
	set.AddFunction(base);
	TableFunction with_override("zarr", {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR}, ScanCells,
	                                    BindCells, InitCells);
	with_override.projection_pushdown = true;
	with_override.filter_pushdown = true;
	with_override.filter_prune = true;
	set.AddFunction(with_override);
	return set;
}

} // namespace duckdb
