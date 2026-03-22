#include "zarr_metadata.hpp"

#include "duckdb/common/helper.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/function/table_function.hpp"
#include "yyjson.hpp"

#include <algorithm>
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

template <class ENTRY>
struct ZarrBindData : public TableFunctionData {
	explicit ZarrBindData(vector<ENTRY> entries_p) : entries(std::move(entries_p)) {
	}

	vector<ENTRY> entries;
};

struct ZarrGlobalState : public GlobalTableFunctionState {
	idx_t offset = 0;
};

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

static bool TryParseInteger(const string &text, int64_t &value) {
	if (text.empty()) {
		return false;
	}
	for (auto ch : text) {
		if (!StringUtil::CharacterIsDigit(ch)) {
			return false;
		}
	}
	value = std::stoll(text);
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

static void CollectSlashChunks(FileSystem &fs, const string &store_path, const string &array_path, const string &dir_path,
                               const string &relative_key, vector<ZarrChunkEntry> &entries) {
	fs.ListFiles(dir_path, [&](const string &child_name, bool is_dir) {
		if (StringUtil::StartsWith(child_name, ".")) {
			return;
		}
		auto child_path = fs.JoinPath(dir_path, child_name);
		auto next_key = relative_key.empty() ? child_name : relative_key + "/" + child_name;
		if (is_dir) {
			CollectSlashChunks(fs, store_path, array_path, child_path, next_key, entries);
			return;
		}
		vector<int64_t> coords;
		if (!TryParseChunkCoords(next_key, '/', coords)) {
			return;
		}
		auto handle = fs.OpenFile(child_path, FileFlags::FILE_FLAGS_READ);
		entries.push_back({store_path, array_path, next_key, std::move(coords), child_path,
		                   NumericCast<int64_t>(handle->GetFileSize())});
	});
}

static void CollectDotChunks(FileSystem &fs, const string &store_path, const string &array_path, const string &dir_path,
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
		auto handle = fs.OpenFile(child_path, FileFlags::FILE_FLAGS_READ);
		entries.push_back({store_path, array_path, child_name, std::move(coords), child_path,
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
	auto root = yyjson_doc_get_root(doc.get());
	if (!yyjson_is_obj(root)) {
		throw InvalidInputException("%s does not contain a JSON object", metadata_path);
	}

	auto compressor = JsonToString(yyjson_obj_get(root, "compressor"));
	auto dimension_separator = OptionalString(root, "dimension_separator", ".");
	auto shape = RequireInt64Array(root, "shape");
	auto chunks = RequireInt64Array(root, "chunks");

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

static void TraverseStore(FileSystem &fs, const string &store_path, const string &dir_path, const string &relative_path,
                          vector<ZarrGroupEntry> &groups, vector<ZarrArrayEntry> &arrays,
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
		auto root = yyjson_doc_get_root(doc.get());
		if (!yyjson_is_obj(root)) {
			throw InvalidInputException("%s does not contain a JSON object", group_metadata);
		}
		groups.push_back({store_path, FormatGroupPath(relative_path), RequireInt64(root, "zarr_format"), group_metadata});
	}

	if (fs.FileExists(array_metadata)) {
		auto entry = ParseArrayMetadata(fs, store_path, relative_path, dir_path);
		if (entry.dimension_separator == "/") {
			CollectSlashChunks(fs, store_path, entry.array_path, dir_path, "", chunks);
		} else {
			CollectDotChunks(fs, store_path, entry.array_path, dir_path, chunks);
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
		TraverseStore(fs, store_path, child_path, JoinNodePath(relative_path, child_name), groups, arrays, chunks);
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

static vector<ZarrGroupEntry> DiscoverGroups(ClientContext &context, const string &path) {
	FileSystem &fs = FileSystem::GetFileSystem(context);
	auto store_path = fs.ExpandPath(path);
	if (!fs.DirectoryExists(store_path)) {
		throw InvalidInputException("Zarr store path does not exist or is not a directory: %s", store_path);
	}
	vector<ZarrGroupEntry> groups;
	vector<ZarrArrayEntry> arrays;
	vector<ZarrChunkEntry> chunks;
	TraverseStore(fs, store_path, store_path, "", groups, arrays, chunks);
	std::sort(groups.begin(), groups.end(), [](const ZarrGroupEntry &lhs, const ZarrGroupEntry &rhs) {
		return std::tie(lhs.group_path, lhs.metadata_path) < std::tie(rhs.group_path, rhs.metadata_path);
	});
	return groups;
}

static void DiscoverStore(ClientContext &context, const string &path, vector<ZarrGroupEntry> &groups,
                          vector<ZarrArrayEntry> &arrays, vector<ZarrChunkEntry> &chunks) {
	FileSystem &fs = FileSystem::GetFileSystem(context);
	auto store_path = fs.ExpandPath(path);
	if (!fs.DirectoryExists(store_path)) {
		throw InvalidInputException("Zarr store path does not exist or is not a directory: %s", store_path);
	}
	TraverseStore(fs, store_path, store_path, "", groups, arrays, chunks);
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
	DiscoverStore(context, StringValue::Get(input.inputs[0]), groups, arrays, chunks);

	names = {"store_path", "array_path", "zarr_format", "rank", "shape", "chunk_shape",
	         "dtype",      "order",      "compressor",  "dimension_separator", "metadata_path"};
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
	DiscoverStore(context, StringValue::Get(input.inputs[0]), groups, arrays, chunks);

	names = {"store_path", "array_path", "chunk_key", "chunk_coords", "file_path", "file_size_bytes"};
	return_types = {LogicalType::VARCHAR,
	                LogicalType::VARCHAR,
	                LogicalType::VARCHAR,
	                LogicalType::LIST(LogicalType::BIGINT),
	                LogicalType::VARCHAR,
	                LogicalType::BIGINT};
	return make_uniq<ZarrBindData<ZarrChunkEntry>>(std::move(chunks));
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

TableFunction ZarrMetadata::GetGroupsFunction() {
	return TableFunction("zarr_groups", {LogicalType::VARCHAR}, ScanGroups, BindGroups, ZarrInit);
}

TableFunction ZarrMetadata::GetArraysFunction() {
	return TableFunction("zarr_arrays", {LogicalType::VARCHAR}, ScanArrays, BindArrays, ZarrInit);
}

TableFunction ZarrMetadata::GetChunksFunction() {
	return TableFunction("zarr_chunks", {LogicalType::VARCHAR}, ScanChunks, BindChunks, ZarrInit);
}

} // namespace duckdb
