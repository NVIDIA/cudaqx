/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"
#include "realtime_decoding.h"
#include "cudaq/qec/decoder_config_payload.h"
#include "cudaq/qec/decoder_config_schema.h"
#include "cudaq/qec/logger.h"
#include "cudaq/qec/realtime/decoding_config.h"
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace cudaq::qec::decoding::config {

bool decoder_custom_args_t::operator==(
    const decoder_custom_args_t &other) const {
  return custom_args_maps_equal(map_, other.map_);
}

void decoder_config::validate_custom_args() const {
  config::validate_custom_args(type, decoder_custom_args.map());
}

cudaqx::heterogeneous_map
decoder_config::decoder_custom_args_to_heterogeneous_map() const {
  auto args = decoder_custom_args.map();
  if (const auto *schema = find_decoder_schema(type)) {
    // Same normalization on every consumer path: non-schema keys are
    // warned-and-dropped (they could never round-trip through YAML, so a
    // local decoder must not see them either) and schema-declared defaults
    // are materialized. YAML emission serializes this same map, so a config
    // reaches a local decoder and a remote target identically.
    drop_non_schema_keys(*schema, args);
    materialize_default_args(*schema, args);
  }
  return args;
}

void multi_decoder_config::validate_custom_args() const {
  for (const auto &decoder : decoders)
    decoder.validate_custom_args();
}

// Post-parse pass over a schema-parsed custom-args map: materialize defaulted
// discriminated sections, then run the canonical registry validation
// (required keys, per-schema hooks; its unknown-key check is a no-op here
// because the parser already rejected unknown keys). Only invoked when the
// section was present in the input document, mirroring the previous behavior
// where an absent decoder_custom_args section skipped its mapping (and
// therefore its required-key checks) entirely.
static void finalize_parsed_args(const decoder_schema &schema,
                                 cudaqx::heterogeneous_map &map,
                                 const std::string &context) {
  materialize_default_args(schema, map);
  validate_custom_args(schema, map, context);
}

} // namespace cudaq::qec::decoding::config

LLVM_YAML_IS_SEQUENCE_VECTOR(std::vector<double>)
LLVM_YAML_IS_SEQUENCE_VECTOR(cudaq::qec::decoding::config::decoder_config)

namespace llvm::yaml {

// Binds a heterogeneous_map to the decoder_schema that describes it so the
// generic mapping traits below can convert between the two. The schema drives
// everything: which keys are legal, the canonical storage type of each value,
// and how nested sections resolve their schemas.
struct schema_binding {
  cudaqx::heterogeneous_map *map = nullptr;
  const cudaq::qec::decoding::config::decoder_schema *schema = nullptr;
};

namespace {

template <typename T>
void input_schema_scalar(IO &io, const std::string &key,
                         cudaqx::heterogeneous_map &map) {
  T value{};
  io.mapRequired(key.c_str(), value);
  map.insert(key, value);
}

template <typename T>
void output_schema_scalar(IO &io, const std::string &key,
                          const cudaqx::heterogeneous_map &map) {
  T value = map.get<T>(key);
  io.mapRequired(key.c_str(), value);
}

} // namespace

template <>
struct CustomMappingTraits<schema_binding> {
  using param_kind = cudaq::qec::decoding::config::param_kind;
  using param_spec = cudaq::qec::decoding::config::param_spec;

  static void inputOne(IO &io, StringRef key, schema_binding &binding) {
    const std::string key_str = key.str();
    const param_spec *spec = nullptr;
    for (const auto &candidate : binding.schema->params) {
      if (candidate.key == key_str) {
        spec = &candidate;
        break;
      }
    }
    if (!spec)
      throw std::runtime_error("Unknown key '" + key_str + "' in '" +
                               binding.schema->name + "' parameters.");

    switch (spec->kind) {
    case param_kind::boolean:
      input_schema_scalar<bool>(io, key_str, *binding.map);
      break;
    case param_kind::int32:
      input_schema_scalar<int>(io, key_str, *binding.map);
      break;
    case param_kind::uint64:
      input_schema_scalar<std::size_t>(io, key_str, *binding.map);
      break;
    case param_kind::f64:
      input_schema_scalar<double>(io, key_str, *binding.map);
      break;
    case param_kind::string:
      input_schema_scalar<std::string>(io, key_str, *binding.map);
      break;
    case param_kind::f64_vec:
      input_schema_scalar<std::vector<double>>(io, key_str, *binding.map);
      break;
    case param_kind::f64_matrix:
      input_schema_scalar<std::vector<std::vector<double>>>(io, key_str,
                                                            *binding.map);
      break;
    case param_kind::subschema: {
      const auto *nested_schema =
          cudaq::qec::decoding::config::find_decoder_schema(spec->subschema);
      if (!nested_schema)
        throw std::runtime_error("No schema registered under '" +
                                 spec->subschema + "' (needed to parse '" +
                                 key_str + "').");
      cudaqx::heterogeneous_map nested;
      schema_binding nested_binding{&nested, nested_schema};
      io.mapRequired(key_str.c_str(), nested_binding);
      binding.map->insert(key_str, nested);
      break;
    }
    case param_kind::discriminated: {
      // The nested schema is named by a sibling key. Read it through the IO
      // (document order does not matter; mapping keys are random access).
      std::string discriminator_value;
      io.mapOptional(spec->discriminator.c_str(), discriminator_value);
      if (discriminator_value.empty())
        throw std::runtime_error("'" + key_str + "' is present but '" +
                                 spec->discriminator + "' is not set.");
      const auto *nested_schema =
          cudaq::qec::decoding::config::find_decoder_schema(
              discriminator_value);
      if (!nested_schema)
        throw std::runtime_error(
            "'" + key_str + "' does not support " + spec->discriminator + " '" +
            discriminator_value +
            "': no parameter schema is registered under that name.");
      cudaqx::heterogeneous_map nested;
      schema_binding nested_binding{&nested, nested_schema};
      io.mapRequired(key_str.c_str(), nested_binding);
      binding.map->insert(key_str, nested);
      break;
    }
    }
  }

  static void output(IO &io, schema_binding &binding) {
    // Only schema keys are emitted; surface anything else (a typo in a
    // programmatically built map) instead of dropping it silently.
    for (const auto &kv : *binding.map) {
      bool known = false;
      for (const auto &spec : binding.schema->params) {
        if (spec.key == kv.first) {
          known = true;
          break;
        }
      }
      if (!known)
        CUDA_QEC_WARN("Key '{}' is not in the '{}' parameter schema; it is "
                      "omitted from the emitted YAML.",
                      kv.first, binding.schema->name);
    }
    // Emit in schema declaration order so output is deterministic.
    for (const auto &spec : binding.schema->params) {
      if (!binding.map->contains(spec.key))
        continue;
      switch (spec.kind) {
      case param_kind::boolean:
        output_schema_scalar<bool>(io, spec.key, *binding.map);
        break;
      case param_kind::int32:
        output_schema_scalar<int>(io, spec.key, *binding.map);
        break;
      case param_kind::uint64:
        output_schema_scalar<std::size_t>(io, spec.key, *binding.map);
        break;
      case param_kind::f64:
        output_schema_scalar<double>(io, spec.key, *binding.map);
        break;
      case param_kind::string:
        output_schema_scalar<std::string>(io, spec.key, *binding.map);
        break;
      case param_kind::f64_vec:
        output_schema_scalar<std::vector<double>>(io, spec.key, *binding.map);
        break;
      case param_kind::f64_matrix:
        output_schema_scalar<std::vector<std::vector<double>>>(io, spec.key,
                                                               *binding.map);
        break;
      case param_kind::subschema: {
        const auto *nested_schema =
            cudaq::qec::decoding::config::find_decoder_schema(spec.subschema);
        if (!nested_schema)
          throw std::runtime_error("No schema registered under '" +
                                   spec.subschema + "' (needed to emit '" +
                                   spec.key + "').");
        auto nested = binding.map->get<cudaqx::heterogeneous_map>(spec.key);
        schema_binding nested_binding{&nested, nested_schema};
        io.mapRequired(spec.key.c_str(), nested_binding);
        break;
      }
      case param_kind::discriminated: {
        std::string discriminator_value;
        if (binding.map->contains(spec.discriminator))
          discriminator_value =
              binding.map->get<std::string>(spec.discriminator);
        const auto *nested_schema =
            discriminator_value.empty()
                ? nullptr
                : cudaq::qec::decoding::config::find_decoder_schema(
                      discriminator_value);
        if (!nested_schema)
          throw std::runtime_error("'" + spec.key +
                                   "' is present but no parameter schema is "
                                   "registered for " +
                                   spec.discriminator + " '" +
                                   discriminator_value + "'.");
        auto nested = binding.map->get<cudaqx::heterogeneous_map>(spec.key);
        schema_binding nested_binding{&nested, nested_schema};
        io.mapRequired(spec.key.c_str(), nested_binding);
        break;
      }
      }
    }
  }
};

template <>
struct ScalarEnumerationTraits<cudaq::qec::decoding::config::DecoderDispatch> {
  static void
  enumeration(IO &io, cudaq::qec::decoding::config::DecoderDispatch &value) {
    io.enumCase(value, "host",
                cudaq::qec::decoding::config::DecoderDispatch::host);
    io.enumCase(value, "device_graph",
                cudaq::qec::decoding::config::DecoderDispatch::device_graph);
  }
};

template <>
struct MappingTraits<cudaq::qec::decoding::config::decoder_config> {
  static void mapping(IO &io,
                      cudaq::qec::decoding::config::decoder_config &config) {
    io.mapRequired("id", config.id);
    io.mapRequired("type", config.type);
    io.mapOptional("dispatch", config.dispatch,
                   cudaq::qec::decoding::config::DecoderDispatch::host);
    io.mapOptional("cuda_device_id", config.cuda_device_id);
    io.mapRequired("block_size", config.block_size);
    io.mapRequired("syndrome_size", config.syndrome_size);
    io.mapRequired("H_sparse", config.H_sparse);
    io.mapRequired("O_sparse", config.O_sparse);
    io.mapRequired("D_sparse", config.D_sparse);

    // Validate that the number of rows in the H_sparse vector is equal to
    // syndrome_size.
    auto num_H_rows =
        std::count(config.H_sparse.begin(), config.H_sparse.end(), -1);
    if (num_H_rows != config.syndrome_size) {
      throw std::runtime_error(
          "Number of rows in H_sparse vector is not equal to syndrome_size: " +
          std::to_string(num_H_rows) +
          " != " + std::to_string(config.syndrome_size));
    }

    // Validate that no values in the H_sparse vector are out of range.
    for (auto value : config.H_sparse) {
      if (value < -1 || (value >= 0 && value >= config.block_size)) {
        throw std::runtime_error("Value in H_sparse vector is out of range: " +
                                 std::to_string(value));
      }
    }

    // Validate that no values in the O_sparse vector are out of range.
    for (auto value : config.O_sparse) {
      if (value < -1 || (value >= 0 && value >= config.block_size)) {
        throw std::runtime_error("Value in O_sparse vector is out of range: " +
                                 std::to_string(value));
      }
    }

    // Validate that if the D_sparse is provided, it is a valid D matrix. That
    // means that the number of rows in the D_sparse matrix should be equal to
    // the number of rows in the H_sparse matrix, and no row should be empty.
    if (!config.D_sparse.empty()) {
      auto num_D_rows =
          std::count(config.D_sparse.begin(), config.D_sparse.end(), -1);
      if (num_D_rows != config.syndrome_size) {
        throw std::runtime_error("Number of rows in D_sparse vector is not "
                                 "equal to syndrome_size: " +
                                 std::to_string(num_D_rows) +
                                 " != " + std::to_string(config.syndrome_size));
      }
      // No row should be empty, which means that there should be no
      // back-to-back -1 values.
      for (std::size_t i = 0; i < config.D_sparse.size() - 1; ++i) {
        if (config.D_sparse.at(i) == -1 && config.D_sparse.at(i + 1) == -1) {
          throw std::runtime_error("D_sparse row is empty for decoder " +
                                   std::to_string(config.id));
        }
      }
    }

    // Convert decoder_custom_args through the schema registered for this
    // decoder type. When no schema is registered, the key is intentionally
    // left unconsumed on input so the YAML parser's strict unknown-key check
    // rejects the section -- a decoder must register a schema (from its own
    // plugin library) to accept custom args.
    const auto *schema =
        cudaq::qec::decoding::config::find_decoder_schema(config.type);
    if (io.outputting()) {
      if (!config.decoder_custom_args.empty()) {
        if (!schema) {
          // Match the historical emission behavior (args for unknown types
          // were silently dropped) so configuration flows still fail with a
          // status code at decoder construction rather than throwing here.
          CUDA_QEC_WARN(
              "decoder_custom_args set for decoder type '{}' but no parameter "
              "schema is registered under that name; the args are omitted "
              "from the emitted YAML.",
              config.type);
        } else {
          // Emit the same normalized map the constructor-facing path
          // produces (non-schema keys dropped, defaults materialized), so a
          // programmatically built config serializes identically on first
          // emission and after a YAML round trip -- e.g. a trt config with
          // only `global_decoder` set gains `global_decoder_params: {}`
          // here, not just after re-parsing.
          auto args_map = config.decoder_custom_args_to_heterogeneous_map();
          schema_binding binding{&args_map, schema};
          io.mapRequired("decoder_custom_args", binding);
        }
      }
    } else if (schema) {
      bool args_present = false;
      for (const auto key : io.keys()) {
        if (key == "decoder_custom_args") {
          args_present = true;
          break;
        }
      }
      cudaqx::heterogeneous_map args_map;
      schema_binding binding{&args_map, schema};
      io.mapOptional("decoder_custom_args", binding);
      if (args_present)
        cudaq::qec::decoding::config::finalize_parsed_args(
            *schema, args_map, "decoder_custom_args (" + config.type + ")");
      config.decoder_custom_args = args_map;
    }
  }
};

// transport section mapping traits
template <>
struct MappingTraits<cudaq::qec::decoding::config::transport_shape_override> {
  static void
  mapping(IO &io,
          cudaq::qec::decoding::config::transport_shape_override &override_) {
    io.mapOptional("provider", override_.provider, std::string());
    io.mapOptional("args", override_.args);
  }
};

template <>
struct MappingTraits<cudaq::qec::decoding::config::transport_config> {
  static void
  mapping(IO &io, cudaq::qec::decoding::config::transport_config &transport) {
    io.mapOptional("provider", transport.provider, std::string());
    io.mapOptional("args", transport.args);
    io.mapOptional("device_graph", transport.device_graph,
                   cudaq::qec::decoding::config::transport_shape_override());
  }
};

// multi_decoder_config mapping traits
template <>
struct MappingTraits<cudaq::qec::decoding::config::multi_decoder_config> {
  static void
  mapping(IO &io, cudaq::qec::decoding::config::multi_decoder_config &config) {
    io.mapRequired("decoders", config.decoders);
    io.mapOptional("transport", config.transport,
                   cudaq::qec::decoding::config::transport_config());
  }
};

} // namespace llvm::yaml

// Static method to convert a YAML string to a multi_decoder_config.
cudaq::qec::decoding::config::multi_decoder_config
cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
    const std::string_view yaml_str) {
  multi_decoder_config config;
  llvm::yaml::Input yaml_in(yaml_str);
  yaml_in >> config;
  if (const auto error = yaml_in.error())
    throw std::runtime_error("Invalid decoder configuration YAML: " +
                             error.message());
  return config;
}

std::string cudaq::qec::decoding::config::multi_decoder_config::to_yaml_str(
    int column_wrap) {
  std::string yaml_str;
  llvm::raw_string_ostream yaml_stream(yaml_str);
  llvm::yaml::Output yaml_out(yaml_stream, nullptr, column_wrap);
  yaml_out << *this;
  return yaml_str;
}

cudaq::qec::decoding::config::decoder_config
cudaq::qec::decoding::config::decoder_config::from_yaml_str(
    const std::string &yaml_str) {
  decoder_config config;
  llvm::yaml::Input yaml_in(yaml_str);
  yaml_in >> config;
  if (const auto error = yaml_in.error())
    throw std::runtime_error("Invalid decoder configuration YAML: " +
                             error.message());
  return config;
}

std::string
cudaq::qec::decoding::config::decoder_config::to_yaml_str(int column_wrap) {
  std::string yaml_str;
  llvm::raw_string_ostream yaml_stream(yaml_str);
  llvm::yaml::Output yaml_out(yaml_stream, nullptr, column_wrap);
  yaml_out << *this;
  return yaml_str;
}

namespace cudaq::qec::decoding::config {

// ---------------------------------------------------------------------------
// JSON Schema export
//
// Translates the registered decoder parameter schemas plus the fixed
// decoder_config envelope (the fields MappingTraits<decoder_config> maps
// above) into a JSON Schema draft 2020-12 document, so standard tooling can
// validate user-provided configuration YAML offline. The document is a
// snapshot of what this installation can parse: it enumerates the schemas
// registered at call time, exactly as the runtime parser resolves them.
// ---------------------------------------------------------------------------

namespace {

// JSON-pointer token escaping for schema names used inside $ref paths.
std::string json_pointer_escape(const std::string &name) {
  std::string out;
  for (char c : name) {
    if (c == '~')
      out += "~0";
    else if (c == '/')
      out += "~1";
    else
      out += c;
  }
  return out;
}

std::string params_ref(const std::string &name) {
  return "#/$defs/decoder_params/" + json_pointer_escape(name);
}

llvm::json::Object json_schema_for_param(const param_spec &spec) {
  using k = param_kind;
  switch (spec.kind) {
  case k::boolean:
    return llvm::json::Object{{"type", "boolean"}};
  case k::int32:
    return llvm::json::Object{{"type", "integer"}};
  case k::uint64:
    return llvm::json::Object{{"type", "integer"}, {"minimum", 0}};
  case k::f64:
    return llvm::json::Object{{"type", "number"}};
  case k::string:
    return llvm::json::Object{{"type", "string"}};
  case k::f64_vec:
    return llvm::json::Object{
        {"type", "array"}, {"items", llvm::json::Object{{"type", "number"}}}};
  case k::f64_matrix:
    return llvm::json::Object{
        {"type", "array"},
        {"items", llvm::json::Object{
                      {"type", "array"},
                      {"items", llvm::json::Object{{"type", "number"}}}}}};
  case k::subschema:
    return llvm::json::Object{{"$ref", params_ref(spec.subschema)}};
  case k::discriminated:
    // The concrete shape is selected by the discriminator value; the
    // dispatch clauses emitted below refine this.
    return llvm::json::Object{{"type", "object"}};
  }
  return llvm::json::Object{};
}

llvm::json::Array registered_name_array(const std::vector<std::string> &names) {
  llvm::json::Array arr;
  for (const auto &name : names)
    arr.push_back(name);
  return arr;
}

llvm::json::Object
decoder_params_json_schema(const decoder_schema &schema,
                           const std::vector<std::string> &all_names) {
  llvm::json::Object properties;
  llvm::json::Array required;
  llvm::json::Array all_of;
  for (const auto &spec : schema.params) {
    properties[spec.key] = json_schema_for_param(spec);
    if (spec.required)
      required.push_back(spec.key);
    if (spec.kind == param_kind::discriminated) {
      // When the section is present, its discriminator must be present and
      // name a registered schema (mirrors the parser's checks).
      all_of.push_back(llvm::json::Object{
          {"if", llvm::json::Object{{"required", llvm::json::Array{spec.key}}}},
          {"then",
           llvm::json::Object{
               {"required", llvm::json::Array{spec.discriminator}},
               {"properties",
                llvm::json::Object{
                    {spec.discriminator,
                     llvm::json::Object{
                         {"enum", registered_name_array(all_names)}}}}}}}});
      // Each candidate discriminator value selects that schema for the
      // section.
      for (const auto &name : all_names)
        all_of.push_back(llvm::json::Object{
            {"if",
             llvm::json::Object{
                 {"properties",
                  llvm::json::Object{{spec.discriminator,
                                      llvm::json::Object{{"const", name}}}}},
                 {"required",
                  llvm::json::Array{spec.discriminator, spec.key}}}},
            {"then", llvm::json::Object{
                         {"properties",
                          llvm::json::Object{
                              {spec.key, llvm::json::Object{
                                             {"$ref", params_ref(name)}}}}}}}});
    }
  }
  llvm::json::Object out{{"type", "object"},
                         {"additionalProperties", false},
                         {"properties", std::move(properties)}};
  if (!required.empty())
    out["required"] = std::move(required);
  if (!all_of.empty())
    out["allOf"] = std::move(all_of);
  return out;
}

} // namespace

std::string decoder_config_json_schema() {
  const auto names = registered_decoder_schema_names();

  llvm::json::Object decoder_params;
  for (const auto &name : names)
    decoder_params[name] =
        decoder_params_json_schema(*find_decoder_schema(name), names);

  // The fixed decoder_config envelope; keep in sync with
  // MappingTraits<decoder_config> above.
  llvm::json::Object config_properties{
      {"id", llvm::json::Object{{"type", "integer"}}},
      {"type", llvm::json::Object{{"type", "string"}}},
      {"dispatch",
       llvm::json::Object{{"enum", llvm::json::Array{"host", "device_graph"}}}},
      {"cuda_device_id",
       llvm::json::Object{{"type", "integer"}, {"minimum", 0}}},
      {"block_size", llvm::json::Object{{"type", "integer"}, {"minimum", 0}}},
      {"syndrome_size",
       llvm::json::Object{{"type", "integer"}, {"minimum", 0}}},
      {"H_sparse", llvm::json::Object{{"$ref", "#/$defs/sparse_matrix"}}},
      {"O_sparse", llvm::json::Object{{"$ref", "#/$defs/sparse_matrix"}}},
      {"D_sparse", llvm::json::Object{{"$ref", "#/$defs/sparse_matrix"}}},
      {"decoder_custom_args", llvm::json::Object{{"type", "object"}}},
  };

  // Per-type dispatch of decoder_custom_args, generated from the registry:
  // a registered type's args follow its schema; a type with no registered
  // schema accepts no args (the parser rejects the section outright).
  llvm::json::Array dispatch;
  for (const auto &name : names)
    dispatch.push_back(llvm::json::Object{
        {"if", llvm::json::Object{{"properties",
                                   llvm::json::Object{
                                       {"type",
                                        llvm::json::Object{{"const", name}}}}},
                                  {"required", llvm::json::Array{"type"}}}},
        {"then", llvm::json::Object{
                     {"properties", llvm::json::Object{
                                        {"decoder_custom_args",
                                         llvm::json::Object{
                                             {"$ref", params_ref(name)}}}}}}}});
  dispatch.push_back(llvm::json::Object{
      {"if",
       llvm::json::Object{
           {"properties",
            llvm::json::Object{
                {"type",
                 llvm::json::Object{
                     {"not", llvm::json::Object{{"enum", registered_name_array(
                                                             names)}}}}}}},
           {"required", llvm::json::Array{"type"}}}},
      {"then",
       llvm::json::Object{
           {"properties",
            llvm::json::Object{{"decoder_custom_args",
                                llvm::json::Object{{"maxProperties", 0}}}}}}}});

  llvm::json::Object defs{
      {"sparse_matrix",
       llvm::json::Object{{"type", "array"},
                          {"items", llvm::json::Object{{"type", "integer"},
                                                       {"minimum", -1}}}}},
      // Server-level transport section (see transport_config /
      // transport_shape_override); the wire is deployment config and lives
      // outside the decoders list.
      {"transport_shape_override",
       llvm::json::Object{
           {"type", "object"},
           {"properties",
            llvm::json::Object{
                {"provider", llvm::json::Object{{"type", "string"}}},
                {"args",
                 llvm::json::Object{
                     {"type", "array"},
                     {"items", llvm::json::Object{{"type", "string"}}}}}}},
           {"additionalProperties", false}}},
      {"transport_config",
       llvm::json::Object{
           {"type", "object"},
           {"properties",
            llvm::json::Object{
                {"provider", llvm::json::Object{{"type", "string"}}},
                {"args",
                 llvm::json::Object{
                     {"type", "array"},
                     {"items", llvm::json::Object{{"type", "string"}}}}},
                {"device_graph",
                 llvm::json::Object{
                     {"$ref", "#/$defs/transport_shape_override"}}}}},
           {"additionalProperties", false}}},
      {"decoder_config",
       llvm::json::Object{
           {"type", "object"},
           {"properties", std::move(config_properties)},
           {"required",
            llvm::json::Array{"id", "type", "block_size", "syndrome_size",
                              "H_sparse", "O_sparse", "D_sparse"}},
           {"additionalProperties", false},
           {"allOf", std::move(dispatch)}}},
      {"decoder_params", std::move(decoder_params)},
  };

  llvm::json::Object root{
      {"$schema", "https://json-schema.org/draft/2020-12/schema"},
      {"title", "CUDA-Q QEC realtime decoding configuration"},
      {"description",
       "Validates multi_decoder_config YAML documents. Generated from the "
       "decoder parameter schemas registered in this installation, so it "
       "reflects the decoder plugins loaded at generation time. Per-schema "
       "validate hooks (arbitrary cross-field checks) are not representable "
       "in JSON Schema; a document that passes may still be rejected when "
       "parsed."},
      {"type", "object"},
      {"properties",
       llvm::json::Object{
           {"decoders",
            llvm::json::Object{
                {"type", "array"},
                {"items",
                 llvm::json::Object{{"$ref", "#/$defs/decoder_config"}}}}},
           {"transport",
            llvm::json::Object{{"$ref", "#/$defs/transport_config"}}}}},
      {"required", llvm::json::Array{"decoders"}},
      {"additionalProperties", false},
      {"$defs", std::move(defs)},
  };

  std::string out;
  llvm::raw_string_ostream os(out);
  llvm::json::OStream json_out(os, /*IndentSize=*/2);
  json_out.value(llvm::json::Value(std::move(root)));
  return out;
}

// Stash a copy for consumers that build their own decoder instances from the
// process-wide configuration -- the decoding-server DeviceCallService plugin
// reads it when CUDAQ_QEC_DECODER_CONFIG is not set (in-process path).
// shared_ptr + mutex: the plugin reads this from the realtime dispatcher
// thread while the application thread may call configure_decoders() again;
// shared ownership keeps the reader's config alive across a concurrent
// replacement.
static std::mutex g_last_multi_decoder_config_mutex;
static std::shared_ptr<const multi_decoder_config> g_last_multi_decoder_config;

int configure_decoders(multi_decoder_config &config) {
  CUDA_QEC_INFO("Initializing realtime decoding library with config object");
  {
    std::lock_guard<std::mutex> lock(g_last_multi_decoder_config_mutex);
    g_last_multi_decoder_config =
        std::make_shared<const multi_decoder_config>(config);
  }
  // Publish the decoder configuration so CUDA-Q can inject it into
  // remote-target job requests. The cudaq integration (ExtraPayloadProvider) is
  // installed by cudaq-qec at load time; this call is a no-op when cudaq-qec is
  // not loaded, keeping this library free of any direct cudaq-common
  // dependency.
  cudaq::qec::publish_decoder_config_payload(config.to_yaml_str());
  return cudaq::qec::decoding::host::configure_decoders(config);
}

std::shared_ptr<const multi_decoder_config>
last_configured_multi_decoder_config() {
  std::lock_guard<std::mutex> lock(g_last_multi_decoder_config_mutex);
  return g_last_multi_decoder_config;
}

void log_config(const char *config_str, bool from_file) {
  const bool dump_config = []() {
    if (auto *ch = std::getenv("CUDAQ_QEC_DEBUG_DECODER"))
      if (ch[0] == '1' || ch[0] == 'y' || ch[0] == 'Y')
        return true;
    return false;
  }();

  if (dump_config) {
    if (cudaq::qec::detail::should_log(cudaq::qec::detail::log_level::info)) {
      CUDA_QEC_INFO(
          "Initializing realtime decoding library with config string: {}",
          config_str);
    } else {
      printf("Initializing realtime decoding library with config string: %s\n",
             config_str);
    }
  }
}

int configure_decoders_from_file(const char *config_file) {
  std::string config_file_str(config_file);
  CUDA_QEC_INFO("Initializing realtime decoding library with config file: {}",
                config_file_str);

  // Verify that the file exists.
  if (!std::filesystem::exists(config_file_str)) {
    CUDA_QEC_WARN("Config file does not exist: {}", config_file_str);
    return 1;
  }

  // Read the config file into a string.
  std::string config_str;
  std::ifstream config_file_stream(config_file_str);
  config_str = std::string(std::istreambuf_iterator<char>(config_file_stream),
                           std::istreambuf_iterator<char>());
  log_config(config_str.c_str(), /*from_file=*/true);
  auto config = multi_decoder_config::from_yaml_str(config_str);
  return configure_decoders(config);
}

int configure_decoders_from_str(const char *config_str) {
  CUDA_QEC_INFO(
      "Initializing realtime decoding library with raw config string");
  log_config(config_str, /*from_file=*/false);
  auto config = multi_decoder_config::from_yaml_str(config_str);
  return configure_decoders(config);
}

void finalize_decoders() { cudaq::qec::decoding::host::finalize_decoders(); }

} // namespace cudaq::qec::decoding::config
