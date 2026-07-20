/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_decoding_config.h"

#include "type_casters.h"
#include "cudaq/qec/decoder_config_schema.h"
#include "cudaq/qec/realtime/decoding_config.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>
#include <stdexcept>

namespace nb = nanobind;

namespace cudaq::qec::decoding::config {

namespace {

template <typename T>
T cast_param(const nb::object &value, const std::string &key,
             const std::string &schema_name, const char *kind_name) {
  try {
    return nb::cast<T>(value);
  } catch (...) {
    throw std::runtime_error("Parameter '" + key + "' of '" + schema_name +
                             "' expects a " + kind_name + " value.");
  }
}

// Convert a Python dict to the canonical storage types the decoder's
// registered schema declares (int32 params admit negative ints, f64 params
// admit Python ints, ...). The generic heterogeneous_map caster stores every
// Python int as std::size_t, which rejects negatives at assignment and makes
// f64 params unreadable at YAML emission / decoder construction.
cudaqx::heterogeneous_map
schema_typed_map_from_dict(const decoder_schema &schema, nb::dict dict) {
  cudaqx::heterogeneous_map map;
  nb::dict residual;
  for (auto item : dict) {
    const std::string key = nb::cast<std::string>(item.first);
    nb::object value = nb::borrow<nb::object>(item.second);
    const param_spec *spec = nullptr;
    for (const auto &candidate : schema.params) {
      if (candidate.key == key) {
        spec = &candidate;
        break;
      }
    }
    if (!spec) {
      // Unknown keys keep the generic conversion so validate_custom_args and
      // emission diagnostics can still name them.
      residual[item.first] = item.second;
      continue;
    }
    switch (spec->kind) {
    case param_kind::boolean:
      map.insert(key, cast_param<bool>(value, key, schema.name, "boolean"));
      break;
    case param_kind::int32:
      map.insert(key, cast_param<int>(value, key, schema.name, "32-bit int"));
      break;
    case param_kind::uint64:
      map.insert(key, cast_param<std::size_t>(value, key, schema.name,
                                              "non-negative int"));
      break;
    case param_kind::f64:
      map.insert(key, cast_param<double>(value, key, schema.name, "float"));
      break;
    case param_kind::string:
      map.insert(key,
                 cast_param<std::string>(value, key, schema.name, "string"));
      break;
    case param_kind::f64_vec:
      map.insert(key, cast_param<std::vector<double>>(value, key, schema.name,
                                                      "list-of-float"));
      break;
    case param_kind::f64_matrix:
      map.insert(key, cast_param<std::vector<std::vector<double>>>(
                          value, key, schema.name, "list-of-list-of-float"));
      break;
    case param_kind::subschema: {
      const auto *nested = find_decoder_schema(spec->subschema);
      if (nested && nb::isinstance<nb::dict>(value))
        map.insert(key, schema_typed_map_from_dict(*nested,
                                                   nb::cast<nb::dict>(value)));
      else
        map.insert(key, cast_param<cudaqx::heterogeneous_map>(
                            value, key, schema.name, "dict"));
      break;
    }
    case param_kind::discriminated: {
      const decoder_schema *nested = nullptr;
      if (dict.contains(spec->discriminator.c_str())) {
        nb::object discriminator = dict[spec->discriminator.c_str()];
        if (nb::isinstance<nb::str>(discriminator))
          nested = find_decoder_schema(nb::cast<std::string>(discriminator));
      }
      if (nested && nb::isinstance<nb::dict>(value))
        map.insert(key, schema_typed_map_from_dict(*nested,
                                                   nb::cast<nb::dict>(value)));
      else
        map.insert(key, cast_param<cudaqx::heterogeneous_map>(
                            value, key, schema.name, "dict"));
      break;
    }
    }
  }
  if (nb::len(residual) > 0) {
    auto generic = nb::cast<cudaqx::heterogeneous_map>(nb::object(residual));
    for (const auto &kv : generic)
      map.insert(kv.first, kv.second);
  }
  return map;
}

cudaqx::heterogeneous_map
custom_args_map_from_python(const std::string &decoder_type, nb::object value) {
  std::string schema_name = decoder_type;
  if (!nb::isinstance<nb::dict>(value) &&
      nb::hasattr(value, "to_heterogeneous_map")) {
    // Deprecated typed-config path (the cudaq_qec._compat shims and the
    // pre-schema classes they replace): the object reduces itself to a dict
    // of the explicitly-set parameters, and carries the decoder schema name
    // to convert that dict with, so conversion does not depend on
    // decoder_config.type having been assigned first.
    if (nb::hasattr(value, "_schema_name")) {
      nb::object schema_attr = value.attr("_schema_name");
      if (nb::isinstance<nb::str>(schema_attr))
        schema_name = nb::cast<std::string>(schema_attr);
    }
    value = value.attr("to_heterogeneous_map")();
  }
  if (nb::isinstance<nb::dict>(value))
    if (const auto *schema = find_decoder_schema(schema_name))
      return schema_typed_map_from_dict(*schema, nb::cast<nb::dict>(value));
  return nb::cast<cudaqx::heterogeneous_map>(value);
}

} // namespace

void bindDecodingConfig(nb::module_ &mod) {
  auto qecmod = nb::hasattr(mod, "qecrt")
                    ? nb::cast<nb::module_>(mod.attr("qecrt"))
                    : mod.def_submodule("qecrt");

  auto mod_cfg =
      qecmod.def_submodule("config", "Realtime decoding configuration");

  // decoder_config
  nb::class_<config::decoder_config>(mod_cfg, "decoder_config")
      .def(nb::init<>())
      .def_rw("id", &decoder_config::id)
      .def_rw("type", &decoder_config::type)
      .def_rw("cuda_device_id", &decoder_config::cuda_device_id)
      .def_rw("block_size", &decoder_config::block_size)
      .def_rw("syndrome_size", &decoder_config::syndrome_size)
      .def_rw("H_sparse", &decoder_config::H_sparse)
      .def_rw("O_sparse", &decoder_config::O_sparse)
      .def_rw("D_sparse", &decoder_config::D_sparse)
      .def_prop_rw(
          "decoder_custom_args",
          [](const decoder_config &self) -> nb::object {
            return nb::cast(self.decoder_custom_args.map());
          },
          [](decoder_config &self, nb::object value) {
            self.decoder_custom_args =
                custom_args_map_from_python(self.type, value);
          },
          "The decoder's parameter dict. Keys are governed by the parameter "
          "schema the decoder registered (see decoder_param_schema()); set "
          "`type` before assigning so values are converted to the schema's "
          "declared types. Reading returns a copy: mutate a local dict and "
          "assign it back rather than mutating the returned value in place.")
      .def(
          "set_decoder_custom_args",
          [](config::decoder_config &self, nb::object custom_args) {
            self.decoder_custom_args =
                custom_args_map_from_python(self.type, custom_args);
          },
          nb::arg("custom_args"),
          "Set the decoder parameter dict for this decoder (equivalent to "
          "assigning decoder_custom_args; set `type` first).")
      .def("validate_custom_args", &decoder_config::validate_custom_args,
           "Validate decoder_custom_args against the parameter schema "
           "registered for this decoder type: unknown keys, missing required "
           "keys, and the schema's own validation hook. Raises RuntimeError "
           "on the first violation. YAML parsing applies the same checks "
           "automatically; call this to vet a configuration built "
           "programmatically before using it.")
      .def("to_yaml_str", &decoder_config::to_yaml_str,
           nb::arg("column_wrap") = 80)
      .def_static("from_yaml_str", &decoder_config::from_yaml_str,
                  nb::arg("yaml_str"))
      .def("__eq__", [](const decoder_config &a, const decoder_config &b) {
        return a == b;
      });

  // multi_decoder_config
  nb::class_<multi_decoder_config>(mod_cfg, "multi_decoder_config")
      .def(nb::init<>())
      .def_rw("decoders", &multi_decoder_config::decoders)
      .def("validate_custom_args", &multi_decoder_config::validate_custom_args,
           "Validate every decoder's custom args against its registered "
           "parameter schema (see decoder_config.validate_custom_args).")
      .def("to_yaml_str", &multi_decoder_config::to_yaml_str,
           nb::arg("column_wrap") = 80)
      .def_static("from_yaml_str", &multi_decoder_config::from_yaml_str,
                  nb::arg("yaml_str"))
      .def("__eq__", [](const multi_decoder_config &a,
                        const multi_decoder_config &b) { return a == b; });

  // Library helpers
  mod_cfg.def(
      "configure_decoders", &configure_decoders, nb::arg("config"),
      "Configure decoders in a multi_decoder_config list; returns int status.");
  mod_cfg.def("configure_decoders_from_file", &configure_decoders_from_file,
              nb::arg("config_file"),
              "Configure decoders from a YAML file; returns int status.");
  mod_cfg.def("configure_decoders_from_str", &configure_decoders_from_str,
              nb::arg("config_str"),
              "Configure decoders from a YAML string; returns int status.");
  mod_cfg.def("finalize_decoders", &finalize_decoders,
              "Finalize decoder resources.");
  mod_cfg.def(
      "decoder_param_schema",
      [](const std::string &name) -> nb::object {
        const auto *schema = find_decoder_schema(name);
        if (!schema)
          return nb::none();
        auto kind_name = [](param_kind kind) -> const char * {
          switch (kind) {
          case param_kind::boolean:
            return "bool";
          case param_kind::int32:
            return "int32";
          case param_kind::uint64:
            return "uint64";
          case param_kind::f64:
            return "float64";
          case param_kind::string:
            return "string";
          case param_kind::f64_vec:
            return "float64_vec";
          case param_kind::f64_matrix:
            return "float64_matrix";
          case param_kind::subschema:
            return "subschema";
          case param_kind::discriminated:
            return "discriminated";
          }
          return "unknown";
        };
        nb::list params;
        for (const auto &spec : schema->params) {
          nb::dict entry;
          entry["key"] = spec.key;
          entry["kind"] = kind_name(spec.kind);
          entry["required"] = spec.required;
          if (!spec.subschema.empty())
            entry["subschema"] = spec.subschema;
          if (!spec.discriminator.empty())
            entry["discriminator"] = spec.discriminator;
          params.append(entry);
        }
        return params;
      },
      nb::arg("decoder_name"),
      "Return the registered custom-args parameter schema for a decoder "
      "(list of parameter descriptors), or None if the decoder has not "
      "registered one.");
  mod_cfg.def("registered_decoder_schemas", &registered_decoder_schema_names,
              "Names of all decoders (and nested sections) with registered "
              "custom-args parameter schemas.");
  mod_cfg.def(
      "decoder_config_json_schema", &decoder_config_json_schema,
      "Return a JSON Schema (draft 2020-12) document, as a string, that "
      "validates multi_decoder_config YAML files. Generated from the decoder "
      "parameter schemas registered in this installation (including loaded "
      "third-party decoder plugins), for use with standard tools such as "
      "check-jsonschema or the python jsonschema package. Schema validate "
      "hooks are not representable in JSON Schema, so a passing document may "
      "still be rejected when parsed.");
}
} // namespace cudaq::qec::decoding::config
