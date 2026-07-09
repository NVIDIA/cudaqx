/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_decoding_config.h"

#include "type_casters.h"
#include "cudaq/qec/realtime/decoding_config.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <unordered_set>

namespace nb = nanobind;

namespace cudaq::qec::decoding::config {
void bindDecodingConfig(nb::module_ &mod) {
  auto qecmod = nb::hasattr(mod, "qecrt")
                    ? nb::cast<nb::module_>(mod.attr("qecrt"))
                    : mod.def_submodule("qecrt");

  auto mod_cfg =
      qecmod.def_submodule("config", "Realtime decoding configuration");

  // Allow Python None to clear std::optional<T> fields.
  const auto setter_accepts_none = nb::for_setter(nb::arg("value").none());

  // srelay_bp_config
  nb::class_<config::srelay_bp_config>(mod_cfg, "srelay_bp_config",
                                       "Relay-BP decoder configuration.")
      .def(nb::init<>())
      .def(
          "__init__",
          [](config::srelay_bp_config &self,
             const cudaqx::heterogeneous_map &map) {
            new (&self)
                srelay_bp_config(srelay_bp_config::from_heterogeneous_map(map));
          },
          nb::arg("map"))
      .def_rw("pre_iter", &srelay_bp_config::pre_iter)
      .def_rw("num_sets", &srelay_bp_config::num_sets)
      .def_rw("stopping_criterion", &srelay_bp_config::stopping_criterion)
      .def_rw("stop_nconv", &srelay_bp_config::stop_nconv)
      .def("to_heterogeneous_map", &srelay_bp_config::to_heterogeneous_map,
           nb::rv_policy::move)
      .def_static("from_heterogeneous_map",
                  &srelay_bp_config::from_heterogeneous_map, nb::arg("map"));

  // nv_qldpc_decoder_config
  nb::class_<config::nv_qldpc_decoder_config>(
      mod_cfg, "nv_qldpc_decoder_config", "Optional decoder custom args.")
      .def(nb::init<>())
      .def(
          "__init__",
          [](config::nv_qldpc_decoder_config &self,
             const cudaqx::heterogeneous_map &map) {
            new (&self) nv_qldpc_decoder_config(
                nv_qldpc_decoder_config::from_heterogeneous_map(map));
          },
          nb::arg("map"))
      .def_rw("use_sparsity", &nv_qldpc_decoder_config::use_sparsity)
      .def_rw("error_rate", &nv_qldpc_decoder_config::error_rate)
      .def_rw("error_rate_vec", &nv_qldpc_decoder_config::error_rate_vec)
      .def_rw("max_iterations", &nv_qldpc_decoder_config::max_iterations)
      .def_rw("n_threads", &nv_qldpc_decoder_config::n_threads)
      .def_rw("use_osd", &nv_qldpc_decoder_config::use_osd)
      .def_rw("osd_method", &nv_qldpc_decoder_config::osd_method)
      .def_rw("osd_order", &nv_qldpc_decoder_config::osd_order)
      .def_rw("bp_batch_size", &nv_qldpc_decoder_config::bp_batch_size)
      .def_rw("osd_batch_size", &nv_qldpc_decoder_config::osd_batch_size)
      .def_rw("iter_per_check", &nv_qldpc_decoder_config::iter_per_check)
      .def_rw("clip_value", &nv_qldpc_decoder_config::clip_value)
      .def_rw("bp_method", &nv_qldpc_decoder_config::bp_method)
      .def_rw("scale_factor", &nv_qldpc_decoder_config::scale_factor)
      .def_rw("proc_float", &nv_qldpc_decoder_config::proc_float)
      .def_rw("gamma0", &nv_qldpc_decoder_config::gamma0)
      .def_rw("gamma_dist", &nv_qldpc_decoder_config::gamma_dist)
      .def_rw("explicit_gammas", &nv_qldpc_decoder_config::explicit_gammas)
      .def_rw("srelay_config", &nv_qldpc_decoder_config::srelay_config)
      .def_rw("bp_seed", &nv_qldpc_decoder_config::bp_seed)
      .def_rw("composition", &nv_qldpc_decoder_config::composition)
      .def("to_heterogeneous_map",
           &nv_qldpc_decoder_config::to_heterogeneous_map, nb::rv_policy::move)
      .def_static("from_heterogeneous_map",
                  &nv_qldpc_decoder_config::from_heterogeneous_map,
                  nb::arg("map"));

  // multi_error_lut_config
  nb::class_<config::multi_error_lut_config>(mod_cfg, "multi_error_lut_config",
                                             "Optional decoder custom args.")
      .def(nb::init<>())
      .def(
          "__init__",
          [](config::multi_error_lut_config &self,
             const cudaqx::heterogeneous_map &map) {
            new (&self) multi_error_lut_config(
                multi_error_lut_config::from_heterogeneous_map(map));
          },
          nb::arg("map"))
      .def_rw("lut_error_depth", &multi_error_lut_config::lut_error_depth)
      .def("to_heterogeneous_map",
           &multi_error_lut_config::to_heterogeneous_map, nb::rv_policy::move)
      .def_static("from_heterogeneous_map",
                  &multi_error_lut_config::from_heterogeneous_map,
                  nb::arg("map"));

  // trt_decoder_config
  nb::class_<config::trt_decoder_config>(mod_cfg, "trt_decoder_config",
                                         "TensorRT decoder configuration.")
      .def(nb::init<>())
      .def(
          "__init__",
          [](config::trt_decoder_config &self,
             const cudaqx::heterogeneous_map &map) {
            new (&self) trt_decoder_config(
                trt_decoder_config::from_heterogeneous_map(map));
          },
          nb::arg("map"))
      .def_rw("onnx_load_path", &trt_decoder_config::onnx_load_path,
              setter_accepts_none)
      .def_rw("engine_load_path", &trt_decoder_config::engine_load_path,
              setter_accepts_none)
      .def_rw("engine_save_path", &trt_decoder_config::engine_save_path,
              setter_accepts_none)
      .def_rw("precision", &trt_decoder_config::precision, setter_accepts_none)
      .def_rw("memory_workspace", &trt_decoder_config::memory_workspace,
              setter_accepts_none)
      .def_rw("batch_size", &trt_decoder_config::batch_size,
              setter_accepts_none)
      .def_rw("use_cuda_graph", &trt_decoder_config::use_cuda_graph,
              setter_accepts_none)
      .def_rw("global_decoder", &trt_decoder_config::global_decoder,
              setter_accepts_none)
      .def_prop_rw(
          "global_decoder_params",
          [](const trt_decoder_config &self) -> nb::object {
            if (std::holds_alternative<pymatching_config>(
                    self.global_decoder_params)) {
              return nb::cast(
                  std::get<pymatching_config>(self.global_decoder_params));
            }
            if (std::holds_alternative<chromobius_config>(
                    self.global_decoder_params)) {
              return nb::cast(
                  std::get<chromobius_config>(self.global_decoder_params));
            }
            return nb::none();
          },
          [](trt_decoder_config &self, nb::object value) {
            if (value.is_none()) {
              self.global_decoder_params = std::monostate();
            } else if (nb::isinstance<pymatching_config>(value)) {
              self.global_decoder_params = nb::cast<pymatching_config>(value);
            } else if (nb::isinstance<chromobius_config>(value)) {
              self.global_decoder_params = nb::cast<chromobius_config>(value);
            } else {
              throw nb::type_error(
                  "global_decoder_params must be pymatching_config, "
                  "chromobius_config, or None.");
            }
          },
          setter_accepts_none)
      .def("to_heterogeneous_map", &trt_decoder_config::to_heterogeneous_map,
           nb::rv_policy::move)
      .def_static("from_heterogeneous_map",
                  &trt_decoder_config::from_heterogeneous_map, nb::arg("map"));

  // pymatching_config
  nb::class_<config::pymatching_config>(mod_cfg, "pymatching_config",
                                        "PyMatching decoder configuration.")
      .def(nb::init<>())
      .def(
          "__init__",
          [](config::pymatching_config &self,
             const cudaqx::heterogeneous_map &map) {
            new (&self) pymatching_config(
                pymatching_config::from_heterogeneous_map(map));
          },
          nb::arg("map"))
      .def_rw("error_rate_vec", &pymatching_config::error_rate_vec)
      .def_rw("merge_strategy", &pymatching_config::merge_strategy)
      .def("to_heterogeneous_map", &pymatching_config::to_heterogeneous_map,
           nb::rv_policy::move)
      .def_static("from_heterogeneous_map",
                  &pymatching_config::from_heterogeneous_map, nb::arg("map"));

  // chromobius_config
  nb::class_<config::chromobius_config>(mod_cfg, "chromobius_config",
                                        "Chromobius decoder configuration.")
      .def(nb::init<>())
      .def(
          "__init__",
          [](config::chromobius_config &self,
             const cudaqx::heterogeneous_map &map) {
            new (&self) chromobius_config(
                chromobius_config::from_heterogeneous_map(map));
          },
          nb::arg("map"))
      .def_rw("drop_mobius_errors_involving_remnant_errors",
              &chromobius_config::drop_mobius_errors_involving_remnant_errors,
              setter_accepts_none)
      .def_rw("ignore_decomposition_failures",
              &chromobius_config::ignore_decomposition_failures,
              setter_accepts_none)
      .def_rw("include_coords_in_mobius_dem",
              &chromobius_config::include_coords_in_mobius_dem,
              setter_accepts_none)
      .def_rw("return_weight", &chromobius_config::return_weight,
              setter_accepts_none)
      .def_rw("write_mobius_match_to_stderr",
              &chromobius_config::write_mobius_match_to_stderr,
              setter_accepts_none)
      .def("to_heterogeneous_map", &chromobius_config::to_heterogeneous_map,
           nb::rv_policy::move)
      .def_static("from_heterogeneous_map",
                  &chromobius_config::from_heterogeneous_map, nb::arg("map"));

  // single_error_lut_config
  nb::class_<config::single_error_lut_config>(
      mod_cfg, "single_error_lut_config",
      "Single error LUT decoder configuration.")
      .def(nb::init<>())
      .def(
          "__init__",
          [](config::single_error_lut_config &self,
             const cudaqx::heterogeneous_map &map) {
            new (&self) single_error_lut_config(
                single_error_lut_config::from_heterogeneous_map(map));
          },
          nb::arg("map"))
      .def("to_heterogeneous_map",
           &single_error_lut_config::to_heterogeneous_map, nb::rv_policy::move)
      .def_static("from_heterogeneous_map",
                  &single_error_lut_config::from_heterogeneous_map,
                  nb::arg("map"));

  // sliding_window_config
  nb::class_<config::sliding_window_config>(
      mod_cfg, "sliding_window_config", "Sliding window decoder configuration.")
      .def(nb::init<>())
      .def(
          "__init__",
          [](config::sliding_window_config &self,
             const cudaqx::heterogeneous_map &map) {
            new (&self) sliding_window_config(
                sliding_window_config::from_heterogeneous_map(map));
          },
          nb::arg("map"))
      .def_rw("window_size", &sliding_window_config::window_size)
      .def_rw("step_size", &sliding_window_config::step_size)
      .def_rw("num_syndromes_per_round",
              &sliding_window_config::num_syndromes_per_round)
      .def_rw("straddle_start_round",
              &sliding_window_config::straddle_start_round)
      .def_rw("straddle_end_round", &sliding_window_config::straddle_end_round)
      .def_rw("error_rate_vec", &sliding_window_config::error_rate_vec)
      .def_rw("inner_decoder_name", &sliding_window_config::inner_decoder_name)
      .def_rw("single_error_lut_params",
              &sliding_window_config::single_error_lut_params)
      .def_rw("multi_error_lut_params",
              &sliding_window_config::multi_error_lut_params)
      .def_rw("nv_qldpc_decoder_params",
              &sliding_window_config::nv_qldpc_decoder_params)
      .def("to_heterogeneous_map", &sliding_window_config::to_heterogeneous_map,
           nb::rv_policy::move)
      .def_static("from_heterogeneous_map",
                  &sliding_window_config::from_heterogeneous_map,
                  nb::arg("map"));

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
      .def_rw("decoder_custom_args", &decoder_config::decoder_custom_args)
      .def(
          "set_decoder_custom_args",
          [](config::decoder_config &self, nb::object decoder_config) {
            if (nb::hasattr(decoder_config, "to_heterogeneous_map")) {
              nb::object hm_object =
                  decoder_config.attr("to_heterogeneous_map")();
              cudaqx::heterogeneous_map hm =
                  nb::cast<cudaqx::heterogeneous_map>(hm_object);
              self.set_decoder_custom_args_from_heterogeneous_map(hm);
              return;
            }
            throw nb::type_error("set_decoder_custom_args expects an object "
                                 "with to_heterogeneous_map().");
          },
          nb::arg("custom_args_obj"))
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
}
} // namespace cudaq::qec::decoding::config
