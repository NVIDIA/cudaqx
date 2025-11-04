/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_decoding_config.h"

#include "type_casters.h"
#include "cudaq/qec/realtime/decoding_config.h"
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <unordered_set>
namespace py = pybind11;

namespace cudaq::qec::decoding::config {
void bindDecodingConfig(py::module &mod) {
  auto qecmod = py::hasattr(mod, "qecrt")
                    ? mod.attr("qecrt").cast<py::module_>()
                    : mod.def_submodule("qecrt");

  auto mod_cfg =
      qecmod.def_submodule("config", "Realtime decoding configuration");

  // srelay_bp_config
  py::class_<config::srelay_bp_config>(mod_cfg, "srelay_bp_config",
                                       "Relay-BP decoder configuration.")
      .def(py::init<>())
      .def(py::init([](const cudaqx::heterogeneous_map &map) {
             return srelay_bp_config::from_heterogeneous_map(map);
           }),
           py::arg("map"))
      .def_readwrite("pre_iter", &srelay_bp_config::pre_iter)
      .def_readwrite("num_sets", &srelay_bp_config::num_sets)
      .def_readwrite("stopping_criterion",
                     &srelay_bp_config::stopping_criterion)
      .def_readwrite("stop_nconv", &srelay_bp_config::stop_nconv)
      .def("to_heterogeneous_map", &srelay_bp_config::to_heterogeneous_map,
           py::return_value_policy::move)
      .def_static("from_heterogeneous_map",
                  &srelay_bp_config::from_heterogeneous_map, py::arg("map"));

  // nv_qldpc_decoder_config
  py::class_<config::nv_qldpc_decoder_config>(
      mod_cfg, "nv_qldpc_decoder_config", "Optional decoder custom args.")
      .def(py::init<>())
      .def(py::init([](const cudaqx::heterogeneous_map &map) {
             return nv_qldpc_decoder_config::from_heterogeneous_map(map);
           }),
           py::arg("map"))
      .def_readwrite("use_sparsity", &nv_qldpc_decoder_config::use_sparsity)
      .def_readwrite("error_rate", &nv_qldpc_decoder_config::error_rate)
      .def_readwrite("error_rate_vec", &nv_qldpc_decoder_config::error_rate_vec)
      .def_readwrite("max_iterations", &nv_qldpc_decoder_config::max_iterations)
      .def_readwrite("n_threads", &nv_qldpc_decoder_config::n_threads)
      .def_readwrite("use_osd", &nv_qldpc_decoder_config::use_osd)
      .def_readwrite("osd_method", &nv_qldpc_decoder_config::osd_method)
      .def_readwrite("osd_order", &nv_qldpc_decoder_config::osd_order)
      .def_readwrite("bp_batch_size", &nv_qldpc_decoder_config::bp_batch_size)
      .def_readwrite("osd_batch_size", &nv_qldpc_decoder_config::osd_batch_size)
      .def_readwrite("iter_per_check", &nv_qldpc_decoder_config::iter_per_check)
      .def_readwrite("clip_value", &nv_qldpc_decoder_config::clip_value)
      .def_readwrite("bp_method", &nv_qldpc_decoder_config::bp_method)
      .def_readwrite("scale_factor", &nv_qldpc_decoder_config::scale_factor)
      .def_readwrite("proc_float", &nv_qldpc_decoder_config::proc_float)
      .def_readwrite("gamma0", &nv_qldpc_decoder_config::gamma0)
      .def_readwrite("gamma_dist", &nv_qldpc_decoder_config::gamma_dist)
      .def_readwrite("explicit_gammas",
                     &nv_qldpc_decoder_config::explicit_gammas)
      .def_readwrite("srelay_config", &nv_qldpc_decoder_config::srelay_config)
      .def_readwrite("bp_seed", &nv_qldpc_decoder_config::bp_seed)
      .def_readwrite("composition", &nv_qldpc_decoder_config::composition)
      .def("to_heterogeneous_map",
           &nv_qldpc_decoder_config::to_heterogeneous_map,
           py::return_value_policy::move)
      .def_static("from_heterogeneous_map",
                  &nv_qldpc_decoder_config::from_heterogeneous_map,
                  py::arg("map"));

  // multi_error_lut_config
  py::class_<config::multi_error_lut_config>(mod_cfg, "multi_error_lut_config",
                                             "Optional decoder custom args.")
      .def(py::init<>())
      .def(py::init([](const cudaqx::heterogeneous_map &map) {
             return multi_error_lut_config::from_heterogeneous_map(map);
           }),
           py::arg("map"))
      .def_readwrite("lut_error_depth",
                     &multi_error_lut_config::lut_error_depth)
      .def("to_heterogeneous_map",
           &multi_error_lut_config::to_heterogeneous_map,
           py::return_value_policy::move)
      .def_static("from_heterogeneous_map",
                  &multi_error_lut_config::from_heterogeneous_map,
                  py::arg("map"));

  // single_error_lut_config
  py::class_<config::single_error_lut_config>(
      mod_cfg, "single_error_lut_config",
      "Single error LUT decoder configuration.")
      .def(py::init<>())
      .def(py::init([](const cudaqx::heterogeneous_map &map) {
             return single_error_lut_config::from_heterogeneous_map(map);
           }),
           py::arg("map"))
      .def("to_heterogeneous_map",
           &single_error_lut_config::to_heterogeneous_map,
           py::return_value_policy::move)
      .def_static("from_heterogeneous_map",
                  &single_error_lut_config::from_heterogeneous_map,
                  py::arg("map"));

  // sliding_window_config
  py::class_<config::sliding_window_config>(
      mod_cfg, "sliding_window_config", "Sliding window decoder configuration.")
      .def(py::init<>())
      .def(py::init([](const cudaqx::heterogeneous_map &map) {
             return sliding_window_config::from_heterogeneous_map(map);
           }),
           py::arg("map"))
      .def_readwrite("window_size", &sliding_window_config::window_size)
      .def_readwrite("step_size", &sliding_window_config::step_size)
      .def_readwrite("num_syndromes_per_round",
                     &sliding_window_config::num_syndromes_per_round)
      .def_readwrite("straddle_start_round",
                     &sliding_window_config::straddle_start_round)
      .def_readwrite("straddle_end_round",
                     &sliding_window_config::straddle_end_round)
      .def_readwrite("error_rate_vec", &sliding_window_config::error_rate_vec)
      .def_readwrite("inner_decoder_name",
                     &sliding_window_config::inner_decoder_name)
      .def_readwrite("single_error_lut_params",
                     &sliding_window_config::single_error_lut_params)
      .def_readwrite("multi_error_lut_params",
                     &sliding_window_config::multi_error_lut_params)
      .def_readwrite("nv_qldpc_decoder_params",
                     &sliding_window_config::nv_qldpc_decoder_params)
      .def("to_heterogeneous_map", &sliding_window_config::to_heterogeneous_map,
           py::return_value_policy::move)
      .def_static("from_heterogeneous_map",
                  &sliding_window_config::from_heterogeneous_map,
                  py::arg("map"));

  // decoder_config
  py::class_<config::decoder_config>(mod_cfg, "decoder_config")
      .def(py::init<>())
      .def_readwrite("id", &decoder_config::id)
      .def_readwrite("type", &decoder_config::type)
      .def_readwrite("block_size", &decoder_config::block_size)
      .def_readwrite("syndrome_size", &decoder_config::syndrome_size)
      .def_readwrite("num_syndromes_per_round",
                     &decoder_config::num_syndromes_per_round)
      .def_readwrite("H_sparse", &decoder_config::H_sparse)
      .def_readwrite("O_sparse", &decoder_config::O_sparse)
      .def_readwrite("D_sparse", &decoder_config::D_sparse)
      .def_readwrite("decoder_custom_args",
                     &decoder_config::decoder_custom_args)
      .def(
          "set_decoder_custom_args",
          [](config::decoder_config &self, py::object decoder_config) {
            if (py::hasattr(decoder_config, "to_heterogeneous_map")) {
              py::object hm_object =
                  decoder_config.attr("to_heterogeneous_map")();

              // The type_caster automatically converts dict to
              // heterogeneous_map
              cudaqx::heterogeneous_map hm =
                  py::cast<cudaqx::heterogeneous_map>(hm_object);
              self.set_decoder_custom_args_from_heterogeneous_map(hm);
              return;
            }
            throw py::type_error("set_decoder_custom_args expects an object "
                                 "with to_heterogeneous_map().");
          },
          py::arg("custom_args_obj"))
      .def("to_yaml_str", &decoder_config::to_yaml_str,
           py::arg("column_wrap") = 80)
      .def_static("from_yaml_str", &decoder_config::from_yaml_str,
                  py::arg("yaml_str"))
      .def(py::self == py::self);

  // multi_decoder_config
  py::class_<multi_decoder_config>(mod_cfg, "multi_decoder_config")
      .def(py::init<>())
      .def_readwrite("decoders", &multi_decoder_config::decoders)
      .def("to_yaml_str", &multi_decoder_config::to_yaml_str,
           py::arg("column_wrap") = 80)
      .def_static("from_yaml_str", &multi_decoder_config::from_yaml_str,
                  py::arg("yaml_str"))
      .def(py::self == py::self);

  // Library helpers
  mod_cfg.def(
      "configure_decoders", &configure_decoders, py::arg("config"),
      "Configure decoders in a multi_decoder_config list; returns int status.");
  mod_cfg.def("configure_decoders_from_file", &configure_decoders_from_file,
              py::arg("config_file"),
              "Configure decoders from a YAML file; returns int status.");
  mod_cfg.def("configure_decoders_from_str", &configure_decoders_from_str,
              py::arg("config_str"),
              "Configure decoders from a YAML string; returns int status.");
  mod_cfg.def("finalize_decoders", &finalize_decoders,
              "Finalize decoder resources.");
}
} // namespace cudaq::qec::decoding::config
