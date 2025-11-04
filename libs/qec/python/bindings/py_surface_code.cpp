/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_surface_code.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Disable this warning: "annotate" attribute directive ignored [-Wattributes]
// These warnings occur because GCC does not understand the __qpu__ attribute.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
#include "cudaq/qec/codes/surface_code.h"
#pragma GCC diagnostic pop

namespace py = pybind11;

namespace cudaq::qec::surface_code {
static py::dict
map_vec2d_to_py_tuple_dict(const std::map<vec2d, std::size_t> &m) {
  py::dict d;
  for (const auto &kv : m) {
    const auto &k = kv.first;
    d[py::make_tuple(k.row, k.col)] = kv.second;
  }
  return d;
}

template <typename Fn>
std::string capture_print(Fn &&fn) {
  std::ostringstream oss;
  auto *old_buf = std::cout.rdbuf(oss.rdbuf());
  fn();
  std::cout.rdbuf(old_buf);
  return oss.str();
}

void bindSurfaceCode(py::module &mod) {
  auto qecmod = py::hasattr(mod, "qecrt")
                    ? mod.attr("qecrt").cast<py::module_>()
                    : mod.def_submodule("qecrt");

  py::enum_<surface_role>(qecmod, "surface_role", py::arithmetic())
      .value("amx", surface_role::amx)
      .value("amz", surface_role::amz)
      .value("empty", surface_role::empty)
      .export_values();

  py::class_<vec2d>(qecmod, "vec2d")
      .def(py::init<int, int>(), py::arg("row"), py::arg("col"))
      .def_readwrite("row", &vec2d::row)
      .def_readwrite("col", &vec2d::col)
      .def("__repr__",
           [](const vec2d &v) {
             return "Vec2D(row=" + std::to_string(v.row) +
                    ", col=" + std::to_string(v.col) + ")";
           })
      .def("__add__", [](const vec2d &a, const vec2d &b) { return a + b; })
      .def("__sub__", [](const vec2d &a, const vec2d &b) { return a - b; })
      .def("__eq__", [](const vec2d &a, const vec2d &b) { return a == b; });

  py::class_<stabilizer_grid>(qecmod, "stabilizer_grid")
      .def(py::init<>())
      .def(py::init<std::uint32_t>(), py::arg("distance"))
      .def_readonly("distance", &stabilizer_grid::distance)
      .def_readonly("grid_length", &stabilizer_grid::grid_length)
      .def_readonly("roles", &stabilizer_grid::roles)
      .def_readonly("x_stab_coords", &stabilizer_grid::x_stab_coords)
      .def_readonly("z_stab_coords", &stabilizer_grid::z_stab_coords)
      .def_readonly("data_coords", &stabilizer_grid::data_coords)
      .def_readonly("x_stabilizers", &stabilizer_grid::x_stabilizers)
      .def_readonly("z_stabilizers", &stabilizer_grid::z_stabilizers)
      .def_property_readonly("x_stab_indices",
                             [](const stabilizer_grid &g) {
                               return map_vec2d_to_py_tuple_dict(
                                   g.x_stab_indices);
                             })
      .def_property_readonly("z_stab_indices",
                             [](const stabilizer_grid &g) {
                               return map_vec2d_to_py_tuple_dict(
                                   g.z_stab_indices);
                             })
      .def_property_readonly("data_indices",
                             [](const stabilizer_grid &g) {
                               return map_vec2d_to_py_tuple_dict(
                                   g.data_indices);
                             })
      .def("format_stabilizer_grid",
           [](const stabilizer_grid &g) {
             return capture_print([&] { g.print_stabilizer_grid(); });
           })
      .def("format_stabilizer_coords",
           [](const stabilizer_grid &g) {
             return capture_print([&] { g.print_stabilizer_coords(); });
           })
      .def("format_stabilizer_indices",
           [](const stabilizer_grid &g) {
             return capture_print([&] { g.print_stabilizer_indices(); });
           })
      .def("format_data_grid",
           [](const stabilizer_grid &g) {
             return capture_print([&] { g.print_data_grid(); });
           })
      .def("format_stabilizers",
           [](const stabilizer_grid &g) {
             return capture_print([&] { g.print_stabilizers(); });
           })
      .def("__repr__",
           [](const stabilizer_grid &g) {
             return "<StabilizerGrid distance=" + std::to_string(g.distance) +
                    " grid_length=" + std::to_string(g.grid_length) + ">";
           })
      .def("get_spin_op_stabilizers", &stabilizer_grid::get_spin_op_stabilizers,
           "Return the stabilizers as a list of cudaq::spin_op_term")
      .def("get_spin_op_observables", &stabilizer_grid::get_spin_op_observables,
           "Return the logical observables as a list of cudaq::spin_op_term");

  qecmod.def("role_to_str", [](surface_role r) {
    switch (r) {
    case surface_role::amx:
      return "X";
    case surface_role::amz:
      return "Z";
    default:
      return "e";
    }
  });
}
} // namespace cudaq::qec::surface_code
