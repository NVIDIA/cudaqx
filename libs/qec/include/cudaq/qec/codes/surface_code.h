/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cudaq/qec/code.h"
#include "cudaq/qec/patch.h"

using namespace cudaqx;

namespace cudaq::qec::surface_code {

enum surface_role { data, amx, amz, empty };

struct vec2d {
  int row;
  int col;

  vec2d(int row_in, int col_in);
};

vec2d operator+(const vec2d &lhs, const vec2d &rhs);
vec2d operator-(const vec2d &lhs, const vec2d &rhs);
bool operator==(const vec2d &lhs, const vec2d &rhs);
// impose 2d ordering for reproducibility
bool operator<(const vec2d &lhs, const vec2d &rhs);

// Grid layout from left to right, top to bottom
// (0,0)   (0,1)   (0,2)   (0,3)   (0,4)
// (1,0)   (1,1)   (1,2)   (1,3)   (1,4)
// (2,0)   (2,1)   (2,2)   (2,3)   (2,4)
// (3,0)   (3,1)   (3,2)   (3,3)   (3,4)
// (4,0)   (4,1)   (4,2)   (4,3)   (4,4)

// Represents the x & z stabilizers in the rotated surface code,
// including data about their layout in a 2D grid.
class stabilizer_grid {
  public:
  // The distance of the code,
  // determines the number of data qubits per dimension
  uint32_t distance;

  // length of the grid
  // for distance = d data qubits,
  // the stabilizer grid has length d+1
  uint32_t grid_length;
  // grid idx -> role
  // stored in row major order
  std::vector<surface_role> roles;
  // stab index -> 2d coord
  std::vector<vec2d> z_stab_coords;
  std::vector<vec2d> x_stab_coords;
  // 2d coord -> stab index
  std::map<vec2d, size_t> x_stab_indices;
  std::map<vec2d, size_t> z_stab_indices;

  // data qubits are in an offset 2D coord system
  // data index -> 2d coord
  std::vector<vec2d> data_coords;
  // 2d coord -> data index
  std::map<vec2d, size_t> data_indices;

  // In surface code, can have weight 2 or weight 4 stabs
  // So {x,z}_stabilizer[i].size() == 2 || 4
  std::vector<std::vector<size_t>> x_stabilizers;
  std::vector<std::vector<size_t>> z_stabilizers;

  // The distance determines how many data qubits are on an edge.
  // The grid itself has stabilizers extending beyond this data
  // qubit edge on each side, as well as many empty grid sites
  // along this edge.

  // Construct the grid from the code's distance
  stabilizer_grid(uint32_t distance);
  // Empty constructor
  stabilizer_grid();

  void generate_grid_roles();
  void generate_grid_indices();
  void generate_stabilizers();

  void print_stabilizer_coords();
  void print_stabilizer_indices();
  void print_stabilizer_maps();
  void print_data_grid();
  void print_stabilizer_grid();
  void print_stabilizers();
};



/// \pure_device_kernel
///
/// @brief Apply X gate to a surface_code patch
/// @param p The patch to apply the X gate to
__qpu__ void x(patch p);

/// \pure_device_kernel
///
/// @brief Apply Y gate to a surface_code patch
/// @param p The patch to apply the Y gate to
__qpu__ void y(patch p);

/// \pure_device_kernel
///
/// @brief Apply Z gate to a surface_code patch
/// @param p The patch to apply the Z gate to
__qpu__ void z(patch p);

/// \pure_device_kernel
///
/// @brief Apply Hadamard gate to a surface_code patch
/// @param p The patch to apply the Hadamard gate to
__qpu__ void h(patch p);

/// \pure_device_kernel
///
/// @brief Apply S gate to a surface_code patch
/// @param p The patch to apply the S gate to
__qpu__ void s(patch p);

/// \pure_device_kernel
///
/// @brief Apply controlled-X gate between two surface_code patches
/// @param control The control patch
/// @param target The target patch
__qpu__ void cx(patch control, patch target);

/// \pure_device_kernel
///
/// @brief Apply controlled-Y gate between two surface_code patches
/// @param control The control patch
/// @param target The target patch
__qpu__ void cy(patch control, patch target);

/// \pure_device_kernel
///
/// @brief Apply controlled-Z gate between two surface_code patches
/// @param control The control patch
/// @param target The target patch
__qpu__ void cz(patch control, patch target);

/// \pure_device_kernel
///
/// @brief Prepare a surface_code patch in the |0⟩ state
/// @param p The patch to prepare
__qpu__ void prep0(patch p);

/// \pure_device_kernel
///
/// @brief Prepare a surface_code patch in the |1⟩ state
/// @param p The patch to prepare
__qpu__ void prep1(patch p);

/// \pure_device_kernel
///
/// @brief Prepare a surface_code patch in the |+⟩ state
/// @param p The patch to prepare
__qpu__ void prepp(patch p);

/// \pure_device_kernel
///
/// @brief Prepare a surface_code patch in the |-⟩ state
/// @param p The patch to prepare
__qpu__ void prepm(patch p);

/// \pure_device_kernel
///
/// @brief Perform stabilizer measurements on a surface_code patch
/// @param p The patch to measure
/// @param x_stabilizers Indices of X stabilizers to measure
/// @param z_stabilizers Indices of Z stabilizers to measure
/// @return Vector of measurement results
__qpu__ std::vector<cudaq::measure_result>
stabilizer(patch p, const std::vector<std::size_t> &x_stabilizers,
           const std::vector<std::size_t> &z_stabilizers);

/// @brief surface_code implementation
class surface_code : public cudaq::qec::code {
protected:
  /// @brief The code distance parameter
  std::size_t distance;

  /// @brief Get the number of data qubits in the surface_code
  /// @return Number of data qubits (distance^2 for surface_code)
  std::size_t get_num_data_qubits() const override;

  /// @brief Get the number of total ancilla qubits in the surface_code
  /// @return Number of data qubits (distance^2 - 1 for surface_code)
  std::size_t get_num_ancilla_qubits() const override;

  /// @brief Get the number of X ancilla qubits in the surface_code
  /// @return Number of data qubits ((distance^2 - 1)/2 for surface_code)
  std::size_t get_num_ancilla_x_qubits() const override;

  /// @brief Get the number of Z ancilla qubits in the surface_code
  /// @return Number of data qubits ((distance^2 - 1)/2 for surface_code)
  std::size_t get_num_ancilla_z_qubits() const override;

public:
  /// @brief Constructor for the surface_code
  surface_code(const heterogeneous_map &);
  // Grid constructor would be useful

  /// @brief Extension creator function for the surface_code
  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      surface_code, static std::unique_ptr<cudaq::qec::code> create(
                  const cudaqx::heterogeneous_map &options) {
        return std::make_unique<surface_code>(options);
      })

  /// @brief Grid to keep track of topological arrangement of qubits.
  stabilizer_grid grid;

};

} // namespace cudaq::qec::surface_code
