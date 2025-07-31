/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
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

/// @brief enumerates the role of a grid site in the surface codes stabilizer
/// grid
enum surface_role { amx, amz, empty };

/// @brief describes the 2d coordinate on the stabilizer grid
struct vec2d {
  int row;
  int col;

  vec2d(int row_in, int col_in);
};

vec2d operator+(const vec2d &lhs, const vec2d &rhs);
vec2d operator-(const vec2d &lhs, const vec2d &rhs);
bool operator==(const vec2d &lhs, const vec2d &rhs);
bool operator<(const vec2d &lhs, const vec2d &rhs);

// clang-format off
/// @brief Generates and keeps track of the 2d grid of stabilizers in the
/// rotated surface code.
/// Following same layout convention as in: https://arxiv.org/abs/2311.10687
/// Grid layout is arranged from left to right, top to bottom (row major storage)
/// grid_length = 4 example:
/// ```
/// (0,0)   (0,1)   (0,2)   (0,3)
/// (1,0)   (1,1)   (1,2)   (1,3)
/// (2,0)   (2,1)   (2,2)   (2,3)
/// (3,0)   (3,1)   (3,2)   (3,3)
/// ```
// clang-format on

///
/// Each entry on the grid can be an X stabilizer, Z stabilizer,
/// or empty, as is needed on the edges.
/// The grid length of 4 corresponds to a distance 3 surface code, which results
/// in:
/// ```
/// e(0,0)  e(0,1)  Z(0,2)  e(0,3)
/// X(1,0)  Z(1,1)  X(1,2)  e(1,3)
/// e(2,0)  X(2,1)  Z(2,2)  X(2,3)
/// e(3,0)  Z(3,1)  e(3,2)  e(3,3)
/// ```
///
/// This is seen through the `print_stabilizer_grid()` member function.
/// To get rid of the empty sites, the `print_stabilizer_coords()` function is
/// used:
/// ```
///                 Z(0,2)
/// X(1,0)  Z(1,1)  X(1,2)
///         X(2,1)  Z(2,2)  X(2,3)
///         Z(3,1)
/// ```
///
/// and to get the familiar visualization of the distance three surface code,
/// the `print_stabilizer_indices` results in:
/// ```
///         Z0
/// X0  Z1  X1
///     X2  Z2  X3
///     Z3
/// ```
///
/// The data qubits are located at the four corners of each of the weight-4
/// stabilizers. They are also organized with index increasing from left to
/// right, top to bottom:
/// ```
/// d0  d1  d2
/// d3  d4  d5
/// d6  d7  d8
/// ```
class stabilizer_grid {
private:
  /// @brief Generates this->roles
  void generate_grid_roles();
  /// @brief Generates {x,z}_stab_coords and indices
  void generate_grid_indices();
  /// @brief Generates {x,z}_stabilizers
  void generate_stabilizers();

public:
  /// @brief The distance of the code
  /// determines the number of data qubits per dimension
  uint32_t distance = 0;

  /// @brief length of the stabilizer grid
  /// for distance = d data qubits,
  /// the stabilizer grid has length d+1
  uint32_t grid_length = 0;

  /// @brief flattened vector of the stabilizer grid sites roles'
  /// grid idx -> role
  /// stored in row major order
  std::vector<surface_role> roles;

  /// @brief x stab index -> 2d coord
  std::vector<vec2d> x_stab_coords;

  /// @brief z stab index -> 2d coord
  std::vector<vec2d> z_stab_coords;

  /// @brief 2d coord -> z stab index
  std::map<vec2d, size_t> x_stab_indices;

  /// @brief 2d coord -> z stab index
  std::map<vec2d, size_t> z_stab_indices;

  /// @brief data index -> 2d coord
  /// data qubits are in an offset 2D coord system from stabilizers
  std::vector<vec2d> data_coords;

  /// @brief 2d coord -> data index
  std::map<vec2d, size_t> data_indices;

  /// @brief Each element is an X stabilizer specified by the data qubits it has
  /// support on
  /// In surface code, can have weight 2 or weight 4 stabs
  /// So {x,z}_stabilizer[i].size() == 2 || 4
  std::vector<std::vector<size_t>> x_stabilizers;

  /// @brief Each element is an Z stabilizer specified by the data qubits it has
  /// support on
  std::vector<std::vector<size_t>> z_stabilizers;

  /// @brief Construct the grid from the code's distance
  stabilizer_grid(uint32_t distance);
  /// @brief Empty constructor
  stabilizer_grid();

  /// @brief Print a 2d grid of stabilizer roles
  void print_stabilizer_grid() const;

  /// @brief Print a 2d grid of stabilizer coords
  void print_stabilizer_coords() const;

  /// @brief Print a 2d grid of stabilizer indices
  void print_stabilizer_indices() const;

  /// @brief Print a 2d grid of data qubit indices
  void print_data_grid() const;

  /// @brief Print the coord <--> indices maps
  void print_stabilizer_maps() const;

  /// @brief Print the stabilizers in sparse pauli format
  void print_stabilizers() const;

  /// @brief Get the stabilizers as a vector of cudaq::spin_op_terms
  std::vector<cudaq::spin_op_term> get_spin_op_stabilizers() const;

  /// @brief Get the observables as a vector of cudaq::spin_op_terms
  std::vector<cudaq::spin_op_term> get_spin_op_observables() const;
};

/// \pure_device_kernel
///
/// @brief Apply X gate to a surface_code patch
/// @param p The patch to apply the X gate to
__qpu__ void x(patch p);

/// \pure_device_kernel
///
/// @brief Apply Z gate to a surface_code patch
/// @param p The patch to apply the Z gate to
__qpu__ void z(patch p);

/// \pure_device_kernel
///
/// @brief Apply controlled-X gate between two surface_code patches
/// @param control The control patch
/// @param target The target patch
__qpu__ void cx(patch control, patch target);

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

  /// @brief Get number of X stabilizer that can be measured
  /// @return Number of X-type stabilizers
  std::size_t get_num_x_stabilizers() const override;

  /// @brief Get number of Z stabilizer that can be measured
  /// @return Number of Z-type stabilizers
  std::size_t get_num_z_stabilizers() const override;

public:
  /// @brief Constructor for the surface_code
  surface_code(const heterogeneous_map &);
  // Grid constructor would be useful

  /// @brief Extension creator function for the surface_code
  CUDAQ_ADD_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      surface_code, static std::unique_ptr<cudaq::qec::code> create(
                        const cudaqx::heterogeneous_map &options) {
        return std::make_unique<surface_code>(options);
      })

  /// @brief Grid to keep track of topological arrangement of qubits.
  stabilizer_grid grid;
};

} // namespace cudaq::qec::surface_code
