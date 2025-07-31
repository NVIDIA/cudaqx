/*******************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq/qec/codes/surface_code.h"
#include <iomanip>

using cudaq::qec::operation;

namespace cudaq::qec::surface_code {

vec2d::vec2d(int row_in, int col_in) : row(row_in), col(col_in) {}

vec2d operator+(const vec2d &lhs, const vec2d &rhs) {
  return {lhs.row + rhs.row, lhs.col + rhs.col};
}

vec2d operator-(const vec2d &lhs, const vec2d &rhs) {
  return {lhs.row - rhs.row, lhs.col - rhs.col};
}

bool operator==(const vec2d &lhs, const vec2d &rhs) {
  return lhs.row == rhs.row && lhs.col == rhs.col;
}

// impose 2d ordering for reproducibility
bool operator<(const vec2d &lhs, const vec2d &rhs) {
  // sort by row component.
  // if row is tied, sort by col.
  if (lhs.row != rhs.row) {
    return lhs.row < rhs.row;
  }
  return lhs.col < rhs.col;
}

void stabilizer_grid::generate_grid_roles() {
  // init grid to all empty
  for (size_t row = 0; row < grid_length; ++row) {
    for (size_t col = 0; col < grid_length; ++col) {
      size_t idx = row * grid_length + col;
      surface_role role = surface_role::empty;
      roles[idx] = role;
    }
  }

  // set alternating x/z interior
  for (size_t row = 1; row < grid_length - 1; ++row) {
    for (size_t col = 1; col < grid_length - 1; ++col) {
      size_t idx = row * grid_length + col;
      bool parity = (row + col) % 2;
      if (!parity) {
        roles[idx] = surface_role::amz;
      } else {
        roles[idx] = surface_role::amx;
      }
    }
  }

  // set top/bottom boundaries for weight 2 z-stabs
  for (size_t row = 0; row < grid_length; row += grid_length - 1) {
    for (size_t col = 1; col < grid_length - 1; ++col) {
      size_t idx = row * grid_length + col;
      bool parity = (row + col) % 2;
      if (!parity)
        roles[idx] = surface_role::amz;
    }
  }

  // set left/right boundaries for weight 2 x-stabs
  for (size_t row = 1; row < grid_length - 1; ++row) {
    for (size_t col = 0; col < grid_length; col += grid_length - 1) {
      size_t idx = row * grid_length + col;
      bool parity = (row + col) % 2;
      if (parity)
        roles[idx] = surface_role::amx;
    }
  }
}

void stabilizer_grid::generate_grid_indices() {
  size_t z_count = 0;
  size_t x_count = 0;
  for (size_t row = 0; row < grid_length; ++row) {
    for (size_t col = 0; col < grid_length; ++col) {
      size_t idx = row * grid_length + col;
      switch (roles[idx]) {
      case surface_role::amz:
        z_stab_coords.push_back(vec2d(row, col));
        z_stab_indices[vec2d(row, col)] = z_count;
        z_count++;
        break;
      case surface_role::amx:
        x_stab_coords.push_back(vec2d(row, col));
        x_stab_indices[vec2d(row, col)] = x_count;
        x_count++;
        break;
      case surface_role::empty:
        // nothing
        break;
      default:
        throw std::runtime_error(
            "Grid index without role should be impossible\n");
      }
    }
  }

  // Data qubit grid
  // This grid is a on a different coordinate system than the stabilizer grid
  // Can think of this grid as offset from the stabilizer grid by a half unit
  // right and down. data_row = stabilizer_row + 0.5 data_col = stabilizer_col +
  // 0.5
  for (size_t row = 0; row < distance; ++row) {
    for (size_t col = 0; col < distance; ++col) {
      size_t idx = row * distance + col;
      data_coords.push_back(vec2d(row, col));
      data_indices[vec2d(row, col)] = idx;
    }
  }
}

void stabilizer_grid::generate_stabilizers() {
  for (size_t i = 0; i < x_stab_coords.size(); ++i) {
    std::vector<size_t> current_stab;
    for (int row_offset = -1; row_offset < 1; ++row_offset) {
      for (int col_offset = -1; col_offset < 1; ++col_offset) {
        int row = x_stab_coords[i].row + row_offset;
        int col = x_stab_coords[i].col + col_offset;
        vec2d trial_coord(row, col);
        if (data_indices.find(trial_coord) != data_indices.end()) {
          current_stab.push_back(data_indices[trial_coord]);
        }
      }
    }
    std::sort(current_stab.begin(), current_stab.end());
    x_stabilizers.push_back(current_stab);
  }

  for (size_t i = 0; i < z_stab_coords.size(); ++i) {
    std::vector<size_t> current_stab;
    for (int row_offset = -1; row_offset < 1; ++row_offset) {
      for (int col_offset = -1; col_offset < 1; ++col_offset) {
        int row = z_stab_coords[i].row + row_offset;
        int col = z_stab_coords[i].col + col_offset;
        vec2d trial_coord(row, col);
        if (data_indices.find(trial_coord) != data_indices.end()) {
          current_stab.push_back(data_indices[trial_coord]);
        }
      }
    }
    std::sort(current_stab.begin(), current_stab.end());
    z_stabilizers.push_back(current_stab);
  }
}

stabilizer_grid::stabilizer_grid() {}

stabilizer_grid::stabilizer_grid(uint32_t distance)
    : distance(distance), grid_length(distance + 1),
      roles(grid_length * grid_length) {
  // generate a 2d grid of roles
  generate_grid_roles();
  // now use grid to set coord
  generate_grid_indices();
  // now use coords to set the stabilizers
  generate_stabilizers();
}

void stabilizer_grid::print_stabilizer_coords() const {
  int width = std::to_string(grid_length).length();

  for (size_t row = 0; row < grid_length; ++row) {
    for (size_t col = 0; col < grid_length; ++col) {
      size_t idx = row * grid_length + col;
      switch (roles[idx]) {
      case amz:
        std::cout << "Z(" << std::setw(width) << row << "," << std::setw(width)
                  << col << ")  ";
        break;
      case amx:
        std::cout << "X(" << std::setw(width) << row << "," << std::setw(width)
                  << col << ")  ";
        break;
      case empty:
        std::cout << std::setw(2 * width + 6) << " ";
        break;
      default:
        throw std::runtime_error(
            "Grid index without role should be impossible\n");
      }
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

void stabilizer_grid::print_stabilizer_indices() const {
  int width = std::to_string(z_stab_indices.size()).length() + 2;
  for (size_t row = 0; row < grid_length; ++row) {
    for (size_t col = 0; col < grid_length; ++col) {
      size_t idx = row * grid_length + col;
      switch (roles[idx]) {
      case amz:
        std::cout << "Z" << std::left << std::setw(width)
                  << z_stab_indices.at(vec2d(row, col));
        break;
      case amx:
        std::cout << "X" << std::left << std::setw(width)
                  << x_stab_indices.at(vec2d(row, col));
        break;
      case empty:
        std::cout << std::setw(width + 1) << " ";
        break;
      default:
        throw std::runtime_error(
            "Grid index without role should be impossible\n");
      }
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

void stabilizer_grid::print_stabilizer_maps() const {
  std::cout << x_stab_coords.size() << " mx ancilla qubits:\n";
  for (size_t i = 0; i < x_stab_coords.size(); ++i) {
    std::cout << "amx[" << i << "] @ (" << x_stab_coords[i].row << ", "
              << x_stab_coords[i].col << ")\n";
  }
  std::cout << z_stab_coords.size() << " mz ancilla qubits:\n";
  for (size_t i = 0; i < z_stab_coords.size(); ++i) {
    std::cout << "amz[" << i << "] @ (" << z_stab_coords[i].row << ", "
              << z_stab_coords[i].col << ")\n";
  }
  std::cout << "\n";

  std::cout << x_stab_coords.size() << " mx ancilla qubits:\n";
  for (const auto &[k, v] : x_stab_indices) {
    std::cout << "@(" << k.row << "," << k.col << "): amx[" << v << "]\n";
  }
  std::cout << z_stab_coords.size() << " mz ancilla qubits:\n";
  for (const auto &[k, v] : z_stab_indices) {
    std::cout << "@(" << k.row << "," << k.col << "): amz[" << v << "]\n";
  }
  std::cout << "\n";
}

void stabilizer_grid::print_data_grid() const {
  int width = std::to_string(distance).length() + 2;

  for (size_t row = 0; row < distance; ++row) {
    for (size_t col = 0; col < distance; ++col) {
      size_t idx = row * distance + col;
      std::cout << "d" << std::left << std::setw(width) << idx;
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

void stabilizer_grid::print_stabilizer_grid() const {
  int width = std::to_string(grid_length).length();

  for (size_t row = 0; row < grid_length; ++row) {
    for (size_t col = 0; col < grid_length; ++col) {
      size_t idx = row * grid_length + col;
      switch (roles[idx]) {
      case amz:
        std::cout << "Z(" << std::setw(width) << row << "," << std::setw(width)
                  << col << ")  ";
        break;
      case amx:
        std::cout << "X(" << std::setw(width) << row << "," << std::setw(width)
                  << col << ")  ";
        break;
      case empty:
        std::cout << "e(" << std::setw(width) << row << "," << std::setw(width)
                  << col << ")  ";
        break;
      default:
        throw std::runtime_error(
            "Grid index without role should be impossible\n");
      }
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

void stabilizer_grid::print_stabilizers() const {
  for (size_t s_i = 0; s_i < x_stabilizers.size(); ++s_i) {
    std::cout << "s[" << s_i << "]: ";
    for (size_t op_i = 0; op_i < x_stabilizers[s_i].size(); ++op_i) {
      std::cout << "X" << x_stabilizers[s_i][op_i] << " ";
    }
    std::cout << "\n";
  }

  size_t offset = x_stabilizers.size();
  for (size_t s_i = 0; s_i < z_stabilizers.size(); ++s_i) {
    std::cout << "s[" << s_i + offset << "]: ";
    for (size_t op_i = 0; op_i < z_stabilizers[s_i].size(); ++op_i) {
      std::cout << "Z" << z_stabilizers[s_i][op_i] << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

std::vector<cudaq::spin_op_term>
stabilizer_grid::get_spin_op_stabilizers() const {
  std::vector<cudaq::spin_op_term> spin_op_stabs;
  for (size_t s_i = 0; s_i < x_stabilizers.size(); ++s_i) {
    std::string stab(data_coords.size(), 'I');
    for (auto elem : x_stabilizers[s_i]) {
      stab[elem] = 'X';
    }
    spin_op_stabs.emplace_back(cudaq::spin_op::from_word(stab));
  }
  for (size_t s_i = 0; s_i < z_stabilizers.size(); ++s_i) {
    std::string stab(data_coords.size(), 'I');
    for (auto elem : z_stabilizers[s_i]) {
      stab[elem] = 'Z';
    }
    spin_op_stabs.emplace_back(cudaq::spin_op::from_word(stab));
  }
  return spin_op_stabs;
}

std::vector<cudaq::spin_op_term>
stabilizer_grid::get_spin_op_observables() const {
  std::vector<cudaq::spin_op_term> spin_op_obs;
  // X obs runs along top row of data qubits
  std::string xobs(data_coords.size(), 'I');
  for (size_t i = 0; i < distance; ++i) {
    xobs[i] = 'X';
  }
  spin_op_obs.emplace_back(cudaq::spin_op::from_word(xobs));

  // Z ob runs along left col of data qubits
  std::string zobs(data_coords.size(), 'I');
  for (size_t i = 0; i < data_coords.size(); i += distance) {
    zobs[i] = 'Z';
  }
  spin_op_obs.emplace_back(cudaq::spin_op::from_word(zobs));

  return spin_op_obs;
}

surface_code::surface_code(const heterogeneous_map &options) : code() {
  if (!options.contains("distance"))
    throw std::runtime_error(
        "[surface_code] distance not provided. distance must be provided via "
        "qec::get_code(..., options) options map.");
  distance = options.get<std::size_t>("distance");
  grid = stabilizer_grid(distance);

  m_stabilizers = grid.get_spin_op_stabilizers();
  m_pauli_observables = grid.get_spin_op_observables();

  // Sort now to avoid repeated sorts later.
  sortStabilizerOps(m_stabilizers);
  sortStabilizerOps(m_pauli_observables);

  operation_encodings.insert(std::make_pair(operation::x, x));
  operation_encodings.insert(std::make_pair(operation::z, z));
  operation_encodings.insert(std::make_pair(operation::cx, cx));
  operation_encodings.insert(std::make_pair(operation::cz, cz));
  operation_encodings.insert(
      std::make_pair(operation::stabilizer_round, stabilizer));
  operation_encodings.insert(std::make_pair(operation::prep0, prep0));
  operation_encodings.insert(std::make_pair(operation::prep1, prep1));
  operation_encodings.insert(std::make_pair(operation::prepp, prepp));
  operation_encodings.insert(std::make_pair(operation::prepm, prepm));
}

std::size_t surface_code::get_num_data_qubits() const {
  return distance * distance;
}

std::size_t surface_code::get_num_ancilla_qubits() const {
  return distance * distance - 1;
}

std::size_t surface_code::get_num_ancilla_x_qubits() const {
  return (distance * distance - 1) / 2;
}

std::size_t surface_code::get_num_ancilla_z_qubits() const {
  return (distance * distance - 1) / 2;
}

std::size_t surface_code::get_num_x_stabilizers() const {
  return (distance * distance - 1) / 2;
}

std::size_t surface_code::get_num_z_stabilizers() const {
  return (distance * distance - 1) / 2;
}

/// @brief Register the surace_code type
CUDAQ_REGISTER_EXTENSION(surface_code)

} // namespace cudaq::qec::surface_code
