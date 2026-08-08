/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025-2026 NVIDIA Corporation & Affiliates.                    *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "qldpc_config_loader.h"

#include "cudaq/qec/decoder.h"
#include "cudaq/qec/realtime/decoding_config.h"
#include "cudaq/qec/realtime/sparse_to_csr.h"

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace test_realtime_qldpc {

namespace {
/// Flat, -1-terminated sparse rows (the YAML/config encoding) to a shaped
/// sparse matrix, so O and D can be handed to the decoder as part of its model.
cudaq::qec::sparse_binary_matrix
sparse_matrix_from_flat_rows(const std::vector<std::int64_t> &flat,
                             std::uint32_t num_rows) {
  std::vector<std::vector<std::uint32_t>> rows;
  std::vector<std::uint32_t> row;
  std::uint32_t num_cols = 0;
  for (std::int64_t entry : flat) {
    if (entry < 0) {
      rows.push_back(std::move(row));
      row.clear();
      continue;
    }
    const auto col = static_cast<std::uint32_t>(entry);
    num_cols = std::max(num_cols, col + 1);
    row.push_back(col);
  }
  if (!row.empty())
    rows.push_back(std::move(row));
  rows.resize(num_rows);
  return cudaq::qec::sparse_binary_matrix::from_nested_csr(num_rows, num_cols,
                                                           rows);
}
} // namespace

namespace {

std::string read_file(const std::string &path) {
  std::ifstream f(path);
  if (!f.is_open())
    throw std::runtime_error(
        "test_realtime_qldpc_config_loader: failed to open YAML: " + path);
  return std::string((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
}

} // namespace

LoadedDecoder load_decoder_from_yaml(const std::string &yaml_path) {
  using namespace cudaq::qec;

  auto mdc = decoding::config::multi_decoder_config::from_yaml_str(
      read_file(yaml_path));
  if (mdc.decoders.size() != 1u)
    throw std::runtime_error(
        "test_realtime_qldpc_config_loader: expected exactly one decoder in "
        "YAML, found " +
        std::to_string(mdc.decoders.size()) + ": " + yaml_path);

  auto &dec = mdc.decoders[0];

  // H_sparse -> dense tensor (the plugin reads it through the standard
  // decoder::get() entry point).
  std::vector<std::uint32_t> h_row_ptr, h_col_idx;
  std::size_t h_rows =
      realtime::sparse_vec_to_csr(dec.H_sparse, h_row_ptr, h_col_idx);
  if (h_rows != dec.syndrome_size)
    throw std::runtime_error(
        "test_realtime_qldpc_config_loader: H_sparse row count " +
        std::to_string(h_rows) + " does not match dec.syndrome_size " +
        std::to_string(dec.syndrome_size));

  const std::size_t bs = dec.block_size;
  const std::size_t ss = dec.syndrome_size;
  cudaqx::tensor<std::uint8_t> H_tensor({ss, bs});
  for (std::size_t r = 0; r < ss; ++r)
    for (std::uint32_t j = h_row_ptr[r]; j < h_row_ptr[r + 1]; ++j)
      H_tensor.at({r, static_cast<std::size_t>(h_col_idx[j])}) = 1;

  auto params = dec.decoder_custom_args_to_heterogeneous_map();
  // O and D belong to the model, so they are supplied at construction rather
  // than installed afterwards.
  const auto num_observables = static_cast<std::uint32_t>(
      std::count(dec.O_sparse.begin(), dec.O_sparse.end(), -1));
  auto plugin = decoder::get(
      "nv-qldpc-decoder",
      cudaq::qec::decoder_init(
          cudaq::qec::sparse_binary_matrix(H_tensor),
          sparse_matrix_from_flat_rows(dec.O_sparse, num_observables),
          /*error_rates=*/{},
          sparse_matrix_from_flat_rows(dec.D_sparse,
                                       static_cast<std::uint32_t>(ss))),
      params);
  if (!plugin)
    throw std::runtime_error(
        "test_realtime_qldpc_config_loader: decoder::get(\"nv-qldpc-decoder\","
        " ...) returned nullptr; is the plugin built and discoverable?");

  LoadedDecoder out{};
  out.decoder = std::move(plugin);

  // num_measurements is the highest column index referenced by D_sparse, +1.
  // (D_sparse columns are measurement-bit indices; CSR sparse_vec_to_csr
  // gives us them via `d_ci`.)
  std::vector<std::uint32_t> d_rp, d_ci;
  realtime::sparse_vec_to_csr(dec.D_sparse, d_rp, d_ci);
  out.num_measurements = 0;
  for (auto c : d_ci)
    out.num_measurements =
        std::max(out.num_measurements, static_cast<std::size_t>(c + 1));

  std::vector<std::uint32_t> o_rp, o_ci;
  out.num_observables = realtime::sparse_vec_to_csr(dec.O_sparse, o_rp, o_ci);

  return out;
}

} // namespace test_realtime_qldpc
