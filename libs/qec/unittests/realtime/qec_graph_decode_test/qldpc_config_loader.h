/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025-2026 NVIDIA Corporation & Affiliates.                    *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

// Tiny shim that does the YAML -> decoder construction in a .cpp TU so the
// .cu test file doesn't have to include cudaq/qec/realtime/decoding_config.h.
// That header pulls in C++20-only `bool operator==(...) const = default;`
// patterns that nvcc 13 chokes on when forced to C++20 (libstdc++ 13 ICE).
//
// The test's .cu file calls `load_decoder_from_yaml` to get back exactly the
// state it needs to bring up the dispatchers and replay syndromes (decoder,
// H_tensor's shape via num_measurements / num_observables, decoder_id is
// always 0 for this single-decoder test).

#include <cstddef>
#include <memory>
#include <string>

namespace cudaq::qec {
class decoder;
}

namespace test_realtime_qldpc {

struct LoadedDecoder {
  std::unique_ptr<cudaq::qec::decoder> decoder;
  std::size_t num_measurements = 0; ///< total per-shot, derived from D_sparse
  std::size_t num_observables = 0;  ///< rows of O_sparse
};

/// Read the YAML file at `yaml_path`, parse it via
/// `cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str`, build
/// the H_tensor, instantiate the nv-qldpc-decoder plugin, and feed it the
/// D_sparse + O_sparse from the YAML.  Returns the constructed decoder plus
/// derived shape information.  Aborts via std::runtime_error on YAML/decoder
/// errors so the gtest body can ASSERT_NO_THROW around the call.
LoadedDecoder load_decoder_from_yaml(const std::string &yaml_path);

} // namespace test_realtime_qldpc
