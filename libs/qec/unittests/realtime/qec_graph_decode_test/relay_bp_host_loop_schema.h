/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025-2026 NVIDIA Corporation & Affiliates.                    *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/qec/decoder_config_schema.h"

namespace test_realtime_qldpc {

/// Register the schema supplied by the out-of-tree nv-qldpc plugin when an
/// older plugin build does not yet provide it itself.
inline void register_relay_bp_host_loop_schema() {
  using cudaq::qec::decoding::config::find_decoder_schema;
  using cudaq::qec::decoding::config::param_kind;
  using cudaq::qec::decoding::config::register_decoder_schema;

  if (find_decoder_schema("nv-qldpc-decoder"))
    return;

  constexpr auto srelay_schema = "nv-qldpc-srelay-config-v0.7";
  if (!find_decoder_schema(srelay_schema)) {
    register_decoder_schema({srelay_schema,
                             {{"pre_iter", param_kind::uint64, true},
                              {"num_sets", param_kind::uint64, true},
                              {"stopping_criterion", param_kind::string, true},
                              {"stop_nconv", param_kind::uint64, true}}});
  }

  register_decoder_schema(
      {"nv-qldpc-decoder",
       {{"use_sparsity", param_kind::boolean, true},
        {"error_rate_vec", param_kind::f64_vec, true},
        {"max_iterations", param_kind::int32, true},
        {"bp_method", param_kind::int32, true},
        {"gamma0", param_kind::f64, true},
        {"gamma_dist", param_kind::f64_vec, true},
        {"srelay_config", param_kind::subschema, true, srelay_schema},
        {"composition", param_kind::int32, true},
        {"clip_value", param_kind::f64, true},
        {"repeatable", param_kind::boolean, true}}});
}

} // namespace test_realtime_qldpc
