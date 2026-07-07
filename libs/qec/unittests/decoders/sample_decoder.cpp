/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/decoder.h"
#include <cuda_runtime.h>
#include <vector>

#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

using namespace cudaqx;

namespace cudaq::qec {

/// @brief This is a sample (dummy) decoder that demonstrates how to build a
/// bare bones custom decoder based on the `cudaq::qec::decoder` interface.
class sample_decoder : public decoder {
private:
  bool decode_to_obs = false;

public:
  sample_decoder(const cudaq::qec::sparse_binary_matrix &H,
                 const cudaqx::heterogeneous_map &params)
      : decoder(H) {
    // Decoder-specific constructor arguments can be placed in `params`.
    decode_to_obs = params.get<bool>("decode_to_obs", decode_to_obs);
    if (decode_to_obs)
      set_result_type(decode_result_type::decode_to_obs);
  }

  virtual decoder_result decode(const std::vector<float_t> &syndrome) {
    // This is a simple decoder that simply results
    decoder_result result;
    result.converged = true;
    result.result =
        decode_to_obs ? syndrome : std::vector<float_t>(block_size, 0.0f);
    return result;
  }

  virtual ~sample_decoder() {}

  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      sample_decoder, static std::unique_ptr<decoder> create(
                          const cudaq::qec::decoder_init &init,
                          const cudaqx::heterogeneous_map &params) {
        return cudaq::qec::make_pcm_decoder<sample_decoder>(init, params);
      })
};

CUDAQ_EXT_PT_REGISTER_TYPE(sample_decoder)

/// @brief Test-only decoder that records the calling thread's placement --
/// the CUDA device a real allocation lands on and the raw MPOL_* mempolicy
/// mode -- both DURING construction (inside decoder::get()'s guarded window)
/// and inside each decode() call, and echoes them through result.result:
///   [0] = CUDA device at construction   [1] = CUDA device inside decode()
///   [2] = mempolicy mode at construction [3] = mempolicy mode inside decode()
/// Unavailable probes (no CUDA / non-Linux / blocked syscall) report -1.
class placement_probe_decoder : public decoder {
private:
  int ctor_device_ = -1;
  int ctor_mempolicy_ = -1;

  static int current_cuda_device() {
    int dev = -1;
    void *p = nullptr;
    if (cudaMalloc(&p, 16) == cudaSuccess && p) {
      cudaPointerAttributes attr{};
      if (cudaPointerGetAttributes(&attr, p) == cudaSuccess)
        dev = attr.device; // device the allocation actually landed on
      cudaFree(p);
    } else {
      cudaGetDevice(&dev);
    }
    return dev;
  }

  static int current_mempolicy_mode() {
    int mode = -1;
#if defined(__linux__)
    syscall(SYS_get_mempolicy, &mode, nullptr, 0UL, nullptr, 0UL);
#endif
    return mode;
  }

public:
  placement_probe_decoder(const cudaq::qec::sparse_binary_matrix &H,
                          const cudaqx::heterogeneous_map &params)
      : decoder(H), ctor_device_(current_cuda_device()),
        ctor_mempolicy_(current_mempolicy_mode()) {}

  virtual decoder_result decode(const std::vector<float_t> &syndrome) override {
    decoder_result result;
    result.converged = true;
    result.result =
        std::vector<float_t>{static_cast<float_t>(ctor_device_),
                             static_cast<float_t>(current_cuda_device()),
                             static_cast<float_t>(ctor_mempolicy_),
                             static_cast<float_t>(current_mempolicy_mode())};
    return result;
  }

  virtual ~placement_probe_decoder() {}

  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      placement_probe_decoder, static std::unique_ptr<decoder> create(
                                   const cudaq::qec::decoder_init &init,
                                   const cudaqx::heterogeneous_map &params) {
        return cudaq::qec::make_pcm_decoder<placement_probe_decoder>(init,
                                                                     params);
      })
};

CUDAQ_EXT_PT_REGISTER_TYPE(placement_probe_decoder)

} // namespace cudaq::qec
