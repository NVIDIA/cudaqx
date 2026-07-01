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

/// @brief Test-only decoder shaped like a typical GPU decoder: it overrides
/// ONLY the single-syndrome decode() (NOT decode_batch) and "allocates lazily"
/// on first decode(). It reports, via result.result[0], the CUDA device its
/// allocation actually landed on — so a test can assert cuda_device_id was
/// honored on whichever entry point invoked it.
class device_probe_decoder : public decoder {
public:
  device_probe_decoder(const cudaq::qec::sparse_binary_matrix &H,
                       const cudaqx::heterogeneous_map &params)
      : decoder(H) {}

  virtual decoder_result decode(const std::vector<float_t> &syndrome) override {
    decoder_result result;
    result.converged = true;
    int dev = -1;
    void *p = nullptr;
    if (cudaMalloc(&p, 16) == cudaSuccess && p) {
      cudaPointerAttributes attr{};
      if (cudaPointerGetAttributes(&attr, p) == cudaSuccess)
        dev = attr.device;     // device the lazy allocation landed on
      cudaFree(p);
    } else {
      cudaGetDevice(&dev);
    }
    result.result = std::vector<float_t>{static_cast<float_t>(dev)};
    return result;
  }

  virtual ~device_probe_decoder() {}

  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      device_probe_decoder, static std::unique_ptr<decoder> create(
                                const cudaq::qec::decoder_init &init,
                                const cudaqx::heterogeneous_map &params) {
        return cudaq::qec::make_pcm_decoder<device_probe_decoder>(init, params);
      })
};

CUDAQ_EXT_PT_REGISTER_TYPE(device_probe_decoder)

} // namespace cudaq::qec
