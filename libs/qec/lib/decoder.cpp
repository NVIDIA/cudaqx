/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/decoder.h"
#include "cuda-qx/core/library_utils.h"
#include "cudaq/qec/plugin_loader.h"
#include "cudaq/qec/version.h"
#include <cassert>
#include <dlfcn.h>
#include <filesystem>
#include <vector>

CUDAQ_INSTANTIATE_REGISTRY(cudaq::qec::decoder, const cudaqx::tensor<uint8_t> &)
CUDAQ_INSTANTIATE_REGISTRY(cudaq::qec::decoder, const cudaqx::tensor<uint8_t> &,
                           const cudaqx::heterogeneous_map &)

namespace cudaq::qec {

decoder::decoder(const cudaqx::tensor<uint8_t> &H) : H(H) {
  const auto H_shape = H.shape();
  assert(H_shape.size() == 2 && "H tensor must be of rank 2");
  syndrome_size = H_shape[0];
  block_size = H_shape[1];
}

// Provide a trivial implementation of for tensor<uint8_t> decode call. Child
// classes should override this if they never want to pass through floats.
decoder_result decoder::decode(const cudaqx::tensor<uint8_t> &syndrome) {
  // Check tensor is of order-1
  // If order >1, we could check that other modes are of dim = 1 such that
  // n x 1, or 1 x n tensors are still valid.
  if (syndrome.rank() != 1) {
    throw std::runtime_error("Decode requires rank-1 tensors");
  }
  std::vector<float_t> soft_syndrome(syndrome.shape()[0]);
  std::vector<uint8_t> vec_cast(syndrome.data(),
                                syndrome.data() + syndrome.shape()[0]);
  convert_vec_hard_to_soft(vec_cast, soft_syndrome);
  return decode(soft_syndrome);
}

// Provide a trivial implementation of the multi-syndrome decoder. Child classes
// should override this if they can do it more efficiently than this.
std::vector<decoder_result>
decoder::decode_batch(const std::vector<std::vector<float_t>> &syndrome) {
  std::vector<decoder_result> result;
  result.reserve(syndrome.size());
  for (auto &s : syndrome)
    result.push_back(decode(s));
  return result;
}

std::string decoder::get_version() const {
  std::stringstream ss;
  ss << "CUDA-Q QEC Base Decoder Interface " << cudaq::qec::getVersion() << " ("
     << cudaq::qec::getFullRepositoryVersion() << ")";
  return ss.str();
}

std::future<decoder_result>
decoder::decode_async(const std::vector<float_t> &syndrome) {
  return std::async(std::launch::async,
                    [this, syndrome] { return this->decode(syndrome); });
}

std::unique_ptr<decoder>
decoder::get(const std::string &name, const cudaqx::tensor<uint8_t> &H,
             const cudaqx::heterogeneous_map &param_map) {
  auto &registry = get_registry();
  auto iter = registry.find(name);
  if (iter == registry.end())
    throw std::runtime_error("invalid decoder requested: " + name);
  return iter->second(H, param_map);
}

std::unique_ptr<decoder> get_decoder(const std::string &name,
                                     const cudaqx::tensor<uint8_t> &H,
                                     const cudaqx::heterogeneous_map options) {
  return decoder::get(name, H, options);
}

// Constructor function for auto-loading plugins
__attribute__((constructor)) void load_decoder_plugins() {
  // Load plugins from the decoder-specific plugin directory
  std::filesystem::path libPath{cudaqx::__internal__::getCUDAQXLibraryPath(
      cudaqx::__internal__::CUDAQXLibraryType::QEC)};
  auto pluginPath = libPath.parent_path() / "decoder-plugins";
  load_plugins(pluginPath.string(), PluginType::DECODER);
}

// Destructor function to clean up only decoder plugins
__attribute__((destructor)) void cleanup_decoder_plugins() {
  // Clean up decoder-specific plugins
  cleanup_plugins(PluginType::DECODER);
}
} // namespace cudaq::qec
