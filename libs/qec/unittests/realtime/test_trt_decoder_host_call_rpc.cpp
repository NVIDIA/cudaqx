/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "qec_realtime_session.h"
#include "realtime_decoding.h"

#include "cudaq/qec/realtime/decoding_config.h"

#include <cuda_runtime_api.h>
#include <filesystem>
#include <gtest/gtest.h>

#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace config = cudaq::qec::decoding::config;
namespace host = cudaq::qec::decoding::host;

bool gpu_available() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::string onnx_path() {
#ifdef TRT_HOST_CALL_RPC_ONNX_PATH
  return TRT_HOST_CALL_RPC_ONNX_PATH;
#else
  return "";
#endif
}

class EnvVarGuard {
public:
  explicit EnvVarGuard(const char *name) : name_(name) {
    if (const char *value = std::getenv(name_))
      original_ = value;
  }

  ~EnvVarGuard() {
    if (original_)
      setenv(name_, original_->c_str(), /*overwrite=*/1);
    else
      unsetenv(name_);
  }

  void set(const char *value) { setenv(name_, value, /*overwrite=*/1); }

private:
  const char *name_;
  std::optional<std::string> original_;
};

struct DecoderConfigGuard {
  ~DecoderConfigGuard() { config::finalize_decoders(); }
};

config::multi_decoder_config make_composite_config(const std::string &onnx) {
  config::decoder_config decoder;
  decoder.id = 0;
  decoder.type = "trt_decoder";
  decoder.block_size = 2;
  decoder.syndrome_size = 2;

  // Deliberate column order: syndrome [1, 0] decodes to error column 1, and
  // column 1 flips the observable. If nested PyMatching loses O, it returns the
  // error frame [0, 1]; the composite XOR reads g[0] and returns 0 instead of
  // the observable frame 1.
  decoder.H_sparse = {1, -1, 0, -1};
  decoder.O_sparse = {1, -1};
  decoder.D_sparse = {0, -1, 1, -1};

  config::pymatching_config pymatching;
  pymatching.error_rate_vec = {0.1, 0.1};
  pymatching.merge_strategy = "smallest_weight";

  config::trt_decoder_config trt;
  trt.onnx_load_path = onnx;
  trt.batch_size = 1;
  trt.use_cuda_graph = false;
  trt.global_decoder = "pymatching";
  trt.global_decoder_params = pymatching;
  decoder.decoder_custom_args = trt;

  config::multi_decoder_config multi;
  multi.decoders.push_back(std::move(decoder));
  return multi;
}

} // namespace

TEST(TrtDecoderHostCallRpc,
     CompositeObservableCorrectionViaConfiguredHostCallSession) {
  if (!gpu_available())
    GTEST_SKIP() << "No CUDA GPU available";

  const std::string model_path = onnx_path();
  ASSERT_FALSE(model_path.empty());
  ASSERT_TRUE(std::filesystem::exists(model_path))
      << "Missing generated ONNX: " << model_path;

  auto multi_config = make_composite_config(model_path);
  const std::string yaml = multi_config.to_yaml_str(200);
  auto round_tripped =
      config::multi_decoder_config::from_yaml_str(std::string_view(yaml));
  ASSERT_EQ(round_tripped, multi_config);

  EnvVarGuard realtime_mode("CUDAQ_QEC_REALTIME_MODE");
  realtime_mode.set("inproc_rpc");

  DecoderConfigGuard configured_decoders;
  ASSERT_EQ(config::configure_decoders(round_tripped), 0);

  auto *session = host::get_realtime_session();
  ASSERT_NE(session, nullptr);
  ASSERT_TRUE(session->initialized());
  EXPECT_FALSE(session->device_mode());

  std::vector<std::uint8_t> syndrome{1, 0};
  EXPECT_NO_THROW(host::enqueue_syndromes(/*decoder_id=*/0, syndrome.data(),
                                          syndrome.size(), /*tag=*/1));

  std::vector<std::uint8_t> corrections(1, 0xCC);
  EXPECT_NO_THROW(host::get_corrections(/*decoder_id=*/0, corrections.data(),
                                        corrections.size(), /*reset=*/false));
  EXPECT_EQ(corrections, (std::vector<std::uint8_t>{1}));

  EXPECT_NO_THROW(host::reset_decoder(/*decoder_id=*/0));
  corrections.assign(1, 0xCC);
  EXPECT_NO_THROW(host::get_corrections(/*decoder_id=*/0, corrections.data(),
                                        corrections.size(), /*reset=*/false));
  EXPECT_EQ(corrections, (std::vector<std::uint8_t>{0}));
}
