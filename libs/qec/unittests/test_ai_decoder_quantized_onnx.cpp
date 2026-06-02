/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/realtime/ai_decoder_service.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <string>
#include <vector>

#define CUDA_ASSERT_OK(call)                                                   \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    ASSERT_EQ(err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err);  \
  } while (0)

namespace {

using namespace cudaq::qec::realtime::experimental;
namespace rt_sdk = cudaq::realtime;

static constexpr size_t kNumElements = 8;
static constexpr size_t kPayloadBytes = kNumElements * sizeof(float);
static constexpr size_t kSlotSize = CUDAQ_RPC_HEADER_SIZE + kPayloadBytes;

const std::vector<float> kInputs = {-4.0f, -2.0f, -1.0f, 0.0f,
                                    1.0f,  2.0f,  3.0f,  4.0f};

bool isGpuAvailable() {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  return err == cudaSuccess && device_count > 0;
}

bool isFp8HardwareAvailable() {
  int device = 0;
  cudaError_t err = cudaGetDevice(&device);
  if (err != cudaSuccess)
    return false;

  cudaDeviceProp prop{};
  err = cudaGetDeviceProperties(&prop, device);
  if (err != cudaSuccess)
    return false;
  return prop.major >= 9;
}

void write_rpc_slot(uint8_t *slot_host, const std::vector<float> &input) {
  std::memset(slot_host, 0, kSlotSize);
  rt_sdk::RPCHeader hdr{};
  hdr.magic = rt_sdk::RPC_MAGIC_REQUEST;
  hdr.arg_len = static_cast<uint32_t>(input.size() * sizeof(float));
  std::memcpy(slot_host, &hdr, sizeof(hdr));
  std::memcpy(slot_host + CUDAQ_RPC_HEADER_SIZE, input.data(),
              input.size() * sizeof(float));
}

class AiDecoderQuantizedOnnxSmokeTest : public ::testing::Test {
protected:
  void SetUp() override {
    if (!isGpuAvailable())
      GTEST_SKIP() << "No GPU available, skipping TensorRT smoke test";

    CUDA_ASSERT_OK(cudaSetDevice(0));
    CUDA_ASSERT_OK(cudaHostAlloc(reinterpret_cast<void **>(&slot_host_),
                                 kSlotSize, cudaHostAllocMapped));
    CUDA_ASSERT_OK(cudaHostGetDevicePointer(
        reinterpret_cast<void **>(&slot_dev_), slot_host_, 0));
    CUDA_ASSERT_OK(cudaHostAlloc(reinterpret_cast<void **>(&mailbox_host_),
                                 sizeof(void *), cudaHostAllocMapped));
    CUDA_ASSERT_OK(cudaHostGetDevicePointer(
        reinterpret_cast<void **>(&mailbox_dev_), mailbox_host_, 0));
    mailbox_host_[0] = slot_dev_;
    CUDA_ASSERT_OK(cudaStreamCreate(&stream_));
  }

  void TearDown() override {
    if (stream_)
      cudaStreamDestroy(stream_);
    if (mailbox_host_)
      cudaFreeHost(mailbox_host_);
    if (slot_host_)
      cudaFreeHost(slot_host_);
  }

  void run_service(const std::string &onnx_path,
                   const onnx_quant_info &expected_info,
                   std::vector<float> &output) {
    ai_decoder_service service(onnx_path,
                               reinterpret_cast<void **>(mailbox_dev_), "",
                               network_typing_override::automatic);

    const auto &build_info = service.get_quant_info();
    EXPECT_EQ(build_info.has_fp8, expected_info.has_fp8);
    EXPECT_EQ(build_info.has_int8, expected_info.has_int8);
    EXPECT_TRUE(build_info.requires_strongly_typed());
    EXPECT_EQ(service.get_input_num_elements(), kNumElements);
    EXPECT_EQ(service.get_output_num_elements(), kNumElements);
    EXPECT_EQ(service.get_input_size(), kPayloadBytes);
    EXPECT_EQ(service.get_output_size(), kPayloadBytes);

    service.capture_graph(stream_);
    ASSERT_NE(service.get_executable_graph(), nullptr);

    write_rpc_slot(slot_host_, kInputs);
    CUDA_ASSERT_OK(cudaGraphLaunch(service.get_executable_graph(), stream_));
    CUDA_ASSERT_OK(cudaStreamSynchronize(stream_));

    auto *response = reinterpret_cast<rt_sdk::RPCResponse *>(slot_host_);
    EXPECT_EQ(response->magic, rt_sdk::RPC_MAGIC_RESPONSE);
    EXPECT_EQ(response->status, 0u);
    EXPECT_EQ(response->result_len, kPayloadBytes);

    output.resize(kNumElements);
    std::memcpy(output.data(), slot_host_ + CUDAQ_RPC_HEADER_SIZE,
                kPayloadBytes);
  }

  uint8_t *slot_host_ = nullptr;
  uint8_t *slot_dev_ = nullptr;
  void **mailbox_host_ = nullptr;
  void **mailbox_dev_ = nullptr;
  cudaStream_t stream_ = nullptr;
};

void expect_identity_qdq(const std::vector<float> &actual, float tolerance) {
  ASSERT_EQ(actual.size(), kInputs.size());
  for (size_t i = 0; i < kInputs.size(); ++i) {
    EXPECT_NEAR(actual[i], kInputs[i], tolerance)
        << "Mismatch at element " << i;
  }
}

TEST_F(AiDecoderQuantizedOnnxSmokeTest, Int8QdqRunsWithExpectedNumerics) {
  onnx_quant_info expected{};
  expected.has_int8 = true;

  auto info = inspect_onnx(INT8_QDQ_ONNX_PATH);
  ASSERT_TRUE(info.has_int8);
  EXPECT_FALSE(info.has_fp8);
  EXPECT_TRUE(info.requires_strongly_typed());

  std::vector<float> output;
  run_service(INT8_QDQ_ONNX_PATH, expected, output);
  expect_identity_qdq(output, 0.0f);
}

TEST_F(AiDecoderQuantizedOnnxSmokeTest, Fp8QdqRunsWithExpectedNumerics) {
  onnx_quant_info expected{};
  expected.has_fp8 = true;

  auto info = inspect_onnx(FP8_QDQ_ONNX_PATH);
  ASSERT_TRUE(info.has_fp8);
  EXPECT_FALSE(info.has_int8);
  EXPECT_TRUE(info.requires_strongly_typed());

  if (!isFp8HardwareAvailable())
    GTEST_SKIP() << "FP8 Q/DQ requires FP8-capable GPU hardware";

  std::vector<float> output;
  run_service(FP8_QDQ_ONNX_PATH, expected, output);
  expect_identity_qdq(output, 1.0e-3f);
}

} // namespace
