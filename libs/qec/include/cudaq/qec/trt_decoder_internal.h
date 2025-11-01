/*******************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/qec/decoder.h"
#include <memory>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "NvOnnxParser.h"

namespace cudaq::qec::trt_decoder_internal {

/// @brief Validates TRT decoder parameters
/// @param params The parameter map to validate
/// @throws std::runtime_error if parameters are invalid
void validate_trt_decoder_parameters(const cudaqx::heterogeneous_map &params);

/// @brief Loads a binary file into memory
/// @param filename Path to the file to load
/// @return Vector containing the file contents
/// @throws std::runtime_error if file cannot be opened
std::vector<char> load_file(const std::string &filename);

/// @brief Builds a TensorRT engine from an ONNX model
/// @param onnx_model_path Path to the ONNX model file
/// @param params Configuration parameters
/// @param logger TensorRT logger instance
/// @return Unique pointer to the built TensorRT engine
/// @throws std::runtime_error if engine building fails
std::unique_ptr<nvinfer1::ICudaEngine>
build_engine_from_onnx(const std::string &onnx_model_path,
                       const cudaqx::heterogeneous_map &params,
                       nvinfer1::ILogger &logger);

/// @brief Saves a TensorRT engine to a file
/// @param engine The engine to save
/// @param file_path Path where to save the engine
/// @throws std::runtime_error if saving fails
void save_engine_to_file(nvinfer1::ICudaEngine *engine,
                         const std::string &file_path);

/// @brief Parses and configures precision settings for TensorRT
/// @param precision The precision string (fp16, bf16, int8, fp8, noTF32, best)
/// @param config TensorRT builder config instance
void parse_precision(const std::string &precision,
                     nvinfer1::IBuilderConfig *config);

} // namespace cudaq::qec::trt_decoder_internal
