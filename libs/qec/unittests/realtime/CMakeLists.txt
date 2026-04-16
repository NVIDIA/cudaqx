# ============================================================================ #
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_subdirectory(app_examples)

# =========================================================================== #
# test_realtime_predecoder_w_pymatching
# =========================================================================== #
# Software-only benchmark: AI predecoder (TRT) + PyMatching via
# realtime_pipeline with ring_buffer_injector.

if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY AND TENSORRT_ONNX_PARSER_LIBRARY
   AND TARGET cudaq-realtime-pipeline)

  get_filename_component(_cuda_bin_pred "${CMAKE_CUDA_COMPILER}" DIRECTORY)
  get_filename_component(_cuda_root_pred "${_cuda_bin_pred}" DIRECTORY)
  set(_cuda_cccl_include_pred "${_cuda_root_pred}/include/cccl")

  add_executable(test_realtime_predecoder_w_pymatching
    test_realtime_predecoder_w_pymatching.cpp
    predecoder_pipeline_common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/realtime/ai_decoder_service.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/realtime/ai_predecoder_service.cu
  )

  set_target_properties(test_realtime_predecoder_w_pymatching PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_STANDARD 17
    LINKER_LANGUAGE CUDA
  )

  target_compile_definitions(test_realtime_predecoder_w_pymatching PRIVATE
    ONNX_MODEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../lib/realtime"
  )

  target_include_directories(test_realtime_predecoder_w_pymatching PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${_cuda_cccl_include_pred}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_SOURCE_DIR}/libs/core/include
    ${CUDAQ_REALTIME_INCLUDE_DIR}
  )

  target_link_libraries(test_realtime_predecoder_w_pymatching PRIVATE
    CUDA::cudart
    ${TENSORRT_LIBRARY}
    ${TENSORRT_ONNX_PARSER_LIBRARY}
    ${CUDAQ_REALTIME_LIBRARY}
    ${CUDAQ_REALTIME_DISPATCH_LIBRARY}
    ${CUDAQ_REALTIME_HOST_DISPATCH_LIBRARY}
    cudaq-realtime-pipeline
    cudaq-qec
    cudaq::cudaq
  )

  get_filename_component(_cudaq_rt_lib_dir "${CUDAQ_REALTIME_LIBRARY}" DIRECTORY)
  set_target_properties(test_realtime_predecoder_w_pymatching PROPERTIES
    BUILD_RPATH "${CMAKE_BINARY_DIR}/lib;${_cudaq_rt_lib_dir}"
    INSTALL_RPATH "${CMAKE_BINARY_DIR}/lib;${_cudaq_rt_lib_dir}"
  )

  if(ENABLE_NVTX)
    target_compile_definitions(test_realtime_predecoder_w_pymatching PRIVATE ENABLE_NVTX)
  endif()

else()
  message(WARNING "TensorRT or cudaq-realtime-pipeline not found. "
                  "Skipping test_realtime_predecoder_w_pymatching.")
endif()

# =========================================================================== #
# hololink_predecoder_bridge
# =========================================================================== #
# FPGA bridge: Hololink RDMA -> realtime_pipeline with AI predecoder +
# PyMatching (HOST_LOOP, external ring buffer from Hololink transceiver).

if (GPU_ROCE_TRANSCEIVER_LIB AND CUDAQ_REALTIME_INCLUDE_DIR AND
    CUDAQ_REALTIME_LIBRARY AND CUDAQ_REALTIME_HOST_DISPATCH_LIBRARY AND
    CUDAQ_REALTIME_BRIDGE_HOLOLINK_LIBRARY AND
    DOCA_INCLUDE_DIR AND DOCA_VERBS_LIB AND DOCA_COMMON_LIB AND
    DOCA_GPUNETIO_LIB AND
    TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY AND TENSORRT_ONNX_PARSER_LIBRARY AND
    TARGET cudaq-realtime-pipeline)

  message(STATUS "Building hololink_predecoder_bridge")

  get_filename_component(_cuda_bin_bridge "${CMAKE_CUDA_COMPILER}" DIRECTORY)
  get_filename_component(_cuda_root_bridge "${_cuda_bin_bridge}" DIRECTORY)
  set(_cuda_cccl_include_bridge "${_cuda_root_bridge}/include/cccl")

  add_executable(hololink_predecoder_bridge
    hololink_predecoder_bridge.cpp
    predecoder_pipeline_common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/realtime/ai_predecoder_service.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/realtime/ai_decoder_service.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/cuda_link_stub.cu
  )

  set_target_properties(hololink_predecoder_bridge PROPERTIES
    LINKER_LANGUAGE CUDA
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_STANDARD 17
  )

  target_compile_definitions(hololink_predecoder_bridge PRIVATE
    ONNX_MODEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../lib/realtime"
  )

  target_include_directories(hololink_predecoder_bridge PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${_cuda_cccl_include_bridge}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_SOURCE_DIR}/libs/core/include
    ${CUDAQ_REALTIME_INCLUDE_DIR}
    ${HOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR}/src
  )

  target_link_libraries(hololink_predecoder_bridge PRIVATE
    cudaq-realtime-pipeline
    ${CUDAQ_REALTIME_DISPATCH_LIBRARY}
    ${CUDAQ_REALTIME_HOST_DISPATCH_LIBRARY}
    ${CUDAQ_REALTIME_BRIDGE_HOLOLINK_LIBRARY}
    ${GPU_ROCE_TRANSCEIVER_LIB}
    ${BASE_RECEIVER_OP_LIB}
    ${HOLOLINK_CORE_LIB}
    ${HOLOLINK_COMMON_LIB}
    ${CUDAQ_REALTIME_LIBRARY}
    ${TENSORRT_LIBRARY}
    ${TENSORRT_ONNX_PARSER_LIBRARY}
    cudaq-qec
    cudaq::cudaq
    $<$<BOOL:${_CUDAQ_LIBRARY}>:${_CUDAQ_LIBRARY}>
    $<$<BOOL:${_NVQIR_LIBRARY}>:${_NVQIR_LIBRARY}>
    CUDA::cudart
    CUDA::cuda_driver
    ${DOCA_VERBS_LIB}
    ${DOCA_GPUNETIO_LIB}
    ${DOCA_COMMON_LIB}
    ${IBVERBS_LIB}
    Threads::Threads
    ${CMAKE_DL_LIBS}
  )

  target_link_options(hololink_predecoder_bridge PRIVATE
    "LINKER:--allow-shlib-undefined"
  )

  if (holoscan_FOUND)
    target_link_libraries(hololink_predecoder_bridge PRIVATE holoscan::core)
  endif()

  get_filename_component(_cudaq_rt_lib_dir_b "${CUDAQ_REALTIME_LIBRARY}" DIRECTORY)
  set_target_properties(hololink_predecoder_bridge PROPERTIES
    BUILD_RPATH "${_cudaq_rt_lib_dir_b};${DOCA_LIB_DIR};${CMAKE_BINARY_DIR}/lib;${CMAKE_BINARY_DIR}/lib/decoder-plugins"
    INSTALL_RPATH "${_cudaq_rt_lib_dir_b};${DOCA_LIB_DIR};${CMAKE_BINARY_DIR}/lib;${CMAKE_BINARY_DIR}/lib/decoder-plugins"
  )

  if(ENABLE_NVTX)
    target_compile_definitions(hololink_predecoder_bridge PRIVATE ENABLE_NVTX)
  endif()

else()
  message(WARNING "hololink_predecoder_bridge: skipped (missing deps). "
                  "Requires hololink bridge libs + TensorRT + cudaq-realtime-pipeline.")
endif()
