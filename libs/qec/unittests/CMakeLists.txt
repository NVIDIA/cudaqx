# ============================================================================ #
# Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# External Dependencies 
# ==============================================================================

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.17.0
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(googletest)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Bug in GCC 12 leads to spurious warnings (-Wrestrict)
# https://gcc.gnu.org/bugzilla/show_bug.cgi?id=105329
if (CMAKE_COMPILER_IS_GNUCXX 
  AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0.0 
  AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13.0.0)
  target_compile_options(gtest PUBLIC --param=evrp-mode=legacy)
endif()
include(GoogleTest)

find_package(CUDAToolkit REQUIRED)

# ==============================================================================
add_compile_options(-Wno-attributes) 

add_executable(test_decoders test_decoders.cpp decoders/sample_decoder.cpp)
target_link_libraries(test_decoders PRIVATE GTest::gtest_main cudaq-qec cudaq::cudaq)
add_dependencies(CUDAQXQECUnitTests test_decoders)
gtest_discover_tests(test_decoders)

add_executable(test_qec test_qec.cpp)
target_link_libraries(test_qec PRIVATE GTest::gtest_main cudaq-qec cudaq::cudaq)
add_dependencies(CUDAQXQECUnitTests test_qec)
gtest_discover_tests(test_qec)

# TensorRT decoder test is only built for x86 architectures
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64|amd64)|(^i.86$)")
  add_executable(test_trt_decoder ./decoders/trt_decoder/test_trt_decoder.cpp)

  # Find TensorRT for the test
  find_path(TENSORRT_INCLUDE_DIR NvInfer.h
      PATHS
          /usr/include/x86_64-linux-gnu
          /usr/local/cuda/include
          /usr/local/tensorrt/include
          /opt/tensorrt/include
      NO_DEFAULT_PATH
  )

  target_include_directories(test_trt_decoder PRIVATE
      ${CUDAToolkit_INCLUDE_DIRS}
  )

  target_link_libraries(test_trt_decoder PRIVATE GTest::gtest_main cudaq-qec cudaq-qec-trt-decoder cudaq::cudaq)
  add_dependencies(CUDAQXQECUnitTests test_trt_decoder)
  gtest_discover_tests(test_trt_decoder)
endif()

# ==============================================================================

add_subdirectory(backend-specific)

