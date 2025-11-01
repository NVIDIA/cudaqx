# ============================================================================ #
# Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

set(MODULE_NAME "cudaq-qec-trt-decoder")

project(${MODULE_NAME})

# Set default TensorRT root if not provided
if(NOT DEFINED TENSORRT_ROOT)
    message(STATUS "TENSORRT_ROOT not provided, using default: ${TENSORRT_ROOT}")
else()
    message(STATUS "Using TENSORRT_ROOT: ${TENSORRT_ROOT}")
endif()

# Specify the source file for the plugin
set(PLUGIN_SRC
  trt_decoder.cpp
  # Add additional source files here as needed
)

find_package(CUDAToolkit REQUIRED)

# Create the shared library
add_library(${MODULE_NAME} SHARED ${PLUGIN_SRC})

# Check for TensorRT availability
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    PATHS
        ${TENSORRT_ROOT}/include
        /usr/include/x86_64-linux-gnu
        /usr/include/aarch64-linux-gnu/
        /usr/local/cuda/include
        /usr/local/tensorrt/include
        /opt/tensorrt/include
    NO_DEFAULT_PATH
)

find_library(TENSORRT_LIBRARY nvinfer
    PATHS
        ${TENSORRT_ROOT}/lib
        /usr/lib/x86_64-linux-gnu
        /usr/local/cuda/lib64
        /usr/local/tensorrt/lib
        /opt/tensorrt/lib
    NO_DEFAULT_PATH
)

find_library(TENSORRT_ONNX_LIBRARY nvonnxparser
    PATHS
        ${TENSORRT_ROOT}/lib
        /usr/lib/x86_64-linux-gnu
        /usr/local/cuda/lib64
        /usr/local/tensorrt/lib
        /opt/tensorrt/lib
    NO_DEFAULT_PATH
)

if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY AND TENSORRT_ONNX_LIBRARY)
    message(STATUS "TensorRT found: ${TENSORRT_INCLUDE_DIR}")
    message(STATUS "TensorRT library: ${TENSORRT_LIBRARY}")
    message(STATUS "TensorRT ONNX parser: ${TENSORRT_ONNX_LIBRARY}")
    target_compile_definitions(${MODULE_NAME} PRIVATE TENSORRT_AVAILABLE)
else()
    message(WARNING "TensorRT not found. Building decoder without TensorRT support.")
    message(WARNING "TENSORRT_INCLUDE_DIR: ${TENSORRT_INCLUDE_DIR}")
    message(WARNING "TENSORRT_LIBRARY: ${TENSORRT_LIBRARY}")
    message(WARNING "TENSORRT_ONNX_LIBRARY: ${TENSORRT_ONNX_LIBRARY}")
endif()

# Set the include directories for dependencies
target_include_directories(${MODULE_NAME}
  PUBLIC
    ${CMAKE_SOURCE_DIR}/libs/qec/include
    ${CMAKE_SOURCE_DIR}/libs/core/include
)

# Add TensorRT include directory if found
if(TENSORRT_INCLUDE_DIR)
    target_include_directories(${MODULE_NAME} PRIVATE 
        ${TENSORRT_INCLUDE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
    )
endif()

# Link with required libraries 
target_link_libraries(${MODULE_NAME}
  PUBLIC
    cudaqx-core
    cudaq::cudaq-operator
  PRIVATE
    cudaq::cudaq-common
    cudaq-qec
)

# Conditionally link TensorRT libraries
if(TENSORRT_LIBRARY AND TENSORRT_ONNX_LIBRARY)
    target_link_libraries(${MODULE_NAME} PRIVATE
        ${TENSORRT_LIBRARY}
        ${TENSORRT_ONNX_LIBRARY}
        CUDA::cudart
    )
endif()

set_target_properties(${MODULE_NAME} PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/decoder-plugins
)

# RPATH configuration 
# ==============================================================================

if (NOT SKBUILD)
  set_target_properties(${MODULE_NAME} PROPERTIES
    BUILD_RPATH "$ORIGIN:${TENSORRT_ROOT}/lib"
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../../lib:$ORIGIN/../../../tensorrt_libs"
  )

  # Let CMake automatically add paths of linked libraries to the RPATH:
  set_target_properties(${MODULE_NAME} PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE)
else()
  # CUDA-Q install its libraries in site-packages/lib (or dist-packages/lib)
  # Thus, we need the $ORIGIN/../lib
  set_target_properties(${MODULE_NAME} PROPERTIES
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../../lib:$ORIGIN/../../../tensorrt_libs"
  )
endif()

# Install
# ==============================================================================

install(TARGETS ${MODULE_NAME}
  COMPONENT qec-lib-plugins
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/decoder-plugins
)



