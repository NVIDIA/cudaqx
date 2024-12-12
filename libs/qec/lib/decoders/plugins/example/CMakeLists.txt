# ============================================================================ #
# Copyright (c) 2024 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

cmake_minimum_required(VERSION 3.28 FATAL_ERROR)
project(cudaq-qec-example)

# Specify the source file for the plugin
set(PLUGIN_SRC 
  single_error_lut_example.cpp
  # single_error_lut_example2.cpp // add other decoder source files here
  plugin_exports.cpp
)

# Create the shared library
add_library(cudaq-qec-example SHARED ${PLUGIN_SRC})

# Set the include directories for dependencies
target_include_directories(cudaq-qec-example 
  PUBLIC 
    ${CMAKE_SOURCE_DIR}/libs/qec/include   # Path to cudaq/qec/decoder.h
    ${CMAKE_SOURCE_DIR}/libs/core/include  # Path to cuda-qx/core/ (for tensor.h, extension points, etc)
)

# Link with required libraries (based on earlier CMake you shared)
target_link_libraries(cudaq-qec-example 
  PUBLIC 
    cudaqx-core   # This could be from another library target, or pre-built if needed
    cudaq::cudaq 
    cudaq::cudaq-spin
  PRIVATE 
    cudaq::cudaq-common
)

# Place the shared library in the specified output directory
set_target_properties(cudaq-qec-example PROPERTIES 
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/plugins
)

# Set RPATH so that the shared libraries are properly resolved
set_target_properties(cudaq-qec-example PROPERTIES
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib"
)
