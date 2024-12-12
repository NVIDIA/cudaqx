# ============================================================================ #
# Copyright (c) 2024 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)

# We need 3.28 because of the `EXCLUDE_FROM_ALL` in FetchContent_Declare
cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

project(CUDAQX
  VERSION 0.0.0
  LANGUAGES C CXX)

set(CUDAQX_ALL_LIBS "qec;solvers")
set(CUDAQX_ENABLE_LIBS "" CACHE STRING
  "Semicolon-separated list of libs to build (${CUDAQX_ALL_LIBS}), or \"all\".")

# We don't want to handle "all" later, thus expand it here.
if(CUDAQX_ENABLE_LIBS STREQUAL "all" )
  set(CUDAQX_ENABLE_LIBS ${CUDAQX_ALL_LIBS})
endif()

# Sanity check.
foreach(lib ${CUDAQX_ENABLE_LIBS})
  if (NOT "${lib}" IN_LIST CUDAQX_ALL_LIBS)
    message(FATAL_ERROR "${lib} isn't a known library: ${CUDAQX_ALL_LIBS}.")
  endif()
endforeach()

# Project setup
# ==============================================================================

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add our Modules to the path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

# Include custom CUDA-QX modules
include(CUDA-QX)

# Options
# ==============================================================================

option(CUDAQX_INCLUDE_TESTS "Generate build targets for unit tests." OFF)
option(CUDAQX_INCLUDE_DOCS "Generate build targets for the docs." OFF)
option(CUDAQX_BINDINGS_PYTHON "Generate build targets for python bindings." OFF)

# Top-level External Dependencies
# ==============================================================================

# FIXME for now, we only use library mode
set(CUDAQ_LIBRARY_MODE ON)
find_package(CUDAQ REQUIRED)

# Top-level targets
# ==============================================================================

# Helper targets to collect libraries and python modules
add_custom_target(cudaqx-pymodules)

# Top-level testing
# ==============================================================================

if (CUDAQX_INCLUDE_TESTS)
  include(CTest)
  add_custom_target(CUDAQXUnitTests)
  add_custom_target(run_tests
    COMMAND ${CMAKE_COMMAND} -E env
            PYTHONPATH="${CUDAQ_INSTALL_DIR}:${CMAKE_BINARY_DIR}/python"
            ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS CUDAQXUnitTests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  if (CUDAQX_BINDINGS_PYTHON)
    set(PYTHON_TEST_DIRS "")
    foreach(lib ${CUDAQX_ENABLE_LIBS})
      list(APPEND PYTHON_TEST_DIRS ../libs/${lib}/python/tests)
    endforeach()
    add_custom_target(run_python_tests
      COMMAND ${CMAKE_COMMAND} -E env
              PYTHONPATH="${CUDAQ_INSTALL_DIR}:${CMAKE_BINARY_DIR}/python"
              pytest -v ${PYTHON_TEST_DIRS}
      DEPENDS cudaqx-pymodules
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
  endif()
endif()

# Check for CUDA Support
# ==============================================================================
include(CheckLanguage)
check_language(CUDA)
set(CUDA_FOUND FALSE)
# Generate -gencode arch=compute_XX,code=sm_XX for list of supported
# arch values.
# List should be sorted in increasing order.
function(CUDA_get_gencode_args out_args_string arch_values)
  # allow the user to pass the list like a normal variable
  set(arch_list ${arch_values} ${ARGN})
  set(out "")
  foreach(arch IN LISTS arch_list)
    set(out "${out} -gencode arch=compute_${arch},code=sm_${arch}")
  endforeach(arch)

  # Repeat the last one as to ensure the generation of PTX for most
  # recent virtual architecture for forward compatibility
  list(GET arch_list -1 last_arch)
  set(out "${out} -gencode arch=compute_${last_arch},code=compute_${last_arch}")
  set(${out_args_string} ${out} PARENT_SCOPE)
endfunction()

if(CMAKE_CUDA_COMPILER)
  if (NOT CUDA_TARGET_ARCHS)
    # Volta, Ampere, Hopper
    set(CUDA_TARGET_ARCHS  "70;80;90")
  endif()
  CUDA_get_gencode_args(CUDA_gencode_flags ${CUDA_TARGET_ARCHS})
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -shared -std=c++17 ${CUDA_gencode_flags} --compiler-options -fPIC")

  enable_language(CUDA)
  set(CUDA_FOUND TRUE)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
  message(STATUS "Cuda language found.")
endif()

# Directory setup
# ==============================================================================

add_subdirectory(libs/core)

if (CUDAQX_INCLUDE_DOCS)
  add_subdirectory(docs)
endif()

foreach(lib ${CUDAQX_ENABLE_LIBS})
  add_subdirectory(libs/${lib})
endforeach()

