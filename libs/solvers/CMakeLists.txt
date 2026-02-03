# ============================================================================ #
# Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Do not merge

# We need 3.28 because of the `EXCLUDE_FROM_ALL` in FetchContent_Declare
cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

# Set a default build type if none was specified. Must set this before
# project().
set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel")

# Set a default install prefix if none was specified.
set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/.cudaqx" CACHE STRING
    "Install path prefix, prepended onto install directories")

# This policy was added in version 3.30 in which calling `FetchContent_Populate()`
# with a single argument (the name of a declared dependency) is deprecated.
# We use the functionality to import the solvers libraries.
if(POLICY CMP0169)
  cmake_policy(SET CMP0169 OLD)
endif()

# Project setup
# ==============================================================================

add_compile_options(-Wno-attributes) 

# Check if core is built as a standalone project.
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(cudaq-solvers)
  set(CUDAQX_SOLVERS_STANDALONE_BUILD TRUE)

  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

  # Add our Modules to the path
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../../cmake/Modules")

  # Include custom CUDA-QX modules
  include(CUDA-QX)

  # Helper target to collect python modules
  add_custom_target(cudaqx-pymodules)
endif()

enable_language(Fortran)

# The following must go after `project(...)` 
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

set(CUDAQX_SOLVERS_SOURCE_DIR  ${CMAKE_CURRENT_SOURCE_DIR})
set(CUDAQX_SOLVERS_INCLUDE_DIR ${CUDAQX_SOLVERS_SOURCE_DIR}/include)

# Options
# ==============================================================================

option(CUDAQX_SOLVERS_INCLUDE_TESTS
       "Generate build targets for the CUDA-QX Solvers unit tests."
       ${CUDAQX_INCLUDE_TESTS})

option(CUDAQX_SOLVERS_BINDINGS_PYTHON
       "Generate build targets for python bindings."
       ${CUDAQX_BINDINGS_PYTHON})

option(CUDAQX_SOLVERS_INSTALL_PYTHON
       "Install python files alongside the library."
       ${CUDAQX_INSTALL_PYTHON})

# Check for CUDA Support (ref: cuda-quantum/CMakeLists.txt)
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
    find_package(CUDAToolkit REQUIRED)

    message(STATUS "Found CUDA Toolkit version: ${CUDAToolkit_VERSION}")

    if (CUDAToolkit_VERSION VERSION_LESS 13.0)
      # Ampere, Hopper
      set(CUDA_TARGET_ARCHS  "75;80;90")
    else()
      # Ampere, Hopper, Blackwell
      set(CUDA_TARGET_ARCHS  "75;80;90;100")
    endif()
  endif()
  CUDA_get_gencode_args(CUDA_gencode_flags ${CUDA_TARGET_ARCHS})
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -shared -std=c++17 ${CUDA_gencode_flags} --compiler-options -fPIC")

  enable_language(CUDA)
  set(CUDA_FOUND TRUE)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
  message(STATUS "Cuda language found.")
endif()

# External Dependencies 
# ==============================================================================

include(FetchContent)

FetchContent_Declare(
  cppitertools
  GIT_REPOSITORY https://github.com/ryanhaining/cppitertools.git
  GIT_TAG master
  EXCLUDE_FROM_ALL
)

# We need version 3.11.1 because that is what CUDA-Q uses. If CUDA-Q updates,
# then we need to remember to update.
FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json
  GIT_TAG v3.11.1
  EXCLUDE_FROM_ALL
)

FetchContent_MakeAvailable(cppitertools json)

if (CUDAQX_SOLVERS_STANDALONE_BUILD)
  # FIXME for now, we only use library mode
  set(CUDAQ_LIBRARY_MODE ON)
  find_package(CUDAQ REQUIRED)

  # FIXME
  add_subdirectory(../core core_build)
endif()

# Wheel building setup
# ==============================================================================

if (SKBUILD)
  # When building with scikit, i.e., building wheels, we want all the install
  # to be on the package directory.
  set(CMAKE_INSTALL_BINDIR cudaq_solvers/bin)
  set(CMAKE_INSTALL_INCLUDEDIR cudaq_solvers/include)
  set(CMAKE_INSTALL_LIBDIR cudaq_solvers/lib)
endif()

# Version
# ==============================================================================
if (DEFINED ENV{CUDAQX_SOLVERS_VERSION})
  # The version was defined by the user (likely a bot performing the build), so
  # use the value provided as is.
  set(CUDAQX_SOLVERS_VERSION "$ENV{CUDAQX_SOLVERS_VERSION}")
  message(STATUS "CUDAQX_SOLVERS_VERSION is set to ${CUDAQX_SOLVERS_VERSION}")
else()
  # Otherwise, create a 0.0.0 version string.
  set(CUDAQX_SOLVERS_VERSION "0.0.0")
endif()

# Retrieve the commit SHA for full revision description
execute_process(COMMAND git rev-parse --verify HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE CUDAQX_SOLVERS_COMMIT_SHA OUTPUT_STRIP_TRAILING_WHITESPACE)

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/lib/version.cpp.in"
               "${CMAKE_CURRENT_BINARY_DIR}/lib/version.cpp" @ONLY)

# Directory setup
# ==============================================================================

add_subdirectory(lib)
add_subdirectory(tools)

if (CUDAQX_SOLVERS_BINDINGS_PYTHON)
  add_subdirectory(python)
endif()

if (CUDAQX_SOLVERS_INCLUDE_TESTS)
  add_custom_target(CUDAQXSolversUnitTests)
  if (CUDAQX_SOLVERS_STANDALONE_BUILD)
    include(CTest)

    add_custom_target(run_tests
      COMMAND ${CMAKE_COMMAND} -E env 
              PYTHONPATH="${CUDAQ_INSTALL_DIR}:${CMAKE_BINARY_DIR}/python"
              ${CMAKE_CTEST_COMMAND} --output-on-failure
      DEPENDS CUDAQXSolversUnitTests
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )

    if (CUDAQX_SOLVERS_BINDINGS_PYTHON)
      add_custom_target(run_python_tests
        COMMAND ${CMAKE_COMMAND} -E env 
                PYTHONPATH="${CUDAQ_INSTALL_DIR}:${CMAKE_BINARY_DIR}/python"
                pytest -v ${CUDAQX_SOLVERS_SOURCE_DIR}/python/tests
        DEPENDS cudaqx-pymodules
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      )
    endif()
  else()
    add_dependencies(CUDAQXUnitTests CUDAQXSolversUnitTests)
  endif()
  add_subdirectory(unittests)
endif()
