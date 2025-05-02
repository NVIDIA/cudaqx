# ============================================================================ #
# Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Requiring the same version as the others.
cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

# Set a default build type if none was specified. Must set this before
# project().
set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel")

# Set a default install prefix if none was specified.
set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/.cudaqx" CACHE STRING
    "Install path prefix, prepended onto install directories")

# Project setup
# ==============================================================================

# Check if core is built as a standalone project.
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(cudaqx-qec)
  set(CUDAQX_QEC_STANDALONE_BUILD TRUE)

  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

  # Add our Modules to the path
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../../cmake/Modules")

  # Include custom CUDA-QX modules
  include(CUDA-QX)

  # Helper target to collect python modules
  add_custom_target(cudaqx-pymodules)
endif()

# The following must go after `project(...)` 
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

set(CUDAQX_QEC_SOURCE_DIR  ${CMAKE_CURRENT_SOURCE_DIR})
set(CUDAQX_QEC_INCLUDE_DIR ${CUDAQX_QEC_SOURCE_DIR}/include)

# Options
# ==============================================================================

option(CUDAQX_QEC_INCLUDE_TESTS
       "Generate build targets for the CUDA-QX QEC unit tests."
       ${CUDAQX_INCLUDE_TESTS})

option(CUDAQX_QEC_BINDINGS_PYTHON
       "Generate build targets for python bindings."
       ${CUDAQX_BINDINGS_PYTHON})

option(CUDAQX_QEC_INSTALL_PYTHON
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

# External Dependencies 
# ==============================================================================

if (CUDAQX_QEC_STANDALONE_BUILD)
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
  set(CMAKE_INSTALL_BINDIR cudaq_qec/bin)
  set(CMAKE_INSTALL_INCLUDEDIR cudaq_qec/include)
  set(CMAKE_INSTALL_LIBDIR cudaq_qec/lib)
endif()

# Version
# ==============================================================================
if (DEFINED ENV{CUDAQX_QEC_VERSION})
  # The version was defined by the user (likely a bot performing the build), so
  # use the value provided as is.
  set(CUDAQX_QEC_VERSION "$ENV{CUDAQX_QEC_VERSION}")
  message(STATUS "CUDAQX_QEC_VERSION is set to ${CUDAQX_QEC_VERSION}")
else()
  # Otherwise, create a version based on the nearest tag in the git repo.
  set(CUDAQX_QEC_VERSION "0.0.0")
endif()

# Retrieve the commit SHA for full revision description
execute_process(COMMAND git rev-parse --verify HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE CUDAQX_QEC_COMMIT_SHA OUTPUT_STRIP_TRAILING_WHITESPACE)

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/lib/version.cpp.in"
               "${CMAKE_CURRENT_BINARY_DIR}/lib/version.cpp" @ONLY)

# Directory setup
# ==============================================================================

add_subdirectory(lib)

if (CUDAQX_QEC_BINDINGS_PYTHON)
  add_subdirectory(python)
endif()

if (CUDAQX_QEC_INCLUDE_TESTS)
  add_custom_target(CUDAQXQECUnitTests)
  if (CUDAQX_QEC_STANDALONE_BUILD)
    include(CTest)

    add_custom_target(run_tests
      COMMAND ${CMAKE_COMMAND} -E env 
              PYTHONPATH="${CUDAQ_INSTALL_DIR}:${CMAKE_BINARY_DIR}/python"
              ${CMAKE_CTEST_COMMAND} --output-on-failure
      DEPENDS CUDAQXQECUnitTests
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )

    if (CUDAQX_QEC_BINDINGS_PYTHON)
      add_custom_target(run_python_tests
        COMMAND ${CMAKE_COMMAND} -E env 
                PYTHONPATH="${CUDAQ_INSTALL_DIR}:${CMAKE_BINARY_DIR}/python"
                pytest -v ${CUDAQX_QEC_SOURCE_DIR}/python/tests
        DEPENDS cudaqx-pymodules
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      )
    endif()
  else()
    add_dependencies(CUDAQXUnitTests CUDAQXQECUnitTests)
  endif()
  add_subdirectory(unittests)
endif()

# External decoder setup
# ==============================================================================
# Make an external decoder available inside wheel. This will modify the rpath of
# any .so files provided.
function(add_target_libs_to_wheel decoder_lib)
  if (NOT EXISTS "${decoder_lib}")
    message(FATAL_ERROR "Invalid file path to external decoder: ${decoder_lib}.")
  endif()
  message(STATUS "Installing external decoder '${decoder_lib}'")
  # Patch the file's rpath during configuration time
  execute_process(
    COMMAND patchelf --set-rpath $ORIGIN/.. "${decoder_lib}"
    RESULT_VARIABLE patchelf_result
    OUTPUT_VARIABLE patchelf_output
    ERROR_VARIABLE patchelf_error
  )
  if (NOT patchelf_result EQUAL 0)
    message(FATAL_ERROR "Failed to patch rpath on ${decoder_lib}: ${patchelf_error}")
  endif()
  message(STATUS "Successfully patched rpath on ${decoder_lib}")
  install(FILES ${decoder_lib}
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/decoder-plugins
          COMPONENT qec-lib-plugins)
endfunction()

if(NOT QEC_EXTERNAL_DECODERS)
  SET(QEC_EXTERNAL_DECODERS $ENV{QEC_EXTERNAL_DECODERS})
endif()

if (QEC_EXTERNAL_DECODERS)
  while(QEC_EXTERNAL_DECODERS)
    list(POP_FRONT QEC_EXTERNAL_DECODERS LIB_FILE)
    add_target_libs_to_wheel(${LIB_FILE})
  endwhile()
endif()
