# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

set(MODULE_NAME "cudaq-qec-chromobius")

# External Dependencies
# ==============================================================================

include(FetchContent)

FetchContent_Declare(
  chromobius
  GIT_REPOSITORY https://github.com/quantumlib/chromobius.git
  GIT_TAG efae0325e4a8fc13be981434d38e499aa06b989f # v1.1.1
)

FetchContent_GetProperties(chromobius)
if(NOT chromobius_POPULATED)
  FetchContent_Populate(chromobius)
endif()

if(NOT TARGET libpymatching)
  message(FATAL_ERROR
    "Chromobius decoder plugin requires the libpymatching target. "
    "Ensure decoders/plugins/pymatching is added before chromobius.")
endif()

if(NOT TARGET libstim)
  message(FATAL_ERROR
    "Chromobius decoder plugin requires the libstim target provided by PyMatching.")
endif()

file(STRINGS
  ${chromobius_SOURCE_DIR}/file_lists/source_files_no_main
  CHROMOBIUS_SOURCE_FILES
)
list(TRANSFORM CHROMOBIUS_SOURCE_FILES
  PREPEND "${chromobius_SOURCE_DIR}/"
)

set(CHROMOBIUS_STATIC_TARGET cudaq-qec-chromobius-static)
add_library(${CHROMOBIUS_STATIC_TARGET} STATIC ${CHROMOBIUS_SOURCE_FILES})
set_target_properties(${CHROMOBIUS_STATIC_TARGET} PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)
target_include_directories(${CHROMOBIUS_STATIC_TARGET}
  PUBLIC
    ${chromobius_SOURCE_DIR}/src
)
target_link_libraries(${CHROMOBIUS_STATIC_TARGET}
  PRIVATE
    libstim
    libpymatching
)
target_compile_options(${CHROMOBIUS_STATIC_TARGET}
  PRIVATE
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O3 -Wall -Wpedantic -fno-strict-aliasing>
)

project(${MODULE_NAME})

# Specify the source file for the plugin
set(PLUGIN_SRC
  chromobius.cpp
)

# Create the shared library
add_library(${MODULE_NAME} SHARED ${PLUGIN_SRC})

# Don't export any symbols (specifically from static libs like Chromobius,
# Stim, and PyMatching) to avoid conflicts with other Python packages.
target_link_options(${MODULE_NAME} PRIVATE "-Wl,--exclude-libs,ALL")

# Set the include directories for dependencies
target_include_directories(${MODULE_NAME}
  PUBLIC
    ${CMAKE_SOURCE_DIR}/libs/qec/include
    ${CMAKE_SOURCE_DIR}/libs/core/include
    ${chromobius_SOURCE_DIR}/src
)

# Link with required libraries
target_link_libraries(${MODULE_NAME}
  PUBLIC
    cudaqx-core
    cudaq::cudaq-operator
  PRIVATE
    cudaq::cudaq-common
    cudaq-qec
    ${CHROMOBIUS_STATIC_TARGET}
    libstim
    libpymatching
)

set_target_properties(${MODULE_NAME} PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/decoder-plugins
)

# RPATH configuration
# ==============================================================================

if (NOT SKBUILD)
  set_target_properties(${MODULE_NAME} PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN:$ORIGIN/.."
  )

  # Let CMake automatically add paths of linked libraries to the RPATH:
  set_target_properties(${MODULE_NAME} PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE)
else()
  # CUDA-Q install its libraries in site-packages/lib (or dist-packages/lib)
  # Thus, we need the $ORIGIN/../lib
  set_target_properties(${MODULE_NAME} PROPERTIES
    INSTALL_RPATH "$ORIGIN/../../lib"
  )
endif()

# Install
# ==============================================================================

install(TARGETS ${MODULE_NAME}
  COMPONENT qec-lib-plugins
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/decoder-plugins
)
