# ============================================================================ #
# Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LIBRARY_NAME cudaq-qec)

add_compile_options(-Wno-attributes) 

# FIXME?: This must be a shared library. Trying to build a static one will fail.
add_library(${LIBRARY_NAME} SHARED
  code.cpp
  decoder.cpp
  detector_error_model.cpp
  experiments.cpp
  pcm_utils.cpp
  plugin_loader.cpp
  stabilizer_utils.cpp 
  decoders/single_error_lut.cpp
  decoders/sliding_window.cpp
  version.cpp
)

add_subdirectory(decoders/plugins/example)
add_subdirectory(codes)
add_subdirectory(device) 

if (CUDAQX_QEC_USE_FLOAT)
   target_compile_definitions(${LIBRARY_NAME} PUBLIC -DCUDAQX_QEC_FLOAT_TYPE=float)
endif()

target_include_directories(${LIBRARY_NAME}
  PUBLIC 
    $<BUILD_INTERFACE:${CUDAQX_QEC_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${CUDAQ_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_options(${LIBRARY_NAME} PUBLIC
  $<$<CXX_COMPILER_ID:GNU>:-Wl,--no-as-needed>
)

target_link_libraries(${LIBRARY_NAME}
  PUBLIC 
    cudaqx-core
    cudaq::cudaq-operator
  PRIVATE
    cudaq::cudaq-common
)

set_target_properties(${LIBRARY_NAME} PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# RPATH configuration 
# ==============================================================================

if (NOT SKBUILD)
  set_target_properties(${LIBRARY_NAME} PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib"
  )

  # Let CMake automatically add paths of linked libraries to the RPATH:
  set_target_properties(${LIBRARY_NAME} PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE)
else()
  # CUDA-Q install its libraries in site-packages/lib (or dist-packages/lib)
  # Thus, we need the $ORIGIN/../lib
  set_target_properties(${LIBRARY_NAME} PROPERTIES
    INSTALL_RPATH "$ORIGIN/../../lib"
  )
endif()

# Install 
# ==============================================================================

install(TARGETS ${LIBRARY_NAME}
  COMPONENT qec-lib
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
    
install(DIRECTORY ${CUDAQX_QEC_INCLUDE_DIR}/cudaq
  COMPONENT qec-headers
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.h"
)

