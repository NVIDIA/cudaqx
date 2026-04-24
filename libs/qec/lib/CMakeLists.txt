# ============================================================================ #
# Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LIBRARY_NAME cudaq-qec)

add_compile_options(-Wno-attributes) 

find_package(CUDAToolkit REQUIRED)

# cuStabilizer is shipped inside the cuquantum-python-cuXX pip wheels
# (FindcuStabilizer auto-discovers it from the active Python environment).
# If it is missing, surface a clear, actionable error rather than CMake's
# default "Could NOT find cuStabilizer" message.
find_package(cuStabilizer QUIET)
if(NOT cuStabilizer_FOUND)
  set(_cuda_major "${CUDAToolkit_VERSION_MAJOR}")
  if(NOT _cuda_major)
    set(_cuda_major "<CUDA major version, e.g. 12 or 13>")
  endif()
  message(FATAL_ERROR
    "cuStabilizer was not found.\n"
    "It is required to build cudaq-qec and is distributed as part of the "
    "cuQuantum SDK pip wheels. Install it into the Python environment used "
    "for this build with:\n"
    "    pip install cuquantum-python-cu${_cuda_major}\n"
    "If the wheel is installed but discovery still fails (e.g. because the "
    "build runs in an isolated venv such as `python -m build`), point CMake "
    "at the install prefix explicitly via either of:\n"
    "    -DCUQUANTUM_ROOT=<site-packages>/cuquantum\n"
    "    -DCUSTABILIZER_ROOT=/path/to/custabilizer\n"
    "(both are also honored as environment variables).")
endif()

set(QEC_SOURCES
  code.cpp
  decoder.cpp
  detector_error_model.cpp
  experiments.cpp
  pcm_utils.cpp
  plugin_loader.cpp
  stabilizer_utils.cpp 
  decoders/lut.cpp
  decoders/sliding_window.cpp
  version.cpp
)

list(APPEND QEC_SOURCES
  dem_sampling/dem_sampling_utils.cu
  dem_sampling/dem_sampling_gpu.cpp
)

# FIXME?: This must be a shared library. Trying to build a static one will fail.
add_library(${LIBRARY_NAME} SHARED ${QEC_SOURCES})

add_subdirectory(decoders/plugins/example)
add_subdirectory(decoders/plugins/pymatching)

if(CUDAQ_QEC_BUILD_TRT_DECODER)
  add_subdirectory(decoders/plugins/trt_decoder)
endif()

add_subdirectory(codes)
add_subdirectory(device) 

if (CUDAQX_QEC_USE_FLOAT)
   target_compile_definitions(${LIBRARY_NAME} PUBLIC -DCUDAQX_QEC_FLOAT_TYPE=float)
endif()

target_include_directories(${LIBRARY_NAME}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
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

target_link_libraries(${LIBRARY_NAME} PRIVATE
  cuStabilizer::cuStabilizer
  CUDA::cudart
)
set_source_files_properties(
  dem_sampling/dem_sampling_utils.cu
  PROPERTIES LANGUAGE CUDA
)

set_target_properties(${LIBRARY_NAME} PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# RPATH configuration 
# ==============================================================================

if (NOT SKBUILD)
  # Append the cuStabilizer library directory (typically inside a
  # cuquantum-python pip wheel) so the loader can find libcustabilizer.so.0
  # at run-time without requiring LD_LIBRARY_PATH.
  set(_qec_build_rpath "$ORIGIN")
  if(cuStabilizer_LIBRARY_DIR)
    list(APPEND _qec_build_rpath "${cuStabilizer_LIBRARY_DIR}")
  endif()

  set_target_properties(${LIBRARY_NAME} PROPERTIES
    BUILD_RPATH "${_qec_build_rpath}"
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib"
  )

  # Let CMake automatically add paths of linked libraries to the RPATH
  # (this picks up cuStabilizer_LIBRARY_DIR for INSTALL_RPATH as well):
  set_target_properties(${LIBRARY_NAME} PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE)
else()
  # CUDA-Q install its libraries in site-packages/lib (or dist-packages/lib)
  # Thus, we need the $ORIGIN/../lib.
  #
  # cuStabilizer ships in the sibling cuquantum-python-cuXX pip wheel (declared
  # as a runtime dependency in pyproject.toml).  Its layout in site-packages is
  #   <sp>/cudaq_qec/lib/libcudaq-qec.so   (this target -- $ORIGIN)
  #   <sp>/cuquantum/lib/libcustabilizer.so.0
  # so we add a relative RPATH entry pointing into that sibling wheel.  Using a
  # relative path (rather than the absolute cuStabilizer_LIBRARY_DIR resolved
  # at build time) keeps the wheel portable across environments.
  set_target_properties(${LIBRARY_NAME} PROPERTIES
    INSTALL_RPATH "$ORIGIN/../../lib:$ORIGIN/../../cuquantum/lib"
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

add_subdirectory(realtime)
