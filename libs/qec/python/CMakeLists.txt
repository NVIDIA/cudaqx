# ============================================================================ #
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# External Dependencies
# ==============================================================================

# Find Python before nanobind so Python_INCLUDE_DIRS is available when nanobind
# builds its nanobind-static library target.
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

if(NOT COMMAND nanobind_add_module)
  FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind
    GIT_TAG v2.12.0
    EXCLUDE_FROM_ALL
  )
  FetchContent_MakeAvailable(nanobind)
endif()

# ==============================================================================

set(MODULE_NAME _pycudaqx_qec_the_suffix_matters_cudaq_qec)

set(_qec_pymodule_sources
  bindings/cudaqx_qec.cpp
  bindings/py_code.cpp
  bindings/py_decoder.cpp
  bindings/py_decoding_config.cpp
  bindings/py_decoding.cpp
  bindings/py_dem_sampling.cpp
  bindings/py_surface_code.cpp
  bindings/py_patch.cpp
)

# The playback emulator's Python binding source lives with the rest of the
# tool under tools/playback-emulator/python/ (self-contained per that tool's
# own layout), not under python/bindings/ with the other QEC bindings -- it
# only builds where the emulator itself does (a realtime-enabled CUDA-Q
# install; see tools/playback-emulator/lib/CMakeLists.txt). cudaqx_qec.cpp's
# call into bindPlaybackEmulator() is itself guarded by the same
# CUDAQX_QEC_HAS_PLAYBACK_EMULATOR define, set below.
set(_playback_emulator_python_dir
  ${CMAKE_CURRENT_SOURCE_DIR}/../tools/playback-emulator/python)
if(TARGET cudaq-qec-playback-emulator)
  list(APPEND _qec_pymodule_sources
    ${_playback_emulator_python_dir}/py_playback_emulator.cpp)
endif()

cudaqx_add_pymodule(${MODULE_NAME} ${_qec_pymodule_sources})

if(TARGET cudaq-qec-playback-emulator)
  target_compile_definitions(${MODULE_NAME} PRIVATE CUDAQX_QEC_HAS_PLAYBACK_EMULATOR)
  target_include_directories(${MODULE_NAME} PRIVATE ${_playback_emulator_python_dir})
  # cudaq_memory_source is its own small static library (see
  # tools/playback-emulator/lib/CMakeLists.txt) -- py_playback_emulator.cpp
  # binds cudaq_memory_source, so it needs both.
  target_link_libraries(${MODULE_NAME} PRIVATE
    cudaq-qec-playback-emulator
    cudaq-qec-playback-emulator-cudaq-memory-source)
endif()

target_link_libraries(${MODULE_NAME}
  PRIVATE
    cudaq-qec
    cudaq-qec-decoders
    cudaq-qec-realtime-decoding
    cudaq::cudaq-python-interop
    cudaq::cudaq-mlir-runtime
)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(${MODULE_NAME} PRIVATE CUDA::cudart)
target_include_directories(${MODULE_NAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

set_target_properties(${MODULE_NAME} PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/cudaq_qec")

if (NOT SKBUILD)
  add_custom_target(
    copy_qec_python_files ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/cudaq_qec
            ${CMAKE_BINARY_DIR}/python/cudaq_qec
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cudaq_qec
    COMMENT "Copying Python files to binary directory"
  )

  add_dependencies(${MODULE_NAME} copy_qec_python_files)
endif()

# RPATH configuration
# ==============================================================================

if (NOT SKBUILD)
  set_target_properties(${MODULE_NAME} PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
  )

  # Let CMake automatically add paths of linked libraries to the RPATH:
  set_target_properties(${MODULE_NAME} PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
  )
else()
  # CUDA-Q install its libraries in site-packages/lib (or dist-packages/lib)
  # Thus, we need the $ORIGIN/../lib
  set_target_properties(${MODULE_NAME} PROPERTIES
    INSTALL_RPATH "$ORIGIN/lib:$ORIGIN/../lib"
  )
endif()

# Install
# ==============================================================================

install(DIRECTORY cudaq_qec
  COMPONENT qec-python
  DESTINATION .
)

install(TARGETS ${MODULE_NAME}
  COMPONENT qec-python
  DESTINATION cudaq_qec/
)
