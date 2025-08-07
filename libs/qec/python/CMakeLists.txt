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
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11
  GIT_TAG v2.13
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(pybind11)

find_package(Python COMPONENTS Interpreter)

# ==============================================================================

set(MODULE_NAME _pycudaqx_qec_the_suffix_matters_cudaq_qec)

cudaqx_add_pymodule(${MODULE_NAME}
  bindings/cudaqx_qec.cpp
  bindings/py_code.cpp
  bindings/py_decoder.cpp
)

target_link_libraries(${MODULE_NAME} 
  PRIVATE 
    cudaq-qec
)

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

