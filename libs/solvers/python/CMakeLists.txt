# ============================================================================ #
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                   #
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

find_package(Python COMPONENTS Interpreter REQUIRED)

# ==============================================================================

set(MODULE_NAME _pycudaqx_solvers_the_suffix_matters_cudaq_solvers)

cudaqx_add_pymodule(${MODULE_NAME}
  bindings/cudaqx_solvers.cpp
  bindings/solvers/py_optim.cpp
  bindings/solvers/py_solvers.cpp
)

target_include_directories(${MODULE_NAME} PRIVATE .)

# TEMP change
link_directories(${CUDAQ_INSTALL_DIR}/lib)

# TODO - Remove dependency on cudaq::cudaq in order to avoid automatic linking
# of a specific simulator like libnvqir-qpp.so.
target_link_libraries(${MODULE_NAME} 
  PRIVATE 
    cudaq-solvers
    cudaq::cudaq
    cudaq::cudaq-python-interop
    cudaq-mlir-runtime # TEMP change
)

set_target_properties(${MODULE_NAME} PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/cudaq_solvers")

if (NOT SKBUILD)
  add_custom_target(
    copy_solvers_python_files ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/cudaq_solvers
            ${CMAKE_BINARY_DIR}/python/cudaq_solvers
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cudaq_solvers
    COMMENT "Copying Python files to binary directory"
  )

  add_dependencies(${MODULE_NAME} copy_solvers_python_files)
endif()

# RPATH configuration 
# ==============================================================================

if (NOT SKBUILD)
  set_target_properties(${MODULE_NAME} PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN/../lib"
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

install(DIRECTORY cudaq_solvers
  COMPONENT solvers-python
  DESTINATION .
)

install(TARGETS ${MODULE_NAME}
  COMPONENT solvers-python
  DESTINATION cudaq_solvers/
)

