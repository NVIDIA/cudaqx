# ============================================================================ #
# Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_library(cudaq-solvers SHARED
  observe_gradients/central_difference.cpp
  observe_gradients/forward_difference.cpp
  observe_gradients/observe_gradient.cpp
  observe_gradients/parameter_shift.cpp
  operators/molecule/drivers/process.cpp
  operators/molecule/drivers/pyscf_driver.cpp
  operators/molecule/fermion_compilers/fermion_compiler.cpp
  operators/molecule/fermion_compilers/jordan_wigner.cpp
  operators/molecule/fermion_compilers/bravyi_kitaev.cpp
  operators/molecule/molecule.cpp
  operators/graph/max_cut.cpp
  operators/graph/clique.cpp
  operators/block_encoding.cpp
  quantum_exact_lanczos.cpp
  operators/operator_pools/operator_pool.cpp
  operators/operator_pools/spin_complement_gsd.cpp
  operators/operator_pools/uccsd_operator_pool.cpp
  operators/operator_pools/qaoa_operator_pool.cpp
  operators/operator_pools/uccgsd_operator_pool.cpp
  operators/operator_pools/ceo_operator_pool.cpp
  operators/operator_pools/upccgsd_operator_pool.cpp
  operators/uccgsd_excitation_utils.cpp
  operators/ceo_excitation_utils.cpp
  version.cpp
)

add_subdirectory(adapt)
add_subdirectory(optimizers)
add_subdirectory(stateprep)
add_subdirectory(qaoa)

target_include_directories(cudaq-solvers
  PUBLIC
    $<BUILD_INTERFACE:${CUDAQX_SOLVERS_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${CUDAQ_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_options(cudaq-solvers PUBLIC
  $<$<CXX_COMPILER_ID:GNU>:-Wl,--no-as-needed>
)

target_link_libraries(cudaq-solvers
  PUBLIC
    cudaqx-core
    cudaq::cudaq-operator
  PRIVATE
    cudaq::cudaq-common
    nlohmann_json::nlohmann_json
    cppitertools::cppitertools
)

set_target_properties(cudaq-solvers PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# RPATH configuration
# ==============================================================================

if (NOT SKBUILD)
  set_target_properties(cudaq-solvers PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib"
  )

  # Let CMake automatically add paths of linked libraries to the RPATH:
  set_target_properties(cudaq-solvers PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE)
else()
  # CUDA-Q install its libraries in site-packages/lib (or dist-packages/lib)
  # Thus, we need the $ORIGIN/../lib
  set_target_properties(cudaq-solvers PROPERTIES
    INSTALL_RPATH "$ORIGIN/../../lib"
  )
endif()

# Install
# ==============================================================================

install(TARGETS cudaq-solvers
  COMPONENT solvers-lib
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY ${CUDAQX_SOLVERS_INCLUDE_DIR}/cudaq
  COMPONENT solvers-headers
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.h"
)

