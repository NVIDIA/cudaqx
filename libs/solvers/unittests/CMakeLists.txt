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
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.17.0
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(googletest)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Bug in GCC 12 leads to spurious warnings (-Wrestrict)
# https://gcc.gnu.org/bugzilla/show_bug.cgi?id=105329
if (CMAKE_COMPILER_IS_GNUCXX 
  AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0.0 
  AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13.0.0)
  target_compile_options(gtest PUBLIC --param=evrp-mode=legacy)
endif()
include(GoogleTest)

# ==============================================================================

add_subdirectory(nvqpp)

# ==============================================================================

add_executable(test_adapt test_adapt.cpp)
target_link_libraries(test_adapt PRIVATE GTest::gtest_main cudaq-solvers test-kernels cudaq::cudaq)
add_dependencies(CUDAQXSolversUnitTests test_adapt)
gtest_discover_tests(test_adapt)

add_executable(test_molecule test_molecule.cpp)
target_link_libraries(test_molecule PRIVATE GTest::gtest_main cudaq-solvers cudaq::cudaq)
add_dependencies(CUDAQXSolversUnitTests test_molecule)
gtest_discover_tests(test_molecule)

add_executable(test_bravyi_kitaev test_bravyi_kitaev.cpp)
target_link_libraries(test_bravyi_kitaev PRIVATE GTest::gtest_main cudaq-solvers cudaq::cudaq)
add_dependencies(CUDAQXSolversUnitTests test_bravyi_kitaev)
gtest_discover_tests(test_bravyi_kitaev)

add_executable(test_optimizers test_optimizers.cpp)
target_link_libraries(test_optimizers PRIVATE GTest::gtest_main cudaq-solvers cudaq::cudaq)
add_dependencies(CUDAQXSolversUnitTests test_optimizers)
gtest_discover_tests(test_optimizers)

add_executable(test_operator_pools test_operator_pools.cpp)
target_link_libraries(test_operator_pools PRIVATE GTest::gtest_main cudaq-solvers cudaq::cudaq)
add_dependencies(CUDAQXSolversUnitTests test_operator_pools)
gtest_discover_tests(test_operator_pools)

add_executable(test_vqe test_vqe.cpp)
target_link_libraries(test_vqe PRIVATE GTest::gtest_main cudaq-solvers test-kernels cudaq::cudaq)
add_dependencies(CUDAQXSolversUnitTests test_vqe)
gtest_discover_tests(test_vqe)

add_executable(test_uccsd test_uccsd.cpp)
target_link_libraries(test_uccsd PRIVATE GTest::gtest_main cudaq-solvers test-kernels cudaq::cudaq)
add_dependencies(CUDAQXSolversUnitTests test_uccsd)
gtest_discover_tests(test_uccsd)

add_executable(test_uccgsd_operator_pool test_uccgsd_operator_pool.cpp)
target_link_libraries(test_uccgsd_operator_pool PRIVATE GTest::gtest_main cudaq-solvers cudaq::cudaq)
add_dependencies(CUDAQXSolversUnitTests test_uccgsd_operator_pool)
gtest_discover_tests(test_uccgsd_operator_pool)

add_executable(test_upccgsd_operator_pool test_upccgsd_operator_pool.cpp)
target_link_libraries(test_upccgsd_operator_pool PRIVATE GTest::gtest_main cudaq-solvers cudaq::cudaq)
add_dependencies(CUDAQXSolversUnitTests test_upccgsd_operator_pool)
gtest_discover_tests(test_upccgsd_operator_pool)

add_executable(test_qaoa test_qaoa.cpp)
target_link_libraries(test_qaoa PRIVATE GTest::gtest_main cudaq-solvers test-kernels cudaq::cudaq)
gtest_discover_tests(test_qaoa)
