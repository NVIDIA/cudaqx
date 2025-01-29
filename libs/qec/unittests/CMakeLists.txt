# ============================================================================ #
# Copyright (c) 2024 NVIDIA Corporation & Affiliates.                          #
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
  GIT_TAG v1.15.2
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
add_compile_options(-Wno-attributes) 

add_executable(test_decoders test_decoders.cpp decoders/sample_decoder.cpp)
target_link_libraries(test_decoders PRIVATE GTest::gtest_main cudaq-qec)
add_dependencies(CUDAQXQECUnitTests test_decoders)
gtest_discover_tests(test_decoders)

add_executable(test_qec test_qec.cpp)
target_link_libraries(test_qec PRIVATE GTest::gtest_main cudaq-qec)
add_dependencies(CUDAQXQECUnitTests test_qec)
gtest_discover_tests(test_qec)

# ==============================================================================

add_subdirectory(backend-specific)

