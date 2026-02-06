# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         #
# All rights reserved.                                                        #
#                                                                             #
# This source code and the accompanying materials are made available under    #
# the terms of the Apache License 2.0 which accompanies this distribution.   #
# ============================================================================ #

# Hololink FPGA syndrome playback tool
# ==============================================================================
# This target is gated by CUDAQX_QEC_ENABLE_HOLOLINK_TOOLS (defined in the
# parent libs/qec/CMakeLists.txt) and requires an external
# holoscan-sensor-bridge build.  It is NOT a CI test — it needs FPGA hardware.

if (NOT HOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR)
  message(FATAL_ERROR
    "HOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR must be set when building hololink tools.")
endif()
if (NOT HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR)
  message(FATAL_ERROR
    "HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR must be set when building hololink tools.")
endif()

find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)

find_library(HOLOLINK_CORE_LIB
  NAMES hololink_core
  PATHS
    "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}"
    "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}/src/hololink/core"
    "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}/lib"
  NO_DEFAULT_PATH)

if (NOT HOLOLINK_CORE_LIB)
  message(FATAL_ERROR
    "Could not find hololink_core library under ${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}.")
endif()

add_executable(hololink_fpga_syndrome_playback
  hololink_fpga_syndrome_playback.cpp)

target_include_directories(hololink_fpga_syndrome_playback
  PRIVATE "${HOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR}/src")

if (DEFINED CUDAQ_INCLUDE_DIR AND CUDAQ_INCLUDE_DIR)
  target_include_directories(hololink_fpga_syndrome_playback
    PRIVATE "${CUDAQ_INCLUDE_DIR}")
endif()

if (DEFINED CUDAQ_INSTALL_DIR AND CUDAQ_INSTALL_DIR)
  target_include_directories(hololink_fpga_syndrome_playback
    PRIVATE "${CUDAQ_INSTALL_DIR}/include")
endif()

if (CUDAQ_REALTIME_INCLUDE_DIR)
  target_include_directories(hololink_fpga_syndrome_playback
    PRIVATE "${CUDAQ_REALTIME_INCLUDE_DIR}")
endif()

target_include_directories(hololink_fpga_syndrome_playback
  PRIVATE "${CUDAToolkit_INCLUDE_DIRS}")

target_link_libraries(hololink_fpga_syndrome_playback
  PRIVATE CUDA::cudart)

target_link_libraries(hololink_fpga_syndrome_playback
  PRIVATE ${HOLOLINK_CORE_LIB} Threads::Threads)
