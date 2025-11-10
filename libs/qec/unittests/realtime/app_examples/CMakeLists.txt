# ============================================================================ #
# Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

#------------------------------------------------------------------------------#
# Surface code 1, fully local, regular Stim simulator
add_executable(surface_code-1-local)

cudaqx_add_device_code(surface_code-1-local
  SOURCES
    surface_code-1.cpp
  COMPILER_FLAGS
    --target stim
)

target_link_directories(surface_code-1-local
  PRIVATE ${CUDAQ_INSTALL_DIR}/lib)

target_link_libraries(surface_code-1-local
  PRIVATE
    cudaq-qec
    cudaq-qec-realtime-decoding
    cudaq-qec-realtime-decoding-simulation
    cudaq::cudaq cudaq::cudaq-common cudaq-mlir-runtime
    nvqir nvqir-stim)

add_test(
  NAME app_examples.surface_code-1-local-test-distance-3
  COMMAND
    bash "${CMAKE_CURRENT_SOURCE_DIR}/surface_code-1-test.sh"
      ${CMAKE_CURRENT_BINARY_DIR}/surface_code-1-local
      ${CMAKE_CURRENT_BINARY_DIR}/surface_code-1-local
      3 40 40 NULL 12 6
      ${CMAKE_BINARY_DIR}/lib
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# Test with sliding_window decoder
add_test(
  NAME app_examples.surface_code-1-local-test-distance-3-sliding-window
  COMMAND
    bash "${CMAKE_CURRENT_SOURCE_DIR}/surface_code-1-test.sh"
      ${CMAKE_CURRENT_BINARY_DIR}/surface_code-1-local
      ${CMAKE_CURRENT_BINARY_DIR}/surface_code-1-local
      3 40 40 NULL 12 6
      ${CMAKE_BINARY_DIR}/lib
      sliding_window 6 1
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# This must be disabled for now because the multi_error_lut decoder is not
# powerful enough to pass this test. The nv-qldpc-decoder can pass this test,
# but that is not available on the GitHub repo.
# add_test(
#   NAME app_examples.surface_code-1-local-test-distance-5
#   COMMAND
#     bash "${CMAKE_CURRENT_SOURCE_DIR}/surface_code-1-test.sh"
#       ${CMAKE_CURRENT_BINARY_DIR}/surface_code-1-local
#       ${CMAKE_CURRENT_BINARY_DIR}/surface_code-1-local
#       5 40 40 NULL 20 10
#       ${CMAKE_BINARY_DIR}/lib
#   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
# )

# Surface code 1 Python test
add_test(
  NAME app_examples.py-surface_code-1-test
  COMMAND
    pytest ${CMAKE_CURRENT_SOURCE_DIR}/surface_code-1-test.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# End of surface code 1, fully local, regular Stim simulator
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
# Surface code 1, --target quantinuum --emulate
add_executable(surface_code-1-quantinuum-emulate)

add_dependencies(surface_code-1-quantinuum-emulate surface_code-1-local)

cudaqx_add_device_code(surface_code-1-quantinuum-emulate
  SOURCES
    surface_code-1.cpp
  COMPILER_FLAGS
    --target quantinuum --emulate --quantinuum-machine Helios-Fake -DMANUALLY_INJECT_ERRORS=1 
  DEPENDS_ON
    # Must wait for the local executable to be built because it has the same
    # surface_code-1.cpp file, and the nvq++ compiler will use the same filename
    # prefix (surface_code-1) for intermediate files. They will stomp on each
    # other if you don't add this.
    surface_code-1-local
)

target_link_directories(surface_code-1-quantinuum-emulate
  PRIVATE ${CUDAQ_INSTALL_DIR}/lib)

# Add -Wl,--export-dynamic to the link line.
target_link_options(surface_code-1-quantinuum-emulate PRIVATE -Wl,--export-dynamic)
target_link_libraries(surface_code-1-quantinuum-emulate
  PRIVATE
    cudaq-qec
    cudaq-qec-realtime-decoding
    cudaq-qec-realtime-decoding-quantinuum
    cudaq::cudaq cudaq::cudaq-common cudaq-mlir-runtime cudaq-rest-qpu
    nvqir nvqir-stim)

add_test(
  NAME app_examples.surface_code-1-quantinuum-emulate-test-distance-3-in-process
  COMMAND
    bash "${CMAKE_CURRENT_SOURCE_DIR}/surface_code-1-test.sh"
      ${CMAKE_CURRENT_BINARY_DIR}/surface_code-1-local
      ${CMAKE_CURRENT_BINARY_DIR}/surface_code-1-quantinuum-emulate
      3 0 1000 NULL 12 6
      ${CMAKE_BINARY_DIR}/lib
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_test(
  NAME app_examples.surface_code-1-quantinuum-emulate-test-distance-5-in-process
  COMMAND
    bash "${CMAKE_CURRENT_SOURCE_DIR}/surface_code-1-test.sh"
      ${CMAKE_CURRENT_BINARY_DIR}/surface_code-1-local
      ${CMAKE_CURRENT_BINARY_DIR}/surface_code-1-quantinuum-emulate
      5 0 0 NULL 20 10
      ${CMAKE_BINARY_DIR}/lib
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)


# End of surface code 1, --target quantinuum --emulate
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
# Surface code 2, fully local, regular Stim simulator
add_executable(surface_code-2-local)

cudaqx_add_device_code(surface_code-2-local
  SOURCES
    surface_code-2.cpp
  COMPILER_FLAGS
    --target stim
)

target_link_directories(surface_code-2-local
  PRIVATE ${CUDAQ_INSTALL_DIR}/lib)

target_link_libraries(surface_code-2-local
  PRIVATE
    cudaq-qec
    cudaq-qec-realtime-decoding
    cudaq-qec-realtime-decoding-simulation
    cudaq::cudaq cudaq::cudaq-common cudaq-mlir-runtime
    nvqir nvqir-stim)

add_test(
  NAME app_examples.surface_code-2-local-test-distance-3
  COMMAND
    bash "${CMAKE_CURRENT_SOURCE_DIR}/surface_code-2-test.sh"
      ${CMAKE_CURRENT_BINARY_DIR}/surface_code-2-local
      ${CMAKE_CURRENT_BINARY_DIR}/surface_code-2-local
      3 10 10 NULL
      ${CMAKE_BINARY_DIR}/lib
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
# This must be disabled for now because the multi_error_lut decoder is not
# powerful enough to pass this test. The nv-qldpc-decoder can pass this test,
# but that is not available on the GitHub repo.
# add_test(
#   NAME app_examples.surface_code-2-local-test-distance-5
#   COMMAND
#     bash "${CMAKE_CURRENT_SOURCE_DIR}/surface_code-2-test.sh"
#       ${CMAKE_CURRENT_BINARY_DIR}/surface_code-2-local
#       ${CMAKE_CURRENT_BINARY_DIR}/surface_code-2-local
#       5 10 10 NULL
#       ${CMAKE_BINARY_DIR}/lib
#   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
# )

# End of surface code 2, fully local, regular Stim simulator
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
# Surface code 2, --target quantinuum --emulate
add_executable(surface_code-2-quantinuum-emulate)

add_dependencies(surface_code-2-quantinuum-emulate surface_code-2-local)

cudaqx_add_device_code(surface_code-2-quantinuum-emulate
  SOURCES
    surface_code-2.cpp
  COMPILER_FLAGS
    --target quantinuum --emulate --quantinuum-machine Helios-Fake -DMANUALLY_INJECT_ERRORS=1
  DEPENDS_ON
    # Must wait for the local executable to be built because it has the same
    # surface_code-2.cpp file, and the nvq++ compiler will use the same filename
    # prefix (surface_code-2) for intermediate files. They will stomp on each
    # other if you don't add this.
    surface_code-2-local
)

target_link_directories(surface_code-2-quantinuum-emulate
  PRIVATE ${CUDAQ_INSTALL_DIR}/lib)

# Add -Wl,--export-dynamic to the link line.
target_link_options(surface_code-2-quantinuum-emulate PRIVATE -Wl,--export-dynamic)
target_link_libraries(surface_code-2-quantinuum-emulate
  PRIVATE
    cudaq-qec
    cudaq-qec-realtime-decoding
    cudaq-qec-realtime-decoding-quantinuum
    cudaq::cudaq cudaq::cudaq-common cudaq-mlir-runtime cudaq-rest-qpu
    nvqir nvqir-stim)

add_test(
  NAME app_examples.surface_code-2-quantinuum-emulate-test-distance-3-in-process
  COMMAND
    bash "${CMAKE_CURRENT_SOURCE_DIR}/surface_code-2-test.sh"
      ${CMAKE_CURRENT_BINARY_DIR}/surface_code-2-local
      ${CMAKE_CURRENT_BINARY_DIR}/surface_code-2-quantinuum-emulate
      3 0 1000 NULL
      ${CMAKE_BINARY_DIR}/lib
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_test(
  NAME app_examples.surface_code-2-quantinuum-emulate-test-distance-5-in-process
  COMMAND
    bash "${CMAKE_CURRENT_SOURCE_DIR}/surface_code-2-test.sh"
      ${CMAKE_CURRENT_BINARY_DIR}/surface_code-2-local
      ${CMAKE_CURRENT_BINARY_DIR}/surface_code-2-quantinuum-emulate
      5 0 0 NULL
      ${CMAKE_BINARY_DIR}/lib
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)


# End of surface code 2, --target quantinuum --emulate
#------------------------------------------------------------------------------#


macro(add_surface_code_test base_name exe_local exe_remote dist server)
  # For now, disable 0.015 error rate because the multi_error_lut decoder is not
  # powerful enough to pass this test. The nv-qldpc-decoder can pass this test,
  # but that is not available on the GitHub repo.
  foreach(err_rate IN ITEMS 0.010)
    foreach(prep IN ITEMS prep0 prepp)
      add_test(
        NAME ${base_name}-dist-${dist}-${prep}-err-${err_rate}
        COMMAND
          bash "${CMAKE_CURRENT_SOURCE_DIR}/surface_code-3-test.sh"
            ${exe_local}
            ${exe_remote}
            ${dist} ${server} ${err_rate} "${prep}"
            ${CMAKE_BINARY_DIR}/quantinuum_decoding_server/lib
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      )
    endforeach()
  endforeach()
endmacro()

#------------------------------------------------------------------------------#
# Surface code 3, fully local, regular Stim simulator
add_executable(surface_code-3-local)

cudaqx_add_device_code(surface_code-3-local
  SOURCES
    surface_code-3.cpp
  COMPILER_FLAGS
    --target stim
)

target_link_directories(surface_code-3-local
  PRIVATE ${CUDAQ_INSTALL_DIR}/lib)

target_link_libraries(surface_code-3-local
  PRIVATE
    cudaq-qec
    cudaq-qec-realtime-decoding
    cudaq-qec-realtime-decoding-simulation
    cudaq::cudaq cudaq::cudaq-common cudaq-mlir-runtime
    nvqir nvqir-stim)

# --- Local tests ---
add_surface_code_test(
  app_examples.surface_code-3-local-test
  ${CMAKE_CURRENT_BINARY_DIR}/surface_code-3-local
  ${CMAKE_CURRENT_BINARY_DIR}/surface_code-3-local
  3 NULL
)
add_surface_code_test(
  app_examples.surface_code-3-local-test
  ${CMAKE_CURRENT_BINARY_DIR}/surface_code-3-local
  ${CMAKE_CURRENT_BINARY_DIR}/surface_code-3-local
  5 NULL
)

# End of surface code 3, fully local, regular Stim simulator
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
# Surface code 3, --target quantinuum --emulate
add_executable(surface_code-3-quantinuum-emulate)

add_dependencies(surface_code-3-quantinuum-emulate surface_code-3-local)

cudaqx_add_device_code(surface_code-3-quantinuum-emulate
  SOURCES
    surface_code-3.cpp
  COMPILER_FLAGS
    --target quantinuum --emulate --quantinuum-machine Helios-Fake -DMANUALLY_INJECT_ERRORS=1
  DEPENDS_ON
    # Must wait for the local executable to be built because it has the same
    # surface_code-3.cpp file, and the nvq++ compiler will use the same filename
    # prefix (surface_code-3) for intermediate files. They will stomp on each
    # other if you don't add this.
    surface_code-3-local
)

target_link_directories(surface_code-3-quantinuum-emulate
  PRIVATE ${CUDAQ_INSTALL_DIR}/lib)

# Add -Wl,--export-dynamic to the link line.
target_link_options(surface_code-3-quantinuum-emulate PRIVATE -Wl,--export-dynamic)
target_link_libraries(surface_code-3-quantinuum-emulate
  PRIVATE
    cudaq-qec
    cudaq-qec-realtime-decoding
    cudaq-qec-realtime-decoding-quantinuum
    cudaq::cudaq cudaq::cudaq-common cudaq-mlir-runtime cudaq-rest-qpu
    nvqir nvqir-stim)


    # --- Quantinuum emulate (in-process) ---
add_surface_code_test(
  app_examples.surface_code-3-quantinuum-emulate-test-in-process
  ${CMAKE_CURRENT_BINARY_DIR}/surface_code-3-local
  ${CMAKE_CURRENT_BINARY_DIR}/surface_code-3-quantinuum-emulate
  3 NULL
)
add_surface_code_test(
  app_examples.surface_code-3-quantinuum-emulate-test-in-process
  ${CMAKE_CURRENT_BINARY_DIR}/surface_code-3-local
  ${CMAKE_CURRENT_BINARY_DIR}/surface_code-3-quantinuum-emulate
  5 NULL
)

# End of surface code 3, --target quantinuum --emulate
#------------------------------------------------------------------------------#


