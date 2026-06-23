# ============================================================================ #
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

#------------------------------------------------------------------------------#
# Maintenance note: to generate QIR files and corresponding config files for
# Quantinuum, you may manually run tests like the following:
#    CUDAQ_DUMP_JIT_IR=1 KEEP_LOG_FILES=1 ctest -R quantinuum
# The resulting files will be in:
#    build/libs/qec/unittests/realtime/app_examples/config-*.yml
#    build/libs/qec/unittests/realtime/app_examples/qir-*.ll
#------------------------------------------------------------------------------#

# Resolve cudaq-realtime dispatch for app examples that opt into
# CUDAQ_QEC_REALTIME_MODE=inproc_rpc.
set(_app_cudaq_realtime_prefixes "")
if(CUDAQ_REALTIME_ROOT)
  list(APPEND _app_cudaq_realtime_prefixes "${CUDAQ_REALTIME_ROOT}")
  list(APPEND _app_cudaq_realtime_prefixes "${CUDAQ_REALTIME_ROOT}/build")
endif()
if(CUDAQ_INSTALL_PREFIX)
  list(APPEND _app_cudaq_realtime_prefixes "${CUDAQ_INSTALL_PREFIX}")
endif()
if(CUDAQ_INSTALL_DIR)
  list(APPEND _app_cudaq_realtime_prefixes "${CUDAQ_INSTALL_DIR}")
endif()
find_library(_APP_CUDAQ_REALTIME_DISPATCH_LIB cudaq-realtime-dispatch
  PATHS ${_app_cudaq_realtime_prefixes} PATH_SUFFIXES lib)
set(_app_device_call_runtime_link "")
if(NOT CUDAQ_DEVICE_CALL_RUNTIME_LIBRARY)
  find_library(CUDAQ_DEVICE_CALL_RUNTIME_LIBRARY
    NAMES cudaq-device-call-runtime
    PATHS ${_app_cudaq_realtime_prefixes}
    PATH_SUFFIXES lib)
endif()
if(CUDAQ_DEVICE_CALL_RUNTIME_LIBRARY)
  set(_app_device_call_runtime_link ${CUDAQ_DEVICE_CALL_RUNTIME_LIBRARY})
elseif(TARGET cudaq::cudaq-device-call-runtime)
  set(_app_device_call_runtime_link cudaq::cudaq-device-call-runtime)
elseif(TARGET cudaq-device-call-runtime)
  set(_app_device_call_runtime_link cudaq-device-call-runtime)
endif()

function(qec_realtime_app_common_link_options _tgt)
  target_link_options(${_tgt} PRIVATE
    "LINKER:--allow-shlib-undefined"
    "LINKER:--export-dynamic"
  )
  if(_APP_CUDAQ_REALTIME_DISPATCH_LIB)
    target_link_libraries(${_tgt} PRIVATE
      ${_APP_CUDAQ_REALTIME_DISPATCH_LIB})
  endif()
endfunction()

function(qec_realtime_app_link_options _tgt)
  qec_realtime_app_common_link_options(${_tgt})
  if(TARGET cudaq-qec-realtime-cudevice-proprietary)
    target_link_libraries(${_tgt} PRIVATE
      CUDA::cudart
      $<LINK_LIBRARY:WHOLE_ARCHIVE,cudaq-qec-realtime-cudevice-proprietary>)
    set_property(TARGET ${_tgt} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET ${_tgt} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    set_property(TARGET ${_tgt} PROPERTY CUDA_STANDARD 17)
    set_property(TARGET ${_tgt} PROPERTY LINKER_LANGUAGE CUDA)
  endif()
endfunction()

#------------------------------------------------------------------------------#
# Surface code 1, fully local, regular Stim simulator
add_executable(surface_code-1-local)
set_target_properties(surface_code-1-local PROPERTIES LINKER_LANGUAGE CXX)

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
qec_realtime_app_link_options(surface_code-1-local)

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

if(_APP_CUDAQ_REALTIME_DISPATCH_LIB
   AND TARGET cudaq-qec-realtime-cudevice-proprietary)
  add_test(
    NAME app_examples.surface_code-1-local-test-distance-3-inproc-rpc
    COMMAND
      bash "${CMAKE_CURRENT_SOURCE_DIR}/surface_code-1-test.sh"
        ${CMAKE_CURRENT_BINARY_DIR}/surface_code-1-local
        ${CMAKE_CURRENT_BINARY_DIR}/surface_code-1-local
        3 60 40 NULL 12 6
        ${CMAKE_BINARY_DIR}/lib
        nv-qldpc-decoder
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_tests_properties(
    app_examples.surface_code-1-local-test-distance-3-inproc-rpc
    PROPERTIES ENVIRONMENT
      "CUDAQ_QEC_REALTIME_MODE=inproc_rpc;EXTRA_CLI_ARGS=--use-relay-bp")
endif()

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

if(_app_device_call_runtime_link
   AND TARGET cudaq-qec-realtime-decoding-host-dispatch
   AND TARGET cudaq-qec-pymatching)
  add_executable(surface_code-1-host-dispatch)
  set_target_properties(surface_code-1-host-dispatch PROPERTIES
    LINKER_LANGUAGE CXX)

  add_dependencies(surface_code-1-host-dispatch surface_code-1-local)

  cudaqx_add_device_code(surface_code-1-host-dispatch
    SOURCES
      surface_code-1.cpp
    COMPILER_FLAGS
      --target stim
      -frealtime-lowering
      -DCUDAQ_QEC_ENABLE_HOST_DISPATCH_DEVICE_CALLS=1
      -DCUDAQ_QEC_ENABLE_PYMATCHING_DECODER_TYPE=1
    DEPENDS_ON
      # Must wait for the local executable to be built because it has the same
      # surface_code-1.cpp file, and the nvq++ compiler will use the same
      # filename prefix (surface_code-1) for intermediate files.
      surface_code-1-local
  )

  target_link_directories(surface_code-1-host-dispatch
    PRIVATE ${CUDAQ_INSTALL_DIR}/lib)

  if(UNIX AND NOT APPLE)
    target_link_options(surface_code-1-host-dispatch PRIVATE
      "LINKER:--no-as-needed")
  endif()

  target_link_libraries(surface_code-1-host-dispatch
    PRIVATE
      cudaq-qec
      cudaq-qec-pymatching
      cudaq-qec-realtime-decoding
      cudaq-qec-realtime-decoding-host-dispatch
      ${_app_device_call_runtime_link}
      cudaq::cudaq cudaq::cudaq-common cudaq-mlir-runtime
      nvqir nvqir-stim)
  target_link_options(surface_code-1-host-dispatch PRIVATE
    "LINKER:--allow-shlib-undefined"
    "LINKER:--export-dynamic"
  )

  set(_surface_code_host_dispatch_rpath
    "${CMAKE_BINARY_DIR}/lib"
    "${CMAKE_BINARY_DIR}/lib/decoder-plugins")
  if(_APP_CUDAQ_REALTIME_DISPATCH_LIB)
    get_filename_component(_app_cudaq_realtime_dispatch_lib_dir
      "${_APP_CUDAQ_REALTIME_DISPATCH_LIB}" DIRECTORY)
    list(APPEND _surface_code_host_dispatch_rpath
      "${_app_cudaq_realtime_dispatch_lib_dir}")
  endif()
  if(CUDAQ_INSTALL_DIR)
    list(APPEND _surface_code_host_dispatch_rpath "${CUDAQ_INSTALL_DIR}/lib")
  endif()
  set_target_properties(surface_code-1-host-dispatch PROPERTIES
    BUILD_RPATH "${_surface_code_host_dispatch_rpath}")

  add_dependencies(surface_code-1-host-dispatch
    cudaq-qec-pymatching
    cudaq-qec-realtime-decoding-host-dispatch)

  add_test(
    NAME app_examples.surface_code-1-host-dispatch-pymatching-smoke
    COMMAND
      ${CMAKE_CURRENT_BINARY_DIR}/surface_code-1-host-dispatch
        --pymatching-host-dispatch-smoke
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

else()
  message(STATUS "Skipping surface_code-1-host-dispatch PyMatching app test "
                 "(requires cudaq-device-call-runtime, "
                 "cudaq-qec-realtime-decoding-host-dispatch, and "
                 "cudaq-qec-pymatching).")
endif()

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
set_target_properties(surface_code-1-quantinuum-emulate PROPERTIES LINKER_LANGUAGE CXX)

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
qec_realtime_app_link_options(surface_code-1-quantinuum-emulate)

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
set_target_properties(surface_code-2-local PROPERTIES LINKER_LANGUAGE CXX)

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
qec_realtime_app_link_options(surface_code-2-local)

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
set_target_properties(surface_code-2-quantinuum-emulate PROPERTIES LINKER_LANGUAGE CXX)

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
qec_realtime_app_link_options(surface_code-2-quantinuum-emulate)

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
set_target_properties(surface_code-3-local PROPERTIES LINKER_LANGUAGE CXX)

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
qec_realtime_app_link_options(surface_code-3-local)

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
set_target_properties(surface_code-3-quantinuum-emulate PROPERTIES LINKER_LANGUAGE CXX)

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
qec_realtime_app_link_options(surface_code-3-quantinuum-emulate)


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
