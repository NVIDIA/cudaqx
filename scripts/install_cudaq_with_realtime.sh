#!/bin/bash

# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Rebuild and install CUDA-Q *with realtime + device_call support*, pinned to
# the SHA recorded in cudaqx's .cudaq_version, into CUDAQ_INSTALL_PREFIX.
#
# device_call (the host-dispatch service the QEC realtime libraries link
# against) is gated in CUDA-Q on CUDAQ_ENABLE_REALTIME. The actual build/install
# is delegated to the canonical CI recipe,
# .github/actions/get-cudaq-build/build_cudaq.sh (it installs cudaq-realtime and
# then configures the top-level build with -DCUDAQ_REALTIME_DIR, which flips
# CUDAQ_ENABLE_REALTIME on). This wrapper just adds the .cudaq_version SHA
# checkout up front and a post-build check that the device_call artifacts landed.
#
# Configuration (all overridable via environment):
#   CUDAQ_SRC             cuda-quantum checkout dir         [/workspaces/cudaq]
#   CUDAQ_INSTALL_PREFIX  install prefix                    [/usr/local/cudaq]
#   LLVM_INSTALL_PREFIX   LLVM used to build CUDA-Q         [/usr/local/llvm]
#   BUILD_TYPE            CMAKE_BUILD_TYPE                  [Release]
#   CC / CXX              host compilers                    [gcc / g++]
#   CUDAQ_REPO/CUDAQ_REF  override the .cudaq_version pin   [from .cudaq_version]
#   FORCE_CHECKOUT=1      allow checkout over a dirty tree  (DISCARDS changes)
#
# Hololink / holoscan-sensor-bridge (HSB) support (built into cudaq-realtime so
# the QEC realtime libraries can drive the FPGA bridge). These deps are
# installed/built once per container; re-runs detect the prior install and skip
# the expensive clone+build:
#   ENABLE_HOLOLINK=1    build DOCA/Holoscan/HSB + hololink tools   [1]
#   HSB_REPO/HSB_REF     holoscan-sensor-bridge source pin
#   HSB_ROOT/HSB_BUILD   HSB checkout / build dir                   [/tmp/...]
#   DOCA_VERSION         DOCA repo version                          [3.3.0]
#   DOCA_UBUNTU          DOCA repo ubuntu flavor                    [ubuntu24.04]
#   CUDA_NATIVE_ARCH     GPU arch HSB is compiled for               [80]
#   HOLOLINK_REALTIME_TESTS
#                        realtime unit tests toggle; forced ON as a temporary
#                        workaround for the pinned CUDA-Q SHA (see block below) [ON]
#
# Usage:
#   scripts/install_cudaq_with_realtime.sh
#   ENABLE_HOLOLINK=0 scripts/install_cudaq_with_realtime.sh   # skip HSB/hololink

set -euo pipefail

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)
version_file="$repo_root/.cudaq_version"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CUDAQ_SRC=${CUDAQ_SRC:-/workspaces/cudaq}
# Resolve to absolute path so later cd's don't confuse relative paths.
# The parent directory must already exist (we're about to clone into it or it's
# already a git checkout).
_cudaq_parent=$(cd "$(dirname "$CUDAQ_SRC")" 2>/dev/null && pwd) || \
  die "parent directory of CUDAQ_SRC does not exist: $(dirname "$CUDAQ_SRC")"
CUDAQ_SRC="$_cudaq_parent/$(basename "$CUDAQ_SRC")"
unset _cudaq_parent
CUDAQ_INSTALL_PREFIX=${CUDAQ_INSTALL_PREFIX:-/usr/local/cudaq}
LLVM_INSTALL_PREFIX=${LLVM_INSTALL_PREFIX:-/usr/local/llvm}
BUILD_TYPE=${BUILD_TYPE:-Release}
CC=${CC:-gcc}
CXX=${CXX:-g++}
FORCE_CHECKOUT=${FORCE_CHECKOUT:-0}

# Hololink / holoscan-sensor-bridge configuration.
ENABLE_HOLOLINK=${ENABLE_HOLOLINK:-1}
HSB_REPO=${HSB_REPO:-https://github.com/nvidia-holoscan/holoscan-sensor-bridge.git}
HSB_REF=${HSB_REF:-2.6.0-EA2}
HSB_ROOT=${HSB_ROOT:-/tmp/holoscan-sensor-bridge}
HSB_BUILD=${HSB_BUILD:-$HSB_ROOT/build}
DOCA_VERSION=${DOCA_VERSION:-3.3.0}
DOCA_UBUNTU=${DOCA_UBUNTU:-ubuntu24.04}
CUDA_NATIVE_ARCH=${CUDA_NATIVE_ARCH:-80}

# ---------------------------------------------------------------------------
# TEMPORARY WORKAROUND -- pinned CUDA-Q SHA predates the hololink cleanup.
#
# The CUDA-Q commit currently pinned in .cudaq_version still defines the
# `hololink_wrapper_generic` target under realtime/unittests/utils, so the
# hololink bridge library only links when the realtime unit tests are ALSO
# built. We therefore force CUDAQ_REALTIME_BUILD_TESTS=ON while hololink is
# enabled, otherwise the realtime build fails with:
#     /usr/bin/ld: cannot find -lhololink_wrapper_generic
#
# >>> TO REMOVE THIS WORKAROUND once .cudaq_version is bumped to a CUDA-Q commit
# >>> that includes "[realtime] Remove redundant hololink_wrapper_generic static
# >>> library" (cuda-quantum@360f37681beaccbba276c785ca2325f674000b44):
# >>>   1. Delete this block and the CUDAQ_REALTIME_BUILD_TESTS entry added to
# >>>      build_env below.
# >>> The hololink bridge then builds correctly with the realtime tests OFF.
# ---------------------------------------------------------------------------
HOLOLINK_REALTIME_TESTS=${HOLOLINK_REALTIME_TESTS:-ON}

# apt/dpkg/curl-to-system-dirs need root; use sudo when not already root.
if [ "$(id -u)" -eq 0 ]; then
  SUDO=""
elif command -v sudo >/dev/null 2>&1; then
  SUDO="sudo"
else
  SUDO=""
fi

command -v jq  >/dev/null 2>&1 || die "jq is required (parses .cudaq_version)"
command -v git >/dev/null 2>&1 || die "git is required"
command -v cmake >/dev/null 2>&1 || die "cmake is required"
[ -f "$version_file" ] || die "$version_file not found"

# SHA (and repo) come from .cudaq_version unless overridden.
CUDAQ_REPO=${CUDAQ_REPO:-$(jq -r '.cudaq.repository' "$version_file")}
CUDAQ_REF=${CUDAQ_REF:-$(jq -r '.cudaq.ref' "$version_file")}
[ -n "$CUDAQ_REPO" ] && [ "$CUDAQ_REPO" != "null" ] || die "could not read .cudaq.repository from $version_file"
[ -n "$CUDAQ_REF" ]  && [ "$CUDAQ_REF"  != "null" ] || die "could not read .cudaq.ref from $version_file"

log "CUDA-Q ${CUDAQ_REPO}@${CUDAQ_REF}"
log "  source : $CUDAQ_SRC"
log "  install: $CUDAQ_INSTALL_PREFIX"
log "  llvm   : $LLVM_INSTALL_PREFIX"
log "  build  : $BUILD_TYPE (CC=$CC CXX=$CXX)"

# ---------------------------------------------------------------------------
# 1. Get the source at the pinned SHA
# ---------------------------------------------------------------------------
if [ -d "$CUDAQ_SRC/.git" ]; then
  cd "$CUDAQ_SRC"
  if [ "$(git rev-parse HEAD)" = "$(git rev-parse "$CUDAQ_REF" 2>/dev/null || echo none)" ]; then
    log "Source already at ${CUDAQ_REF}; skipping checkout"
  else
    if [ -n "$(git status --porcelain)" ] && [ "$FORCE_CHECKOUT" != "1" ]; then
      die "$CUDAQ_SRC has local changes; commit/stash them or set FORCE_CHECKOUT=1 to discard."
    fi
    log "Fetching + checking out ${CUDAQ_REF}"
    git fetch origin "$CUDAQ_REF" || git fetch origin
    git checkout --force "$CUDAQ_REF"
  fi
else
  # No submodules: CUDA-Q uses the prebuilt LLVM_INSTALL_PREFIX / NANOBIND_INSTALL_PREFIX
  # and FetchContent for tpls, so initializing submodules would needlessly clone
  # the huge tpls/llvm tree. (Matches the CI checkout, which inits no submodules.)
  log "Cloning ${CUDAQ_REPO} into $CUDAQ_SRC (shallow)"
  # --no-checkout: skip checking out the default branch; we only want CUDAQ_REF.
  git clone --no-checkout "https://github.com/${CUDAQ_REPO}.git" "$CUDAQ_SRC"
  cd "$CUDAQ_SRC"
  # Fetch only the pinned SHA with a shallow window so we don't pull full history.
  git fetch --depth=10 origin "$CUDAQ_REF"
  git checkout --force FETCH_HEAD
fi

[ -x "$LLVM_INSTALL_PREFIX/bin/llvm-config" ] || \
  log "WARNING: no llvm-config at $LLVM_INSTALL_PREFIX/bin; set LLVM_INSTALL_PREFIX if the build can't find LLVM."

# ---------------------------------------------------------------------------
# 1.5 Install DOCA / Holoscan SDK / holoscan-sensor-bridge (once per container).
#
#     cudaq-realtime's hololink tools link against DOCA GPUNetIO, the Holoscan
#     SDK, and a locally-built holoscan-sensor-bridge (HSB). These are heavy to
#     install/build, so each step is guarded and skipped when already present.
#     Mirrors .github/actions/build-lib/build_qec.sh but made idempotent.
#
#     NOTE: the cudaqx CI container ships Mellanox OFED pre-installed, so we must
#     NOT run doca's install_dev_prerequisites.sh / doca-all (it conflicts with
#     the container OFED). We install only the GPUNetIO dev headers we need.
# ---------------------------------------------------------------------------

# True when HSB has already been built (its libs exist under $HSB_BUILD).
hsb_built() {
  [ -d "$HSB_BUILD" ] && \
    find "$HSB_BUILD" -name 'libhololink_core*' -print -quit 2>/dev/null | grep -q .
}

install_hsb_deps() {
  command -v nvcc >/dev/null 2>&1 || \
    die "nvcc not found; DOCA/Holoscan/HSB build requires the CUDA toolkit on PATH"

  local cuda_major cuda_full cuda_dash doca_arch doca_repo
  cuda_major=$(nvcc --version | sed -n 's/^.*release \([0-9]\+\).*$/\1/p')
  cuda_full=$(nvcc --version | sed -n 's/^.*release \([0-9]\+\.[0-9]\+\).*$/\1/p')
  cuda_dash=$(echo "$cuda_full" | sed 's/\./-/')

  log "Installing build tools (ninja, curl, pkg-config)"
  $SUDO apt-get update
  $SUDO apt-get install -y --no-install-recommends ninja-build curl pkg-config

  # --- DOCA GPUNetIO dev headers ---
  if [ -d /opt/mellanox/doca/include ]; then
    log "DOCA already present at /opt/mellanox/doca; skipping"
  else
    log "Installing DOCA GPUNetIO dev package (DOCA $DOCA_VERSION / $DOCA_UBUNTU)"
    doca_arch=$(uname -m)
    case "$doca_arch" in aarch64|arm64) doca_arch="arm64-sbsa" ;; esac
    doca_repo="https://linux.mellanox.com/public/repo/doca/${DOCA_VERSION}/${DOCA_UBUNTU}/$doca_arch"
    $SUDO curl -fsSL "$doca_repo/GPG-KEY-Mellanox.pub" \
      -o /usr/share/keyrings/GPG-KEY-Mellanox.pub
    echo "deb [signed-by=/usr/share/keyrings/GPG-KEY-Mellanox.pub] $doca_repo /" \
      | $SUDO tee /etc/apt/sources.list.d/doca.list >/dev/null
    $SUDO apt-get update
    $SUDO apt-get install -y --no-install-recommends libdoca-sdk-gpunetio-dev
  fi

  # hololink_core links CUDA::nvrtc -- must match the exact toolkit version.
  $SUDO apt-get install -y "cuda-nvrtc-dev-$cuda_dash" 2>/dev/null || true

  # --- Holoscan SDK ---
  if [ -d /opt/nvidia/holoscan ]; then
    log "Holoscan SDK already present at /opt/nvidia/holoscan; skipping"
  else
    log "Installing Holoscan SDK (cuda $cuda_major)"
    $SUDO apt-get install -y --no-install-recommends "holoscan-cuda-$cuda_major" || {
      # Force-install if normal install fails due to missing (OFED) deps.
      local hsdk_tmp
      hsdk_tmp=$(mktemp -d)
      ( cd "$hsdk_tmp" \
          && apt-get download holoscan "holoscan-cuda-$cuda_major" \
          && $SUDO dpkg --force-depends -i holoscan*.deb )
      rm -rf "$hsdk_tmp"
    }
  fi

  [ -d /opt/mellanox/doca/include ] || die "DOCA SDK installation failed"
  [ -d /opt/nvidia/holoscan ]       || die "Holoscan SDK installation failed"

  # --- holoscan-sensor-bridge (hololink) ---
  if hsb_built; then
    log "holoscan-sensor-bridge already built at $HSB_BUILD; skipping clone+build"
  else
    log "Building holoscan-sensor-bridge $HSB_REF at $HSB_ROOT"
    rm -rf "$HSB_ROOT"
    git clone --depth 1 --branch "$HSB_REF" "$HSB_REPO" "$HSB_ROOT"
    (
      cd "$HSB_ROOT"
      # Strip operators we don't need to avoid configure failures from missing deps.
      sed -i '/add_subdirectory(audio_packetizer)/d; /add_subdirectory(compute_crc)/d;
              /add_subdirectory(csi_to_bayer)/d; /add_subdirectory(image_processor)/d;
              /add_subdirectory(iq_dec)/d; /add_subdirectory(iq_enc)/d;
              /add_subdirectory(linux_coe_receiver)/d; /add_subdirectory(linux_receiver)/d;
              /add_subdirectory(packed_format_converter)/d; /add_subdirectory(sub_frame_combiner)/d;
              /add_subdirectory(udp_transmitter)/d; /add_subdirectory(emulator)/d;
              /add_subdirectory(sig_gen)/d; /add_subdirectory(sig_viewer)/d' \
        src/hololink/operators/CMakeLists.txt
      CUDA_NATIVE_ARCH="$CUDA_NATIVE_ARCH" cmake -G Ninja -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DHOLOLINK_BUILD_ONLY_NATIVE=OFF \
        -DHOLOLINK_BUILD_PYTHON=OFF \
        -DHOLOLINK_BUILD_TESTS=OFF \
        -DHOLOLINK_BUILD_TOOLS=OFF \
        -DHOLOLINK_BUILD_EXAMPLES=OFF \
        -DHOLOLINK_BUILD_EMULATOR=OFF
      cmake --build build --target gpu_roce_transceiver hololink_core
    )
    hsb_built || die "holoscan-sensor-bridge build finished but libhololink_core is missing under $HSB_BUILD"
    log "holoscan-sensor-bridge built at $HSB_BUILD"
  fi
}

if [ "$ENABLE_HOLOLINK" = "1" ]; then
  install_hsb_deps
else
  log "ENABLE_HOLOLINK=0 -- skipping DOCA/Holoscan/HSB; building realtime without hololink tools"
fi

# ---------------------------------------------------------------------------
# 2. Clean the build dirs, then build + install realtime AND the full CUDA-Q
#    (with device_call) by delegating to the canonical CI recipe instead of
#    duplicating it. That script uses a positional contract (BUILD_TYPE,
#    LAUNCHER, CC, CXX) and reads CUDAQ_SRC / CUDAQ_INSTALL_PREFIX /
#    LLVM_INSTALL_PREFIX from the env.
# ---------------------------------------------------------------------------
build_script="$repo_root/.github/actions/get-cudaq-build/build_cudaq.sh"
[ -f "$build_script" ] || die "delegate build script not found: $build_script"

# Force a fresh build. ($CUDAQ_SRC/build is also wiped by the inner build script,
# but $CUDAQ_SRC/realtime/build is incremental, so clean both explicitly.)
log "Cleaning build directories"
rm -rf "$CUDAQ_SRC/realtime/build" "$CUDAQ_SRC/build"

log "Building realtime + CUDA-Q via $build_script"
build_env=(
  "CUDAQ_SRC=$CUDAQ_SRC"
  "CUDAQ_INSTALL_PREFIX=$CUDAQ_INSTALL_PREFIX"
  "LLVM_INSTALL_PREFIX=$LLVM_INSTALL_PREFIX"
)
if [ "$ENABLE_HOLOLINK" = "1" ]; then
  # Turn on the hololink bridge tools in the realtime build and point them at
  # the HSB checkout we installed/built above.
  build_env+=(
    "CUDAQ_REALTIME_ENABLE_HOLOLINK_TOOLS=ON"
    "HOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR=$HSB_ROOT"
    "HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR=$HSB_BUILD"
    # TEMPORARY: required by the pinned CUDA-Q SHA -- see the HOLOLINK_REALTIME_TESTS
    # workaround block near the top of this script. Remove once .cudaq_version is
    # bumped past the hololink_wrapper_generic cleanup.
    "CUDAQ_REALTIME_BUILD_TESTS=$HOLOLINK_REALTIME_TESTS"
  )
fi
env "${build_env[@]}" bash "$build_script" "$BUILD_TYPE" "" "$CC" "$CXX"

# ---------------------------------------------------------------------------
# 3. Verify device_call support actually installed (the markers from manual
#    inspection: runtime library + public header).
# ---------------------------------------------------------------------------
log "Verifying device_call support in $CUDAQ_INSTALL_PREFIX"
dc_lib="$CUDAQ_INSTALL_PREFIX/lib/libcudaq-device-call-runtime.so"
dc_hdr="$CUDAQ_INSTALL_PREFIX/include/cudaq/realtime/device_call_service.h"
ok=1
if [ -f "$dc_lib" ]; then echo "  [ok]      $dc_lib"; else echo "  [MISSING] $dc_lib"; ok=0; fi
if [ -f "$dc_hdr" ]; then echo "  [ok]      $dc_hdr"; else echo "  [MISSING] $dc_hdr"; ok=0; fi
[ "$ok" = "1" ] || die "build finished but device_call artifacts are missing -- check that CUDA was found and CUDAQ_ENABLE_REALTIME was TRUE."

log "Done. CUDA-Q with device_call support installed at $CUDAQ_INSTALL_PREFIX"

# ---------------------------------------------------------------------------
# 4. Print a copy-pasteable cmake command for configuring cudaqx.
#    (When hololink is enabled we include the HOLOSCAN_SENSOR_BRIDGE flags so
#    the GPU-RoCE decoding server is built instead of silently disabled.)
# ---------------------------------------------------------------------------
printf '\n'
log "Next step -- configure cudaqx (from a build dir under the repo root):"
if [ "$ENABLE_HOLOLINK" = "1" ]; then
  cat <<EOF

  mkdir -p "$repo_root/build" && cd "$repo_root/build"
  cmake -G Ninja \\
    -DCUDAQ_REALTIME_ROOT="$CUDAQ_INSTALL_PREFIX" \\
    -DCUDAQ_DIR="$CUDAQ_INSTALL_PREFIX/lib/cmake/cudaq" \\
    -DCUDAQX_QEC_ENABLE_HOLOLINK_TOOLS=ON \\
    -DHOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR="$HSB_ROOT" \\
    -DHOLOSCAN_SENSOR_BRIDGE_BUILD_DIR="$HSB_BUILD" \\
    ..
  cmake --build .

EOF
else
  cat <<EOF

  mkdir -p "$repo_root/build" && cd "$repo_root/build"
  cmake -G Ninja \\
    -DCUDAQ_REALTIME_ROOT="$CUDAQ_INSTALL_PREFIX" \\
    -DCUDAQ_DIR="$CUDAQ_INSTALL_PREFIX/lib/cmake/cudaq" \\
    ..
  cmake --build .

EOF
fi
