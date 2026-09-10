# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under    #
# the terms of the Apache License 2.0 which accompanies this distribution.    #
# ============================================================================ #
#
# Release image for the QEC decoding server. Multi-stage: the "toolchain"
# and "builder" stages (compilers, dev headers, source tree) are discarded --
# the final "runtime" stage ships only compiled binaries, shared libraries,
# and a runtime subset of apt dependencies. No source code is included.
#
# Two artifacts (the NV QLDPC decoder plugin and the cudevice archive
# backing device_graph dispatch) aren't part of this source tree and are
# supplied as prebuilt files via an optional build context -- the flat
# layout `nv-qldpc-decoder-<arch>_ubuntu24.04_cuda<version>_release.tar.gz`
# extracts to. In CI, build_decoding_server_docker.yaml populates this
# from a draft-release asset:
#
#   artifacts/
#     libcudaq-qec-nv-qldpc-decoder.so
#     libcudaq-qec-realtime-cudevice-proprietary.a
#
# Both are optional independently: without the cudevice archive, device_graph
# dispatch is compiled but not linked (host-dispatch decoders still work);
# without the nv-qldpc plugin, that decoder is simply absent from
# lib/decoder-plugins.
#
# CUDA-Q itself is NOT built here: it is taken verbatim from the official
# published CUDA-Q container (cudaq_image below) -- the /opt/nvidia/cudaq
# install, including the proprietary components (custatevec/tensornet
# simulator backends) that the open-source cuda-quantum-dev build lacks.
# The build toolchain (gcc-12, CUDA toolkit, and the LLVM/MLIR dev install
# CUDA-Q is compiled against) comes from cuda-quantum-devcontainer, the
# same lineage the official build itself uses. Only two things are
# compiled here:
#   cudaq-realtime -- rebuilt with HSB tools (the official install ships
#     it without them), from the cuda-quantum commit recorded as
#     source-sha in the consumed image's own
#     /opt/nvidia/cudaq/build_info.txt, i.e. the exact commit that
#     image's binaries were built from.
#   cudaqx/qec -- this repo's own code, including the TensorRT decoder
#     plugin, decoding_server, and hsb_fpga_syndrome_playback.
#
# The CUDA-Q commit used above is therefore chosen by cudaq_image, not by
# this repo's .cudaq_version: nothing in this build reads that file. It
# does still record the CUDA-Q commit this repo's qec sources are
# developed against, so a cudaq_image far from that pin is where build or
# ABI breakage would be expected to surface.
#
# Build (from the repo root):
#   DOCKER_BUILDKIT=1 docker build -f docker/decoding-server/Dockerfile \
#     --build-arg CUDA_VERSION=13.0 \
#     --build-arg cuda_native_arch=90 \
#     --build-context artifacts=./path/to/artifacts \
#     -t cudaqx-decoding-server:prod .
#
# If overriding CUDA_VERSION (e.g. to 12.6), also pass matching
# --build-arg cudaq_image=nvcr.io/nvidia/nightly/cuda-quantum:cu12-latest
# and --build-arg cuda_runtime_image_tag=12.6.3-runtime-ubuntu24.04 (these
# aren't derived from each other -- see the args' definitions below).
#
# The --build-context flag may be omitted; the image still builds (minus
# the pieces it would have supplied).

# syntax=docker/dockerfile:1.7

ARG CUDA_VERSION=13.0
ARG TARGETARCH
# The official published CUDA-Q container whose /opt/nvidia/cudaq install
# is consumed at build time and shipped verbatim in the final image.
#   nightly (tracks cuda-quantum main): nvcr.io/nvidia/nightly/cuda-quantum:cu<major>-latest
#   releases (retained, versioned):     nvcr.io/nvidia/quantum/cuda-quantum:cu<major>-<version>
# Release builds should pin a versioned release image. Note: CUDA-Q
# releases up to and including 0.15.1 predate realtime support being
# built into these containers and cannot be used.
ARG cudaq_image=nvcr.io/nvidia/nightly/cuda-quantum:cu13-latest
# Toolchain for compiling cudaqx/qec and cudaq-realtime: gcc-12, CUDA
# toolkit, and the LLVM/MLIR dev install CUDA-Q builds against -- the same
# lineage the official CUDA-Q build uses, at the same /usr/local/llvm
# path, so nvq++ and MLIR linking resolve consistently.
#
# Only branch-suffixed floating tags exist for this image. That's fine
# for current builds (nothing from it ships, and libs/qec requires the
# LLVM version EXACT, so toolchain drift fails at configure time rather
# than yielding a mismatched binary), but it makes rebuilding an old
# release image risky: by then this tag may serve a newer toolchain than
# that release builds against.
ARG devcontainer_image=ghcr.io/nvidia/cuda-quantum-devcontainer
ARG devcontainer_tag_suffix=gcc12-main
# nvidia/cuda on Docker Hub tags by full patch version (e.g.
# 13.0.3-runtime-ubuntu24.04), not the bare CUDA_VERSION minor used
# elsewhere in this file. Kept as a separate arg since a FROM line can't
# derive one from the other; update both together when bumping either.
ARG cuda_runtime_image_tag=13.0.3-runtime-ubuntu24.04

# Empty placeholder for the optional artifacts build context. Naming a stage
# the same as a `--build-context NAME=path` flag lets that flag override the
# stage entirely; when the flag is omitted, this empty scratch image is used
# instead, so the mount below sees zero files rather than erroring.
FROM scratch AS artifacts

# The official CUDA-Q container, pulled so its /opt/nvidia/cudaq
# install can be COPY'd into the builder and runtime stages.
FROM ${cudaq_image} AS cudaq-official

# ---------------------------------------------------------------------------
# Stage: toolchain -- RDMA/DOCA/Holoscan/TensorRT dev headers needed to
# compile the decoding server, plus patchelf (needed at CMake configure time
# when an nv-qldpc .so artifact is supplied).
# ---------------------------------------------------------------------------
FROM ${devcontainer_image}:${TARGETARCH}-cu${CUDA_VERSION}-${devcontainer_tag_suffix} AS toolchain

# Build tooling and RDMA userspace, plus what the qec build needs beyond
# the devcontainer: gfortran + libblas-dev (xtensor-blas), cuda-nvtx
# (profiling hooks), git-lfs. NVTX's package name is keyed to the toolkit
# version, derived from nvcc like the DOCA step below. cuStabilizer (also
# required by the qec build) needs no install: the devcontainer's pinned
# cuquantum wheel already provides it where FindcuStabilizer.cmake looks;
# if a future devcontainer drops it, the qec configure fails with an
# explicit cuStabilizer error.
RUN set -e; \
    CUDA_FULL_VERSION=$(nvcc --version | sed -n 's/^.*release \([0-9]\+\.[0-9]\+\).*$/\1/p'); \
    CUDA_VER_DASH=$(echo "$CUDA_FULL_VERSION" | sed 's/\./-/'); \
    apt-get update && apt-get install -y --no-install-recommends \
        ninja-build curl pkg-config jq patchelf \
        rdma-core ibverbs-providers \
        gfortran libblas-dev cuda-nvtx-$CUDA_VER_DASH git-lfs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# DOCA 3.3.0: GPUNetIO dev package only (doca-all conflicts with the base
# image's preinstalled OFED), plus the matching cuda-nvrtc-dev
# (hololink_core links CUDA::nvrtc).
RUN set -e; \
    DOCA_ARCH=$(uname -m); \
    case "$DOCA_ARCH" in aarch64|arm64) DOCA_ARCH="arm64-sbsa" ;; esac; \
    DOCA_REPO="https://linux.mellanox.com/public/repo/doca/3.3.0/ubuntu24.04/$DOCA_ARCH"; \
    curl -fsSL "$DOCA_REPO/GPG-KEY-Mellanox.pub" -o /usr/share/keyrings/GPG-KEY-Mellanox.pub; \
    echo "deb [signed-by=/usr/share/keyrings/GPG-KEY-Mellanox.pub] $DOCA_REPO /" \
        > /etc/apt/sources.list.d/doca.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends libdoca-sdk-gpunetio-dev; \
    CUDA_FULL_VERSION=$(nvcc --version | sed -n 's/^.*release \([0-9]\+\.[0-9]\+\).*$/\1/p'); \
    CUDA_VER_DASH=$(echo "$CUDA_FULL_VERSION" | sed 's/\./-/'); \
    apt-get install -y cuda-nvrtc-dev-$CUDA_VER_DASH 2>/dev/null || true; \
    apt-get clean && rm -rf /var/lib/apt/lists/*; \
    test -d /opt/mellanox/doca/include

# Holoscan SDK (force-install fallback: the package's dependency list can
# miss on the devcontainer base).
RUN set -e; \
    CUDA_MAJOR_VERSION=$(nvcc --version | sed -n 's/^.*release \([0-9]\+\).*$/\1/p'); \
    apt-get update; \
    apt-get install -y --no-install-recommends holoscan-cuda-$CUDA_MAJOR_VERSION || { \
        _hsdk_tmp=$(mktemp -d); \
        (cd "$_hsdk_tmp" && apt-get download holoscan holoscan-cuda-$CUDA_MAJOR_VERSION \
            && dpkg --force-depends -i holoscan*.deb); \
        rm -rf "$_hsdk_tmp"; \
    }; \
    apt-get clean && rm -rf /var/lib/apt/lists/*; \
    test -d /opt/nvidia/holoscan

# TensorRT dev, pinned to the toolkit's CUDA flavor (needed at build time so
# CUDAQ_QEC_BUILD_TRT_DECODER=AUTO detects it). No arm64 packages exist for
# CUDA 12.x, only 13.x, so skip the install on that combination rather than
# fail the build -- trt_decoder just won't be built there (AUTO mode).
RUN set -e; \
    ARCH=$(uname -m); \
    CUDA_FULL_VERSION=$(nvcc --version | sed -n 's/^.*release \([0-9]\+\.[0-9]\+\).*$/\1/p'); \
    if [ "$ARCH" != "x86_64" ] && [ "${CUDA_FULL_VERSION%%.*}" = "12" ]; then \
        echo "== tensorrt-dev: skipping (no arm64 packages for CUDA 12.x)"; \
    else \
        apt-get update; \
        apt-cache search tensorrt \
            | awk -v v="$CUDA_FULL_VERSION" '{print "Package: "$1"\nPin: version *+cuda"v"\nPin-Priority: 1001\n"}' \
            > /etc/apt/preferences.d/tensorrt-cuda$CUDA_FULL_VERSION.pref; \
        apt-get install -y tensorrt-dev; \
        apt-get clean && rm -rf /var/lib/apt/lists/*; \
    fi

# holoscan-sensor-bridge (HSB): the FPGA<->GPU RDMA transport library.
# Both the cudaq-realtime gpu_roce bridge and cudaqx/qec's HSB tools
# (hsb_fpga_syndrome_playback) compile against this source + build tree.
# Only the two needed targets are built; the unrelated operators are
# stripped from the build to avoid pulling in their dependencies.
#
# cuda_native_arch: CMake CUDA architecture list for all GPU code built in
# this file (HSB, cudaq-realtime, cudaqx/qec) -- e.g. 90 (GH200),
# 100 (GB200), 121 (DGX Spark GB10).
ARG cuda_native_arch=90
RUN set -e; \
    git clone --depth 1 --branch 2.6.0-EA2 \
        https://github.com/nvidia-holoscan/holoscan-sensor-bridge.git \
        /opt/holoscan-sensor-bridge; \
    cd /opt/holoscan-sensor-bridge; \
    sed -i '/add_subdirectory(audio_packetizer)/d; /add_subdirectory(compute_crc)/d; \
            /add_subdirectory(csi_to_bayer)/d; /add_subdirectory(image_processor)/d; \
            /add_subdirectory(iq_dec)/d; /add_subdirectory(iq_enc)/d; \
            /add_subdirectory(linux_coe_receiver)/d; /add_subdirectory(linux_receiver)/d; \
            /add_subdirectory(packed_format_converter)/d; /add_subdirectory(sub_frame_combiner)/d; \
            /add_subdirectory(udp_transmitter)/d; /add_subdirectory(emulator)/d; \
            /add_subdirectory(sig_gen)/d; /add_subdirectory(sig_viewer)/d' \
        src/hololink/operators/CMakeLists.txt; \
    export CUDA_NATIVE_ARCH="${cuda_native_arch}"; \
    cmake -G Ninja -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DHOLOLINK_BUILD_ONLY_NATIVE=OFF \
        -DHOLOLINK_BUILD_PYTHON=OFF \
        -DHOLOLINK_BUILD_TESTS=OFF \
        -DHOLOLINK_BUILD_TOOLS=OFF \
        -DHOLOLINK_BUILD_EXAMPLES=OFF \
        -DHOLOLINK_BUILD_EMULATOR=OFF; \
    cmake --build build --target gpu_roce_transceiver hololink_core

# ---------------------------------------------------------------------------
# Stage: builder -- compile cudaq-realtime (HSB tools ON, producing
# libcudaq-realtime-bridge-gpu-roce.so) and cudaqx/qec.
#
# Install prefixes are fixed absolute paths reused verbatim by the runtime
# stage: decoding_server's ELF carries absolute RPATH entries for the
# CUDA-Q and cudaq-realtime lib dirs (see
# libs/qec/tools/decoding-server/CMakeLists.txt INSTALL_RPATH), so those
# directories must land at the identical path in both stages.
# ---------------------------------------------------------------------------
FROM toolchain AS builder
# Re-declared: ARG scope is per-stage; one --build-arg on the command line
# still overrides both declarations.
ARG cuda_native_arch=90
ENV CUDAQ_INSTALL_PREFIX=/opt/nvidia/cudaq \
    CUDAQ_REALTIME_ROOT=/opt/nvidia/cudaq-realtime \
    CUDAQX_INSTALL_PREFIX=/opt/nvidia/cudaqx

# The official CUDA-Q install, at the same absolute path it occupies in
# the published container (load-bearing: RPATHs baked into what we build
# reference it, and the runtime stage ships it at this path).
COPY --from=cudaq-official /opt/nvidia/cudaq /opt/nvidia/cudaq

WORKDIR /workspaces/cudaqx
COPY . .

# cudaq-realtime, rebuilt with HSB tools ON (the official install's
# cudaq-realtime is built without them). Built from the exact commit the
# official /opt/nvidia/cudaq binaries were built at (recorded in the
# install's build_info.txt) with a devcontainer toolchain from the same
# lineage the official build uses -- though not necessarily the identical
# devcontainer build (see the devcontainer_image note above). Same source
# commit and same compiler family keep this install binary-compatible with
# the consumed CUDA-Q. CUDA-Q itself is left untouched; this second,
# HSB-enabled realtime install lands alongside it.
#
# NOTE: the checkout ref comes from build_info.txt, deliberately not from
# this repo's .cudaq_version (which governs cudaqx-dev and CI, and is not
# read by this build). It has to track the consumed image: realtime and
# CUDA-Q share a co-developed dispatch/RPC protocol, so pairing realtime
# from one commit with CUDA-Q binaries from another is what breaks.
#
# TODO: once release builds pin a realtime-enabled versioned cudaq_image
# (see the cudaq_image note above), consider replacing this build with
# the official prebuilt realtime installer attached to the same CUDA-Q
# release (install_cuda_quantum_realtime_*): it is HSB-enabled and gcc-12/
# libstdc++-built. Needs verification first that device-linking its
# dispatch archive (built for CUDA archs "80-real;90" only) into our
# cuda_native_arch-targeted link works on sm_100/sm_121 targets.
RUN --mount=type=cache,target=/root/.ccache,sharing=locked \
    set -e; \
    . scripts/cudaq_realtime_cmake_flags.sh; \
    export CUDA_NATIVE_ARCH="${cuda_native_arch}"; \
    CUDAQ_SHA=$(sed -n 's/^source-sha: //p' "$CUDAQ_INSTALL_PREFIX/build_info.txt"); \
    test -n "$CUDAQ_SHA"; \
    echo "== cudaq-realtime source: NVIDIA/cuda-quantum@$CUDAQ_SHA (from build_info.txt)"; \
    git clone --filter=blob:none --no-checkout https://github.com/NVIDIA/cuda-quantum.git /tmp/cudaq-src; \
    git -C /tmp/cudaq-src sparse-checkout init --cone; \
    git -C /tmp/cudaq-src sparse-checkout set realtime cmake; \
    git -C /tmp/cudaq-src checkout "$CUDAQ_SHA"; \
    cmake -G Ninja -S /tmp/cudaq-src/realtime -B /tmp/realtime-build \
        -DCMAKE_INSTALL_PREFIX="$CUDAQ_REALTIME_ROOT" \
        -DCMAKE_CUDA_FLAGS="$(cudaq_realtime_cmake_cuda_flags)" \
        -DCMAKE_CUDA_ARCHITECTURES="${cuda_native_arch}" \
        -DCUDAQ_REALTIME_ENABLE_HSB_TOOLS=ON \
        -DHOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR=/opt/holoscan-sensor-bridge \
        -DHOLOSCAN_SENSOR_BRIDGE_BUILD_DIR=/opt/holoscan-sensor-bridge/build; \
    cmake --build /tmp/realtime-build; \
    cmake --install /tmp/realtime-build; \
    rm -rf /tmp/realtime-build /tmp/cudaq-src

# cudaqx/qec itself: decoding_server, hsb_fpga_syndrome_playback, the QEC
# and realtime libraries, and the decoder plugins (pymatching,
# multi_error_lut, trt_decoder, plus nv-qldpc if supplied via the
# artifacts context). Python bindings are off -- this is a binary runtime
# image, not a wheel build. CUDAQX_INCLUDE_TESTS stays ON despite that
# name: hsb_fpga_syndrome_playback is an actual deliverable that lives
# under unittests/utils/ and is only built/installed when tests are
# enabled. The gtest unit test binaries get compiled as a side effect but
# are never installed into CUDAQX_INSTALL_PREFIX, so they don't reach the
# runtime stage.
#
# The artifacts context is bind-mounted read-only, but
# add_target_libs_to_wheel() (libs/qec/CMakeLists.txt) patchelfs the
# nv-qldpc .so's rpath in place at configure time, so it needs a writable
# copy first. The cudevice archive is only ever read (imported as a static
# lib), so the read-only mount is fine for it as-is.
RUN --mount=type=bind,from=artifacts,target=/mnt/artifacts \
    --mount=type=cache,target=/root/.ccache,sharing=locked \
    set -e; \
    _cudevice_archive=""; \
    if [ -f /mnt/artifacts/libcudaq-qec-realtime-cudevice-proprietary.a ]; then \
        _cudevice_archive=/mnt/artifacts/libcudaq-qec-realtime-cudevice-proprietary.a; \
        echo "== cudevice archive: $_cudevice_archive"; \
    else \
        echo "== cudevice archive: absent (device_graph dispatch will not be linked)"; \
    fi; \
    if [ -f /mnt/artifacts/libcudaq-qec-nv-qldpc-decoder.so ]; then \
        mkdir -p /tmp/nv-qldpc-plugin; \
        cp -f /mnt/artifacts/libcudaq-qec-nv-qldpc-decoder.so /tmp/nv-qldpc-plugin/; \
        export QEC_EXTERNAL_DECODERS=/tmp/nv-qldpc-plugin/libcudaq-qec-nv-qldpc-decoder.so; \
        echo "== nv-qldpc plugin: $QEC_EXTERNAL_DECODERS"; \
    else \
        echo "== nv-qldpc plugin: absent (nv-qldpc-decoder will not be available)"; \
    fi; \
    cmake -S libs/qec -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=gcc-12 \
        -DCMAKE_CXX_COMPILER=g++-12 \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CUDA_ARCHITECTURES="${cuda_native_arch}" \
        -DCUDAQ_DIR="$CUDAQ_INSTALL_PREFIX/lib/cmake/cudaq/" \
        -DCUDAQX_INCLUDE_TESTS=ON \
        -DCUDAQX_BINDINGS_PYTHON=OFF \
        -DCMAKE_INSTALL_PREFIX="$CUDAQX_INSTALL_PREFIX" \
        -DCUDAQ_REALTIME_ROOT="$CUDAQ_REALTIME_ROOT" \
        -DCUDAQX_QEC_ENABLE_HSB_TOOLS=ON \
        -DHOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR=/opt/holoscan-sensor-bridge \
        -DHOLOSCAN_SENSOR_BRIDGE_BUILD_DIR=/opt/holoscan-sensor-bridge/build \
        ${_cudevice_archive:+-DCUDAQ_QEC_REALTIME_CUDEVICE_PROPRIETARY_ARCHIVE="$_cudevice_archive"}; \
    cmake --build build -j"$(nproc)"; \
    cmake --install build; \
    test -x "$CUDAQX_INSTALL_PREFIX/bin/decoding_server"; \
    test -x "$CUDAQX_INSTALL_PREFIX/bin/hsb_fpga_syndrome_playback"

# ---------------------------------------------------------------------------
# Stage: runtime -- minimal production image. No compilers, no -dev
# packages, no source tree.
# ---------------------------------------------------------------------------
FROM nvidia/cuda:${cuda_runtime_image_tag} AS runtime
ARG TARGETARCH
ARG CUDA_VERSION

# Runtime package set: RDMA userspace (rdma-core/ibverbs + their libnl
# deps), the DOCA GPUNetIO and Holoscan runtimes behind the gpu_roce
# transport, NVRTC (a JIT dependency of hololink_core that the CUDA
# "runtime" base image doesn't include), and the TensorRT runtime libs for
# the trt_decoder plugin -- version-pinned to the toolkit's CUDA flavor,
# and skipped on arm64+CUDA 12.x where no TensorRT packages exist (the
# toolchain stage skips building trt_decoder there for the same reason).
# curl/gnupg are only needed to register the DOCA apt repo and are removed
# again afterwards.
RUN set -e; \
    apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg \
    && rm -rf /var/lib/apt/lists/*; \
    DOCA_ARCH=$(uname -m); case "$DOCA_ARCH" in aarch64|arm64) DOCA_ARCH="arm64-sbsa" ;; esac; \
    DOCA_REPO="https://linux.mellanox.com/public/repo/doca/3.3.0/ubuntu24.04/$DOCA_ARCH"; \
    curl -fsSL "$DOCA_REPO/GPG-KEY-Mellanox.pub" -o /usr/share/keyrings/GPG-KEY-Mellanox.pub; \
    echo "deb [signed-by=/usr/share/keyrings/GPG-KEY-Mellanox.pub] $DOCA_REPO /" \
        > /etc/apt/sources.list.d/doca.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends doca-sdk-gpunetio; \
    CUDA_MAJOR="${CUDA_VERSION%%.*}"; \
    CUDA_VER_DASH="$(echo "$CUDA_VERSION" | tr '.' '-')"; \
    apt-get install -y --no-install-recommends \
        rdma-core ibverbs-providers \
        holoscan-cuda-${CUDA_MAJOR} \
        cuda-nvrtc-${CUDA_VER_DASH} \
        libnl-3-200 libnl-route-3-200; \
    if [ "$TARGETARCH" != "arm64" ] || [ "$CUDA_MAJOR" != "12" ]; then \
        apt-cache search tensorrt \
            | awk -v v="$CUDA_VERSION" '{print "Package: "$1"\nPin: version *+cuda"v"\nPin-Priority: 1001\n"}' \
            > /etc/apt/preferences.d/tensorrt-cuda$CUDA_VERSION.pref; \
        apt-get install -y --no-install-recommends tensorrt-libs; \
    else \
        echo "== tensorrt-libs: skipping (no arm64 packages for CUDA 12.x)"; \
    fi; \
    apt-get purge -y curl gnupg; \
    apt-get autoremove -y; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/* /etc/apt/sources.list.d/doca.list /etc/apt/preferences.d/tensorrt-cuda$CUDA_VERSION.pref

# CUDA-Q (the official container's install, shipped verbatim) +
# cudaq-realtime + cudaqx/qec runtime trees, at the SAME absolute paths
# used in the builder stage (load-bearing for decoding_server's baked-in
# RPATH). The nv-qldpc plugin (if supplied) is already inside
# CUDAQX_INSTALL_PREFIX, rpath-patched, via `cmake --install`; the cudevice
# archive (if supplied) is already statically linked into the
# decoding_server binary itself -- neither artifact needs a separate COPY
# here. Note the official install's GPU simulator backends
# (custatevec/tensornet) ride along but their cuquantum wheel dependencies
# are not shipped; the decoding server never loads them.
COPY --from=cudaq-official /opt/nvidia/cudaq /opt/nvidia/cudaq
COPY --from=builder /opt/nvidia/cudaq-realtime /opt/nvidia/cudaq-realtime
COPY --from=builder /opt/nvidia/cudaqx /opt/nvidia/cudaqx

# libcudaq-qec.so is unconditionally linked against cuStabilizer (provided
# by the devcontainer's cuquantum wheel) with an absolute RPATH entry into
# that wheel's site-packages location, so this single file has to land at
# the same path here too. Only the one .so, not the whole cuquantum wheel
# dir -- the other libraries there are unrelated cuQuantum components
# decoding_server never touches, and libcustabilizer itself only depends
# on glibc.
COPY --from=builder \
    /usr/local/lib/python3.12/dist-packages/cuquantum/lib/libcustabilizer.so.0 \
    /usr/local/lib/python3.12/dist-packages/cuquantum/lib/libcustabilizer.so.0

ENV CUDAQX_INSTALL_DIR=/opt/nvidia/cudaqx \
    CUDA_QUANTUM_PATH=/opt/nvidia/cudaq \
    CUDAQ_REALTIME_DIR=/opt/nvidia/cudaq-realtime \
    PATH=/opt/nvidia/cudaqx/bin:${PATH}

WORKDIR /opt/nvidia/cudaqx/bin
ENTRYPOINT ["/opt/nvidia/cudaqx/bin/decoding_server"]
