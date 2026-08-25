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
# supplied as prebuilt files via an optional build context -- the same flat
# layout `nv-qldpc-decoder-<arch>_ubuntu24.04_cuda<version>_release.tar.gz`
# extracts to (see all_libs_release.yaml's "Fetch assets" step):
#
#   artifacts/
#     libcudaq-qec-nv-qldpc-decoder.so
#     libcudaq-qec-realtime-cudevice-proprietary.a
#
# Both are optional independently: without the cudevice archive, device_graph
# dispatch is compiled but not linked (host-dispatch decoders still work);
# without the nv-qldpc plugin, that decoder is simply absent from
# lib/decoder-plugins. Everything else -- CUDA-Q, cudaq-realtime, cudaqx/qec
# (including the TensorRT decoder plugin), decoding_server, and
# hsb_fpga_syndrome_playback -- is compiled here from this repo's source.
#
# Build (from the repo root):
#   DOCKER_BUILDKIT=1 docker build -f docker/decoding-server/Dockerfile \
#     --build-arg CUDA_VERSION=13.0 \
#     --build-arg cuda_native_arch=90 \
#     --build-context artifacts=./path/to/artifacts \
#     --build-context ising_bundle=./path/to/ising-bundle \
#     -t cudaqx-decoding-server:prod .
#
# If overriding CUDA_VERSION (e.g. to 12.6), also pass a matching
# --build-arg cuda_runtime_image_tag=12.6.3-runtime-ubuntu24.04 (the two
# aren't derived from each other -- see the arg's definition below).
#
# For a reproducible release build, pin the dev image instead of floating
# on latest: --build-arg dev_base_image_version=0.7.0 (no leading "v").
#
# Both --build-context flags may be omitted; the image still builds (minus
# whichever pieces they would have supplied).

# syntax=docker/dockerfile:1.7

ARG CUDA_VERSION=13.0
ARG TARGETARCH
ARG dev_base_image=ghcr.io/nvidia/cudaqx-dev
# The version segment of the dev_base_image tag: "latest" (floating, tracks
# main) or a release version like "0.7.0" (ghcr.io/nvidia/cudaqx-dev is
# tagged <version>-<arch>-cu<cuda_version> when built from a releases/vX.Y.Z
# branch -- no leading "v"). Pin this to a release version for a
# reproducible release build instead of floating on whatever's on main.
ARG dev_base_image_version=latest
# nvidia/cuda on Docker Hub tags by full patch version (e.g.
# 13.0.3-runtime-ubuntu24.04), not the bare CUDA_VERSION minor used
# elsewhere in this file. Kept as a separate arg since a FROM line can't
# derive one from the other; update both together when bumping either.
ARG cuda_runtime_image_tag=13.0.3-runtime-ubuntu24.04

# Empty placeholders for the two optional build contexts. Naming a stage the
# same as a `--build-context NAME=path` flag lets that flag override the
# stage entirely; when a flag is omitted, the matching empty scratch image
# is used instead, so the COPY/mount below sees zero files rather than
# erroring.
FROM scratch AS artifacts
FROM scratch AS ising_bundle

# ---------------------------------------------------------------------------
# Stage: toolchain -- RDMA/DOCA/Holoscan/TensorRT dev headers needed to
# compile the decoding server, plus patchelf (needed at CMake configure time
# when an nv-qldpc .so artifact is supplied).
# ---------------------------------------------------------------------------
FROM ${dev_base_image}:${dev_base_image_version}-${TARGETARCH}-cu${CUDA_VERSION} AS toolchain

RUN apt-get update && apt-get install -y --no-install-recommends \
        ninja-build curl pkg-config jq patchelf \
        rdma-core ibverbs-providers \
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

# holoscan-sensor-bridge 2.6.0-EA2, prebuilt (native gpu_roce_transceiver +
# hololink_core only).
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
ARG cuda_native_arch=90
ENV CUDAQ_INSTALL_PREFIX=/usr/local/cudaq \
    CUDAQ_REALTIME_ROOT=/opt/nvidia/cudaq-realtime \
    CUDAQX_INSTALL_PREFIX=/opt/nvidia/cudaqx

WORKDIR /workspaces/cudaqx
COPY . .

# cudaq-realtime, rebuilt with HSB tools ON (the base image's cudaq-realtime
# is built with HSB tools OFF). CUDA-Q itself (/usr/local/cudaq) is left
# untouched; only this second, HSB-enabled realtime install is added
# alongside it.
RUN --mount=type=cache,target=/root/.ccache,sharing=locked \
    set -e; \
    . scripts/cudaq_realtime_cmake_flags.sh; \
    export CUDA_NATIVE_ARCH="${cuda_native_arch}"; \
    CUDAQ_REPO=$(jq -r '.cudaq.repository' .cudaq_version); \
    CUDAQ_REF=$(jq -r '.cudaq.ref' .cudaq_version); \
    git clone --filter=blob:none --no-checkout "https://github.com/${CUDAQ_REPO}.git" /tmp/cudaq-realtime-src; \
    cd /tmp/cudaq-realtime-src; \
    git sparse-checkout init --cone; \
    git sparse-checkout set realtime cmake; \
    git checkout "$CUDAQ_REF"; \
    cd realtime; mkdir -p build; cd build; \
    cmake -G Ninja -DCMAKE_INSTALL_PREFIX="$CUDAQ_REALTIME_ROOT" \
        -DCMAKE_CUDA_FLAGS="$(cudaq_realtime_cmake_cuda_flags)" \
        -DCUDAQ_REALTIME_ENABLE_HSB_TOOLS=ON \
        -DHOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR=/opt/holoscan-sensor-bridge \
        -DHOLOSCAN_SENSOR_BRIDGE_BUILD_DIR=/opt/holoscan-sensor-bridge/build \
        ..; \
    ninja; \
    ninja install; \
    rm -rf /tmp/cudaq-realtime-src

# Python bindings are off -- this
# is a binary runtime image, not a wheel build. CUDAQX_INCLUDE_TESTS stays
# ON despite that name: hsb_fpga_syndrome_playback is an actual deliverable
# that lives under unittests/utils/ and is only built/installed when tests
# are enabled. The gtest unit test binaries get compiled as a side effect
# but are never installed into CUDAQX_INSTALL_PREFIX, so they don't reach
# the runtime stage.
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

# CUDA-Q + cudaq-realtime + cudaqx/qec runtime trees, copied at the SAME
# absolute paths used in the builder stage (load-bearing for
# decoding_server's baked-in RPATH). The nv-qldpc plugin (if supplied) is
# already inside CUDAQX_INSTALL_PREFIX, rpath-patched, via `cmake --install`;
# the cudevice archive (if supplied) is already statically linked into the
# decoding_server binary itself -- neither artifact needs a separate COPY
# here.
COPY --from=builder /usr/local/cudaq /usr/local/cudaq
COPY --from=builder /opt/nvidia/cudaq-realtime /opt/nvidia/cudaq-realtime
COPY --from=builder /opt/nvidia/cudaqx /opt/nvidia/cudaqx

# libcudaq-qec.so is unconditionally linked against cuStabilizer (part of a
# cuquantum-python pip wheel baked into the dev_base_image) with an absolute
# RPATH entry into that wheel's site-packages location, so this single file
# has to land at the same path here too. Only the one .so, not the whole
# ~181MB cuquantum wheel dir -- the other libraries there are unrelated
# cuQuantum components decoding_server never touches, and libcustabilizer
# itself only depends on glibc.
COPY --from=builder \
    /usr/local/lib/python3.12/dist-packages/cuquantum/lib/libcustabilizer.so.0 \
    /usr/local/lib/python3.12/dist-packages/cuquantum/lib/libcustabilizer.so.0

# Ising-Decoder bundle: gated by a Hugging Face license
# click-through, so it is NOT fetched inside this Dockerfile. 
# Optional and empty by default; if omitted, trt_decoder(ising) has nothing
# to load unless QEC_ISING_ARTIFACTS_DIR is bind-mounted in at `docker run`
# time instead.
COPY --from=ising_bundle . /opt/nvidia/ising-bundle

ENV CUDAQX_INSTALL_DIR=/opt/nvidia/cudaqx \
    CUDA_QUANTUM_PATH=/usr/local/cudaq \
    CUDAQ_REALTIME_DIR=/opt/nvidia/cudaq-realtime \
    QEC_ISING_ARTIFACTS_DIR=/opt/nvidia/ising-bundle \
    PATH=/opt/nvidia/cudaqx/bin:${PATH}

WORKDIR /opt/nvidia/cudaqx/bin
ENTRYPOINT ["/opt/nvidia/cudaqx/bin/decoding_server"]
