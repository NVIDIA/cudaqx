# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under    #
# the terms of the Apache License 2.0 which accompanies this distribution.    #
# ============================================================================ #

# Development image for building AND running the QEC decoding server on
# RDMA/FPGA hardware (NVQLink lab, DGX Spark / GB200).
#
# The base image (ghcr.io/nvidia/cudaqx-dev, built by
# docker/build_env/cudaqx.dev.Dockerfile) already carries the toolchain plus
# CUDA-Q and cudaq-realtime built at the .cudaq_version pin, with a copy of
# that pin at /cudaq_version.  This file adds only the environment the
# hardware paths need -- the same packages CI installs at test time in
# .github/actions/build-lib/build_qec.sh, moved into cached image layers:
#
#   - RDMA userspace (rdma-core, ibverbs providers incl. SoftRoCE/rxe)
#   - DOCA 3.3.0 GPUNetIO dev headers (NOT doca-all: it conflicts with the
#     Mellanox OFED preinstalled in the devcontainer base)
#   - Holoscan SDK
#   - TensorRT dev (for the trt_decoder plugin; arm64 packages exist for
#     CUDA 13 only, hence the cu13.0 default base)
#   - cuStabilizer (cuquantum-python wheel)
#   - the Ising-artifact exporter's Python environment + `hf` CLI, so the
#     hardware CI can download the gated model and rebuild the bundle on
#     every run
#   - a prebuilt holoscan-sensor-bridge 2.6.0-EA2 at /opt/holoscan-sensor-bridge
#
# What is deliberately NOT baked in: cudaq-realtime-with-HSB-tools and cudaqx
# itself -- the hardware CI builds both from the commit under test (see
# docker/decoding-server/hw_ci/container_build.sh).  Proprietary artifacts
# (nv-qldpc plugin, cudevice archive) are bind-mounted at run time, never
# baked into a layer.
#
# Build (normally done by hw_ci/run_hw_ci.sh, context = this directory):
#   docker build -f docker/decoding-server/dev.Dockerfile \
#     --build-arg base_image=ghcr.io/nvidia/cudaqx-dev:<tag> \
#     --build-arg cuda_native_arch=100 \
#     -t cudaqx-decoding-hwci docker/decoding-server

ARG base_image=ghcr.io/nvidia/cudaqx-dev:latest-arm64-cu13.0
FROM ${base_image}

# CUDA architecture the prebaked holoscan-sensor-bridge kernels target:
# 100 = GB200 (sm_100), 121 = DGX Spark GB10 (sm_121).  The hardware-CI
# runner auto-detects and passes this.
ARG cuda_native_arch=100

# ---------------------------------------------------------------------------
# Build tools + RDMA userspace.
# NOTE: the base image ships Mellanox OFED's rdma-core fork, whose
# ibverbs-providers outranks Ubuntu's and contains ONLY the mlx5 provider --
# no rxe (SoftRoCE).  The apt line below therefore keeps the Mellanox
# package, and `--roce-pair rxe` runs will SKIP with a named reason until a
# rxe provider matching the Mellanox libibverbs ABI is built into the image
# (see the hw_ci README).  perftest (ib_write_bw) is for fabric smoke tests.
# The `sudo` binary must exist because the example scripts' network helpers
# invoke it literally (a no-op when already root).
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ninja-build curl pkg-config jq \
        rdma-core ibverbs-providers ibverbs-utils infiniband-diags perftest \
        iproute2 ethtool iputils-ping sudo \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# DOCA 3.3.0: only the GPUNetIO dev package (mirrors build_qec.sh; doca-all
# conflicts with the base image's preinstalled OFED), plus cuda-nvrtc-dev
# matching the toolkit (hololink_core links CUDA::nvrtc).
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Holoscan SDK (force-install fallback mirrors build_qec.sh: the package's
# dependency list can miss on the devcontainer base).
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# TensorRT dev, pinned to the toolkit's CUDA flavor (mirrors the arm64 steps
# in .github/workflows/lib_qec.yaml).
# ---------------------------------------------------------------------------
RUN set -e; \
    CUDA_FULL_VERSION=$(nvcc --version | sed -n 's/^.*release \([0-9]\+\.[0-9]\+\).*$/\1/p'); \
    apt-get update; \
    apt-cache search tensorrt \
        | awk -v v="$CUDA_FULL_VERSION" '{print "Package: "$1"\nPin: version *+cuda"v"\nPin-Priority: 1001\n"}' \
        > /etc/apt/preferences.d/tensorrt-cuda$CUDA_FULL_VERSION.pref; \
    apt-get install -y tensorrt-dev; \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# cuStabilizer (pre-bakes .github/actions/build-lib/setup_custabilizer.sh's
# pip install; sourcing that script at build time then becomes a no-op).
# ---------------------------------------------------------------------------
RUN set -e; \
    CUDA_MAJOR_VERSION=$(nvcc --version | sed -n 's/^.*release \([0-9]\+\).*$/\1/p'); \
    pip install --no-cache-dir "cuquantum-python-cu${CUDA_MAJOR_VERSION}>=26.3.0"

# ---------------------------------------------------------------------------
# Ising exporter environment: the `hf` CLI plus every Python package
# examples/qec/realtime_decoding_demo/prepare_ising_artifacts.py checks for,
# so the hardware CI downloads the gated model and regenerates the bundle on
# every run.  Torch must be a CUDA build: the pinned Ising-Decoding exporter
# is GPU-only (its local_run.sh preflights torch.cuda.is_available(), so CPU
# torch fails the lane before inference starts).  The cu130 index matches
# the image's toolkit and covers GB200 (sm_100) and Spark GB10 (sm_121);
# the multi-GB nvidia-* dependency wheels are the accepted cost.
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu130 \
    && pip install --no-cache-dir \
        "huggingface_hub[cli]" \
        stim ldpc beliefmatching hydra-core omegaconf onnx \
        pymatching safetensors scipy matplotlib numpy \
    && hf version

# ---------------------------------------------------------------------------
# Prebuilt holoscan-sensor-bridge 2.6.0-EA2 (mirrors build_qec.sh: same
# operator strip, same targets).  Both the source and build trees are kept:
# cudaq-realtime's HSB-tools build and cudaqx's HSB-tools build consume them
# via HOLOSCAN_SENSOR_BRIDGE_{SOURCE,BUILD}_DIR.  The /workspaces symlink
# satisfies hsb_fpga_decoding_server_test.sh's default HSB_DIR.
# ---------------------------------------------------------------------------
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
    # In-tree mlx5 drivers (e.g. the DGX Spark's -nvidia kernel, no OFED)
    # reject BlueFlame UAR allocation -- doca_uar_create fails and every
    # gpu_roce lane dies at transceiver start.  NONCACHE doorbells are
    # functionally equivalent for the CI lanes.
    sed -i 's/#define DOCA_SEND_BLUE_FLAME 1/#define DOCA_SEND_BLUE_FLAME 0/' \
        src/hololink/operators/gpu_roce_transceiver/gpu_roce_transceiver_common.hpp; \
    export CUDA_NATIVE_ARCH=${cuda_native_arch}; \
    cmake -G Ninja -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DHOLOLINK_BUILD_ONLY_NATIVE=OFF \
        -DHOLOLINK_BUILD_PYTHON=OFF \
        -DHOLOLINK_BUILD_TESTS=OFF \
        -DHOLOLINK_BUILD_TOOLS=OFF \
        -DHOLOLINK_BUILD_EXAMPLES=OFF \
        -DHOLOLINK_BUILD_EMULATOR=OFF; \
    cmake --build build --target gpu_roce_transceiver hololink_core; \
    mkdir -p /workspaces; \
    ln -sfn /opt/holoscan-sensor-bridge /workspaces/holoscan-sensor-bridge

# ---------------------------------------------------------------------------
# Tools the base image lacks:
#  - ibdev2netdev: the in-tree --setup-network helpers shell out to it to map
#    IB devices to netdevs.  It lives in mlnx-ofed-kernel-utils (from the
#    DOCA repo configured above) -- NOT in modern mlnx-tools, which dropped
#    it.  Userspace deps only; no DKMS/kernel modules ride along with
#    --no-install-recommends.
#  - patchelf: libs/qec's add_target_libs_to_wheel patches the rpath of
#    staged external decoder plugins at configure time.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        mlnx-ofed-kernel-utils patchelf \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && test -x /usr/sbin/ibdev2netdev && command -v patchelf

ENV HOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR=/opt/holoscan-sensor-bridge \
    HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR=/opt/holoscan-sensor-bridge/build
