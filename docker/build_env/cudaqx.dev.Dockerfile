# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

ARG base_image=ghcr.io/nvidia/cuda-quantum-devdeps:ext-amd64-cu12.6-gcc11-main
FROM $base_image

LABEL org.opencontainers.image.description="Dev tools for building and testing CUDA-QX libraries"
LABEL org.opencontainers.image.source="https://github.com/NVIDIA/cudaqx"
LABEL org.opencontainers.image.title="cudaqx-dev"
LABEL org.opencontainers.image.url="https://github.com/NVIDIA/cudaqx"

# FIXME: Remove the cmake install once private repos are updated.
RUN apt-get update && apt-get install -y gfortran libblas-dev jq cuda-nvtx-12-6 \
  && python3 -m pip install "cmake<4" --user \
  && apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY .cudaq_version /cudaq_version

ENV CUDAQ_INSTALL_PREFIX=/usr/local/cudaq

RUN mkdir -p /workspaces/cudaq && cd /workspaces/cudaq \
  && git init \
  && CUDAQ_REPO=$(jq -r '.cudaq.repository' /cudaq_version) \
  && CUDAQ_COMMIT=$(jq -r '.cudaq.ref' /cudaq_version) \
  && git remote add origin https://github.com/${CUDAQ_REPO} \
  && git fetch -q --depth=1 origin ${CUDAQ_COMMIT} \
  && git reset --hard FETCH_HEAD \
  && bash scripts/build_cudaq.sh -v \
  && rm -rf build

#RUN mkdir -p /workspaces/cudaqx && cd /workspaces/cudaqx \
#  && cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DCUDAQ_DIR=/usr/local/cudaq/lib/cmake/cudaq .. \
#  && ninja install
