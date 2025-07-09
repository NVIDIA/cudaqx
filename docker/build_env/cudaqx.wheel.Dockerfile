# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

ARG base_image=ghcr.io/nvidia/cuda-quantum-devdeps:manylinux-amd64-cu12.0-gcc11-main
FROM ${base_image}

ARG python_version=3.10

LABEL org.opencontainers.image.description="Dev tools for building and testing CUDA-QX libraries"
LABEL org.opencontainers.image.source="https://github.com/NVIDIA/cudaqx"
LABEL org.opencontainers.image.title="cudaqx-dev"
LABEL org.opencontainers.image.url="https://github.com/NVIDIA/cudaqx"

ENV CUDAQ_INSTALL_PREFIX=/usr/local/cudaq

RUN dnf install -y jq cuda-nvtx-12-0
RUN mkdir -p /workspaces/cudaqx
COPY .cudaq_version /workspaces/cudaqx
COPY .github/workflows/scripts/build_cudaq.sh /workspaces/cudaqx
RUN mkdir /cudaq-wheels
COPY cudaq-wheels/ /cudaq-wheels/

RUN mkdir -p /workspaces/cudaqx/cudaq && cd /workspaces/cudaqx/cudaq \
  && git init \
  && CUDAQ_REPO=$(jq -r '.cudaq.repository' /workspaces/cudaqx/.cudaq_version) \
  && CUDAQ_COMMIT=$(jq -r '.cudaq.ref' /workspaces/cudaqx/.cudaq_version) \
  && git remote add origin https://github.com/${CUDAQ_REPO} \
  && git fetch -q --depth=1 origin ${CUDAQ_COMMIT} \
  && git reset --hard FETCH_HEAD \
  && cd .. \
  && bash build_cudaq.sh --python-version ${python_version} \
  && rm -rf cudaq
