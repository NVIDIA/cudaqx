#!/usr/bin/env bash

# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Reset CUDA-QX build artifacts and (optionally) the install prefix.
#
# Usage:
#   scripts/clean.sh                      # remove ./build, ./build_qec, ./build_solvers, ./_skbuild
#   scripts/clean.sh --install            # also remove $CUDAQX_INSTALL_PREFIX
#   scripts/clean.sh --install --pip      # also `pip uninstall` the cudaq-{qec,solvers}* packages
#   scripts/clean.sh --docker             # also stop+remove the cudaqx_wheel_builder container
#   scripts/clean.sh --all                # all of the above (use with care)
#   scripts/clean.sh --dry-run            # print actions, do not execute

set -euo pipefail

DO_INSTALL=0
DO_PIP=0
DO_DOCKER=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install)  DO_INSTALL=1 ;;
    --pip)      DO_PIP=1 ;;
    --docker)   DO_DOCKER=1 ;;
    --all)      DO_INSTALL=1; DO_PIP=1; DO_DOCKER=1 ;;
    --dry-run)  DRY_RUN=1 ;;
    -h|--help)
      sed -n '12,21p' "$0"; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

CUDAQX_INSTALL_PREFIX_VAL=${CUDAQX_INSTALL_PREFIX:-$HOME/.cudaqx}

run() {
  echo "+ $*"
  if [[ $DRY_RUN -eq 0 ]]; then
    "$@"
  fi
}

# Build dirs (in repo root) -------------------------------------------------
echo "Cleaning build artifacts"
for d in build build_qec build_solvers _skbuild; do
  if [[ -e "$d" ]]; then
    run rm -rf "$d"
  fi
done

# Install prefix ------------------------------------------------------------
if [[ $DO_INSTALL -eq 1 ]]; then
  echo "Cleaning install prefix"
  if [[ -d "$CUDAQX_INSTALL_PREFIX_VAL" ]]; then
    run rm -rf "$CUDAQX_INSTALL_PREFIX_VAL"
  else
    echo "  (install prefix $CUDAQX_INSTALL_PREFIX_VAL does not exist)"
  fi
fi

# Pip --------------------------------------------------------------------
if [[ $DO_PIP -eq 1 ]]; then
  echo "Uninstalling pip packages"
  run python3 -m pip uninstall -y \
    cudaq-qec-cu12 cudaq-qec-cu13 cudaq-qec \
    cudaq-solvers-cu12 cudaq-solvers-cu13 cudaq-solvers || true
fi

# Wheel-builder container ---------------------------------------------------
if [[ $DO_DOCKER -eq 1 ]]; then
  echo "Resetting wheel-builder container"
  if command -v docker >/dev/null 2>&1; then
    if docker ps -a --format '{{.Names}}' | grep -q '^cudaqx_wheel_builder$'; then
      run docker stop cudaqx_wheel_builder
      run docker rm cudaqx_wheel_builder
    else
      echo "  (cudaqx_wheel_builder is not present)"
    fi
  else
    echo "  (docker not available; skipping)"
  fi
fi

echo "Done."
