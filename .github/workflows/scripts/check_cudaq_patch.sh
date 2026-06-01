#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Verify that the embedded CUDAQ_PATCH in the CUDA-Q build scripts still
# applies cleanly against the CUDA-Q commit pinned in .cudaq_version.
#
# The patches are sourced and piped to `git apply` exactly the way the build
# scripts apply them (assign the CUDAQ_PATCH heredoc, then
# `echo "$CUDAQ_PATCH" | git apply`). This is important because the trailing
# newline added by `echo` is part of the patch payload (it supplies the final
# blank context line in the second hunk). Anything that extracts the patch
# via a different mechanism may produce a payload that git apply rejects even
# though the build scripts apply the same bytes successfully.

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
WORKFLOW_SCRIPT="$REPO_ROOT/.github/workflows/scripts/build_cudaq.sh"
WHEEL_SCRIPT="$REPO_ROOT/scripts/ci/build_cudaq_wheel.sh"
CUDAQ_VERSION_FILE="$REPO_ROOT/.cudaq_version"

# Source the CUDAQ_PATCH='...' heredoc out of a build script and echo it back
# the same way the build script does, so the trailing newline matches.
emit_patch_from_script() {
  local script=$1
  local shell=$2
  "${shell}" -c "$(sed -n "/^CUDAQ_PATCH='/,/^'\$/p" "${script}")"$'\n''echo "$CUDAQ_PATCH"'
}

# Apply (or check) the patch via the same code path the build script uses.
apply_patch_like_build_script() {
  local script=$1
  local shell=$2
  local mode=$3  # "--check" or empty
  emit_patch_from_script "${script}" "${shell}" \
    | git apply ${mode} --verbose
}

if [ ! -f "${CUDAQ_VERSION_FILE}" ]; then
  echo "Missing ${CUDAQ_VERSION_FILE}" >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required" >&2
  exit 1
fi

CUDAQ_REPO=$(jq -r '.cudaq.repository' "${CUDAQ_VERSION_FILE}")
CUDAQ_REF=$(jq -r '.cudaq.ref' "${CUDAQ_VERSION_FILE}")

echo "Checking CUDAQ_PATCH against ${CUDAQ_REPO}@${CUDAQ_REF}"

WORKDIR=$(mktemp -d)
trap 'rm -rf "${WORKDIR}"' EXIT

# Compare the two embedded patches byte-for-byte, using the same emission
# path each build script uses.
WORKFLOW_PATCH_BYTES="${WORKDIR}/workflow.patch"
WHEEL_PATCH_BYTES="${WORKDIR}/wheel.patch"
emit_patch_from_script "${WORKFLOW_SCRIPT}" bash > "${WORKFLOW_PATCH_BYTES}"
emit_patch_from_script "${WHEEL_SCRIPT}"    sh   > "${WHEEL_PATCH_BYTES}"

if ! cmp -s "${WORKFLOW_PATCH_BYTES}" "${WHEEL_PATCH_BYTES}"; then
  echo "CUDAQ_PATCH differs between:" >&2
  echo "  ${WORKFLOW_SCRIPT}" >&2
  echo "  ${WHEEL_SCRIPT}" >&2
  diff -u "${WORKFLOW_PATCH_BYTES}" "${WHEEL_PATCH_BYTES}" >&2 || true
  exit 1
fi
echo "CUDAQ_PATCH matches in both build scripts."

git clone --filter=blob:none --no-checkout \
  "https://github.com/${CUDAQ_REPO}.git" "${WORKDIR}/cudaq"
git -C "${WORKDIR}/cudaq" checkout "${CUDAQ_REF}"

echo "Checking patch from .github/workflows/scripts/build_cudaq.sh (bash) ..."
(
  cd "${WORKDIR}/cudaq"
  apply_patch_like_build_script "${WORKFLOW_SCRIPT}" bash --check
)

echo "Checking patch from scripts/ci/build_cudaq_wheel.sh (sh) ..."
(
  cd "${WORKDIR}/cudaq"
  apply_patch_like_build_script "${WHEEL_SCRIPT}" sh --check
)

echo "CUDAQ_PATCH applies cleanly to ${CUDAQ_REPO}@${CUDAQ_REF}."
