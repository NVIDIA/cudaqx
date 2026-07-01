#!/usr/bin/env bash

# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# CUDA-QX environment doctor: prints versions and configuration relevant to
# building, packaging, and running CUDA-QX. Use this when users report
# "ImportError" or version-mismatch issues — capturing the output up front
# is cheaper than guessing.
#
# Usage:
#   scripts/doctor.sh           # human-readable output
#   scripts/doctor.sh --json    # JSON for piping into a bug report

set -u

JSON=0
if [[ "${1:-}" == "--json" ]]; then
  JSON=1
fi

# Helpers --------------------------------------------------------------------

_cmd() {
  # Run command, return its first line (or empty if not found / failure).
  local out
  if ! command -v "$1" >/dev/null 2>&1; then
    echo ""
    return
  fi
  out=$("$@" 2>/dev/null | head -n 1) || out=""
  echo "$out"
}

_pip_pkg() {
  # Return "name version" for a pip package matching the regex, or "".
  python3 -m pip list --format=freeze 2>/dev/null \
    | grep -E "^$1==" | head -n 1
}

_kv() {
  # Text mode only. JSON output is emitted in one shot at the end via Python.
  printf '  %-30s %s\n' "$1:" "$2"
}

_section() {
  echo ""
  echo "=== $1 ==="
}

# Collect --------------------------------------------------------------------

CMAKE_VER=$(_cmd cmake --version)
NINJA_VER=$(_cmd ninja --version)
GCC_VER=$(_cmd gcc --version)
CLANG_VER=$(_cmd clang --version)
PYTHON_VER=$(_cmd python3 --version)
PIP_VER=$(_cmd python3 -m pip --version)
DOXYGEN_VER=$(_cmd doxygen --version)

NVIDIA_SMI_HEAD=""
if command -v nvidia-smi >/dev/null 2>&1; then
  NVIDIA_SMI_HEAD=$(nvidia-smi --query-gpu=name,driver_version,compute_cap \
                    --format=csv,noheader 2>/dev/null | head -n 1)
fi

NVCC_VER=$(_cmd nvcc --version | tr '\n' ' ')

PIP_CUDAQ=$(_pip_pkg 'cuda-quantum(-cu1[23])?')
PIP_QEC=$(_pip_pkg 'cudaq-qec(-cu1[23])?')
PIP_SOLVERS=$(_pip_pkg 'cudaq-solvers(-cu1[23])?')
PIP_CUQUANTUM=$(_pip_pkg 'cuquantum-python-cu1[23]')
PIP_TENSORRT=$(_pip_pkg 'tensorrt-cu1[23]')
PIP_TORCH=$(_pip_pkg 'torch')

# Env vars
CUDAQ_INSTALL_PREFIX_VAL=${CUDAQ_INSTALL_PREFIX:-$HOME/.cudaq}
CUDAQX_INSTALL_PREFIX_VAL=${CUDAQX_INSTALL_PREFIX:-$HOME/.cudaqx}

# Sanity: do install prefixes exist?
CUDAQ_PREFIX_OK="missing"
[[ -d "$CUDAQ_INSTALL_PREFIX_VAL" ]] && CUDAQ_PREFIX_OK="ok ($CUDAQ_INSTALL_PREFIX_VAL)"
CUDAQX_PREFIX_OK="missing"
[[ -d "$CUDAQX_INSTALL_PREFIX_VAL" ]] && CUDAQX_PREFIX_OK="ok ($CUDAQX_INSTALL_PREFIX_VAL)"

# Imports
IMPORT_CUDAQ=$(python3 -c 'import cudaq; print(cudaq.__file__)' 2>&1 | tr '\n' ' ')
IMPORT_QEC=$(python3 -c 'import cudaq_qec; print(cudaq_qec.__file__)' 2>&1 | tr '\n' ' ')
IMPORT_SOLVERS=$(python3 -c 'import cudaq_solvers; print(cudaq_solvers.__file__)' 2>&1 | tr '\n' ' ')

# Quick CUDA-suffix consistency check.
QEC_SUFFIX=$(echo "$PIP_QEC" | grep -oE 'cu1[23]' | head -n 1)
SOLVERS_SUFFIX=$(echo "$PIP_SOLVERS" | grep -oE 'cu1[23]' | head -n 1)
CUDAQ_SUFFIX=$(echo "$PIP_CUDAQ" | grep -oE 'cu1[23]' | head -n 1)
SUFFIX_WARN=""
for pair in "qec=$QEC_SUFFIX" "solvers=$SOLVERS_SUFFIX"; do
  pkg=${pair%%=*}
  pkg_suffix=${pair##*=}
  if [[ -n "$CUDAQ_SUFFIX" && -n "$pkg_suffix" && "$CUDAQ_SUFFIX" != "$pkg_suffix" ]]; then
    SUFFIX_WARN+="cuda-quantum-$CUDAQ_SUFFIX vs cudaq-$pkg-$pkg_suffix; "
  fi
done

# Print ---------------------------------------------------------------------

if [[ $JSON -eq 1 ]]; then
  # Hand all gathered values to Python so it can emit valid JSON.
  # (The previous hand-rolled printf approach left trailing commas and
  # could not safely escape values containing quotes.)
  export CMAKE_VER NINJA_VER GCC_VER CLANG_VER DOXYGEN_VER
  export PYTHON_VER PIP_VER
  export NVIDIA_SMI_HEAD NVCC_VER
  export PIP_CUDAQ PIP_QEC PIP_SOLVERS PIP_CUQUANTUM PIP_TENSORRT PIP_TORCH
  export CUDAQ_PREFIX_OK CUDAQX_PREFIX_OK
  export IMPORT_CUDAQ IMPORT_QEC IMPORT_SOLVERS
  export SUFFIX_WARN CUDAQ_SUFFIX

  python3 - <<'PY'
import json, os

def env(k, default=""):
    return os.environ.get(k, default).strip()

sanity = {}
if env("SUFFIX_WARN"):
    sanity["abi_mismatch_warning"] = env("SUFFIX_WARN")
elif env("CUDAQ_SUFFIX"):
    sanity["cuda_suffix_consistency"] = f"ok ({env('CUDAQ_SUFFIX')})"

out = {
    "build_toolchain": {
        "cmake":   env("CMAKE_VER")   or "MISSING",
        "ninja":   env("NINJA_VER")   or "MISSING",
        "gcc":     env("GCC_VER")     or "MISSING",
        "clang":   env("CLANG_VER")   or "not present",
        "doxygen": env("DOXYGEN_VER") or "not present",
    },
    "python": {
        "python3": env("PYTHON_VER") or "MISSING",
        "pip":     env("PIP_VER")    or "MISSING",
    },
    "gpu_cuda": {
        "nvidia_smi": env("NVIDIA_SMI_HEAD") or "not present",
        "nvcc":       env("NVCC_VER")        or "not present",
    },
    "pip_packages": {
        "cuda-quantum":     env("PIP_CUDAQ")     or "not installed",
        "cudaq-qec":        env("PIP_QEC")       or "not installed",
        "cudaq-solvers":    env("PIP_SOLVERS")   or "not installed",
        "cuquantum-python": env("PIP_CUQUANTUM") or "not installed",
        "tensorrt":         env("PIP_TENSORRT") or "not installed",
        "torch":            env("PIP_TORCH")    or "not installed",
    },
    "install_prefixes": {
        "CUDAQ_INSTALL_PREFIX":  env("CUDAQ_PREFIX_OK"),
        "CUDAQX_INSTALL_PREFIX": env("CUDAQX_PREFIX_OK"),
    },
    "imports": {
        "cudaq":         env("IMPORT_CUDAQ"),
        "cudaq_qec":     env("IMPORT_QEC"),
        "cudaq_solvers": env("IMPORT_SOLVERS"),
    },
    "sanity": sanity,
}
print(json.dumps(out, indent=2))
PY
  exit 0
fi

_section "Build toolchain"
_kv "cmake"        "${CMAKE_VER:-MISSING}"
_kv "ninja"        "${NINJA_VER:-MISSING}"
_kv "gcc"          "${GCC_VER:-MISSING}"
_kv "clang"        "${CLANG_VER:-not present}"
_kv "doxygen"      "${DOXYGEN_VER:-not present}"

_section "Python"
_kv "python3"      "${PYTHON_VER:-MISSING}"
_kv "pip"          "${PIP_VER:-MISSING}"

_section "GPU / CUDA"
_kv "nvidia-smi"   "${NVIDIA_SMI_HEAD:-not present}"
_kv "nvcc"         "${NVCC_VER:-not present}"

_section "CUDA-Q / CUDA-QX (pip)"
_kv "cuda-quantum" "${PIP_CUDAQ:-not installed}"
_kv "cudaq-qec"    "${PIP_QEC:-not installed}"
_kv "cudaq-solvers" "${PIP_SOLVERS:-not installed}"
_kv "cuquantum-python" "${PIP_CUQUANTUM:-not installed}"
_kv "tensorrt"     "${PIP_TENSORRT:-not installed}"
_kv "torch"        "${PIP_TORCH:-not installed}"

_section "Install prefixes"
_kv "CUDAQ_INSTALL_PREFIX"  "$CUDAQ_PREFIX_OK"
_kv "CUDAQX_INSTALL_PREFIX" "$CUDAQX_PREFIX_OK"

_section "Imports"
_kv "import cudaq"         "$IMPORT_CUDAQ"
_kv "import cudaq_qec"     "$IMPORT_QEC"
_kv "import cudaq_solvers" "$IMPORT_SOLVERS"

_section "Sanity"
if [[ -n "$SUFFIX_WARN" ]]; then
  _kv "WARNING: ABI mismatch" "$SUFFIX_WARN"
elif [[ -n "$CUDAQ_SUFFIX" ]]; then
  _kv "CUDA suffix consistency" "ok ($CUDAQ_SUFFIX)"
fi
