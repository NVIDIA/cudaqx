#!/usr/bin/env bash
# preflight.sh — gather environment, toolchain, GPU/CPU, install state.
#
# This is the FIRST script every CUDA-Q Libraries skill calls. The output (especially
# in --json mode) is consumed by `pick_workflow.py` to deterministically
# pick the next reference file and the next commands to run.
#
# Supersedes `scripts/doctor.sh` (the existing repo-root tool): preflight
# re-implements the same checks and adds GPU count, CPU count, venv
# detection, submodule status, build state, and structured JSON output.
# `doctor.sh` is kept around for the human-readable troubleshooting flow.
#
# Usage:
#   preflight.sh           # human-readable
#   preflight.sh --json    # machine-readable; pipe into pick_workflow.py

set -u

JSON=0
[[ "${1:-}" == "--json" ]] && JSON=1

# Locate repo root. Prefer `git rev-parse` so moving this script up or down
# a directory level does not silently break path arithmetic. Fall back to the
# legacy 4-levels-up relative path when not in a git checkout (e.g. extracted
# tarball, or running outside the repo).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if command -v git >/dev/null 2>&1 \
    && REPO_ROOT="$(git -C "$HERE" rev-parse --show-toplevel 2>/dev/null)" \
    && [[ -n "$REPO_ROOT" ]]; then
  :
else
  REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_first_line() {
  command -v "$1" >/dev/null 2>&1 || { echo ""; return; }
  "$@" 2>/dev/null | head -n 1
}

_pip_pkg_version() {
  python3 -m pip list --format=freeze 2>/dev/null \
    | awk -F'==' -v re="^$1$" '$1 ~ re { print $2; exit }'
}

# ---------------------------------------------------------------------------
# Gather: system
# ---------------------------------------------------------------------------

OS_KERNEL="$(uname -srm 2>/dev/null || echo unknown)"
OS_ID=""
OS_VERSION=""
if [[ -f /etc/os-release ]]; then
  # shellcheck source=/dev/null
  OS_ID="$(. /etc/os-release && echo "${ID:-}")"
  # shellcheck source=/dev/null
  OS_VERSION="$(. /etc/os-release && echo "${VERSION_ID:-}")"
fi
CPU_COUNT="$(nproc 2>/dev/null || echo 0)"
CPU_MODEL="$(awk -F: '/^model name/ {gsub(/^ +/, "", $2); print $2; exit}' /proc/cpuinfo 2>/dev/null || echo '')"

# ---------------------------------------------------------------------------
# Gather: GPU
# ---------------------------------------------------------------------------

GPU_COUNT=0
GPU_LINES=""
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_LINES="$(nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total \
                          --format=csv,noheader 2>/dev/null || echo '')"
  if [[ -n "$GPU_LINES" ]]; then
    GPU_COUNT=$(printf '%s\n' "$GPU_LINES" | wc -l)
  fi
fi

NVCC_LINE="$(_first_line nvcc --version | sed 's/^[[:space:]]*//')"
[[ -z "$NVCC_LINE" ]] && NVCC_LINE="$(nvcc --version 2>/dev/null | grep -E '^Cuda' | head -n 1 || true)"

# ---------------------------------------------------------------------------
# Gather: python / venv
# ---------------------------------------------------------------------------

PY_INFO_JSON=$(REPO_ROOT="$REPO_ROOT" python3 - <<'PY' 2>/dev/null || echo '{}'
import json, os, subprocess, sys
from pathlib import Path

repo_root = Path(os.environ.get("REPO_ROOT", "."))

# Find candidate venvs in the repo root: any .venv, venv, env, .env that
# contains a `pyvenv.cfg` and a python interpreter we can invoke.
candidates = []
for name in (".venv", "venv", "env", ".env"):
    cand = repo_root / name
    if not cand.is_dir():
        continue
    if not (cand / "pyvenv.cfg").exists():
        continue
    py = cand / "bin" / "python"
    if not py.exists():
        py = cand / "bin" / "python3"
    if not py.exists():
        continue
    info = {"path": str(cand), "interpreter": str(py)}
    try:
        ver = subprocess.run(
            [str(py), "-c", "import sys; print(sys.version.split()[0])"],
            capture_output=True, text=True, timeout=5,
        )
        info["version"] = ver.stdout.strip() if ver.returncode == 0 else ""
        info["valid"] = ver.returncode == 0
    except Exception:
        info["version"] = ""
        info["valid"] = False
    # NOTE: compare unresolved paths. .venv/bin/python is typically a symlink
    # to the system python, so resolve() would falsely match. Treat the venv
    # as "in use" if sys.executable lives inside its bin/ directory (handles
    # python vs python3 aliases inside the same venv).
    try:
        active = Path(sys.executable)
        bin_dir = (cand / "bin").resolve(strict=False)
        info["in_use"] = active.parent.resolve(strict=False) == bin_dir
    except Exception:
        info["in_use"] = str(py) == sys.executable
    candidates.append(info)

out = {
    "executable": sys.executable,
    "version": sys.version.split()[0],
    "prefix": sys.prefix,
    "base_prefix": getattr(sys, "base_prefix", sys.prefix),
    "in_venv": getattr(sys, "base_prefix", sys.prefix) != sys.prefix,
    "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV", ""),
    "candidates": candidates,
}
print(json.dumps(out))
PY
)

# ---------------------------------------------------------------------------
# Gather: toolchain
# ---------------------------------------------------------------------------

CMAKE_VER="$(_first_line cmake --version)"
NINJA_VER="$(_first_line ninja --version)"
GCC_VER="$(_first_line gcc --version)"
CLANG_VER="$(_first_line clang --version)"
DOXYGEN_VER="$(_first_line doxygen --version)"
GFORTRAN_VER="$(_first_line gfortran --version)"

# ---------------------------------------------------------------------------
# Gather: pip packages relevant to CUDA-Q Libraries
# ---------------------------------------------------------------------------

PIP_CUDAQ="$(_pip_pkg_version 'cuda-quantum(-cu1[23])?')"
PIP_QEC="$(_pip_pkg_version 'cudaq-qec(-cu1[23])?')"
PIP_SOLVERS="$(_pip_pkg_version 'cudaq-solvers(-cu1[23])?')"
PIP_CUQUANTUM="$(_pip_pkg_version 'cuquantum-python-cu1[23]')"
PIP_TENSORRT="$(_pip_pkg_version 'tensorrt(-cu1[23])?')"
PIP_TORCH="$(_pip_pkg_version 'torch')"
PIP_QUIMB="$(_pip_pkg_version 'quimb')"
PIP_LIGHTNING="$(_pip_pkg_version 'lightning')"
PIP_MPI4PY="$(_pip_pkg_version 'mpi4py')"
PIP_TRANSFORMERS="$(_pip_pkg_version 'transformers')"
PIP_OPENFERMION="$(_pip_pkg_version 'openfermion')"

# Suffix consistency (cu12 vs cu13).
_pkg_suffix() { echo "$1" | grep -oE 'cu1[23]' | head -n 1; }
NAME_CUDAQ="$(python3 -m pip list --format=freeze 2>/dev/null | grep -E '^cuda-quantum(-cu1[23])?==' | head -n 1)"
NAME_QEC="$(python3 -m pip list --format=freeze 2>/dev/null | grep -E '^cudaq-qec(-cu1[23])?==' | head -n 1)"
NAME_SOLVERS="$(python3 -m pip list --format=freeze 2>/dev/null | grep -E '^cudaq-solvers(-cu1[23])?==' | head -n 1)"
SUFFIX_CUDAQ="$(_pkg_suffix "$NAME_CUDAQ")"
SUFFIX_QEC="$(_pkg_suffix "$NAME_QEC")"
SUFFIX_SOLVERS="$(_pkg_suffix "$NAME_SOLVERS")"

# ---------------------------------------------------------------------------
# Gather: env vars + install prefixes
# ---------------------------------------------------------------------------

CUDAQ_INSTALL_PREFIX_VAL="${CUDAQ_INSTALL_PREFIX:-$HOME/.cudaq}"
CUDAQX_INSTALL_PREFIX_VAL="${CUDAQX_INSTALL_PREFIX:-$HOME/.cudaq}"
CUDAQ_DIR_VAL="${CUDAQ_DIR:-}"
CUDAQX_ENABLE_LIBS_VAL="${CUDAQX_ENABLE_LIBS:-}"
LD_LIBRARY_PATH_VAL="${LD_LIBRARY_PATH:-}"
PYTHONPATH_VAL="${PYTHONPATH:-}"

CUDAQ_PREFIX_EXISTS=false
[[ -d "$CUDAQ_INSTALL_PREFIX_VAL" ]] && CUDAQ_PREFIX_EXISTS=true
CUDAQX_PREFIX_EXISTS=false
[[ -d "$CUDAQX_INSTALL_PREFIX_VAL" ]] && CUDAQX_PREFIX_EXISTS=true

CUDAQX_INSTALL_FILECOUNT=0
[[ -d "$CUDAQX_INSTALL_PREFIX_VAL" ]] && \
  CUDAQX_INSTALL_FILECOUNT=$(find "$CUDAQX_INSTALL_PREFIX_VAL" -type f 2>/dev/null | wc -l)

BUILD_DIR_EXISTS=false
[[ -d "$REPO_ROOT/build" ]] && BUILD_DIR_EXISTS=true

# ---------------------------------------------------------------------------
# Gather: git submodules
# ---------------------------------------------------------------------------

SUBMODULES_JSON='[]'
if [[ -f "$REPO_ROOT/.gitmodules" ]] && command -v git >/dev/null 2>&1; then
  # `python3 -c` (not `python3 - <<HEREDOC`) so the piped stdin from
  # `git submodule status` reaches Python. A heredoc would shadow stdin
  # and we'd silently emit '[]' regardless of submodule state.
  SUBMODULES_JSON=$(cd "$REPO_ROOT" && git submodule status 2>/dev/null \
    | python3 -c '
import json, sys
out = []
for line in sys.stdin:
    line = line.rstrip("\n")
    if not line:
        continue
    initialized = not (line.startswith("-") or line.startswith("U"))
    parts = line[1:].split() if line[0] in "-+U" else line.split()
    if len(parts) >= 2:
        out.append({
            "path": parts[1],
            "sha": parts[0],
            "initialized": initialized,
        })
print(json.dumps(out))
')
fi

# ---------------------------------------------------------------------------
# Quick warnings (high-signal; pick_workflow.py uses these)
# ---------------------------------------------------------------------------

WARNINGS=()
if [[ -n "$SUFFIX_CUDAQ" && -n "$SUFFIX_QEC" && "$SUFFIX_CUDAQ" != "$SUFFIX_QEC" ]]; then
  WARNINGS+=("ABI mismatch: cuda-quantum-$SUFFIX_CUDAQ vs cudaq-qec-$SUFFIX_QEC")
fi
if [[ -n "$SUFFIX_CUDAQ" && -n "$SUFFIX_SOLVERS" && "$SUFFIX_CUDAQ" != "$SUFFIX_SOLVERS" ]]; then
  WARNINGS+=("ABI mismatch: cuda-quantum-$SUFFIX_CUDAQ vs cudaq-solvers-$SUFFIX_SOLVERS")
fi
if [[ -z "$GFORTRAN_VER" && -n "$PIP_SOLVERS" ]]; then
  WARNINGS+=("gfortran missing: cobyla/lbfgs will crash at runtime; install libgfortran5 or gfortran")
fi
[[ -z "$CMAKE_VER" ]] && WARNINGS+=("cmake not on PATH")
[[ -z "$NINJA_VER" ]] && WARNINGS+=("ninja not on PATH")

# Venv discovery: warn when a candidate exists but isn't being used.
VENV_WARN=$(echo "$PY_INFO_JSON" | python3 -c '
import json, sys
try:
    d = json.loads(sys.stdin.read() or "{}")
except json.JSONDecodeError:
    sys.exit(0)
cands = d.get("candidates", [])
if not cands:
    sys.exit(0)
in_use = next((c for c in cands if c.get("in_use")), None)
valid = [c for c in cands if c.get("valid") and not c.get("in_use")]
if in_use:
    sys.exit(0)
if valid:
    cand = valid[0]
    path = cand.get("path", "")
    ver = cand.get("version", "?")
    exe = d.get("executable", "")
    print(f"venv detected at {path} (python {ver}) "
          f"but the active interpreter is {exe}; "
          f"consider: source {path}/bin/activate")
' 2>/dev/null)
[[ -n "$VENV_WARN" ]] && WARNINGS+=("$VENV_WARN")

# ---------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------

if [[ $JSON -eq 1 ]]; then
  # Hand all gathered values to Python so it can emit clean JSON.
  export OS_KERNEL OS_ID OS_VERSION CPU_COUNT CPU_MODEL
  export GPU_COUNT GPU_LINES NVCC_LINE
  export PY_INFO_JSON
  export CMAKE_VER NINJA_VER GCC_VER CLANG_VER DOXYGEN_VER GFORTRAN_VER
  export PIP_CUDAQ PIP_QEC PIP_SOLVERS PIP_CUQUANTUM PIP_TENSORRT PIP_TORCH
  export PIP_QUIMB PIP_LIGHTNING PIP_MPI4PY PIP_TRANSFORMERS PIP_OPENFERMION
  export NAME_CUDAQ NAME_QEC NAME_SOLVERS SUFFIX_CUDAQ SUFFIX_QEC SUFFIX_SOLVERS
  export CUDAQ_INSTALL_PREFIX_VAL CUDAQX_INSTALL_PREFIX_VAL CUDAQ_DIR_VAL CUDAQX_ENABLE_LIBS_VAL
  export LD_LIBRARY_PATH_VAL PYTHONPATH_VAL
  export CUDAQ_PREFIX_EXISTS CUDAQX_PREFIX_EXISTS CUDAQX_INSTALL_FILECOUNT BUILD_DIR_EXISTS
  export SUBMODULES_JSON REPO_ROOT
  WARN_JSON=$(printf '%s\n' "${WARNINGS[@]}" | python3 -c 'import json,sys; print(json.dumps([l for l in sys.stdin.read().splitlines() if l]))')
  export WARN_JSON

  python3 - <<'PY'
import json, os
def env(k, default=""):
    return os.environ.get(k, default)
def b(k):
    return env(k) == "true"

gpus = []
for line in env("GPU_LINES").splitlines():
    if not line.strip():
        continue
    parts = [p.strip() for p in line.split(",")]
    while len(parts) < 4:
        parts.append("")
    gpus.append({
        "name": parts[0],
        "driver_version": parts[1],
        "compute_cap": parts[2],
        "memory_total": parts[3],
    })

try:
    py_info = json.loads(env("PY_INFO_JSON") or "{}")
except json.JSONDecodeError:
    py_info = {}
try:
    submodules = json.loads(env("SUBMODULES_JSON") or "[]")
except json.JSONDecodeError:
    submodules = []
try:
    warnings = json.loads(env("WARN_JSON") or "[]")
except json.JSONDecodeError:
    warnings = []

out = {
    "system": {
        "os_kernel": env("OS_KERNEL"),
        "os_id": env("OS_ID"),
        "os_version": env("OS_VERSION"),
        "cpu_count": int(env("CPU_COUNT") or 0),
        "cpu_model": env("CPU_MODEL"),
    },
    "gpu": {
        "count": int(env("GPU_COUNT") or 0),
        "devices": gpus,
        "nvcc": env("NVCC_LINE"),
    },
    "python_env": py_info,
    "toolchain": {
        "cmake": env("CMAKE_VER"),
        "ninja": env("NINJA_VER"),
        "gcc": env("GCC_VER"),
        "clang": env("CLANG_VER"),
        "doxygen": env("DOXYGEN_VER"),
        "gfortran": env("GFORTRAN_VER"),
    },
    "pip_packages": {
        "cuda-quantum":     {"version": env("PIP_CUDAQ"),      "name": env("NAME_CUDAQ").split('==')[0] if '==' in env("NAME_CUDAQ") else env("NAME_CUDAQ"),    "cuda_suffix": env("SUFFIX_CUDAQ")},
        "cudaq-qec":        {"version": env("PIP_QEC"),        "name": env("NAME_QEC").split('==')[0] if '==' in env("NAME_QEC") else env("NAME_QEC"),          "cuda_suffix": env("SUFFIX_QEC")},
        "cudaq-solvers":    {"version": env("PIP_SOLVERS"),    "name": env("NAME_SOLVERS").split('==')[0] if '==' in env("NAME_SOLVERS") else env("NAME_SOLVERS"), "cuda_suffix": env("SUFFIX_SOLVERS")},
        "cuquantum-python": {"version": env("PIP_CUQUANTUM")},
        "tensorrt":         {"version": env("PIP_TENSORRT")},
        "torch":            {"version": env("PIP_TORCH")},
        "quimb":            {"version": env("PIP_QUIMB")},
        "lightning":        {"version": env("PIP_LIGHTNING")},
        "mpi4py":           {"version": env("PIP_MPI4PY")},
        "transformers":     {"version": env("PIP_TRANSFORMERS")},
        "openfermion":      {"version": env("PIP_OPENFERMION")},
    },
    "env_vars": {
        "CUDAQ_INSTALL_PREFIX": env("CUDAQ_INSTALL_PREFIX_VAL"),
        "CUDAQX_INSTALL_PREFIX": env("CUDAQX_INSTALL_PREFIX_VAL"),
        "CUDAQ_DIR": env("CUDAQ_DIR_VAL"),
        "CUDAQX_ENABLE_LIBS": env("CUDAQX_ENABLE_LIBS_VAL"),
        "LD_LIBRARY_PATH": env("LD_LIBRARY_PATH_VAL"),
        "PYTHONPATH": env("PYTHONPATH_VAL"),
    },
    "build_state": {
        "repo_root": env("REPO_ROOT"),
        "cudaq_prefix_exists": b("CUDAQ_PREFIX_EXISTS"),
        "cudaqx_prefix_exists": b("CUDAQX_PREFIX_EXISTS"),
        "cudaqx_install_filecount": int(env("CUDAQX_INSTALL_FILECOUNT") or 0),
        "build_dir_exists": b("BUILD_DIR_EXISTS"),
    },
    "git_submodules": submodules,
    "warnings": warnings,
}
print(json.dumps(out, indent=2))
PY
  exit 0
fi

# Human-readable mode -------------------------------------------------------

printf "=== System ===\n"
printf "  %-22s %s\n" "Kernel:"   "$OS_KERNEL"
printf "  %-22s %s %s\n" "Distro:" "${OS_ID:-?}" "${OS_VERSION:-?}"
printf "  %-22s %s\n" "CPUs:"     "$CPU_COUNT"
[[ -n "$CPU_MODEL" ]] && printf "  %-22s %s\n" "CPU model:" "$CPU_MODEL"

printf "\n=== GPU / CUDA ===\n"
if [[ "$GPU_COUNT" -eq 0 ]]; then
  printf "  no GPUs visible (nvidia-smi unavailable or empty)\n"
else
  printf "  %-22s %s\n" "GPU count:" "$GPU_COUNT"
  printf "%s\n" "$GPU_LINES" | sed 's/^/    /'
fi
[[ -n "$NVCC_LINE" ]] && printf "  %-22s %s\n" "nvcc:" "$NVCC_LINE"

printf "\n=== Python / venv ===\n"
echo "$PY_INFO_JSON" | python3 -c '
import json, sys
d = json.load(sys.stdin) if sys.stdin else {}
for k in ("executable", "version", "prefix", "in_venv", "VIRTUAL_ENV"):
    val = d.get(k, "")
    print(f"  {k:22s} {val}")
cands = d.get("candidates", [])
if cands:
    print(f"  candidates ({len(cands)}):")
    for c in cands:
        flag = "ACTIVE" if c.get("in_use") else ("ok" if c.get("valid") else "broken")
        path = c.get("path", "")
        ver = c.get("version", "?")
        print(f"    [{flag}] {path} (python {ver})")
else:
    print("  candidates             (none in repo root)")
'

printf "\n=== Toolchain ===\n"
printf "  %-22s %s\n" "cmake:"    "${CMAKE_VER:-MISSING}"
printf "  %-22s %s\n" "ninja:"    "${NINJA_VER:-MISSING}"
printf "  %-22s %s\n" "gcc:"      "${GCC_VER:-MISSING}"
printf "  %-22s %s\n" "clang:"    "${CLANG_VER:-not present}"
printf "  %-22s %s\n" "doxygen:"  "${DOXYGEN_VER:-not present}"
printf "  %-22s %s\n" "gfortran:" "${GFORTRAN_VER:-not present}"

printf "\n=== pip (CUDA-Q Libraries-relevant) ===\n"
printf "  %-22s %s (%s)\n" "cuda-quantum:"     "${PIP_CUDAQ:-not installed}"   "${SUFFIX_CUDAQ:--}"
printf "  %-22s %s (%s)\n" "cudaq-qec:"        "${PIP_QEC:-not installed}"     "${SUFFIX_QEC:--}"
printf "  %-22s %s (%s)\n" "cudaq-solvers:"    "${PIP_SOLVERS:-not installed}" "${SUFFIX_SOLVERS:--}"
printf "  %-22s %s\n" "cuquantum-python:"  "${PIP_CUQUANTUM:-not installed}"
printf "  %-22s %s\n" "tensorrt:"          "${PIP_TENSORRT:-not installed}"
printf "  %-22s %s\n" "torch:"             "${PIP_TORCH:-not installed}"

printf "\n=== Install prefixes ===\n"
printf "  %-22s %s\n" "CUDAQ_INSTALL_PREFIX:"  "$CUDAQ_INSTALL_PREFIX_VAL ($([[ $CUDAQ_PREFIX_EXISTS == true ]] && echo present || echo missing))"
printf "  %-22s %s (%d files)\n" "CUDAQX_INSTALL_PREFIX:" "$CUDAQX_INSTALL_PREFIX_VAL ($([[ $CUDAQX_PREFIX_EXISTS == true ]] && echo present || echo missing))" "$CUDAQX_INSTALL_FILECOUNT"
printf "  %-22s %s\n" "build/ in repo:"        "$([[ $BUILD_DIR_EXISTS == true ]] && echo present || echo missing)"

printf "\n=== Git submodules ===\n"
echo "$SUBMODULES_JSON" | python3 -c '
import json, sys
mods = json.load(sys.stdin) if sys.stdin else []
if not mods:
    print("  (no submodules)")
for m in mods:
    flag = "init" if m["initialized"] else "NOT init (run: git submodule update --init)"
    print(f"  {m[\"path\"]:30s} {m[\"sha\"][:10]} [{flag}]")
'

if [[ ${#WARNINGS[@]} -gt 0 ]]; then
  printf "\n=== Warnings ===\n"
  for w in "${WARNINGS[@]}"; do
    printf "  - %s\n" "$w"
  done
fi
