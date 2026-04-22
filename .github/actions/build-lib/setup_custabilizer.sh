#!/bin/sh
# Shared helper: install cuquantum-python (which bundles custabilizer) and
# export CUSTABILIZER_ROOT + LD_LIBRARY_PATH so cmake and the runtime linker
# can find the library.  Must be *sourced*, not executed.

if [ "${BASH_SOURCE[0]:-$0}" = "$0" ] 2>/dev/null; then
  echo "ERROR: this script must be sourced, not executed." >&2
  echo "Usage: . $(basename "$0")" >&2
  exit 1
fi

if [ -z "$CUSTABILIZER_ROOT" ] && [ -x "$(command -v python3)" ]; then
  NVCC_BIN=${CUDACXX:-$(command -v nvcc)}
  CUDA_MAJOR=""
  if [ -n "$NVCC_BIN" ] && [ -x "$NVCC_BIN" ]; then
    CUDA_MAJOR=$("$NVCC_BIN" --version | sed -nE 's/.*release ([0-9]+)\..*/\1/p' | head -n 1)
  fi

  pip install "cuquantum-python-cu${CUDA_MAJOR:-12}>=26.3.0"

  CUSTABILIZER_ROOT=$(python3 -c "
import pathlib, importlib.util
for pkg in ('custabilizer', 'cuquantum'):
    spec = importlib.util.find_spec(pkg)
    if spec and spec.origin:
        root = pathlib.Path(spec.origin).parent
        if (root / 'include' / 'custabilizer.h').exists():
            print(root); raise SystemExit(0)
print('')
")
fi

if [ -n "$CUSTABILIZER_ROOT" ]; then
  echo "Resolved cuStabilizer root: $CUSTABILIZER_ROOT"
  export CUSTABILIZER_ROOT
  export LD_LIBRARY_PATH="$CUSTABILIZER_ROOT/lib:$CUSTABILIZER_ROOT/lib64:${LD_LIBRARY_PATH:-}"
else
  echo "WARNING: could not resolve CUSTABILIZER_ROOT" >&2
fi
