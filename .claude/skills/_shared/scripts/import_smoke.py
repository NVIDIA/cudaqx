#!/usr/bin/env python3
"""import_smoke.py — try every import a CUDA-QX agent might need, classify failures.

Designed to run *fast* and never crash: each import lives in its own
subprocess and the worst case is "module missing". The output (especially
in --json mode) is consumed by `pick_workflow.py` to deterministically
recommend an install command before the agent burns tokens guessing.

Probes:

  Core:        cudaq, cudaq_qec, cudaq_solvers
  QEC extras:  tensor_network_decoder (needs quimb + cuquantum-python),
               trt_decoder (needs tensorrt),
               nv-qldpc-decoder availability
  Solvers:     gqe (needs torch + lightning + mpi4py + transformers)

Each probe returns one of: ``ok``, ``missing``, ``import_error``,
``runtime_error``. Errors are classified into a small set of likely causes
with a suggested fix string (so the agent doesn't have to translate the
traceback into action).

Usage::

    import_smoke.py            # human-readable
    import_smoke.py --json     # machine-readable
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from typing import Any

PROBES: list[dict[str, Any]] = [
    # name              code to run                                            fix string when missing                       feature group
    {
        "name": "cudaq",
        "code": "import cudaq; print(cudaq.__file__)",
        "fix": "pip install cuda-quantum-cu12  # or -cu13",
        "group": "core"
    },
    {
        "name": "cudaq_qec",
        "code": "import cudaq_qec; print(cudaq_qec.__file__)",
        "fix": "pip install cudaq-qec        # picks cu12/cu13",
        "group": "core"
    },
    {
        "name": "cudaq_solvers",
        "code": "import cudaq_solvers; print(cudaq_solvers.__file__)",
        "fix": "pip install cudaq-solvers",
        "group": "core"
    },

    # QEC extras
    {
        "name": "quimb",
        "code": "import quimb",
        "fix": "pip install 'cudaq-qec[tensor-network-decoder]'",
        "group": "qec-tensor-network"
    },
    {
        "name": "cuquantum",
        "code": "import cuquantum",
        "fix": "pip install 'cuquantum-python-cu12>=26.03.0'    # or -cu13",
        "group": "qec-tensor-network"
    },
    {
        "name":
            "tensorrt",
        "code":
            "import tensorrt",
        "fix":
            "pip install 'cudaq-qec[trt-decoder]'             # or pip install tensorrt-cu12",
        "group":
            "qec-trt"
    },
    {
        "name": "nv-qldpc-decoder",
        "code":
            "import cudaq_qec as qec; import numpy as np; "
            "qec.get_decoder('nv-qldpc-decoder', np.zeros((2,2), dtype=np.uint8))",
        "fix": "Closed-source plugin from NVIDIA; see libs/qec/README.md",
        "group": "qec-nv-qldpc"
    },

    # Solvers extras
    {
        "name": "torch",
        "code": "import torch; print(torch.cuda.is_available())",
        "fix": "pip install 'cudaq-solvers[gqe]'",
        "group": "solvers-gqe"
    },
    {
        "name": "lightning",
        "code": "import lightning",
        "fix": "pip install 'cudaq-solvers[gqe]'",
        "group": "solvers-gqe"
    },
    {
        "name": "mpi4py",
        "code": "import mpi4py",
        "fix": "pip install 'cudaq-solvers[gqe]'",
        "group": "solvers-gqe"
    },
    {
        "name": "transformers",
        "code": "import transformers",
        "fix": "pip install 'cudaq-solvers[gqe]'",
        "group": "solvers-gqe"
    },

    # Chemistry
    {
        "name": "openfermionpyscf",
        "code": "import openfermionpyscf",
        "fix": "pip install openfermion openfermionpyscf",
        "group": "solvers-chemistry"
    },
]


def _classify(stderr: str) -> tuple[str, str]:
    """Return (status, hint) for an import failure."""
    s = stderr.lower()
    if "modulenotfounderror" in s or "no module named" in s:
        return "missing", "Module not installed (or different python interpreter)."
    if "libcustabilizer" in s:
        return "import_error", "ABI mismatch on libcustabilizer; install matching cuquantum-python (>=26.03)."
    if "libcudart" in s:
        return "import_error", "Missing CUDA runtime; install matching nvidia-cuda-runtime-cuXX."
    if "libgfortran" in s:
        return "import_error", "Missing libgfortran; apt install libgfortran5 (or gfortran)."
    if "decoder" in s and "not found" in s:
        return "runtime_error", "Decoder plugin not registered; ensure plugin install / configure_decoders_from_file()."
    if "importerror" in s or "load library" in s or ".so: cannot open" in s:
        return "import_error", "Shared library missing — check LD_LIBRARY_PATH and CUDA suffix consistency."
    if "cuda" in s and "device" in s and "available" in s:
        return "runtime_error", "CUDA runtime can't reach a GPU (driver vs runtime mismatch?)."
    return "runtime_error", "See stderr above for details."


def _run(code: str, timeout: float = 25.0) -> tuple[int, str, str]:
    """Run a snippet in a subprocess; return (exit_code, stdout, stderr)."""
    try:
        r = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout after {timeout}s"
    except FileNotFoundError:
        return 127, "", "python interpreter not found"


def _probe(spec: dict[str, Any]) -> dict[str, Any]:
    rc, out, err = _run(spec["code"])
    if rc == 0:
        return {
            "name": spec["name"],
            "group": spec["group"],
            "status": "ok",
            "stdout": out.strip(),
            "fix": "",
            "hint": "",
        }
    status, hint = _classify(err)
    return {
        "name": spec["name"],
        "group": spec["group"],
        "status": status,
        "stdout": out.strip(),
        "stderr": err.strip().splitlines()[-1] if err.strip() else "",
        "fix": spec.get("fix", ""),
        "hint": hint,
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json",
                   action="store_true",
                   help="Emit JSON instead of text.")
    p.add_argument("--feature",
                   action="append",
                   default=None,
                   help="Probe only the named feature group (repeatable). "
                   "Groups: core, qec-tensor-network, qec-trt, qec-nv-qldpc, "
                   "solvers-gqe, solvers-chemistry.")
    args = p.parse_args()

    probes = PROBES
    if args.feature:
        wanted = set(args.feature)
        probes = [s for s in PROBES if s["group"] in wanted]

    results = [_probe(s) for s in probes]

    # Compute group-level rollup ("ok" iff every probe in group is ok).
    groups: dict[str, dict[str, Any]] = {}
    for r in results:
        g = groups.setdefault(r["group"], {"status": "ok", "members": []})
        g["members"].append(r["name"])
        if r["status"] != "ok":
            g["status"] = "broken"

    summary = {
        "python_executable":
            sys.executable,
        "pip_available":
            shutil.which("pip") is not None,
        "groups":
            groups,
        "probes":
            results,
        "blockers": [{
            "name": r["name"],
            "group": r["group"],
            "status": r["status"],
            "fix": r["fix"],
            "hint": r["hint"]
        } for r in results if r["status"] != "ok"],
    }

    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    print(f"python  : {summary['python_executable']}")
    print(f"pip     : {'available' if summary['pip_available'] else 'MISSING'}")
    print()
    print(f"{'group':<22}{'status':<10}members")
    for g, info in groups.items():
        print(f"  {g:<20}{info['status']:<10}{','.join(info['members'])}")
    print()
    if summary["blockers"]:
        print("Blockers:")
        for b in summary["blockers"]:
            print(f"  [{b['status']}] {b['name']:<22} {b['hint']}")
            if b["fix"]:
                print(f"      fix: {b['fix']}")
    else:
        print("All probes green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
