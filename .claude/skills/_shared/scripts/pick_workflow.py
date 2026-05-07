#!/usr/bin/env python3
"""pick_workflow.py — pick the next reference file + commands, deterministically.

Given a user *intent* (e.g. ``vqe``, ``qec-realtime``, ``build-docs``),
plus the JSON snapshots from ``preflight.sh --json`` and
``import_smoke.py --json``, this script returns:

* ``reference``  — which intent-named markdown file to read next
* ``commands``   — concrete commands to run (build, install, serve docs, ...)
* ``verify``     — how to confirm the fix worked
* ``blockers``   — environmental problems that must be fixed first

The mapping is fully rule-based; no LLM call is made. Adding a new intent
or rule means editing the ``INTENT_TABLE`` / ``RULES`` below.

Usage::

    python pick_workflow.py --intent vqe \\
        --preflight /tmp/preflight.json \\
        --imports   /tmp/import_smoke.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Intent → reference map
# ---------------------------------------------------------------------------

# The reference path is repo-root-relative.
INTENT_TABLE: dict[str, dict[str, Any]] = {
    # cuda-qx-build intents
    "build": {
        "skill": "cuda-qx-build",
        "reference": ".claude/skills/cuda-qx-build/references/build.md",
        "commands": [
            "mkdir -p build && cd build",
            "cmake -G Ninja -S .. "
            "-DCUDAQ_DIR=$CUDAQ_INSTALL_PREFIX/lib/cmake/cudaq "
            "-DCMAKE_INSTALL_PREFIX=$CUDAQX_INSTALL_PREFIX "
            "-DCUDAQX_ENABLE_LIBS=all -DCUDAQX_INCLUDE_TESTS=ON "
            "-DCUDAQX_BINDINGS_PYTHON=ON -DCMAKE_BUILD_TYPE=Release",
            "ninja install",
            "ctest",
        ],
        "verify": "python3 -c 'import cudaq_qec, cudaq_solvers; print(cudaq_qec, cudaq_solvers)'",
        "required_features": ["core"],
    },
    "build-wheels": {
        "skill": "cuda-qx-build",
        "reference": ".claude/skills/cuda-qx-build/references/wheels.md",
        "commands": ["scripts/build_wheels.sh"],
        "verify": "ls wheels/ | grep -E 'cudaq_(qec|solvers)_cu1[23].*manylinux'",
        "required_features": [],
    },
    "build-docs": {
        "skill": "cuda-qx-build",
        "reference": ".claude/skills/cuda-qx-build/references/docs.md",
        "commands": [
            "export PYTHONPATH=$CUDAQX_INSTALL_PREFIX:$PYTHONPATH",
            "bash scripts/build_docs.sh",
            "python3 -m http.server --directory $CUDAQX_INSTALL_PREFIX/docs 8000 &",
            "echo 'Open http://localhost:8000 in a browser'",
        ],
        "verify": "test -s $CUDAQX_INSTALL_PREFIX/docs/index.html",
        "required_features": ["core"],
    },
    "debug-import": {
        "skill": "cuda-qx-build",
        "reference": ".claude/skills/cuda-qx-build/references/triage.md",
        "commands": [
            "bash scripts/doctor.sh",
            "pip list | grep -E 'cuda-quantum|cudaq-(qec|solvers)|cuquantum|tensorrt|torch'",
        ],
        "verify": "python3 -c 'import cudaq, cudaq_qec, cudaq_solvers'",
        "required_features": [],
    },

    # cuda-qx-qec intents
    "qec-decode": {
        "skill": "cuda-qx-qec",
        "reference": ".claude/skills/cuda-qx-qec/references/decode.md",
        "commands": [
            "python3 -c \"import cudaq_qec as qec; "
            "code = qec.get_code('steane'); print(code.get_parity_z().shape)\"",
        ],
        "verify": "echo 'cudaq.set_target(\"stim\") for kernel workflows; "
                  "decode against dem.detector_error_matrix for circuit-level'",
        "required_features": ["core"],
    },
    "qec-custom": {
        "skill": "cuda-qx-qec",
        "reference": ".claude/skills/cuda-qx-qec/references/extend.md",
        "commands": [
            "ls libs/qec/include/cudaq/qec/codes/",
            "ls libs/qec/python/cudaq_qec/plugins/decoders/",
        ],
        "verify": "python3 -c 'import cudaq_qec as qec; "
                  "print(qec.get_code(\"my_code\").get_stabilizers())'",
        "required_features": ["core"],
    },
    "qec-realtime": {
        "skill": "cuda-qx-qec",
        "reference": ".claude/skills/cuda-qx-qec/references/realtime.md",
        "commands": [
            "echo 'Phase 1: build DEM; Phase 2: write config.yaml; "
            "Phase 3: configure_decoders_from_file BEFORE cudaq.run; "
            "Phase 4: in-kernel reset/enqueue/get_corrections; finalize at end'",
        ],
        "verify": "CUDAQ_QEC_DEBUG_DECODER=1 python3 your_script.py "
                  "2>&1 | grep 'Initializing realtime decoding library'",
        "required_features": ["core"],
    },
    "qec-debug": {
        "skill": "cuda-qx-qec",
        "reference": ".claude/skills/cuda-qx-qec/references/triage.md",
        "commands": [
            "echo '90% of \"LER looks wrong\" cases are: "
            "(1) didnt slice X-stab half, (2) decoded against code.get_parity instead of dem.detector_error_matrix, "
            "(3) different noise object passed to sample vs DEM helper'",
        ],
        "verify": "python3 your_script.py  # at p=0, LER should be 0",
        "required_features": ["core"],
    },

    # cuda-qx-solvers intents
    "vqe": {
        "skill": "cuda-qx-solvers",
        "reference": ".claude/skills/cuda-qx-solvers/references/vqe.md",
        "commands": [
            "python3 -c \"import cudaq_solvers as solvers; print(solvers.vqe.__doc__)\"",
        ],
        "verify": "energy < hf_energy after the run",
        "required_features": ["core"],
    },
    "qaoa": {
        "skill": "cuda-qx-solvers",
        "reference": ".claude/skills/cuda-qx-solvers/references/qaoa.md",
        "commands": [
            "python3 -c \"import cudaq_solvers as solvers; "
            "print('use cobyla; lbfgs needs gradients QAOA does not auto-wire')\"",
        ],
        "verify": "QAOAResult unpack: optval, optp, config = result",
        "required_features": ["core"],
    },
    "gqe": {
        "skill": "cuda-qx-solvers",
        "reference": ".claude/skills/cuda-qx-solvers/references/gqe.md",
        "commands": [
            "python3 -c \"import torch; print('cuda available:', torch.cuda.is_available())\"",
            "python3 -c \"from cudaq_solvers.gqe_algorithm.gqe import get_default_config; print(get_default_config())\"",
        ],
        "verify": "min_energy decreases across iterations; no sys.exit(1) on import",
        "required_features": ["core", "solvers-gqe"],
    },
    "chemistry": {
        "skill": "cuda-qx-solvers",
        "reference": ".claude/skills/cuda-qx-solvers/references/chemistry.md",
        "commands": [
            "export OMP_NUM_THREADS=1   # reproducible PySCF coefficients",
            "lsof -n -i :8000 || true   # verify no stale cudaq-pyscf server",
        ],
        "verify": "mol.energies['hf_energy'] is finite",
        "required_features": ["core"],
    },
}

# ---------------------------------------------------------------------------
# Cross-cutting rules: detected blockers → recommended fix command
# ---------------------------------------------------------------------------


def _rule_blockers(preflight: dict, imports: dict, intent: str) -> list[dict]:
    blockers: list[dict] = []
    table_entry = INTENT_TABLE.get(intent, {})
    needed_features: list[str] = table_entry.get("required_features", [])

    # ABI mismatch: hard block.
    for w in preflight.get("warnings", []):
        if "ABI mismatch" in w:
            blockers.append({
                "kind": "abi_mismatch",
                "detail": w,
                "fix": "Reinstall both with the same -cuXX suffix. "
                       "E.g. pip install cuda-quantum-cu12 cudaq-qec-cu12 cudaq-solvers-cu12",
            })

    # gfortran missing: blocks any optimizer-using workflow.
    if intent in {"vqe", "qaoa", "chemistry"} and not preflight.get("toolchain", {}).get("gfortran"):
        blockers.append({
            "kind": "gfortran_missing",
            "detail": "gfortran not on PATH; cobyla/lbfgs will crash at runtime",
            "fix": "sudo apt install -y libgfortran5 gfortran libblas-dev",
        })

    # Build intents need cmake/ninja.
    if intent.startswith("build") and intent != "build-docs":
        if not preflight.get("toolchain", {}).get("cmake"):
            blockers.append({
                "kind": "missing_tool",
                "detail": "cmake not on PATH",
                "fix": "Install CMake >= 3.28 (apt install cmake / brew install cmake)",
            })
        if not preflight.get("toolchain", {}).get("ninja"):
            blockers.append({
                "kind": "missing_tool",
                "detail": "ninja not on PATH",
                "fix": "Install Ninja >= 1.10 (apt install ninja-build / pip install ninja)",
            })

    # Wheel build needs docker (rule of thumb).
    if intent == "build-wheels":
        # We can't easily detect docker in preflight (didn't probe it).
        # Add a soft hint instead of a hard blocker.
        blockers.append({
            "kind": "info",
            "detail": "Wheel build runs inside the cudaqx_wheel_builder container",
            "fix": "Ensure `docker` is on PATH; scripts/build_wheels.sh handles the rest",
        })

    # Docs build needs PYTHONPATH to include the install prefix and a built install.
    if intent == "build-docs":
        if not preflight.get("build_state", {}).get("cudaqx_prefix_exists"):
            blockers.append({
                "kind": "no_install",
                "detail": "$CUDAQX_INSTALL_PREFIX does not exist; docs autodoc imports will fail",
                "fix": "Run intent=build first (cmake + ninja install), then retry docs",
            })

    # GQE feature gating.
    if intent == "gqe":
        gqe_status = imports.get("groups", {}).get("solvers-gqe", {}).get("status")
        if gqe_status == "broken":
            broken = [b for b in imports.get("blockers", []) if b["group"] == "solvers-gqe"]
            for b in broken:
                blockers.append({
                    "kind": "missing_extra",
                    "detail": f"{b['name']}: {b['hint']}",
                    "fix": b["fix"],
                })
        # Torch + GPU SM mismatch (best-effort).
        torch_ver = (preflight.get("pip_packages", {}).get("torch") or {}).get("version")
        if torch_ver and not preflight.get("gpu", {}).get("count", 0):
            blockers.append({
                "kind": "info",
                "detail": "torch installed but no GPU visible; GQE will run on CPU and be slow.",
                "fix": "If you have a GPU but nvidia-smi fails, fix the driver before training.",
            })

    # Core imports must work.
    if "core" in needed_features:
        core_status = imports.get("groups", {}).get("core", {}).get("status")
        if core_status == "broken":
            broken = [b for b in imports.get("blockers", []) if b["group"] == "core"]
            for b in broken:
                blockers.append({
                    "kind": "missing_core",
                    "detail": f"{b['name']}: {b['hint']}",
                    "fix": b["fix"],
                })

    return blockers


def _ensure_first_three_lookup(intent: str) -> dict[str, Any] | None:
    if intent not in INTENT_TABLE:
        return None
    entry = INTENT_TABLE[intent]
    return {
        "skill": entry["skill"],
        "reference": entry["reference"],
        "commands": list(entry["commands"]),
        "verify": entry["verify"],
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--intent", required=True,
                   help=f"User intent. Known: {sorted(INTENT_TABLE)}")
    p.add_argument("--preflight", type=Path, default=None,
                   help="Path to preflight.sh --json output. Optional but recommended.")
    p.add_argument("--imports", type=Path, default=None,
                   help="Path to import_smoke.py --json output. Optional but recommended.")
    p.add_argument("--format", choices=["json", "text"], default="json")
    args = p.parse_args()

    if args.intent not in INTENT_TABLE:
        sys.stderr.write(
            f"Unknown intent '{args.intent}'. Known: {sorted(INTENT_TABLE)}\n"
        )
        return 2

    preflight = json.loads(args.preflight.read_text()) if args.preflight else {}
    imports = json.loads(args.imports.read_text()) if args.imports else {}

    pick = _ensure_first_three_lookup(args.intent)
    blockers = _rule_blockers(preflight, imports, args.intent)

    out: dict[str, Any] = {
        "intent": args.intent,
        **pick,
        "blockers": blockers,
        "ok": all(b["kind"] == "info" for b in blockers),
    }

    if args.format == "json":
        print(json.dumps(out, indent=2))
        return 0

    print(f"intent     : {args.intent}")
    print(f"skill      : {out['skill']}")
    print(f"reference  : {out['reference']}")
    print(f"commands   :")
    for c in out["commands"]:
        print(f"  - {c}")
    print(f"verify     : {out['verify']}")
    if blockers:
        print()
        print("BLOCKERS (fix these first):")
        for b in blockers:
            print(f"  [{b['kind']}] {b['detail']}")
            if b["fix"]:
                print(f"      fix: {b['fix']}")
    else:
        print("No blockers detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
