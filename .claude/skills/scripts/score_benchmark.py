#!/usr/bin/env python3
"""Score CUDA-QX skill benchmarks against agent responses.

Usage:
    python score_benchmark.py --skill solvers --responses responses.json
    python score_benchmark.py --skill qec     --responses responses.json

The responses file is JSON, mapping prompt id ("S1", "S2", ..., "A1", ...)
to the agent's plain-text reply.  Substrings are matched case-insensitively.

This is a coarse proxy for the full rubric (correctness, specificity,
coverage, no-hallucinations).  It only checks `must_include` and
`must_not_include` substrings.  Always pair with a human pass for
correctness and specificity scoring.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

BENCHMARKS: dict[str, dict] = {
    "solvers": {
        "scenarios": [
            {
                "id": "S1",
                "must_include": ["OMP_NUM_THREADS=1", "PySCF", "eigenvalue"],
                "must_not_include": ["CUDA-QX bug", "jordan_wigner bug"],
            },
            {
                "id": "S2",
                "must_include": ["localhost:8000", "lsof", "cudaq-pyscf"],
                "must_not_include": ["reinstall CUDA-Q", "change basis"],
            },
            {
                "id": "S3",
                "must_include": ["nele_cas", "norb_cas", "casci",
                                  "natorb", "MP2"],
                "must_not_include": ["natorb works without MP2"],
            },
            {
                "id": "S4",
                "must_include": ["gradient", "lbfgs", "cobyla"],
                "must_not_include": ["LBFGS bug"],
            },
            {
                "id": "S5",
                "must_include": ["max_iter", "30"],
                "must_not_include": ["max_iterations is correct"],
            },
            {
                "id": "S6",
                "must_include": ["cudaq-solvers[gqe]", "mqpu", "mpiexec",
                                  "cudaq.mpi.initialize", "PMIX_MCA_gds=hash"],
                "must_not_include": ["GQE supports CPU multi-process"],
            },
            {
                "id": "S7",
                "must_include": ["num_qubits", "num_electrons",
                                  "num_orbitals", "uccsd", "uccgsd"],
                "must_not_include": [],
            },
            {
                "id": "S8",
                "must_include": ["scipy.optimize.minimize", "method", "jac",
                                  "tol", "gradient", "optimizer", "verbose",
                                  "shots"],
                "must_not_include": ["any callable optimizer is supported"],
            },
            {
                "id": "S9",
                "must_include": ["networkx", "get_maxcut_hamiltonian",
                                  "get_num_qaoa_parameters", "qaoa"],
                "must_not_include": ["use stim for graph problems"],
            },
            {
                "id": "S10",
                "must_include": ["libgfortran", "system package"],
                "must_not_include": ["macOS", "Windows"],
            },
            {
                "id": "S11",
                "must_include": ["gradient-based", "method"],
                "must_not_include": ["always returns just a float"],
            },
            {
                "id": "S12",
                "must_include": ["cold", "warm"],
                "must_not_include": ["auto-detect"],
            },
        ],
        "activation": [
            {"id": "A1", "should_activate": True},
            {"id": "A2", "should_activate": True},
            {"id": "A3", "should_activate": True},
            {"id": "A4", "should_activate": False},
            {"id": "A5", "should_activate": False},
            {"id": "A6", "should_activate": False},
            {"id": "A7", "should_activate": True},
            {"id": "A8", "should_activate": True},
            {"id": "A9", "should_activate": False},
            {"id": "A10", "should_activate": True},
        ],
        "activation_marker": "cuda-qx-solvers",
    },
    "qec": {
        "scenarios": [
            {
                "id": "S1",
                "must_include": ["C-order"],
                "must_not_include": ["F-order is supported"],
            },
            {
                "id": "S2",
                "must_include": ["cuquantum-python", "26.03.0"],
                "must_not_include": ["uninstall cudaq_qec"],
            },
            {
                "id": "S3",
                "must_include": ['backend="gpu"', "RuntimeError", "auto",
                                  "cpu"],
                "must_not_include": ["silently falls back when gpu requested"],
            },
            {
                "id": "S4",
                "must_include": ["NumPy"],
                "must_not_include": ["yes, supported"],
            },
            {
                "id": "S5",
                "must_include": ["(40, 6)", "numShots * numRounds"],
                "must_not_include": ["(10, 4, 6)"],
            },
            {
                "id": "S6",
                "must_include": ["stabilizers", "pauli_observables",
                                  "operation_encodings",
                                  "get_num_data_qubits",
                                  "get_num_ancilla_qubits"],
                "must_not_include": ["only stabilizers are required"],
            },
            {
                "id": "S7",
                "must_include": ["sliding_window", "window_size",
                                  "step_size", "num_syndromes_per_round",
                                  "inner_decoder_name", "error_rate_vec"],
                "must_not_include": [
                    "sliding_window does not require an inner decoder"
                ],
            },
            {
                "id": "S8",
                "must_include": ["use_osd", "bp_method", "proc_float",
                                  "error_rate_vec", "max_iterations",
                                  "default to None"],
                "must_not_include": ["all fields are required"],
            },
            {
                "id": "S9",
                "must_include": ["tensor_network_decoder", "quimb",
                                  "cuquantum-python"],
                "must_not_include": ["quimb is included by default"],
            },
            {
                "id": "S10",
                "must_include": ["LicenseRef-NVIDIA-Proprietary",
                                  "libs/qec/pyproject.toml"],
                "must_not_include": ["yes, both are Apache 2.0"],
            },
            {
                "id": "S11",
                "must_include": ["prep0", "prep1", "prepp", "prepm",
                                  "stabilizer_round", "cx", "cz", "h", "s"],
                "must_not_include": ["swap", "measure_x"],
            },
            {
                "id": "S12",
                "must_include": ["X errors", "Z errors",
                                  "DetectorErrorModel",
                                  "detector_error_matrix"],
                "must_not_include": ["they are aliases"],
            },
        ],
        "activation": [
            {"id": "A1", "should_activate": True},
            {"id": "A2", "should_activate": True},
            {"id": "A3", "should_activate": True},
            {"id": "A4", "should_activate": True},
            {"id": "A5", "should_activate": False},
            {"id": "A6", "should_activate": False},
            {"id": "A7", "should_activate": False},
            {"id": "A8", "should_activate": True},
            {"id": "A9", "should_activate": False},
            {"id": "A10", "should_activate": True},
        ],
        "activation_marker": "cuda-qx-qec",
    },
}


@dataclass
class ScenarioScore:
    id: str
    coverage: int = 0
    coverage_max: int = 0
    purity: int = 0
    purity_max: int = 0
    missing: list[str] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)


def score_scenario(spec: dict, response: str) -> ScenarioScore:
    text = response.lower()
    must = [s.lower() for s in spec["must_include"]]
    must_not = [s.lower() for s in spec["must_not_include"]]
    s = ScenarioScore(id=spec["id"])
    s.coverage_max = max(len(must), 1)
    s.purity_max = max(len(must_not), 1)

    hit = sum(1 for m in must if m in text)
    s.coverage = hit
    s.missing = [m for m in spec["must_include"] if m.lower() not in text]

    bad = sum(1 for m in must_not if m in text)
    s.purity = s.purity_max - bad
    s.forbidden = [m for m in spec["must_not_include"] if m.lower() in text]
    return s


def score_activation(spec: dict, response: str, marker: str) -> bool:
    activated = marker.lower() in response.lower()
    return activated == spec["should_activate"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skill", choices=sorted(BENCHMARKS), required=True)
    p.add_argument("--responses", type=Path, required=True,
                   help="JSON file mapping prompt id -> agent response")
    p.add_argument("--report", type=Path, default=None,
                   help="Optional JSON path to dump full scores")
    args = p.parse_args()

    bench = BENCHMARKS[args.skill]
    responses: dict[str, str] = json.loads(args.responses.read_text())

    scenarios: list[ScenarioScore] = []
    for spec in bench["scenarios"]:
        resp = responses.get(spec["id"], "")
        scenarios.append(score_scenario(spec, resp))

    activation_correct = 0
    activation_total = len(bench["activation"])
    for spec in bench["activation"]:
        resp = responses.get(spec["id"], "")
        if score_activation(spec, resp, bench["activation_marker"]):
            activation_correct += 1

    coverage = sum(s.coverage for s in scenarios)
    coverage_max = sum(s.coverage_max for s in scenarios)
    purity = sum(s.purity for s in scenarios)
    purity_max = sum(s.purity_max for s in scenarios)

    print(f"Skill: {args.skill}")
    print(f"  Coverage:   {coverage} / {coverage_max}")
    print(f"  Purity:     {purity} / {purity_max}")
    print(f"  Activation: {activation_correct} / {activation_total}")
    total = coverage + purity + activation_correct
    total_max = coverage_max + purity_max + activation_total
    print(f"  Total:      {total} / {total_max}")

    for s in scenarios:
        if s.missing or s.forbidden:
            print()
            print(f"  [{s.id}] coverage {s.coverage}/{s.coverage_max} "
                  f"purity {s.purity}/{s.purity_max}")
            if s.missing:
                print(f"    missing: {s.missing}")
            if s.forbidden:
                print(f"    forbidden hit: {s.forbidden}")

    if args.report:
        report = {
            "skill": args.skill,
            "coverage": coverage,
            "coverage_max": coverage_max,
            "purity": purity,
            "purity_max": purity_max,
            "activation": activation_correct,
            "activation_max": activation_total,
            "total": total,
            "total_max": total_max,
            "scenarios": [s.__dict__ for s in scenarios],
        }
        args.report.write_text(json.dumps(report, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
