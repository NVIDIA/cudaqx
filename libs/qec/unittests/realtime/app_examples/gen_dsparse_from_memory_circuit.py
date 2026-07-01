# ============================================================================ #
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Generate D_sparse.txt for the trt+Ising path of surface_code-4-yaml.

The Ising decoding project (https://github.com/NVIDIA/Ising-Decoding) provides a
pretrained surface-code predecoder, and its generate_test_data.py exports a
bundle of H_csr.bin/O_csr.bin/priors.bin in Ising detector order. Because
surface_code-4-yaml reads the cudaqx live measurement buffer, D_sparse.txt
bridges the two: one row per Ising detector, whose entries are cudaqx live-buffer
measurement indices, so each row reproduces one cudaqx detector bit in Ising's
detector row order. The cudaqx live measurement stream then feeds Ising's
H/O/priors.

Recipe (see surface_code-4-yaml-test.sh header for the full flow):
  1. In the Ising repo: generate_test_data.py --distance D --n-rounds T \
       --basis Z --code-rotation XV   -> H_csr.bin/O_csr.bin/priors.bin bundle
  2. Run surface_code-4-yaml --save_dem ... once; it prints cnot_schedX_flat /
     cnot_schedZ_flat lines. Save that stdout to a sched.txt.
  3. python gen_dsparse_from_memory_circuit.py D T Z XV sched.txt \
       <bundle>/D_sparse.txt --ising-repo /path/to/ising/code
  4. Run surface_code-4-yaml --ising_bundle <bundle> ...

Geometry: cudaqx surface_code orientation XV corresponds to Ising code_rotation
"XV" (first_bulk X, rotated_type V) under an identity data- and X-ancilla
mapping; only the Z-ancillas are permuted. This script derives that Z-ancilla
permutation by matching cudaqx Z-stabilizer supports to Ising hz rows, then
translates Ising's detector->measurement map into the cudaqx buffer order.

Usage:
  gen_dsparse_from_memory_circuit.py <distance> <n_rounds> <basis> \
      <code_rotation> <sched.txt> <out D_sparse.txt> [--ising-repo PATH]

Positional arguments:
  distance       Surface code distance D.
  n_rounds       Number of rounds T (Ising n_rounds; counts prep + final).
  basis          Measurement basis (Z).
  code_rotation  Ising code_rotation string, e.g. XV (first_bulk=X, rotated=V).
  sched.txt      File containing the app's printed cnot_schedX_flat /
                 cnot_schedZ_flat lines.
  out            Output D_sparse.txt path.

Options:
  --ising-repo PATH  Path to the Ising repo's `code` directory (the one
                     containing qec/surface_code). Defaults to
                     /work/github/ising/code.
"""
import argparse
import sys
import types
from pathlib import Path


def parse_args(argv):
    ap = argparse.ArgumentParser(
        description="Generate D_sparse.txt for surface_code-4-yaml's trt+Ising "
        "path.")
    ap.add_argument("distance", type=int, help="surface code distance D")
    ap.add_argument("n_rounds", type=int, help="number of rounds T")
    ap.add_argument("basis", help="measurement basis (Z)")
    ap.add_argument("code_rotation", help="Ising code_rotation string, e.g. XV")
    ap.add_argument("sched",
                    help="file with the app's printed cnot_schedX/Z_flat lines")
    ap.add_argument("out", help="output D_sparse.txt path")
    ap.add_argument("--ising-repo",
                    default="/work/github/ising/code",
                    help="path to the Ising repo's `code` directory "
                    "(contains qec/surface_code); default %(default)s")
    return ap.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    d = args.distance
    NR = args.n_rounds
    BASIS = args.basis.upper()
    ROT = args.code_rotation.upper()
    SCHED = args.sched
    OUT = args.out
    ISING = args.ising_repo

    # --- cudaqx supports from the app's printed schedule ---
    def parse_sched(line):
        nums = [int(x) for x in line.split(":", 1)[1].replace(",", " ").split()]
        sup = {}
        for i in range(0, len(nums), 2):
            sup.setdefault(nums[i], set()).add(nums[i + 1])
        return [frozenset(sup[s]) for s in sorted(sup)]

    sX = sZ = None
    for ln in open(SCHED):
        if ln.startswith("cnot_schedX_flat:"):
            sX = parse_sched(ln)
        elif ln.startswith("cnot_schedZ_flat:"):
            sZ = parse_sched(ln)
    assert sX and sZ, "sched.txt must contain cnot_schedX_flat / cnot_schedZ_flat"
    nx, nz = len(sX), len(sZ)

    # --- Ising geometry + DEM (m2d) ---
    p = Path(ISING)
    if not (p / "qec" / "surface_code").is_dir():
        sys.exit(f"--ising-repo '{ISING}' does not contain qec/surface_code")
    sys.path.insert(0, str(p))
    _sc = types.ModuleType("qec.surface_code")
    _sc.__path__ = [str(p / "qec" / "surface_code")]
    _sc.__package__ = "qec.surface_code"
    sys.modules.setdefault("qec.surface_code", _sc)
    from qec.surface_code.memory_circuit import SurfaceCode, MemoryCircuit
    from qec.noise_model import NoiseModel

    sc = SurfaceCode(d, first_bulk_syndrome_type=ROT[0], rotated_type=ROT[1])
    ihz = [
        frozenset(i
                  for i in range(sc.hz.shape[1])
                  if sc.hz[r, i] == 1)
        for r in range(sc.hz.shape[0])
    ]
    ihx = [
        frozenset(i
                  for i in range(sc.hx.shape[1])
                  if sc.hx[r, i] == 1)
        for r in range(sc.hx.shape[0])
    ]

    def perm(cqx, ising, lbl):
        pi = [None] * len(cqx)
        used = set()
        for c, supp in enumerate(cqx):
            m = [
                r for r, isup in enumerate(ising)
                if isup == supp and r not in used
            ]
            assert len(
                m) == 1, f"{lbl} ancilla {c} support {sorted(supp)} matches {m}"
            pi[c] = m[0]
            used.add(m[0])
        return pi

    pi_z = perm(sZ, ihz, "Z")  # cudaqx Z-anc c -> ising Z row
    pi_x = perm(sX, ihx, "X")
    assert pi_x == list(range(nx)), "expected identity X-ancilla mapping"

    # inv_z: ising Z position -> cudaqx Z position
    inv_z = [None] * nz
    for c, q in enumerate(pi_z):
        inv_z[q] = c

    # buffer translate: ising buffer index -> cudaqx buffer index
    # per round r: X block [48r..], Z block [48r+nx..]; data [nanc..]
    per_round = nx + nz
    nanc = per_round * NR

    def translate(m):
        if m >= nanc:
            return m  # data: identity
        r = m // per_round
        off = m % per_round
        if off < nx:
            return per_round * r + off  # X identity
        return per_round * r + nx + inv_z[off - nx]  # Z permuted

    # --- Ising DEM detector->measurement map ---
    DEFAULT = {
        "p_prep_X": 0.002,
        "p_prep_Z": 0.002,
        "p_meas_X": 0.002,
        "p_meas_Z": 0.002,
        "p_idle_cnot_X": 0.001,
        "p_idle_cnot_Y": 0.001,
        "p_idle_cnot_Z": 0.001,
        "p_idle_spam_X": 0.001996,
        "p_idle_spam_Y": 0.001996,
        "p_idle_spam_Z": 0.001996,
        "p_cnot_IX": 0.0002,
        "p_cnot_IY": 0.0002,
        "p_cnot_IZ": 0.0002,
        "p_cnot_XI": 0.0002,
        "p_cnot_XX": 0.0002,
        "p_cnot_XY": 0.0002,
        "p_cnot_XZ": 0.0002,
        "p_cnot_YI": 0.0002,
        "p_cnot_YX": 0.0002,
        "p_cnot_YY": 0.0002,
        "p_cnot_YZ": 0.0002,
        "p_cnot_ZI": 0.0002,
        "p_cnot_ZX": 0.0002,
        "p_cnot_ZY": 0.0002,
        "p_cnot_ZZ": 0.0002
    }
    nm = NoiseModel(**DEFAULT)
    pp = float(nm.get_max_probability())
    circ = MemoryCircuit(distance=d,
                         idle_error=pp,
                         sqgate_error=pp,
                         tqgate_error=pp,
                         spam_error=(2.0 / 3.0) * pp,
                         n_rounds=NR,
                         basis=BASIS,
                         code_rotation=ROT,
                         noise_model=nm,
                         add_boundary_detectors=True)
    circ.set_error_rates()
    stim_c = circ.stim_circuit
    det_rows = []
    meas = 0
    for inst in stim_c.flattened():
        if inst.name in ("M", "MR", "MX", "MZ", "MRX", "MRZ"):
            meas += len(inst.targets_copy())
        elif inst.name == "DETECTOR":
            det_rows.append(sorted(meas + t.value for t in inst.targets_copy()))

    # D_sparse[j] = translate(ising_m2d[j])
    flat = []
    for row in det_rows:
        for m in row:
            flat.append(translate(m))
        flat.append(-1)
    with open(OUT, "w") as f:
        f.write(" ".join(str(x) for x in flat) + "\n")
    print(f"wrote {OUT}: {len(det_rows)} detectors, nmeas={meas}, "
          f"Z-anc perm nontrivial={pi_z != list(range(nz))}")


if __name__ == "__main__":
    main(sys.argv[1:])
