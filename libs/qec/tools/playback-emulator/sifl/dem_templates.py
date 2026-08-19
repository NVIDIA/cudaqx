#!/usr/bin/env python3
"""Per-round-count DEM templates shared by sifl_demo_pymatching.py and
sifl_demo_nv_qldpc.py: for every r in [MIN_ROUNDS, max_rounds], a fresh r-round circuit is
asked for its own DEM directly -- no chunking, no seams, no stitching. multi_round_decoder.cpp
loads those monolithic per-round matrices straight off disk."""
import os

TEMPLATE_ROOT = os.path.join(os.path.dirname(__file__), "templates")
# Kept in step with multi_round_decoder.cpp's kMinRounds. Each round count gets an
# independently built full DEM (no chunk chain), so 1 round is a valid shot.
MIN_ROUNDS = 1

def _detector_bits(circuit):
    """The raw measurement bits behind each detector, read off the circuit itself."""
    n, rows = 0, []
    for inst in circuit.flattened():
        if inst.name in ("M", "MR", "MX", "MZ"): n += len(inst.targets_copy())
        elif inst.name == "DETECTOR": rows.append([n + t.value for t in inst.targets_copy()])
    return rows

def _dem_sparse(circuit):
    """H (row=detector), O (row=observable) and error_rates, read directly off Stim's DEM
    targets. qec.dem_from_stim_text takes this same decomposed-separator walk but lands it
    in a num_detectors x num_faults dense tensor; at max_rounds=160 that tensor is billions
    of cells per round count, and it's rebuilt from scratch max_rounds - 2 times, that's what
    made template building unusably slow. Building the sparse rows ourselves, straight from
    Stim's DemTarget stream, keeps every round count's cost proportional to its (sparse)
    number of error mechanisms instead of num_detectors x num_faults.

    Columns with identical (detector, observable) support are merged as they're collected,
    same as qec.dem_merge_duplicate_columns's default or_combine: decompose_errors=True
    splits some mechanisms into pieces that land back on the same support from different
    circuit locations, and without merging them the fault count -- and every sub-decoder's
    build cost -- is several times larger than it needs to be."""
    dem = circuit.detector_error_model(decompose_errors=True)
    groups = {}
    for inst in dem.flattened():
        if inst.type != "error": continue
        prob = inst.args_copy()[0]
        dets, obs = set(), set()
        def flush():
            if dets or obs:
                key = (tuple(sorted(dets)), tuple(sorted(obs)))
                term = 1.0 - 2.0 * prob
                groups[key] = groups[key] * term if key in groups else term
        for t in inst.targets_copy():
            if t.is_separator():
                flush(); dets.clear(); obs.clear()
                continue
            if t.is_relative_detector_id(): dets ^= {t.val}
            elif t.is_logical_observable_id(): obs ^= {t.val}
        flush()
    H_rows = [[] for _ in range(dem.num_detectors)]
    O_rows = [[] for _ in range(dem.num_observables)]
    rates = []
    for col, ((dets, obs), term) in enumerate(groups.items()):
        rates.append(0.5 * (1.0 - term))
        for d in dets: H_rows[d].append(col)
        for o in obs: O_rows[o].append(col)
    return H_rows, O_rows, rates

def build_templates(gen, out_dir, max_rounds):
    """One full DEM per round count, analysed independently -- no chunk reuse across r."""
    if os.path.exists(f"{out_dir}/.done"): return
    os.makedirs(out_dir, exist_ok=True)
    write = lambda name, rows: open(f"{out_dir}/{name}", "w").write(
        " ".join(str(v) for row in rows for v in list(row) + [-1]))
    for r in range(MIN_ROUNDS, max_rounds + 1):
        circuit = gen(r)
        H_rows, O_rows, rates = _dem_sparse(circuit)
        write(f"r{r}.H", H_rows)
        write(f"r{r}.O", O_rows)
        open(f"{out_dir}/r{r}.rates", "w").write(" ".join(map(str, rates)))
        write(f"r{r}.D", _detector_bits(circuit))
    open(f"{out_dir}/.done", "w").close()
