#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
#
# This source code and the accompanying materials are made available under
# the terms of the Apache License 2.0 which accompanies this distribution.
# ============================================================================ #
"""NMOptimizer d=5/r=5 memory benchmark for large surface-code DEMs.

This script is meant for the d=5/r=5 scalability discussion where the DEM has
roughly the same shape as Nico's case: 120 detectors and 1677 error mechanisms.
It can either generate a Stim rotated-memory surface-code DEM or load an exact
``.dem`` file.

The script has two modes:
  * path diagnostics: construct NMOptimizer and print reduced-path largest
    intermediate / opt cost for one or more batch sizes.
  * optional execution: run a tiny Adam step to expose actual forward/backward
    memory failures. Use ``--run-step`` for this because the hard DEM can OOM.

Example:
    python3 scripts/benchmark_nmoptimizer_d5r5_nico_dem.py \
      --device cuda:0 --dtype float64 --execute all \
      --batch-sizes 1 2 4 1000

With an exact DEM file:
    python3 scripts/benchmark_nmoptimizer_d5r5_nico_dem.py \
      --dem-path surface_code_bZ_d5_r05_center_5_5.dem \
      --batch-sizes 1 2 4 1000
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import stim
except ImportError as exc:  # pragma: no cover
    raise SystemExit("stim is required for this benchmark") from exc

try:
    from beliefmatching.belief_matching import detector_error_model_to_check_matrices
except ImportError as exc:  # pragma: no cover
    raise SystemExit("beliefmatching is required to convert DEMs") from exc

from cudaq_qec import NMOptimizer, make_training_step

EXECUTE_MODES = ("codegen", "unrolled", "opt_einsum", "cutensornet")


@dataclass
class Problem:
    H: np.ndarray
    logical: np.ndarray
    priors: np.ndarray
    circuit: stim.Circuit | None
    dem: stim.DetectorErrorModel


@contextlib.contextmanager
def timed(label: str):
    start = time.perf_counter()
    print(
        f"[START] {label}: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        flush=True)
    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - start
        print(
            f"[FAILED] {label}: {datetime.now(timezone.utc).isoformat(timespec='seconds')} "
            f"(elapsed {elapsed:.2f}s)",
            flush=True)
        raise
    else:
        elapsed = time.perf_counter() - start
        print(
            f"[END] {label}: {datetime.now(timezone.utc).isoformat(timespec='seconds')} "
            f"(elapsed {elapsed:.2f}s)",
            flush=True)


def parse_detector_error_model(dem: stim.DetectorErrorModel):
    matrices = detector_error_model_to_check_matrices(dem)
    H = np.zeros(matrices.check_matrix.shape, dtype=np.float64)
    matrices.check_matrix.astype(np.float64).toarray(out=H)
    logical = np.zeros(matrices.observables_matrix.shape, dtype=np.float64)
    matrices.observables_matrix.astype(np.float64).toarray(out=logical)
    priors = np.array([float(p) for p in matrices.priors], dtype=np.float64)
    return H, logical, priors


def make_surface_code_circuit(args: argparse.Namespace) -> stim.Circuit:
    noise_kwargs = {
        "after_clifford_depolarization": args.after_clifford_p,
        "after_reset_flip_probability": args.after_reset_p,
        "before_measure_flip_probability": args.before_measure_p,
        "before_round_data_depolarization": args.before_round_data_p,
    }
    noise_kwargs = {
        k: v for k, v in noise_kwargs.items() if v is not None and v > 0.0
    }
    return stim.Circuit.generated(
        args.code,
        distance=args.distance,
        rounds=args.rounds,
        **noise_kwargs,
    )


def build_problem(args: argparse.Namespace) -> Problem:
    if args.dem_path is not None:
        dem_text = Path(args.dem_path).read_text(encoding="utf-8")
        dem = stim.DetectorErrorModel(dem_text)
        circuit = None
    else:
        circuit = make_surface_code_circuit(args)
        dem = circuit.detector_error_model(decompose_errors=True)
    H, logical, priors = parse_detector_error_model(dem)
    return Problem(H=H,
                   logical=logical,
                   priors=priors,
                   circuit=circuit,
                   dem=dem)


def make_circuit_sampler(circuit: stim.Circuit, seed: int):
    try:
        return circuit.compile_detector_sampler(seed=seed)
    except TypeError:
        return circuit.compile_detector_sampler()


def sample_from_circuit(circuit: stim.Circuit, shots: int, seed: int):
    sampler = make_circuit_sampler(circuit, seed)
    dets, obs = sampler.sample(shots, separate_observables=True)
    return dets.astype(np.float64), obs.reshape(-1).astype(bool)


def sample_from_dem(dem: stim.DetectorErrorModel, shots: int, seed: int):
    try:
        sampler = dem.compile_sampler(seed=seed)
    except TypeError:
        sampler = dem.compile_sampler()

    # Stim versions have varied here. Try the more descriptive API first,
    # then fall back to tuple shape inspection.
    try:
        dets, obs = sampler.sample(shots, separate_observables=True)
        return dets.astype(np.float64), obs.reshape(-1).astype(bool)
    except TypeError:
        result = sampler.sample(shots, bit_packed=False)

    if isinstance(result, tuple):
        if len(result) >= 2:
            return result[0].astype(
                np.float64), result[1].reshape(-1).astype(bool)
    raise RuntimeError(
        "Could not sample observables from DEM sampler for this Stim version. "
        "Use generated-circuit mode or pass --diagnostics-only.")


def sample_problem(problem: Problem, shots: int, seed: int):
    if problem.circuit is not None:
        return sample_from_circuit(problem.circuit, shots, seed)
    return sample_from_dem(problem.dem, shots, seed)


def path_largest_intermediate(info: Any) -> float:
    value = getattr(info, "largest_intermediate", None)
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return float("nan")


def path_opt_cost(info: Any) -> float:
    value = getattr(info, "opt_cost", None)
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return float("nan")


def einsum_rank(eq: str) -> int:
    if "->" not in eq:
        return 0
    lhs, rhs = eq.split("->", 1)
    terms = [term for term in lhs.split(",") if term]
    terms.append(rhs)
    return max((len(term) for term in terms), default=0)


def path_max_einsum_rank(info: Any) -> int:
    ranks = [
        einsum_rank(step[2]) for step in getattr(info, "contraction_list", ())
    ]
    return max(ranks, default=0)


def fmt_float(value: float) -> str:
    if math.isnan(value):
        return "n/a"
    return f"{value:.6e}"


def element_size(dtype: str) -> int:
    return torch.tensor([], dtype=getattr(torch, dtype)).element_size()


def clear_cuda(device: str):
    gc.collect()
    if "cuda" in device and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(torch.device(device))


def peak_mem_mib(device: str) -> str:
    if "cuda" not in device or not torch.cuda.is_available():
        return "n/a"
    torch.cuda.synchronize(torch.device(device))
    return f"{torch.cuda.max_memory_allocated(torch.device(device)) / 2**20:.1f}"


def make_initial_priors(priors: np.ndarray, mode: str) -> np.ndarray:
    if mode == "true":
        return priors.copy()
    if mode == "uniform":
        return np.full_like(priors, priors.mean())
    raise ValueError(f"unknown init mode: {mode}")


def probs_to_logits(probs: np.ndarray, dtype: str) -> torch.Tensor:
    eps = 1e-6 if dtype == "float32" else 1e-12
    clipped = np.clip(probs, eps, 1.0 - eps)
    return torch.as_tensor(np.log(clipped / (1.0 - clipped)),
                           dtype=getattr(torch, dtype))


def construct_optimizer(problem: Problem, syn: np.ndarray, flips: np.ndarray,
                        priors: np.ndarray, args: argparse.Namespace,
                        execute: str) -> NMOptimizer:
    return NMOptimizer(problem.H,
                       problem.logical,
                       priors.tolist(),
                       syn,
                       flips,
                       dtype=args.dtype,
                       device=args.device,
                       execute=execute,
                       compile=args.compile)


def run_case(problem: Problem,
             batch_size: int,
             execute: str,
             args: argparse.Namespace,
             fixed_path: Any | None = None,
             fixed_path_source: str = "") -> tuple[dict[str, str], Any | None]:
    label = f"batch={batch_size}, execute={execute}"
    row = {
        "batch": str(batch_size),
        "construct_batch": str(batch_size),
        "execute": execute,
        "path_source": fixed_path_source if fixed_path is not None else "self",
        "status": "FAIL",
        "construct_s": "n/a",
        "largest_intermediate": "n/a",
        "intermediate_gib": "n/a",
        "opt_cost": "n/a",
        "max_einsum_rank": "n/a",
        "step_s": "n/a",
        "peak_mem_mib": "n/a",
        "error": "",
    }
    init_priors = make_initial_priors(problem.priors, args.init)
    syn, flips = sample_problem(problem, batch_size, args.seed + batch_size)

    selected_path = None
    try:
        clear_cuda(args.device)
        with timed(f"Construct NMOptimizer ({label})"):
            t0 = time.perf_counter()
            if args.run_step:
                logits = probs_to_logits(init_priors, args.dtype)
                logits = logits.to(torch.device(args.device))
                logits.requires_grad_(True)
                torch_opt = torch.optim.Adam([logits], lr=args.lr)
                progress_every = max(1, args.progress_every)

                def _progress(done: int, total: int, elapsed: float) -> None:
                    if done == 1 or done == total or done % progress_every == 0:
                        rate = done / elapsed if elapsed > 0 else float("nan")
                        print(
                            f"  batch slice {done}/{total}: "
                            f"elapsed={elapsed:.1f}s, rate={rate:.3f}/s, "
                            f"peak_mem_mib={peak_mem_mib(args.device)}",
                            flush=True)

                opt, step_fn = make_training_step(
                    problem.H,
                    problem.logical,
                    init_priors.tolist(),
                    syn,
                    flips,
                    logits,
                    torch_opt,
                    batch_slicing="auto",
                    batch_slice_size=args.batch_slice_size,
                    progress_callback=_progress,
                    dtype=args.dtype,
                    device=args.device,
                    execute=execute,
                    compile=args.compile)
            else:
                opt = construct_optimizer(problem, syn, flips, init_priors,
                                          args, execute)
                step_fn = None
            if fixed_path is not None:
                print(f"Reusing reduced path from {fixed_path_source}",
                      flush=True)
                opt.optimize_path(fixed_path)
            row["construct_s"] = f"{time.perf_counter() - t0:.3f}"

        construct_batch = int(getattr(opt, "_batch_size", batch_size))
        row["construct_batch"] = str(construct_batch)
        run_label = label
        if construct_batch != batch_size:
            run_label += f", construct_batch={construct_batch}"

        info = getattr(opt, "_reduced_info", None)
        selected_path = getattr(opt, "path_batch", None)
        largest = path_largest_intermediate(info)
        cost = path_opt_cost(info)
        row["largest_intermediate"] = fmt_float(largest)
        row["opt_cost"] = fmt_float(cost)
        row["max_einsum_rank"] = str(path_max_einsum_rank(info))
        if not math.isnan(largest):
            gib = largest * element_size(args.dtype) / 2**30
            row["intermediate_gib"] = f"{gib:.3f}"
        print(
            "Path info: largest_intermediate="
            f"{row['largest_intermediate']}, estimated_one_tensor_GiB="
            f"{row['intermediate_gib']}, opt_cost={row['opt_cost']}",
            flush=True)

        if args.run_step:
            with timed(f"Adam step ({run_label})"):
                t0 = time.perf_counter()
                loss = step_fn()
                if "cuda" in args.device and torch.cuda.is_available():
                    torch.cuda.synchronize(torch.device(args.device))
                row["step_s"] = f"{time.perf_counter() - t0:.3f}"
            print(f"Loss={float(loss.detach().cpu()):.6e}", flush=True)

        row["peak_mem_mib"] = peak_mem_mib(args.device)
        row["status"] = "OK"
        return row, selected_path
    except Exception as exc:  # intentionally broad: this is an OOM probe.
        row["error"] = repr(exc)
        with contextlib.suppress(Exception):
            row["peak_mem_mib"] = peak_mem_mib(args.device)
        print(f"FAILED CASE {label}: {exc!r}", flush=True)
        if args.stop_on_failure:
            raise
        return row, selected_path
    finally:
        with contextlib.suppress(Exception):
            del opt  # type: ignore[name-defined]
        gc.collect()
        if "cuda" in args.device and torch.cuda.is_available():
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()


def parse_execute(value: str) -> list[str]:
    if value == "all":
        return list(EXECUTE_MODES)
    modes = [v.strip() for v in value.split(",") if v.strip()]
    bad = sorted(set(modes) - set(EXECUTE_MODES))
    if bad:
        raise ValueError(f"invalid execute mode(s): {bad}")
    return modes


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dem-path",
                        type=str,
                        default=None,
                        help="Optional exact .dem file to benchmark.")
    parser.add_argument("--code", default="surface_code:rotated_memory_z")
    parser.add_argument("--distance", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument(
        "--noise-p",
        type=float,
        default=0.005,
        help="Default probability for all enabled Stim noise mechanisms.")
    parser.add_argument("--after-clifford-p", type=float, default=None)
    parser.add_argument("--after-reset-p", type=float, default=None)
    parser.add_argument("--before-measure-p", type=float, default=None)
    parser.add_argument("--before-round-data-p", type=float, default=None)
    parser.add_argument("--batch-sizes",
                        type=int,
                        nargs="+",
                        default=[1, 2, 4, 1000])
    parser.add_argument(
        "--execute",
        default="all",
        help="all, codegen, unrolled, opt_einsum, cutensornet, or comma list")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype",
                        choices=["float32", "float64"],
                        default="float64")
    parser.add_argument("--init",
                        choices=["uniform", "true"],
                        default="uniform")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--run-step",
        action="store_true",
        help="Actually execute one forward/backward Adam step. May OOM.")
    parser.add_argument(
        "--batch-slice-size",
        type=int,
        default=None,
        help=("Construct with this smaller batch size and stream the "
              "requested batch through that fixed path."))
    parser.add_argument("--microbatch-size",
                        dest="batch_slice_size",
                        type=int,
                        help=argparse.SUPPRESS)
    parser.add_argument("--progress-every",
                        type=int,
                        default=10,
                        help="Print batch-slice progress every N chunks.")
    parser.add_argument(
        "--reuse-first-path",
        action="store_true",
        help=("For each requested batch size, reuse the first execute mode's "
              "reduced contraction path for later execute modes."))
    parser.add_argument("--lr",
                        type=float,
                        default=1e-2,
                        help="Adam learning rate used with --run-step.")
    parser.add_argument("--compile",
                        action="store_true",
                        help="Pass compile=True to NMOptimizer.")
    parser.add_argument("--stop-on-failure", action="store_true")
    return parser


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()
    if args.after_clifford_p is None:
        args.after_clifford_p = args.noise_p
    if args.after_reset_p is None:
        args.after_reset_p = args.noise_p
    if args.before_measure_p is None:
        args.before_measure_p = args.noise_p
    if args.before_round_data_p is None:
        args.before_round_data_p = args.noise_p

    if args.batch_slice_size is not None and args.batch_slice_size <= 0:
        parser.error("--batch-slice-size must be a positive integer.")

    execute_modes = parse_execute(args.execute)
    print(
        "Config: "
        f"source={'dem-file' if args.dem_path else 'stim-generated'}, "
        f"d={args.distance}, r={args.rounds}, device={args.device}, "
        f"dtype={args.dtype}, execute={execute_modes}, "
        f"batch_sizes={args.batch_sizes}, run_step={args.run_step}, "
        f"reuse_first_path={args.reuse_first_path}, "
        f"batch_slice_size={args.batch_slice_size}",
        flush=True)

    with timed("Build/load DEM"):
        problem = build_problem(args)
    print(
        f"DEM shape: H={problem.H.shape}, logical={problem.logical.shape}, "
        f"priors={problem.priors.shape}",
        flush=True)
    print(
        f"True priors: mean={problem.priors.mean():.6e}, "
        f"min={problem.priors.min():.6e}, max={problem.priors.max():.6e}",
        flush=True)
    if problem.H.shape != (120, 1677):
        print(
            "WARNING: DEM shape is not Nico's reported d=5/r=5 size "
            "H=(120, 1677). If you have the exact DEM, rerun with --dem-path.",
            flush=True)

    rows = []
    for batch_size in args.batch_sizes:
        fixed_path = None
        fixed_path_source = ""
        for execute in execute_modes:
            row, selected_path = run_case(problem,
                                          batch_size,
                                          execute,
                                          args,
                                          fixed_path=fixed_path,
                                          fixed_path_source=fixed_path_source)
            rows.append(row)
            if (args.reuse_first_path and fixed_path is None and
                    row["status"] == "OK" and selected_path is not None):
                fixed_path = selected_path
                fixed_path_source = f"execute={execute}"

    print("\n## NMOptimizer d=5/r=5 Large-DEM Memory Matrix")
    headers = [
        "batch", "construct_batch", "execute", "path_source", "status",
        "construct_s", "largest_intermediate", "intermediate_gib", "opt_cost",
        "max_einsum_rank", "step_s", "peak_mem_mib", "error"
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        print("| " + " | ".join(row[h].replace("\n", " ") for h in headers) +
              " |")

    return 0 if all(r["status"] == "OK" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
