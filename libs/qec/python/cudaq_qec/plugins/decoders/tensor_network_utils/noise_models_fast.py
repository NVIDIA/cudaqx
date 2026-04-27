# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Faster torch variant of :class:`NMOptimizer`.

Applies the same tricks the JAX variant uses:

  * snapshot the full tensor-network arrays once; hoist all non-noise /
    non-syndrome tensors out of the per-iteration hot loop.
  * rebuild the ``[1-p, p]`` noise tensors from a single 1-D parameter
    tensor via one ``torch.stack``, instead of walking the quimb TN.
  * expand the cotengra-optimised contraction path into a static
    sequence of :func:`torch.einsum` calls so :func:`torch.compile`
    can fuse the whole thing (forward + backward via AOTAutograd).

The ``torch.compile`` forward is optional.  Even without it, the
per-iteration cost already drops a lot compared to the base
:class:`NMOptimizer` because the Python loop over ``n_errors`` tensors
and the repeated ``quimb`` bookkeeping disappear.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import opt_einsum as oe
import torch

from .noise_models import NMOptimizer

_ASCII_POOL = ("abcdefghijklmnopqrstuvwxyz"
               "ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _remap_eq_to_ascii(eq: str) -> str:
    """Rewrite an einsum equation so every label is in ``[a-zA-Z]``.

    :mod:`opt_einsum` emits non-ASCII unicode labels once the total
    number of distinct indices exceeds 52; :func:`torch.einsum` rejects
    those.  Each pairwise step, however, uses at most a handful of
    labels, so remapping them locally is safe.
    """
    if "->" in eq:
        lhs, rhs = eq.split("->")
    else:
        lhs, rhs = eq, None

    mapping: dict[str, str] = {}
    out_lhs_chars: list[str] = []
    for c in lhs:
        if c == ",":
            out_lhs_chars.append(c)
            continue
        if c not in mapping:
            if len(mapping) >= len(_ASCII_POOL):
                raise ValueError(
                    f"Einsum step '{eq}' has more than {len(_ASCII_POOL)} "
                    "distinct labels; cannot remap to ASCII.")
            mapping[c] = _ASCII_POOL[len(mapping)]
        out_lhs_chars.append(mapping[c])
    out_lhs = "".join(out_lhs_chars)
    if rhs is None:
        return out_lhs
    out_rhs = "".join(mapping.get(c, c) for c in rhs)
    return f"{out_lhs}->{out_rhs}"


class NMOptimizerCompiled(NMOptimizer):
    """Drop-in torch replacement for :class:`NMOptimizer` with a
    hoisted-array, fused forward pass.

    Constructor signature is identical to :class:`NMOptimizer`.  After
    construction, call :meth:`optimize_path` with a quimb/cotengra
    optimiser (e.g. ``cotengra.HyperOptimizer()``) to pin a contraction
    path; the JIT is rebuilt automatically.

    Args:
        compile: whether to wrap the forward in :func:`torch.compile`.
            ``torch.compile`` currently struggles to trace through the
            ``opt_einsum`` Python machinery, so by default we unroll the
            cotengra path into a static sequence of ``torch.einsum``
            calls (see ``execute="unrolled"``) — that is the version
            which actually benefits from ``torch.compile``.
        execute: one of ``"unrolled"`` (default) which generates a
            static list of ``torch.einsum`` steps from the cotengra
            path, or ``"opt_einsum"`` which calls
            ``opt_einsum.contract_expression`` once and invokes it on
            every step.  ``"unrolled"`` is faster and compile-friendly.
        dynamic_syndromes: if ``True`` (default) the generated predict
            function takes syndrome tensors as runtime arguments instead
            of capturing them in its closure.  This makes
            :meth:`update_dataset` / :meth:`_update_data` a pure data
            refresh — the cotengra path, the codegen, and any
            ``torch.compile``-d graph all stay valid as long as shapes
            and dtypes don't change, so online / streaming training
            compiles exactly once.  The cost is a modest reduction in
            how many contraction steps the partial-evaluator can fold
            (steps that touch syndromes become runtime-only).  Set
            ``False`` to restore the original behavior of baking the
            current syndromes into the closure as constants (marginally
            faster for purely offline training, but forces a rebuild +
            retrace on every :meth:`update_dataset` call).
    """

    def __init__(self,
                 *args,
                 compile: bool = False,
                 execute: str = "codegen",
                 compile_mode: str | None = None,
                 dynamic_syndromes: bool = True,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if execute not in ("unrolled", "opt_einsum", "codegen"):
            raise ValueError(f"Invalid execute mode: {execute!r}")
        self._use_torch_compile = compile
        self._execute_mode = execute
        self._torch_compile_mode = compile_mode
        self._dynamic_syndromes = dynamic_syndromes
        self._compiled_predict: Any | None = None
        self._snapshot_arrays_and_eq()

    # -- internal layout caches ------------------------------------------------

    def _snapshot_arrays_and_eq(self) -> None:
        self._eq_batch = self.full_tn.get_equation(
            output_inds=("batch_index", self.logical_obs_inds[0]))
        tensors = list(self.full_tn.tensors)
        self._tensors_ref = tensors

        noise_ids = {id(t) for t in self.noise_model.tensors}
        syndrome_ids = {id(t) for t in self.syndrome_tn.tensors}

        self._noise_pos_for_error: dict[str, int] = {}
        self._syndrome_positions: list[tuple[int, str]] = []
        self._static_positions: list[int] = []

        for i, t in enumerate(tensors):
            if id(t) in noise_ids:
                self._noise_pos_for_error[t.inds[0]] = i
            elif id(t) in syndrome_ids:
                tag = next(tg for tg in t.tags if tg.startswith("SYN_"))
                self._syndrome_positions.append((i, tag))
            else:
                self._static_positions.append(i)

        self._noise_pos_ordered = tuple(
            self._noise_pos_for_error[ei] for ei in self.error_inds)

        torch_dtype = getattr(torch, self._dtype)
        dev = self.torch_device

        def _as_torch(x):
            if isinstance(x, torch.Tensor):
                return x.detach().to(device=dev, dtype=torch_dtype)
            return torch.as_tensor(np.asarray(x), dtype=torch_dtype, device=dev)

        self._static_arrays: dict[int, torch.Tensor] = {
            i: _as_torch(self._tensors_ref[i].data)
            for i in self._static_positions
        }
        self._syndrome_arrays: list[torch.Tensor] = [
            _as_torch(self._tensors_ref[i].data)
            for i, _tag in self._syndrome_positions
        ]

        if self._execute_mode == "opt_einsum":
            shapes = tuple(t.shape for t in tensors)
            self._oe_expr = oe.contract_expression(
                self._eq_batch,
                *shapes,
                optimize=self.path_batch
                if self.path_batch not in (None, "auto") else "auto",
            )
            self._path_steps = None
        else:
            self._oe_expr = None
            # Flatten the cotengra path into a static [(eq, (i,j)), ...] list,
            # remapping any non-ASCII subscripts opt_einsum may have produced
            # (happens once #distinct indices > 52) so torch.einsum accepts them.
            shapes = tuple(t.shape for t in tensors)
            _, info = oe.contract_path(
                self._eq_batch,
                *shapes,
                shapes=True,
                optimize=self.path_batch
                if self.path_batch not in (None, "auto") else "auto",
            )
            self._path_steps = [(_remap_eq_to_ascii(step[2]), tuple(step[0]))
                                for step in info.contraction_list]

        self._compile_predict()

    def _compile_predict(self) -> None:
        static_arrays = self._static_arrays
        syndrome_positions = tuple(p for p, _t in self._syndrome_positions)
        noise_pos_ordered = self._noise_pos_ordered
        n = len(self._tensors_ref)

        if self._execute_mode == "opt_einsum":
            oe_expr = self._oe_expr

            def _predict(
                    noise_probs: torch.Tensor,
                    syndrome_tuple: tuple[torch.Tensor, ...]) -> torch.Tensor:
                noise_stacked = torch.stack((1.0 - noise_probs, noise_probs),
                                            dim=-1)
                arrays: list[torch.Tensor] = [None] * n  # type: ignore
                for pos, arr in static_arrays.items():
                    arrays[pos] = arr
                for pos, arr in zip(syndrome_positions, syndrome_tuple):
                    arrays[pos] = arr
                for k, pos in enumerate(noise_pos_ordered):
                    arrays[pos] = noise_stacked[k]
                out = oe_expr(*arrays, backend="torch")
                return out / out.sum(dim=1, keepdim=True)

        elif self._execute_mode == "unrolled":
            path_steps = self._path_steps

            def _predict(
                    noise_probs: torch.Tensor,
                    syndrome_tuple: tuple[torch.Tensor, ...]) -> torch.Tensor:
                noise_stacked = torch.stack((1.0 - noise_probs, noise_probs),
                                            dim=-1)
                ops: list[torch.Tensor] = [None] * n  # type: ignore
                for pos, arr in static_arrays.items():
                    ops[pos] = arr
                for pos, arr in zip(syndrome_positions, syndrome_tuple):
                    ops[pos] = arr
                for k, pos in enumerate(noise_pos_ordered):
                    ops[pos] = noise_stacked[k]
                for eq_str, idxs in path_steps:
                    sorted_idxs = sorted(idxs, reverse=True)
                    picked = [ops[i] for i in idxs]
                    for i in sorted_idxs:
                        ops.pop(i)
                    ops.append(torch.einsum(eq_str, *picked))
                out = ops[0]
                return out / out.sum(dim=1, keepdim=True)

        else:  # codegen: partial-eval + flat Python with named locals
            syndrome_tensors = list(self._syndrome_arrays)
            codegen_fn = self._build_codegen_predict(
                n,
                static_arrays,
                syndrome_positions,
                noise_pos_ordered,
                self._path_steps,
                syndrome_tensors,
                dynamic_syndromes=self._dynamic_syndromes,
            )
            self._codegen_fn = codegen_fn
            self._codegen_n_folded = getattr(codegen_fn, "_n_folded", 0)
            self._codegen_n_runtime = getattr(codegen_fn, "_n_runtime", 0)

            if self._dynamic_syndromes:

                def _predict(
                        noise_probs: torch.Tensor,
                        syndrome_tuple: tuple[torch.Tensor,
                                              ...]) -> torch.Tensor:
                    return codegen_fn(noise_probs, syndrome_tuple)
            else:

                def _predict(
                        noise_probs: torch.Tensor,
                        syndrome_tuple: tuple[torch.Tensor,
                                              ...]) -> torch.Tensor:
                    # static mode: syndromes are captured in codegen_fn's
                    # closure; the ``syndrome_tuple`` arg is ignored and
                    # would be stale after any :meth:`update_dataset`.
                    return codegen_fn(noise_probs)

        self._predict_fn = _predict
        if self._use_torch_compile:
            try:
                kwargs: dict[str, Any] = {"dynamic": False}
                if self._torch_compile_mode is not None:
                    kwargs["mode"] = self._torch_compile_mode
                self._compiled_predict = torch.compile(_predict, **kwargs)
            except Exception as exc:  # pragma: no cover
                print(f"torch.compile failed ({exc!r}); falling back to eager.")
                self._compiled_predict = _predict
        else:
            self._compiled_predict = _predict

    @staticmethod
    def _build_codegen_predict(n,
                               static_arrays,
                               syndrome_positions,
                               noise_pos_ordered,
                               path_steps,
                               syndrome_tensors,
                               dynamic_syndromes: bool = True):
        """Generate a flat Python function implementing the contraction.

        Performs **partial evaluation** of the contraction path: every
        step whose inputs are all "static" (i.e. don't depend on any
        runtime input) is executed once here and its result is captured
        into the closure of the returned function.  Only the runtime
        steps appear in the emitted function body.

        What counts as "static" depends on ``dynamic_syndromes``:

          * ``dynamic_syndromes=False``: static = code tensors + the
            current syndrome tensors.  The generated signature is
            ``_predict(noise_probs)``.  This folds the most steps but
            forces a full rebuild on every :meth:`update_dataset` call
            (and any ``torch.compile`` wrapper around the result must
            re-trace).
          * ``dynamic_syndromes=True`` (default): static = code tensors
            only.  Syndromes become runtime arguments.  The generated
            signature is ``_predict(noise_probs, syndromes)``.  Folds
            fewer steps but :meth:`update_dataset` is now a pure data
            refresh, so the same ``torch.compile``-d graph serves
            every batch (as long as shapes and dtypes don't change).

        For surface-code d=3 / r=3 the static form folds ~330 steps
        down to ~500 runtime einsums; the dynamic form folds ~210 and
        leaves ~630, a mild ~3-5% per-iter regression but *no*
        recompilation on dataset swap.
        """
        static_positions = sorted(static_arrays.keys())
        noise_pos_set = set(noise_pos_ordered)
        syn_pos_set = set(syndrome_positions)

        # state[pos] = (var_name, is_dynamic, concrete_value_or_None)
        state: list[tuple[str, bool, torch.Tensor | None]] = []
        for pos in range(n):
            if pos in noise_pos_set:
                k = noise_pos_ordered.index(pos)
                state.append((f"_n{k}", True, None))
            elif pos in syn_pos_set:
                sidx = syndrome_positions.index(pos)
                if dynamic_syndromes:
                    state.append((f"_S{sidx}", True, None))
                else:
                    state.append((f"_S{sidx}", False, syndrome_tensors[sidx]))
            else:
                sidx = static_positions.index(pos)
                state.append(
                    (f"_C{sidx}", False, static_arrays[static_positions[sidx]]))

        closure_vars: dict[str, torch.Tensor] = {}
        # capture initial static tensors (only ones actually used)
        # we'll add them lazily as they're referenced
        runtime_lines: list[str] = []
        n_folded = 0

        for step_idx, (eq_str, idxs) in enumerate(path_steps):
            picked = [state[i] for i in idxs]
            for i in sorted(idxs, reverse=True):
                state.pop(i)
            any_dynamic = any(p[1] for p in picked)
            out_name = f"_r{step_idx}"
            if not any_dynamic:
                arrs = [p[2] for p in picked]
                with torch.no_grad():
                    result = torch.einsum(eq_str, *arrs).contiguous()
                static_name = f"_P{step_idx}"
                closure_vars[static_name] = result
                state.append((static_name, False, result))
                n_folded += 1
            else:
                arg_names = [p[0] for p in picked]
                runtime_lines.append(
                    f"    {out_name} = torch.einsum({eq_str!r}, "
                    f"{', '.join(arg_names)})")
                state.append((out_name, True, None))

        assert len(state) == 1
        final_name = state[0][0]
        is_final_dyn = state[0][1]
        if not is_final_dyn:
            # Degenerate case: contraction didn't depend on noise at all.
            static_final = state[0][2]
            closure_vars["_FINAL"] = static_final
            runtime_lines = []
            final_name = "_FINAL"

        # Identify which static tensors are actually referenced
        used_static = set()
        for line in runtime_lines:
            for tok in line.replace(",", " ").split():
                tok = tok.strip("()")
                if tok.startswith("_C") or tok.startswith("_S") \
                        or tok.startswith("_P"):
                    used_static.add(tok)
        if final_name.startswith(("_C", "_S", "_P", "_FINAL")):
            used_static.add(final_name)

        # Bind needed closure values
        for name in list(used_static):
            if name in closure_vars:
                continue
            if name.startswith("_C"):
                sidx = int(name[2:])
                closure_vars[name] = static_arrays[static_positions[sidx]]
            elif name.startswith("_S"):
                if dynamic_syndromes:
                    # dynamic mode: _S* names are function arguments, not
                    # closure constants — nothing to bind here.
                    continue
                sidx = int(name[2:])
                closure_vars[name] = syndrome_tensors[sidx]

        header: list[str] = []
        if dynamic_syndromes:
            header.append("def _predict(noise_probs, syndromes):")
        else:
            header.append("def _predict(noise_probs):")
        if len(runtime_lines) == 0:
            header.append("    _out = _FINAL")
        else:
            header.append("    noise_stacked = torch.stack("
                          "(1.0 - noise_probs, noise_probs), dim=-1)")
            for k in range(len(noise_pos_ordered)):
                header.append(f"    _n{k} = noise_stacked[{k}]")
            if dynamic_syndromes:
                for sidx in range(len(syndrome_positions)):
                    header.append(f"    _S{sidx} = syndromes[{sidx}]")
        body = header + runtime_lines
        body.append(f"    _out = {final_name}")
        body.append("    return _out / _out.sum(dim=1, keepdim=True)")
        source = "\n".join(body)

        ns: dict[str, Any] = {"torch": torch}
        ns.update(closure_vars)
        exec(compile(source, "<nm_compiled_predict>", "exec"), ns)
        fn = ns["_predict"]
        fn._n_folded = n_folded  # type: ignore[attr-defined]
        fn._n_runtime = len(runtime_lines)  # type: ignore[attr-defined]
        return fn

    # -- public API overrides --------------------------------------------------

    def decoder_prediction(self) -> torch.Tensor:
        """Forward pass (compiled if ``compile=True``)."""
        return self._compiled_predict(self._noise_probs,
                                      tuple(self._syndrome_arrays))

    # cross_entropy_ler / logical_error_rate are inherited; they both
    # call ``self.decoder_prediction()`` which is our fused version.

    # -- dataset replacement ---------------------------------------------------

    def _update_data(self,
                     new_syndrome_arrays: torch.Tensor,
                     new_observable_flips: npt.NDArray[Any],
                     enforce_shape: bool = True) -> None:
        super()._update_data(new_syndrome_arrays, new_observable_flips,
                             enforce_shape)
        torch_dtype = getattr(torch, self._dtype)
        dev = self.torch_device
        for k, (pos, _tag) in enumerate(self._syndrome_positions):
            data = self._tensors_ref[pos].data
            if isinstance(data, torch.Tensor):
                self._syndrome_arrays[k] = data.detach().to(device=dev,
                                                            dtype=torch_dtype)
            else:
                self._syndrome_arrays[k] = torch.as_tensor(np.asarray(data),
                                                           dtype=torch_dtype,
                                                           device=dev)
        # In ``dynamic_syndromes`` mode the codegen / opt_einsum expr /
        # torch.compile cache are all parameterised by the *shape* of the
        # syndrome tensors, not their identity — so we can keep everything
        # as-is and the next :meth:`decoder_prediction` call will just use
        # the fresh arrays.  In static mode we must rebuild because the
        # generated function captured the old syndromes in its closure.
        if self._execute_mode == "codegen" and not self._dynamic_syndromes:
            self._snapshot_arrays_and_eq()

    # -- contraction-path optimisation ----------------------------------------

    def optimize_path(self, optimize: Any = None, batch_size: int = -1) -> Any:
        """Compute and cache a contraction path via quimb, rebuild JIT.

        Unlike the parent (which defaults to cuTensorNet's optimiser
        and returns a cuQuantum-only path), this version always uses
        quimb's :meth:`TensorNetwork.contraction_info` so the path is
        compatible with :mod:`opt_einsum` and with manual unrolling.
        """
        info = self.full_tn.contraction_info(
            output_inds=("batch_index", self.logical_obs_inds[0]),
            optimize=optimize if optimize is not None else "auto",
        )
        self.path_batch = info.path
        self.slicing_batch = tuple()
        self._snapshot_arrays_and_eq()
        return info


def make_compiled_step(optimizer: NMOptimizerCompiled,
                       logits: torch.Tensor,
                       torch_optimizer: torch.optim.Optimizer,
                       compile: bool = True):
    """Build a (optionally) ``torch.compile``-d optimisation step.

    Returned callable takes no arguments, runs a full Adam step
    (zero_grad → forward+backward → optimizer.step), and returns the
    scalar loss tensor.  With ``torch.compile`` and AOTAutograd the
    forward and backward are fused into one graph.
    """

    def _step():
        torch_optimizer.zero_grad(set_to_none=True)
        probs = torch.sigmoid(logits)
        preds = optimizer._predict_fn(probs, tuple(optimizer._syndrome_arrays))
        loss = (-torch.log(preds[optimizer.obs_idx_true, 1]).sum() -
                torch.log(preds[optimizer.obs_idx_false, 0]).sum())
        loss.backward()
        torch_optimizer.step()
        return loss

    if compile:
        try:
            return torch.compile(_step, dynamic=False)
        except Exception as exc:  # pragma: no cover
            print(f"torch.compile step failed ({exc!r}); using eager.")
    return _step
