# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Tests for :class:`NMOptimizer` and friends.

Each test is parameterised over ``device in ("cpu", "cuda")``; CUDA
cases are skipped when no GPU is available.
"""

import sys
import warnings

import numpy as np
import pytest
import cudaq_qec as qec

torch = pytest.importorskip(
    "torch", reason="torch not installed; skipping TN noise-learning tests")

if sys.version_info >= (3, 11):
    from cudaq_qec.plugins.decoders.tensor_network_utils.nm_optimizer import (
        NMOptimizer,
        make_compiled_step,
        remap_eq_to_ascii,
    )

pytestmark = pytest.mark.skipif(sys.version_info < (3, 11),
                                reason="Requires Python >= 3.11")


def _gpu_available() -> bool:
    return torch.cuda.is_available()


def _device_params():
    """``device`` parametrize values; cuda is skipped when unavailable."""
    out = ["cpu"]
    if _gpu_available():
        out.append("cuda")
    return out


_EXECUTE_MODES = ("codegen", "unrolled", "opt_einsum")

# -- fixtures / helpers -------------------------------------------------------


def _simple_repetition_code():
    """[[3,1]] repetition-code-like fixture with a single logical observable."""
    H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float64)
    logical = np.array([[1, 0, 1]], dtype=np.float64)
    priors = [0.1, 0.2, 0.3]
    return H, logical, priors


def _nondegenerate_code():
    """3-error code where ``P(l|s)`` genuinely depends on the priors.

    ``L = [1,1,1]`` is **not** in the GF(2) row span of ``H``, so every
    syndrome admits error patterns of both logical values and the
    gradient w.r.t. the noise priors is non-trivial.  Use this whenever
    a test needs to exercise the autograd path; the
    :func:`_simple_repetition_code` fixture has ``L`` in ``row(H)``,
    which makes ``P(l|s)`` deterministic and zeroes the gradient.
    """
    H = np.array([[1, 1, 0], [1, 0, 1]], dtype=np.float64)
    logical = np.array([[1, 1, 1]], dtype=np.float64)
    return H, logical


def _random_code(rng: np.random.Generator,
                 n_checks: int = 5,
                 n_errors: int = 8):
    H = rng.integers(0, 2, size=(n_checks, n_errors)).astype(np.float64)
    logical = rng.integers(0, 2, size=(1, n_errors)).astype(np.float64)
    priors = rng.uniform(0.02, 0.2, size=n_errors).astype(np.float64).tolist()
    return H, logical, priors


def _sample_synthetic_dataset(H: np.ndarray, logical_obs: np.ndarray,
                              priors: list[float], num_shots: int,
                              rng: np.random.Generator):
    """Sample errors from a Bernoulli noise model and derive (syn, flips)."""
    n_errors = H.shape[1]
    p = np.asarray(priors, dtype=np.float64)
    errors = (rng.random((num_shots, n_errors)) < p).astype(np.uint8)
    syndromes = (errors @ H.T) % 2
    flips = (errors @ logical_obs.T).reshape(-1) % 2
    return syndromes.astype(np.float64), flips.astype(bool)


def _make_opt(H, logical, priors, syn, flips, **kwargs):
    """Thin wrapper that forwards kwargs; collapses 7-line constructions."""
    return NMOptimizer(H, logical, priors, syn, flips, **kwargs)


def _naive_cross_entropy(opt: "NMOptimizer") -> torch.Tensor:
    """Reference cross-entropy: predict, then ``-log p`` per shot.

    Mirrors the pre-fusion implementation; used to verify the codegen
    loss in :func:`test_fused_loss_matches_naive`.
    """
    preds = opt.decoder_prediction()
    obs_t = opt.obs_idx_true
    obs_f = opt.obs_idx_false
    return (-torch.log(preds[obs_t, 1]).sum() -
            torch.log(preds[obs_f, 0]).sum())


# -- construction ------------------------------------------------------------


@pytest.mark.parametrize("device", _device_params())
def test_construction_basic(device):
    H, logical, priors = _simple_repetition_code()
    syn, flips = _sample_synthetic_dataset(H,
                                           logical,
                                           priors,
                                           num_shots=8,
                                           rng=np.random.default_rng(0))
    opt = _make_opt(H, logical, priors, syn, flips, device=device)
    assert opt._batch_size == 8
    assert opt._noise_probs.requires_grad
    assert len(opt.noise_params) == 1
    assert opt.noise_params[0] is opt._noise_probs
    np_probs = opt._noise_probs.detach().cpu().numpy()
    assert np_probs.shape == (3,)
    np.testing.assert_allclose(np_probs, priors, atol=1e-6)
    assert np.all((np_probs >= 0.0) & (np_probs <= 1.0))


@pytest.mark.parametrize("device", _device_params())
def test_invalid_execute_mode_rejected(device):
    H, logical, priors = _simple_repetition_code()
    syn, flips = _sample_synthetic_dataset(H,
                                           logical,
                                           priors,
                                           num_shots=4,
                                           rng=np.random.default_rng(1))
    with pytest.raises(ValueError, match="Invalid execute mode"):
        _make_opt(H,
                  logical,
                  priors,
                  syn,
                  flips,
                  device=device,
                  execute="bogus")


@pytest.mark.parametrize("device", _device_params())
def test_invalid_dtype_rejected(device):
    """Unsupported dtypes must be rejected at the constructor boundary,
    before any contraction setup runs."""
    H, logical, priors = _simple_repetition_code()
    syn, flips = _sample_synthetic_dataset(H,
                                           logical,
                                           priors,
                                           num_shots=4,
                                           rng=np.random.default_rng(2))
    with pytest.raises(ValueError, match="Invalid dtype"):
        _make_opt(H,
                  logical,
                  priors,
                  syn,
                  flips,
                  device=device,
                  dtype="float16")


# -- forward pass / gradient -------------------------------------------------


@pytest.mark.parametrize("device", _device_params())
def test_decoder_prediction_shape_and_range(device):
    H, logical, priors = _simple_repetition_code()
    syn, flips = _sample_synthetic_dataset(H,
                                           logical,
                                           priors,
                                           num_shots=16,
                                           rng=np.random.default_rng(2))
    opt = _make_opt(H, logical, priors, syn, flips, device=device)
    pred = opt.decoder_prediction()
    assert pred.shape == (16, 2)
    s = pred.sum(dim=1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-5)
    assert torch.all(pred >= -1e-6) and torch.all(pred <= 1.0 + 1e-6)


@pytest.mark.parametrize("device", _device_params())
@pytest.mark.parametrize("execute", _EXECUTE_MODES)
def test_gradient_flows(device, execute):
    """Backward populates a non-zero gradient on ``_noise_probs``.

    Uses :func:`_nondegenerate_code` plus mismatched init priors so the
    loss surface has a non-trivial gradient.  Parametrized over every
    ``execute`` mode so unrolled and opt_einsum paths can't silently
    regress on the autograd graph.
    """
    rng = np.random.default_rng(3)
    H, logical = _nondegenerate_code()
    true_priors = [0.1, 0.15, 0.25]
    init_priors = [0.5, 0.5, 0.5]
    syn, flips = _sample_synthetic_dataset(H,
                                           logical,
                                           true_priors,
                                           num_shots=64,
                                           rng=rng)
    opt = _make_opt(H,
                    logical,
                    init_priors,
                    syn,
                    flips,
                    device=device,
                    dtype="float64",
                    execute=execute)
    opt._noise_probs.grad = None
    loss = opt.cross_entropy_loss()
    loss.backward()
    assert opt._noise_probs.grad is not None
    assert torch.any(opt._noise_probs.grad != 0.0)


# -- fused-loss correctness --------------------------------------------------


@pytest.mark.parametrize("device", _device_params())
@pytest.mark.parametrize("execute", _EXECUTE_MODES)
def test_fused_loss_matches_naive(device, execute):
    """``cross_entropy_loss`` == predict + manual CE in every execute mode.

    Codegen mode fuses the CE reduction into the contraction graph;
    unrolled/opt_einsum wrap ``predict_fn``.  All three must agree with
    the naive reference up to fp64 round-off.
    """
    rng = np.random.default_rng(11)
    H, logical = _nondegenerate_code()
    init_priors = [0.2, 0.3, 0.4]
    syn, flips = _sample_synthetic_dataset(H,
                                           logical, [0.1, 0.15, 0.25],
                                           num_shots=48,
                                           rng=rng)
    opt = _make_opt(H,
                    logical,
                    init_priors,
                    syn,
                    flips,
                    device=device,
                    dtype="float64",
                    execute=execute)
    with torch.no_grad():
        fused = opt.cross_entropy_loss()
        naive = _naive_cross_entropy(opt)
    assert torch.isfinite(fused) and torch.isfinite(naive)
    assert torch.allclose(fused, naive, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("device", _device_params())
def test_fused_loss_matches_naive_static_codegen(device):
    """Static codegen (``dynamic_syndromes=False``) numerical correctness."""
    rng = np.random.default_rng(13)
    H, logical = _nondegenerate_code()
    init_priors = [0.2, 0.3, 0.4]
    syn, flips = _sample_synthetic_dataset(H,
                                           logical, [0.1, 0.15, 0.25],
                                           num_shots=40,
                                           rng=rng)
    opt = _make_opt(H,
                    logical,
                    init_priors,
                    syn,
                    flips,
                    device=device,
                    dtype="float64",
                    execute="codegen",
                    dynamic_syndromes=False)
    with torch.no_grad():
        fused = opt.cross_entropy_loss()
        naive = _naive_cross_entropy(opt)
    assert torch.isfinite(fused) and torch.isfinite(naive)
    assert torch.allclose(fused, naive, atol=1e-8, rtol=1e-8)
    loss_probs = opt.loss_fn(from_logits=False)
    loss_logits = opt.loss_fn(from_logits=True)
    probs = opt._noise_probs.detach().clone().requires_grad_(False)
    logits = torch.log(probs / (1.0 - probs))
    with torch.no_grad():
        v_probs = loss_probs(probs, ())
        v_logits = loss_logits(logits, ())
    assert torch.allclose(v_probs, fused, atol=1e-8, rtol=1e-8)
    assert torch.allclose(v_logits, fused, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("device", _device_params())
@pytest.mark.parametrize("execute", _EXECUTE_MODES)
def test_loss_fn_from_logits_and_probs(device, execute):
    """``loss_fn(from_logits=True)`` matches ``loss_fn(from_logits=False) o sigmoid``,
    and both agree with ``cross_entropy_loss`` on the optimiser's own probs.

    Parametrized over execute modes so the logit-vs-probs equivalence is
    validated on every supported backend.
    """
    rng = np.random.default_rng(12)
    H, logical = _nondegenerate_code()
    init_priors = [0.2, 0.3, 0.4]
    syn, flips = _sample_synthetic_dataset(H,
                                           logical, [0.1, 0.15, 0.25],
                                           num_shots=32,
                                           rng=rng)
    opt = _make_opt(H,
                    logical,
                    init_priors,
                    syn,
                    flips,
                    device=device,
                    dtype="float64",
                    execute=execute)
    loss_probs = opt.loss_fn(from_logits=False)
    loss_logits = opt.loss_fn(from_logits=True)
    probs = opt._noise_probs.detach().clone().requires_grad_(False)
    logits = torch.log(probs / (1.0 - probs))
    with torch.no_grad():
        v_probs = loss_probs(probs, opt._syndrome_tuple)
        v_logits = loss_logits(logits, opt._syndrome_tuple)
        v_self = opt.cross_entropy_loss()
    assert torch.allclose(v_probs, v_logits, atol=1e-8, rtol=1e-8)
    assert torch.allclose(v_probs, v_self, atol=1e-8, rtol=1e-8)


# -- numerical guards --------------------------------------------------------


@pytest.mark.parametrize("device", _device_params())
def test_small_priors_finite_loss(device):
    """Realistic small priors (``1e-3``) pass through unclamped and yield finite loss."""
    H, logical, _ = _simple_repetition_code()
    small_priors = [1e-3, 0.5, 1.0 - 1e-3]
    syn, flips = _sample_synthetic_dataset(H,
                                           logical, [0.1, 0.2, 0.3],
                                           num_shots=8,
                                           rng=np.random.default_rng(4))
    for dtype in ("float32", "float64"):
        opt = _make_opt(H,
                        logical,
                        small_priors,
                        syn,
                        flips,
                        device=device,
                        dtype=dtype)
        assert torch.all(torch.isfinite(opt._noise_probs))
        loss = opt.cross_entropy_loss()
        assert torch.isfinite(loss), (
            f"non-finite loss at dtype={dtype}: {loss}")


@pytest.mark.parametrize("device", _device_params())
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_boundary_priors_clamped_with_warning(device, dtype):
    """Priors at the (0, 1) boundary are clamped into ``[eps, 1 - eps]``
    with a single :class:`UserWarning`; loss stays finite downstream."""
    H, logical, _ = _simple_repetition_code()
    boundary_priors = [0.0, 0.5, 1.0]
    syn, flips = _sample_synthetic_dataset(H,
                                           logical, [0.1, 0.2, 0.3],
                                           num_shots=8,
                                           rng=np.random.default_rng(20))
    # Mirrors the dtype-eps table inside NMOptimizer.  Hardcoded so the
    # test pins the boundary contract independently of the implementation.
    eps = 1e-12 if dtype == "float64" else 1e-6
    with pytest.warns(UserWarning, match=r"Clamped \d+/\d+"):
        opt = _make_opt(H,
                        logical,
                        boundary_priors,
                        syn,
                        flips,
                        device=device,
                        dtype=dtype)
    probs = opt._noise_probs.detach().cpu().numpy()
    assert np.all(probs >= eps - 1e-9)
    assert np.all(probs <= 1.0 - eps + 1e-9)
    np.testing.assert_allclose(probs[1], 0.5, atol=1e-6)
    loss = opt.cross_entropy_loss()
    assert torch.isfinite(loss)


@pytest.mark.skipif(not _gpu_available(), reason="CUDA not available")
def test_boundary_priors_finite_with_tf32_matmul_enabled():
    """Regression: global TF32 matmul must not make boundary-prior loss NaN."""
    old_precision = torch.get_float32_matmul_precision()
    try:
        # Mirror solver imports that enable TF32 process-wide during full
        # pytest collection; the QEC test must be self-contained.
        torch.set_float32_matmul_precision("high")
        H, logical, _ = _simple_repetition_code()
        boundary_priors = [0.0, 0.5, 1.0]
        syn, flips = _sample_synthetic_dataset(H,
                                               logical, [0.1, 0.2, 0.3],
                                               num_shots=8,
                                               rng=np.random.default_rng(20))
        with pytest.warns(UserWarning, match=r"Clamped \d+/\d+"):
            opt = _make_opt(H,
                            logical,
                            boundary_priors,
                            syn,
                            flips,
                            device="cuda",
                            dtype="float32")

        loss = opt.cross_entropy_loss()
        assert torch.isfinite(loss)
    finally:
        torch.set_float32_matmul_precision(old_precision)


@pytest.mark.parametrize("device", _device_params())
def test_non_finite_priors_raise(device):
    """Non-finite priors are caller bugs, not stability concerns - raise."""
    H, logical, _ = _simple_repetition_code()
    syn, flips = _sample_synthetic_dataset(H,
                                           logical, [0.1, 0.2, 0.3],
                                           num_shots=8,
                                           rng=np.random.default_rng(21))
    for bad_priors in ([0.1, np.nan, 0.3], [0.1, np.inf, 0.3]):
        with pytest.raises(ValueError, match="All priors must be finite"):
            _make_opt(H, logical, bad_priors, syn, flips, device=device)


def test_in_range_priors_no_warning():
    """In-range priors must pass through with zero warnings."""
    H, logical, priors = _simple_repetition_code()
    syn, flips = _sample_synthetic_dataset(H,
                                           logical,
                                           priors,
                                           num_shots=4,
                                           rng=np.random.default_rng(22))
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _make_opt(H, logical, priors, syn, flips, device="cpu")


def test_non_1d_noise_model_rejected():
    """A 2-D ``noise_model`` is rejected at the constructor boundary."""
    H, logical, priors = _simple_repetition_code()
    syn, flips = _sample_synthetic_dataset(H,
                                           logical,
                                           priors,
                                           num_shots=4,
                                           rng=np.random.default_rng(23))
    bad = np.full((2, 3), 0.5)
    with pytest.raises(ValueError, match="must be 1-D"):
        _make_opt(H, logical, bad, syn, flips, device="cpu")


# -- current_syndrome_args ---------------------------------------------------


@pytest.mark.parametrize("device", _device_params())
def test_current_syndrome_args_dynamic_returns_live_tuple(device):
    """Dynamic mode: returns the live syndrome tuple."""
    rng = np.random.default_rng(101)
    H, logical, priors = _simple_repetition_code()
    syn, flips = _sample_synthetic_dataset(H,
                                           logical,
                                           priors,
                                           num_shots=12,
                                           rng=rng)
    opt = _make_opt(H,
                    logical,
                    priors,
                    syn,
                    flips,
                    device=device,
                    execute="codegen",
                    dynamic_syndromes=True)
    args = opt.current_syndrome_args()
    assert args is opt._syndrome_tuple
    assert len(args) > 0
    assert torch.isfinite(
        opt.loss_fn(from_logits=False)(opt.noise_params[0], args))


@pytest.mark.parametrize("device", _device_params())
def test_current_syndrome_args_static_returns_empty(device):
    """Static codegen mode: returns ``()`` (syndromes are closure-baked)."""
    rng = np.random.default_rng(102)
    H, logical, priors = _simple_repetition_code()
    syn, flips = _sample_synthetic_dataset(H,
                                           logical,
                                           priors,
                                           num_shots=12,
                                           rng=rng)
    opt = _make_opt(H,
                    logical,
                    priors,
                    syn,
                    flips,
                    device=device,
                    execute="codegen",
                    dynamic_syndromes=False)
    assert opt.current_syndrome_args() == ()
    assert torch.isfinite(
        opt.loss_fn(from_logits=False)(opt.noise_params[0], ()))


# -- dataset swap ------------------------------------------------------------


@pytest.mark.parametrize("device", _device_params())
def test_update_dataset_dynamic_keeps_predict_fn(device):
    """Dynamic mode: predict function identity unchanged across swaps."""
    rng = np.random.default_rng(5)
    H, logical, priors = _simple_repetition_code()
    syn1, flips1 = _sample_synthetic_dataset(H,
                                             logical,
                                             priors,
                                             num_shots=10,
                                             rng=rng)
    opt = _make_opt(H,
                    logical,
                    priors,
                    syn1,
                    flips1,
                    device=device,
                    dynamic_syndromes=True)
    fn_before = opt._predict_fn
    syn2, flips2 = _sample_synthetic_dataset(H,
                                             logical,
                                             priors,
                                             num_shots=10,
                                             rng=rng)
    opt.update_dataset(syn2, flips2)
    assert opt._predict_fn is fn_before


@pytest.mark.parametrize("device", _device_params())
def test_update_dataset_static_rebuilds_predict_fn(device):
    """Static mode: predict function is re-codegened on swap."""
    rng = np.random.default_rng(6)
    H, logical, priors = _simple_repetition_code()
    syn1, flips1 = _sample_synthetic_dataset(H,
                                             logical,
                                             priors,
                                             num_shots=10,
                                             rng=rng)
    opt = _make_opt(H,
                    logical,
                    priors,
                    syn1,
                    flips1,
                    device=device,
                    dynamic_syndromes=False)
    fn_before = opt._predict_fn
    syn2, flips2 = _sample_synthetic_dataset(H,
                                             logical,
                                             priors,
                                             num_shots=10,
                                             rng=rng)
    opt.update_dataset(syn2, flips2)
    assert opt._predict_fn is not fn_before


@pytest.mark.parametrize("device", _device_params())
@pytest.mark.parametrize("dynamic_syndromes", [True, False])
def test_update_dataset_shape_change_rebuilds_and_decodes(
        device, dynamic_syndromes):
    """A different batch size triggers a full rebuild; loss stays finite
    and matches a freshly constructed optimiser to fp64 precision."""
    rng = np.random.default_rng(77)
    H, logical = _nondegenerate_code()
    init_priors = [0.1, 0.15, 0.25]
    syn1, flips1 = _sample_synthetic_dataset(H,
                                             logical,
                                             init_priors,
                                             num_shots=16,
                                             rng=rng)
    opt = _make_opt(H,
                    logical,
                    init_priors,
                    syn1,
                    flips1,
                    device=device,
                    dtype="float64",
                    dynamic_syndromes=dynamic_syndromes)
    syn2, flips2 = _sample_synthetic_dataset(H,
                                             logical,
                                             init_priors,
                                             num_shots=33,
                                             rng=rng)
    opt.update_dataset(syn2, flips2, enforce_shape=False)
    assert opt._batch_size == 33
    pred = opt.decoder_prediction()
    assert pred.shape == (33, 2)
    loss = opt.cross_entropy_loss()
    assert torch.isfinite(loss)

    ref = _make_opt(H,
                    logical,
                    init_priors,
                    syn2,
                    flips2,
                    device=device,
                    dtype="float64",
                    dynamic_syndromes=dynamic_syndromes)
    with torch.no_grad():
        ref_loss = ref.cross_entropy_loss()
    assert torch.allclose(loss, ref_loss, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("device", _device_params())
def test_update_dataset_enforce_shape_mismatch_raises(device):
    """``enforce_shape=True`` (default) must reject a syndrome batch whose
    per-tensor shape differs from the construction-time batch.  The
    permissive path is already covered by
    :func:`test_update_dataset_shape_change_rebuilds_and_decodes`."""
    rng = np.random.default_rng(78)
    H, logical = _nondegenerate_code()
    init_priors = [0.1, 0.15, 0.25]
    syn1, flips1 = _sample_synthetic_dataset(H,
                                             logical,
                                             init_priors,
                                             num_shots=16,
                                             rng=rng)
    opt = _make_opt(H,
                    logical,
                    init_priors,
                    syn1,
                    flips1,
                    device=device,
                    dtype="float64")
    syn2, flips2 = _sample_synthetic_dataset(H,
                                             logical,
                                             init_priors,
                                             num_shots=33,
                                             rng=rng)
    with pytest.raises(AssertionError, match="Shape mismatch"):
        opt.update_dataset(syn2, flips2)


# -- optimize_path -----------------------------------------------------------


@pytest.mark.parametrize("device", _device_params())
def test_optimize_path_default_preserves_forward(device):
    """``optimize_path()`` with the default ``"auto"`` finder rebuilds the
    JIT but does not change the numerical forward output."""
    rng = np.random.default_rng(88)
    H, logical = _nondegenerate_code()
    priors = [0.1, 0.15, 0.25]
    syn, flips = _sample_synthetic_dataset(H,
                                           logical,
                                           priors,
                                           num_shots=24,
                                           rng=rng)
    opt = _make_opt(H,
                    logical,
                    priors,
                    syn,
                    flips,
                    device=device,
                    dtype="float64")
    with torch.no_grad():
        before = opt.decoder_prediction().detach().cpu().numpy()
    opt.optimize_path()
    with torch.no_grad():
        after = opt.decoder_prediction().detach().cpu().numpy()
    np.testing.assert_allclose(before, after, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("device", _device_params())
def test_optimize_path_with_cotengra(device):
    """A user-supplied ``cotengra.HyperOptimizer`` is accepted by
    ``optimize_path`` and the forward stays numerically consistent."""
    ctg = pytest.importorskip("cotengra")
    rng = np.random.default_rng(89)
    H, logical = _nondegenerate_code()
    priors = [0.1, 0.15, 0.25]
    syn, flips = _sample_synthetic_dataset(H,
                                           logical,
                                           priors,
                                           num_shots=24,
                                           rng=rng)
    opt = _make_opt(H,
                    logical,
                    priors,
                    syn,
                    flips,
                    device=device,
                    dtype="float64")
    with torch.no_grad():
        before = opt.decoder_prediction().detach().cpu().numpy()
    info = opt.optimize_path(
        optimize=ctg.HyperOptimizer(max_repeats=2, parallel=False))
    assert info is not None
    with torch.no_grad():
        after = opt.decoder_prediction().detach().cpu().numpy()
    np.testing.assert_allclose(before, after, atol=1e-10, rtol=1e-10)


# -- remap_eq_to_ascii -------------------------------------------------------


def test_remap_eq_to_ascii_simple():
    eq = "ab,bc->ac"
    out = remap_eq_to_ascii(eq)
    # ASCII input is returned unchanged via the ``isascii()`` fast path.
    assert out == "ab,bc->ac"


def test_remap_eq_to_ascii_unicode_labels():
    """Synthetic equation with non-ASCII labels is remapped to a-zA-Z."""
    eq = "\u0391\u0392,\u0392\u0393->\u0391\u0393"  # greek letters
    out = remap_eq_to_ascii(eq)
    assert "\u0391" not in out and "\u0392" not in out and "\u0393" not in out
    assert "->" in out
    lhs, rhs = out.split("->")
    assert all(c.isascii() and c.isalpha() or c == "," for c in lhs)
    assert all(c.isascii() and c.isalpha() for c in rhs)


def test_remap_eq_to_ascii_too_many_labels():
    """Equations with > 52 distinct labels raise."""
    chars = [chr(0x4E00 + i) for i in range(53)]  # 53 distinct CJK chars
    eq = "".join(chars) + "->" + chars[0]
    with pytest.raises(ValueError, match="more than 52"):
        remap_eq_to_ascii(eq)


# -- logical_error_rate ------------------------------------------------------


@pytest.mark.parametrize("device", _device_params())
def test_logical_error_rate_matches_argmax(device):
    """``logical_error_rate`` equals ``mean(argmax != observable_flips)``."""
    rng = np.random.default_rng(202)
    H, logical = _nondegenerate_code()
    true_priors = [0.05, 0.15, 0.10]
    syn, flips = _sample_synthetic_dataset(H,
                                           logical,
                                           true_priors,
                                           num_shots=256,
                                           rng=rng)
    opt = _make_opt(H,
                    logical,
                    true_priors,
                    syn,
                    flips,
                    device=device,
                    dtype="float64")
    ler = opt.logical_error_rate()
    assert isinstance(ler, float)
    assert 0.0 <= ler <= 1.0

    with torch.no_grad():
        preds = opt.decoder_prediction()
        argmax_pred = (preds[:, 1] > preds[:, 0]).cpu().numpy()
    expected = float(np.mean(argmax_pred != flips.astype(bool)))
    assert abs(ler - expected) < 1e-12


@pytest.mark.parametrize("device", _device_params())
def test_logical_error_rate_improves_with_better_priors(device):
    """Decoding with true Bernoulli rates beats (or matches) uniform priors."""
    rng = np.random.default_rng(303)
    H, logical = _nondegenerate_code()
    true_priors = [0.03, 0.18, 0.07]
    syn, flips = _sample_synthetic_dataset(H,
                                           logical,
                                           true_priors,
                                           num_shots=4096,
                                           rng=rng)
    uniform = [float(np.mean(true_priors))] * H.shape[1]
    opt_true = _make_opt(H,
                         logical,
                         true_priors,
                         syn,
                         flips,
                         device=device,
                         dtype="float64")
    opt_uniform = _make_opt(H,
                            logical,
                            uniform,
                            syn,
                            flips,
                            device=device,
                            dtype="float64")
    assert opt_true.logical_error_rate(
    ) <= opt_uniform.logical_error_rate() + 1e-6


# -- parity vs base TN decoder -----------------------------------------------


@pytest.mark.parametrize("device", _device_params())
def test_forward_parity_with_tn_decoder(device):
    """Forward with frozen probs agrees with the base TN decoder's batch."""
    rng = np.random.default_rng(123)
    H, logical, priors = _random_code(rng, n_checks=4, n_errors=6)
    syn, flips = _sample_synthetic_dataset(H,
                                           logical,
                                           priors,
                                           num_shots=12,
                                           rng=rng)
    opt = _make_opt(H,
                    logical,
                    priors,
                    syn,
                    flips,
                    device=device,
                    dtype="float64")
    with torch.no_grad():
        pred = opt.decoder_prediction().detach().cpu().numpy()
    ref = qec.get_decoder("tensor_network_decoder",
                          H,
                          logical_obs=logical,
                          noise_model=priors,
                          dtype="float64")
    res = ref.decode_batch(syn)
    ref_p_flip = np.array([r.result[0] for r in res], dtype=np.float64)
    np.testing.assert_allclose(pred[:, 1], ref_p_flip, atol=1e-4, rtol=1e-4)


# -- CPU/GPU parity ----------------------------------------------------------


@pytest.mark.skipif(not _gpu_available(),
                    reason="CUDA not available; CPU/GPU parity test skipped")
def test_cpu_gpu_parity_forward():
    """Forward with the same seed agrees CPU vs GPU to atol 1e-4."""
    rng = np.random.default_rng(7)
    H, logical, priors = _random_code(rng, n_checks=4, n_errors=6)
    syn, flips = _sample_synthetic_dataset(H,
                                           logical,
                                           priors,
                                           num_shots=16,
                                           rng=rng)
    opt_cpu = _make_opt(H,
                        logical,
                        priors,
                        syn,
                        flips,
                        device="cpu",
                        dtype="float64")
    opt_gpu = _make_opt(H,
                        logical,
                        priors,
                        syn,
                        flips,
                        device="cuda",
                        dtype="float64")
    with torch.no_grad():
        p_cpu = opt_cpu.decoder_prediction().detach().cpu().numpy()
        p_gpu = opt_gpu.decoder_prediction().detach().cpu().numpy()
    np.testing.assert_allclose(p_cpu, p_gpu, atol=1e-4, rtol=1e-4)


# -- truth-data convergence --------------------------------------------------


@pytest.mark.parametrize("device", _device_params())
def test_recovers_true_priors_within_tol(device):
    """Fitted priors converge to the Bernoulli rates that sampled the data."""
    rng = np.random.default_rng(0xC0DE)
    H, logical = _nondegenerate_code()
    true_priors = [0.03, 0.12, 0.08]
    syn, flips = _sample_synthetic_dataset(H,
                                           logical,
                                           true_priors,
                                           num_shots=2000,
                                           rng=rng)
    init_priors = [0.10] * H.shape[1]
    opt = _make_opt(H,
                    logical,
                    init_priors,
                    syn,
                    flips,
                    device=device,
                    dtype="float64",
                    execute="codegen")
    init_p = opt.noise_params[0].detach()
    logits = torch.logit(init_p).clone().requires_grad_(True)
    torch_opt = torch.optim.Adam([logits], lr=0.05)
    step_fn = make_compiled_step(opt, logits, torch_opt)
    for _ in range(500):
        step_fn()
    fitted = torch.sigmoid(logits).detach().cpu().numpy()
    np.testing.assert_allclose(fitted,
                               np.asarray(true_priors, dtype=np.float64),
                               atol=0.02)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
