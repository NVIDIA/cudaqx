# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under    #
# the terms of the Apache License 2.0 which accompanies this distribution.    #
# ============================================================================ #

# Pure-NumPy tests of the relay_solutions post-processing helper against
# hand-computed expectations. The decoder itself is not needed: the wire
# format is a plain dict of flat arrays, built synthetically here; GPU
# end-to-end coverage lives elsewhere.

import numpy as np
import pytest

import cudaq_qec as qec
from cudaq_qec import relay_solutions as rs

INF = np.inf


def _pack_rows(bit_rows, width):
    """Pack per-record bit vectors into the little-endian int32 word format."""
    words = []
    for bits in bit_rows:
        value = 0
        for i, b in enumerate(bits):
            value |= int(b) << i
        words.append(value)
    assert width <= 32  # single-word records are enough for these tests
    return np.asarray(words, dtype=np.int32)


def _payload(width, max_records, counts, totals, iters, weight, packed_rows):
    return {
        "relay_solutions_width":
            width,
        "relay_solutions_max_records":
            max_records,
        "relay_solutions_counts":
            np.asarray(counts, dtype=np.int32),
        "relay_solutions_totals":
            np.asarray(totals, dtype=np.int32),
        "relay_solutions_iters":
            np.asarray(iters, dtype=np.int32).ravel(),
        "relay_solutions_weight":
            np.asarray(weight, dtype=np.float64).ravel(),
        "relay_solutions_result":
            _pack_rows([b for shot in packed_rows for b in shot], width),
    }


# Correction-mode fixture: width 5, two observables, four shots, R = 3.
#   shot 0: 3 records; the min-weight record changes class between N=1 and 2
#   shot 1: 1 record, then exhausts the schedule
#   shot 2: never converges (scored via the decoder's fallback result)
#   shot 3: 2 records with a weight tie (earliest record must win)
L = np.array([
    [1, 0, 1, 0, 0],
    [0, 1, 0, 0, 1],
])
WIDTH, K, N_SHOTS, R = 5, 2, 4, 3
COUNTS = [3, 1, 0, 2]
ITERS = [[3, 10, 20], [7, -1, -1], [-1, -1, -1], [5, 9, -1]]
WEIGHT = [[5.0, 2.0, 9.0], [4.0, INF, INF], [INF, INF, INF], [3.0, 3.0, INF]]
RECORD_BITS = [
    [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]],  # cls (1,0)(0,1)(1,0)
    [[1, 1, 0, 0, 0], [0] * 5, [0] * 5],  # cls (1,1)
    [[0] * 5, [0] * 5, [0] * 5],
    [[0] * 5, [0, 1, 0, 0, 0], [0] * 5],  # cls (0,0), (0,1)
]
NUM_ITER = [20, 50, 60, 40]  # full-schedule iterations of the recording run
RETURNED = np.array(
    [
        [0, 1, 0, 0, 0],  # shot 0: its overall min-weight record
        [1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1],  # shot 2: fallback, not among the records
        [0, 0, 0, 0, 0],
    ],
    dtype=np.float64)
OBS_TRUTH = np.array([[1, 0], [1, 1], [0, 1], [0, 0]])


def make_results(counts=COUNTS, totals=None, num_iter=NUM_ITER):
    payload = _payload(WIDTH, R, counts, totals if totals else counts, ITERS,
                       WEIGHT, RECORD_BITS)
    opt_results = [{"num_iter": ni} for ni in num_iter]
    return qec.BatchDecoderResult(RETURNED, np.asarray(COUNTS, dtype=bool),
                                  opt_results, payload)


def test_unpack_round_trip():
    recs = rs.unpack(make_results().batch_opt_results, N_SHOTS)
    assert recs.width == WIDTH and recs.max_records == R
    np.testing.assert_array_equal(recs.counts, COUNTS)
    np.testing.assert_array_equal(recs.iters, ITERS)
    np.testing.assert_array_equal(recs.weight, WEIGHT)
    for shot in range(N_SHOTS):
        for record in range(COUNTS[shot]):
            np.testing.assert_array_equal(recs.bits(shot, record),
                                          RECORD_BITS[shot][record])


def test_unpack_single_shot_keys():
    payload = _payload(WIDTH, 2, [2], [2], [[3, 10]], [[5.0, 2.0]],
                       [RECORD_BITS[0][:2]])
    payload["relay_solutions_count"] = payload.pop("relay_solutions_counts")[0]
    payload["relay_solutions_total"] = payload.pop("relay_solutions_totals")[0]
    recs = rs.unpack(payload, 1)
    np.testing.assert_array_equal(recs.counts, [2])
    np.testing.assert_array_equal(recs.bits(0, 1), RECORD_BITS[0][1])


def test_stop_nconv_sweep_correction_mode():
    sweep = rs.stop_nconv_sweep(make_results(),
                                OBS_TRUTH,
                                percentiles=[50, 100],
                                observables=L)

    np.testing.assert_array_equal(sweep.n, [1, 2, 3])
    # N=1: every shot's first record (or fallback) matches the truth.
    # N>=2: shot 0's min-weight record changes to class (0,1) -> one error.
    # Shot 3's N=2 weight tie must resolve to the EARLIEST record (correct
    # class); shot 2's fallback matches the truth at every N.
    np.testing.assert_array_equal(sweep.num_errors, [0, 1, 1])
    np.testing.assert_allclose(sweep.ler, [0, 0.25, 0.25])

    # Iterations: cumulative at the Nth convergence, or the shot's num_iter
    # once it exhausts (shot 2 always, shot 1 from N=2, shot 3 at N=3).
    expected = np.array([[3, 7, 60, 5], [10, 50, 60, 9], [20, 50, 60, 40]])
    np.testing.assert_allclose(sweep.avg_iters, expected.mean(axis=1))
    np.testing.assert_allclose(sweep.iters_percentiles,
                               np.percentile(expected, [50, 100], axis=1))
    assert sweep.iters_percentiles.shape == (2, 3)
    assert sweep.num_unconverged == 1
    np.testing.assert_allclose(sweep.frac_exhausted, [0.25, 0.5, 0.75])


def test_scalar_percentile_and_n_values():
    sweep = rs.stop_nconv_sweep(make_results(),
                                OBS_TRUTH,
                                percentiles=50,
                                n_values=[2],
                                observables=L)
    assert sweep.iters_percentiles.shape == (1,)
    np.testing.assert_allclose(sweep.iters_percentiles,
                               [np.percentile([10, 50, 60, 9], 50)])
    np.testing.assert_array_equal(sweep.num_errors, [1])


def test_n_beyond_max_records_uses_full_schedule():
    sweep = rs.stop_nconv_sweep(make_results(),
                                OBS_TRUTH,
                                percentiles=50,
                                n_values=[5],
                                observables=L)
    # Every shot exhausts: predictions equal the full-record minimum and the
    # iteration count is each shot's num_iter.
    np.testing.assert_array_equal(sweep.num_errors, [1])
    np.testing.assert_allclose(sweep.avg_iters, [np.mean(NUM_ITER)])
    np.testing.assert_allclose(sweep.frac_exhausted, [1.0])


def test_observables_required_in_correction_mode():
    with pytest.raises(ValueError, match="observables"):
        rs.stop_nconv_sweep(make_results(), OBS_TRUTH, percentiles=50)


def test_capped_records_rejected_beyond_cap():
    results = make_results(counts=[2, 1, 0, 2], totals=[3, 1, 0, 2])
    with pytest.raises(ValueError, match="capped at 2"):
        rs.stop_nconv_sweep(results, OBS_TRUTH, percentiles=50, observables=L)
    # Within the cap the sweep is still faithful.
    sweep = rs.stop_nconv_sweep(results,
                                OBS_TRUTH,
                                percentiles=50,
                                n_values=[1, 2],
                                observables=L)
    np.testing.assert_array_equal(sweep.num_errors, [0, 1])


def test_missing_num_iter_raises():
    payload = _payload(WIDTH, R, COUNTS, COUNTS, ITERS, WEIGHT, RECORD_BITS)
    results = qec.BatchDecoderResult(RETURNED, np.asarray(COUNTS, dtype=bool),
                                     None, payload)
    with pytest.raises(ValueError, match="num_iter"):
        rs.stop_nconv_sweep(results, OBS_TRUTH, percentiles=50, observables=L)
    # N=1 with no unconverged shots would not need it -- but shot 2 never
    # converged, so its iteration count is still unavailable.
    with pytest.raises(ValueError, match="num_iter"):
        rs.stop_nconv_sweep(results,
                            OBS_TRUTH,
                            percentiles=50,
                            n_values=[1],
                            observables=L)


def test_observables_mode_uses_records_directly():
    # width == k: records already are observable classes.
    width, k = 2, 2
    payload = _payload(width, 2, [2, 1], [2, 1], [[4, 8], [6, -1]],
                       [[3.0, 1.0], [2.0, INF]],
                       [[[1, 0], [0, 1]], [[1, 1], [0, 0]]])
    returned = np.array([[0, 1], [1, 1]], dtype=np.float64)
    results = qec.BatchDecoderResult(returned, np.array([True, True]), [{
        "num_iter": 9
    }, {
        "num_iter": 9
    }], payload)
    truth = np.array([[1, 0], [1, 1]])
    sweep = rs.stop_nconv_sweep(results, truth, percentiles=50)
    # N=1: both first records match. N=2: shot 0 flips to (0,1) -> error;
    # shot 1 exhausts and keeps (1,1).
    np.testing.assert_array_equal(sweep.num_errors, [0, 1])
    np.testing.assert_allclose(sweep.avg_iters, [(4 + 6) / 2, (8 + 9) / 2])


def test_percentile_validation():
    with pytest.raises(ValueError, match="percentiles"):
        rs.stop_nconv_sweep(make_results(),
                            OBS_TRUTH,
                            percentiles=0,
                            observables=L)
    with pytest.raises(ValueError, match="percentiles"):
        rs.stop_nconv_sweep(make_results(),
                            OBS_TRUTH,
                            percentiles=[50, 101],
                            observables=L)


if __name__ == "__main__":
    pytest.main()
