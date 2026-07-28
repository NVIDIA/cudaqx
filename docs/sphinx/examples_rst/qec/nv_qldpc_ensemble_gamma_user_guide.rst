.. _ensemble_gamma_user_guide:

Improving Relay BP Decoding With Ensemble Gammas
=================================================

The NV-qLDPC Relay-BP decoder supports ensembles of independent gamma distribution trajectories (N "lanes"), run in parallel on a single GPU. When any lane has satisfied the stopping criterion, then the decoder may exit early. This can lead to significantly reduced decoding times as increased parallelism allows for faster exploration of decoding solutions and therefore faster convergence.

While setting multiple lanes of ensembled gammas can reduce the number of RelayBP iterations needed to converge to a solution, it can also lead to increased time per iteration; each leg of RelayBP synchronizes between lanes to check for convergence. Multiple lanes share a fixed number of GPU blocks, so as the number of edges in the Tanner graph increases, the GPU eventually must serialize the execution of each lane at each iteration, and then block for every lane to complete. We can measure these two competing effects directly; we build the Z-component of the circuit-level detector error model (DEM) (i.e. the Z-stabilizer detectors only) for the bivariate-bicycle (BB) codes ``[[72,12,6]]``, ``[[144,12,12]]`` and ``[[288,12,18]]`` and, for each, construct the decoder with ``gamma_ensemble_size`` set to N in {1, 2, 4, 8}. We then decode many sampled syndromes, timing each ``decode()`` call and reading back the number of RelayBP iterations the decoder used to converge:

.. code-block:: python

    import cudaq_qec as qec

    # H, error_rates, syndromes come from a circuit-level DEM of the code
    decoder = qec.get_decoder(
        "nv-qldpc-decoder", H, error_rate_vec=error_rates,
        use_sparsity=True, bp_method=3, composition=1, max_iterations=50,
        gamma0=0.125, gamma_dist=[-0.24, 0.66], clip_value=200.0, repeatable=True,
        proc_float="fp32", gamma_ensemble_size=N,    # sweep N in {1, 2, 4, 8}
        srelay_config={"pre_iter": 1, "num_sets": 600,
                       "stopping_criterion": "FirstConv"},
        opt_results={"num_iter": True})              # return the iteration count

    for syndrome in syndromes:
        result = decoder.decode(syndrome)            # timed for the per-iteration cost
        num_iter = result.opt_results["num_iter"]    # RelayBP iterations to converge

Performance Comparison
++++++++++++++++++++++

All experiments below use the same circuit-level noise and decoder settings, and are run on a single GB200. The bicycle-code DEMs use uniform circuit-level noise at ``p = 0.002`` for all three codes (with ``r = 6``, ``12`` and ``18`` rounds for ``[[72,12,6]]``, ``[[144,12,12]]`` and ``[[288,12,18]]`` respectively). The decoder is constructed as shown above with ``gamma0 = 0.125``, and ``num_sets = 600`` for all codes, and ``gamma_dist = [-0.24, 0.66]`` for ``[[72,12,6]]`` and ``[[144,12,12]]``, and ``gamma_dist = [-0.161, 0.815]`` for ``[[288,12,18]]``.

.. image:: ../../images/relaybp_gamma_ensemble_perf.png
   :align: center
   :alt: Iterations to converge, time per iteration, and mean latency versus ensemble size, for several DEMs

As the ensemble size N grows, the median number of iterations to converge (left) falls for every code — more lanes explore more gamma sets in parallel, so the fastest-converging one wins sooner. At N = 8 the median iteration count drops to 0.67×–0.71× of its single-lane value across the three codes. The time per iteration (middle) rises with N — each lane's parity-check rows are spread over co-resident GPU blocks, so a lane must serially process more work per iteration — growing to 1.53×–1.66× at N = 8, largest for the ``[[288,12,18]]`` DEM. The net mean decode latency (right) combines the two effects; here the higher per-iteration cost roughly cancels the iteration savings, so mean latency stays close to its single-lane value at N = 8, ranging from 0.88× to 1.10×.

Latency Distribution
++++++++++++++++++++

One of the benefits of ensembling is that the latency distribution becomes narrower, leading to more consistent latency for hard-to-decode syndromes. Using the same decoders as above, we now focus on the behavior of the 50th percentile, 90th percentile, and 99th percentile decoding latencies.

.. image:: ../../images/relaybp_latency_percentiles.png
   :align: center
   :alt: Latency percentiles (p50, p90, p99) versus ensemble size N, per code

The worst-case (i.e. p99) latency improves with N for every code — at N = 8 it improves by ~1.55× for ``[[72,12,6]]``, ~2.75× for ``[[144,12,12]]`` and ~2.65× for ``[[288,12,18]]``. The median (p50) rises with N (by ~1.12×–1.23× at N = 8), as the per-iteration overhead is paid on decodes that would have converged quickly anyway. 

Logical Error Rate Under Hard Deadlines
+++++++++++++++++++++++++++++++++++++++

The narrower tail can improve logical error rates considerably for decoders under hard deadlines. Suppose each decode must finish within a wall-clock budget ``t``; a decode is a success only if it both finishes within ``t`` and returns the correct logical outcome, so the logical error rate under that deadline is ``LER(t) = P(latency > t or logical error)``. Using the same decoders as above, we record both the per-syndrome latency and whether the decoded logical is correct. The plot below shows the LER versus deadline for regular RelayBP (N = 1) and for each ensemble size N; each curve is measured over 150,000 sampled syndromes per configuration at the circuit-level noise strength given above (``p = 0.002``), using the ``FirstConv`` stopping criterion.

.. image:: ../../images/relaybp_hard_deadline_ler.png
   :align: center
   :alt: Deadline vs deadline-miss rate — bicycle codes

Ensembling contracts the latency tail for all three codes, resulting in substantially lower LER for certain deadlines: beyond a crossover at the tightest deadlines (where the higher per-iteration cost makes the larger ensembles slightly worse), a larger ensemble misses fewer deadlines at looser budgets and reaches a lower floor. The regime under which ensembling is beneficial for these codes depends on the code and deadline, and likely requires specific tuning based on the problem. Below is a plot of the factor by which each ensemble size lowers LER relative to regular RelayBP (N = 1) as a function of the deadline ``t``; values above 1 mean a lower LER than N = 1. 

.. image:: ../../images/relaybp_ler_multiplier.png
   :align: center
   :alt: LER improvement multiplier over regular RelayBP (N=1) versus hard deadline, per code

At the tightest deadlines the larger ensembles are briefly worse (their higher per-iteration cost delays the fastest decodes), but past that crossover a larger ensemble lowers LER substantially — by up to ~41× for ``[[144,12,12]]`` and ~89× for ``[[288,12,18]]`` at deadlines of ~1-5 ms. For those two codes the multiplier then falls back toward at loose deadlines, where both the ensemble and N = 1 reach their near-zero logical error floors; for ``[[72,12,6]]`` it settles at ~2×.
