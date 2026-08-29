Performance Studies
===================

In-depth performance studies of CUDA-Q QEC decoders on NVIDIA GPUs -- measuring
decode latency, logical error rate, and the trade-offs behind decoder tuning knobs.

The first study shows how **gamma ensembling** narrows the Relay BP decode-latency
tail, improving the logical error rate under hard decode deadlines by up to **~89x**
on bivariate-bicycle codes (measured on a single GB200 with CUDA-Q QEC 0.7.0).

The second study shows how **relay solution recording** replaces a per-``stop_nconv``
sweep of full decode runs with a single recording run plus offline post-processing,
reproducing every RelayBP-N result exactly from one GPU pass.

.. toctree::
   :maxdepth: 1

   Improving Relay BP Decoding With Gamma Ensembles <nv_qldpc_gamma_ensemble_user_guide>
   Sweeping Relay BP Stopping Criteria From a Single Run <nv_qldpc_relay_solutions_user_guide>
