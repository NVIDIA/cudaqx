Performance Studies
===================

In-depth performance studies of CUDA-Q QEC decoders on NVIDIA GPUs -- measuring
decode latency, logical error rate, and the trade-offs behind decoder tuning knobs.

The first study shows how **gamma ensembling** narrows the Relay BP decode-latency
tail, improving the logical error rate under hard decode deadlines by up to **~89x**
on bivariate-bicycle codes (measured on a single GB200 with CUDA-Q QEC 0.7.0).

.. toctree::
   :maxdepth: 1

   Improving Relay BP Decoding With Gamma Ensembles <nv_qldpc_gamma_ensemble_user_guide>
