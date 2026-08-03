.. _dyn_dem_example:

Dynamic DEM Construction
------------------------

A **detector error model** (DEM) relates independent fault mechanisms to the
detectors and logical observables they flip. CUDA-Q QEC already builds DEMs
from Stim circuits (``dem_from_memory_circuit``, ``dem_from_stim_text``). The
*dynamic DEM* interfaces add two complementary paths that do not require a
circuit:

1. **Matrix construction** — ``dem_from_css_matrices`` builds a
   :math:`T`-round code-capacity (or phenomenological) DEM directly from CSS
   generator matrices and noise rates.
2. **Composable chunks** — an :class:`~cudaq_qec.ExtendedDem` is one round (or
   phase) of that model. Stitch chunks across round boundaries and
   ``dem_close`` / ``dem_close_all`` to recover a flat
   :class:`~cudaq_qec.DetectorErrorModel`.

The chunk form is what streaming and real-time decoders need when the round
count is chosen at run time: the same init / bulk / final phase description
expands to any :math:`T \ge 2`, and helpers such as
``dem_chunks_to_d_sparse`` fill the ``D_sparse`` field of a decoder config.

Matrix construction
+++++++++++++++++++

``css_code_matrices`` (Python: ``CssCodes``) holds the four sparse CSS blocks
``hz``, ``hx``, ``lz``, ``lx``. ``css_noise_params`` (Python: ``CssNoise``)
supplies uniform or per-qubit / per-check rates ``px``, ``py``, ``pz``, and
optional measurement-error rate ``pm``.

With ``num_rounds = T``, each data-qubit fault in round :math:`r` contributes
to detector bands :math:`r` and :math:`r+1` (syndrome differences), except the
final round which touches only band :math:`T-1`. Measurement errors flip no
logical observable. The invariant

.. math::

   \mathrm{dem\_close}(\mathrm{dem\_stitch\_all}(
     \underbrace{c,\ldots,c}_{T}))
   \;=\;
   \mathrm{dem\_from\_css\_matrices}(\mathrm{code},\mathrm{noise},T)

holds for one-round chunks :math:`c` built by
``extended_dem_from_css_matrices``. Prefer ``dem_close_all`` when only the
closed DEM is needed: it is :math:`O(T)` rather than the left-fold
:math:`O(T^2)` of ``dem_stitch_all``.

Example
+++++++

.. tab:: Python

   .. literalinclude:: ../../examples/qec/python/dyn_dem.py
      :language: python
      :start-after: [Begin Documentation]
      :end-before: [End Documentation]

.. tab:: C++

   .. literalinclude:: ../../examples/qec/cpp/dyn_dem.cpp
      :language: cpp
      :start-after: [Begin Documentation]
      :end-before: [End Documentation]

   Compile and run with

   .. code-block:: bash

      nvq++ -lcudaq-qec -lcudaq-qec-decoders dyn_dem.cpp
      ./a.out

Phase specs and YAML ``dem_chunks``
+++++++++++++++++++++++++++++++++++

For asymmetric init / bulk / final rounds (for example dropping measurement
errors on a destructive final readout), describe each phase as a
:class:`~cudaq_qec.DemChunkSpec` — the same ``-1``-terminated sparse lists used
by ``H_sparse`` / ``O_sparse`` — and group them in a
:class:`~cudaq_qec.DemChunksSpec`. ``dem_chunks_from_spec(spec, num_rounds)``
expands to ``init``, ``num_rounds - 2`` copies of ``bulk``, then ``final``.

Real-time decoder YAML accepts the same structure under ``dem_chunks`` with
``num_rounds``. When ``H_sparse`` is empty, the configuration is *chunk form*:
``expand_dem_chunks`` (called during decoder construction) derives
``block_size``, ``syndrome_size``, ``H_sparse``, ``O_sparse``, and
``D_sparse``, and supplies ``error_rate_vec`` from the closed DEM priors. A
nonempty ``H_sparse`` keeps the config in flat form (the matrix wins).

.. code-block:: yaml

   decoders:
     - id: 0
       type: single_error_lut
       num_rounds: 5
       dem_chunks:
         init:
           num_faults: 9
           H_mid_sparse: [ 0, 1, 5, -1, 1, 2, 6, -1, 2, 3, 7, -1, 3, 4, 8, -1 ]
           H_out_sparse: [ 5, -1, 6, -1, 7, -1, 8, -1 ]
           O_sparse: [ 0, -1 ]
           error_rates: [ 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02 ]
         bulk:
           num_faults: 9
           H_in_sparse: [ 0, 1, 5, -1, 1, 2, 6, -1, 2, 3, 7, -1, 3, 4, 8, -1 ]
           H_out_sparse: [ 5, -1, 6, -1, 7, -1, 8, -1 ]
           O_sparse: [ 0, -1 ]
           error_rates: [ 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02 ]
         final:
           num_faults: 5
           H_in_sparse: [ 0, 1, -1, 1, 2, -1, 2, 3, -1, 3, 4, -1 ]
           O_sparse: [ 0, -1 ]
           error_rates: [ 0.02, 0.02, 0.02, 0.02, 0.02 ]

Duplicate fault columns
+++++++++++++++++++++++

After stitching, identical row-support columns can appear when the same
physical fault is modelled on both sides of a seam. ``dem_merge_duplicate_columns``
(or ``dem_stitch_merged``) collapses them. The default
:class:`~cudaq_qec.PriorCombineMode` ``or_combine`` uses the GF(2) / XOR rule
:math:`p = \tfrac12\bigl(1 - \prod_i (1 - 2 p_i)\bigr)`, matching DEM
canonicalization elsewhere in CUDA-Q QEC. ``sum_combine`` is a small-:math:`p`
linear approximation, clamped to :math:`[0, 1]`.

Closing drops ``out_syndrome``
++++++++++++++++++++++++++++++

``dem_close`` and ``dem_close_all`` intentionally discard the last chunk's
outgoing seam: there is no later round for it to differ against. Put any
final-boundary detector in ``in_syndrome`` or ``interior`` (for a phase
spec, ``H_in_sparse`` / ``H_mid_sparse``). ``dem_chunks_spec`` validation
rejects a nonempty ``final.H_out_sparse`` for this reason.

See also
++++++++

- :ref:`dyn_dem_python_api` — Python API reference
- :ref:`dyn_dem_cpp_api` — C++ API reference
- :doc:`/examples_rst/qec/realtime_decoding` — decoder configuration YAML
- :doc:`/examples_rst/qec/dem_sampling` — sampling from a flat DEM
