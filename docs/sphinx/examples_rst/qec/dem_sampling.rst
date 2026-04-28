.. _dem_sampling_example:

DEM Sampling — Monte-Carlo Sampling from Detector Error Models
--------------------------------------------------------------

A **detector error model** (DEM) describes the probabilistic relationship
between independent error mechanisms and the detectors (syndrome bits) that
observe them. Given a binary check matrix :math:`H` and a vector of per-mechanism
error probabilities, DEM sampling generates random error vectors and the
corresponding syndromes via

.. math::

   \text{errors}_{ij} \sim \text{Bernoulli}(p_j), \qquad
   \text{syndromes} = \text{errors} \cdot H^T \pmod{2}.

In Python, ``cudaq_qec.dem_sampling`` provides this capability with automatic
backend selection: it uses GPU-accelerated sampling via cuStabilizer when
available and falls back to a CPU implementation otherwise. In C++ the CPU and
GPU paths are exposed as separate functions in the ``cudaq::qec::dem_sampler``
namespace (see :ref:`dem_sampling_cpp_api`).

Example
+++++++

.. tab:: Python

   .. literalinclude:: ../../examples/qec/python/dem_sampling.py
      :language: python
      :start-after: [Begin Documentation]

.. tab:: C++

   .. literalinclude:: ../../examples/qec/cpp/dem_sampling.cpp
      :language: cpp
      :start-after: [Begin Documentation]

   Compile and run with

   .. code-block:: bash

      nvq++ --enable-mlir -lcudaq-qec dem_sampling.cpp -o dem_sampling
      ./dem_sampling

GPU Acceleration
++++++++++++++++

When a CUDA-capable GPU is available, ``dem_sampling`` uses a fully on-device
pipeline that is significantly faster than per-shot CPU sampling, especially
for large numbers of shots and sparse error models (low probabilities):

1. **Sparse Bernoulli sampling** — Errors are generated directly in compressed
   sparse row (CSR) format. For low error probabilities the CSR representation
   is compact, and the sampler skips mechanisms with zero probability entirely
   rather than evaluating a Bernoulli trial for every mechanism in every shot.

2. **GF(2) sparse-dense matrix multiply** — Syndromes are computed as
   :math:`\text{errors} \times H^T \pmod{2}` using a sparse-dense multiply
   over GF(2). The check matrix :math:`H^T` is stored in a bitpacked layout,
   reducing memory bandwidth by 8x compared to one byte per entry.

3. **On-device packing and unpacking** — :math:`H` is transposed and bitpacked
   on the GPU in a single kernel. Syndromes are unpacked from the bitpacked
   result, and the dense error matrix is produced from the CSR representation
   via a fused zero-and-scatter kernel.

The CPU path uses ``std::bernoulli_distribution`` per mechanism per shot
followed by a dense dot product for the syndrome.

Input Types and Backend Selection
+++++++++++++++++++++++++++++++++

The ``backend`` parameter controls where sampling runs:

- ``"auto"`` (default) — try GPU first, fall back to CPU.
- ``"gpu"`` — require GPU; raise ``RuntimeError`` if unavailable.
- ``"cpu"`` — always use the CPU path.

The Python binding accepts several input types, each routed through a different
code path:

1. **NumPy arrays** (most common) — When the GPU is available the bindings
   automatically allocate device memory, copy inputs host-to-device, run
   cuStabilizer, and copy results back as NumPy ``uint8`` arrays. With
   ``backend="cpu"`` the GPU path is skipped entirely. No user action is
   required beyond passing standard ``uint8`` and ``float64`` arrays.

2. **PyTorch CUDA tensors** — The GPU path reads input device pointers directly
   via ``data_ptr()`` and writes outputs into ``torch.empty`` tensors on the
   same device, avoiding any host-device copies. This is the fastest path when
   inputs are already on the GPU. PyTorch is an optional dependency; install
   with ``pip install torch``.

3. **PyTorch CPU tensors** — With ``backend="gpu"`` the tensors are
   automatically moved to CUDA (via ``.to(device)``) before sampling. With
   ``backend="auto"`` CPU tensors are rejected with an error; convert them to
   NumPy with ``.numpy()`` first.

The C++ API exposes two namespaces:

- ``cudaq::qec::dem_sampler::cpu::sample_dem`` — takes a ``cudaqx::tensor``
  check matrix and a ``std::vector<double>`` of probabilities; returns
  ``(syndromes, errors)`` as tensors.
- ``cudaq::qec::dem_sampler::gpu::sample_dem`` — takes raw device pointers and
  writes results into caller-provided device buffers; returns ``false`` if
  cuStabilizer is not available at runtime.

See Also
++++++++

- :doc:`/api/qec/python_api` — ``dem_sampling`` Python API reference
- :doc:`/api/qec/cpp_api` — ``dem_sampler`` C++ API reference
- :doc:`/examples_rst/qec/decoders` — Decoder examples that consume syndromes
