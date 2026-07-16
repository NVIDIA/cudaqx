Realtime Decoding: Driving the Decoding Server from an FPGA or a QPU Kernel
===========================================================================

.. note::

   This page describes a C++ example that you compile against the *installed*
   CUDA-QX SDK. The decoding server and the FPGA playback tool it drives are
   **deliverables** — prebuilt and installed alongside the SDK, not built from
   this example.

This example drives the delivered ``decoding_server`` from two different
syndrome sources, both decoding through the **same prebuilt server**:

- ``--source qpu-kernel`` — a lowered QEC kernel supplies syndromes itself over
  **UDP**, in software. No NIC, no FPGA. This is the portable, hardware-free
  path and is what CI exercises.
- ``--source fpga`` — the delivered ``hololink_fpga_syndrome_playback`` tool
  streams pre-generated syndromes over **RoCE from a real FPGA** into the
  server's RDMA ring. Requires a ConnectX NIC cabled to the FPGA. There is no
  emulator here.

Both sources decode with any of three decoders — ``pymatching``,
``trt_decoder`` (a TensorRT predecoder + PyMatching), and ``nv-qldpc-decoder``
(GPU relay-BP).

For the decoder-specific FPGA walkthroughs see
:doc:`/examples_rst/qec/realtime_predecoder_fpga` (TensorRT predecoder) and
:doc:`/examples_rst/qec/realtime_relay_bp` (nv-qldpc Relay BP); for the
software-only pymatching benchmark see
:doc:`/examples_rst/qec/realtime_predecoder_pymatching`.


Deliverables versus the example
-------------------------------

**Deliverables** (installed, not built by this example):

- ``decoding_server`` — the standalone server that owns the decoder instances.
- ``hololink_fpga_syndrome_playback`` — the FPGA playback tool (present only in
  the hololink-enabled deliverable image).
- the QEC + realtime libraries and the decoder plugins.

**The example** (the only thing you compile) is a single source,
``surface_code_realtime_decoding.cpp``, built two ways against the installed
SDK — using only installed headers and libraries:

- ``surface_code_realtime_decoding`` — the **generator** (``--target stim``):
  writes the decoder configuration and, for the FPGA source, a syndrome file.
- ``surface_code_realtime_decoding-cqr`` — the **lowered kernel**
  (``-frealtime-lowering -DQEC_APP_EXTERNAL_DECODING_SERVER``): the live
  syndrome source that streams to the server over UDP.


Building the example
--------------------

.. code-block:: bash

   cmake -S . -B build \
     -DCUDAQ_INSTALL_DIR=<cuda-quantum install prefix> \
     -DCUDAQX_INSTALL_DIR=<cuda-qx install prefix>
   cmake --build build

In a CUDA-QX container ``CUDAQ_INSTALL_DIR`` defaults to ``/usr/local/cudaq``.
If the realtime libraries are in a separate prefix, add
``-DCUDAQ_REALTIME_DIR=<realtime prefix>``. The lowered kernel links the
realtime dispatch archive (relocatable CUDA device code), so the build needs a
CUDA toolchain; the device-link architecture defaults to ``80`` — override with
``-DCMAKE_CUDA_ARCHITECTURES=90`` to match a Hopper / GB200 GPU.


Running
-------

``run_realtime_decoding.sh`` resolves the deliverables from ``--install-prefix``
and the two example binaries from ``--example-build-dir`` (default ``./build``).

QPU-kernel source (software, UDP, no hardware)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   ./run_realtime_decoding.sh --source qpu-kernel --decoder pymatching            --install-prefix <prefix>
   ./run_realtime_decoding.sh --source qpu-kernel --decoder trt_decoder           --install-prefix <prefix>
   ./run_realtime_decoding.sh --source qpu-kernel --decoder nv-qldpc-decoder --gpu 0 --install-prefix <prefix>

The lowered kernel runs the surface-code memory experiment and streams each
shot's syndromes to the server over UDP; the server decodes on the host-call
path and returns corrections.

FPGA source (real FPGA over RoCE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   ./run_realtime_decoding.sh --source fpga --decoder pymatching \
       --setup-network --device <nic> --bridge-ip <host-ip> --fpga-ip <fpga-ip> \
       --install-prefix <prefix>

``--setup-network`` configures the ConnectX interface (needs ``sudo``);
``--spacing`` (default 10 µs) paces the playback so it does not overrun the
FPGA's fixed 64-slot RDMA RX ring. ``pymatching`` and ``trt_decoder`` decode on
the CPU (``cpu_roce``); ``nv-qldpc-decoder --gpu 0`` decodes on the GPU
device-graph scheduler (``gpu_roce``).

Both sources apply real pass/fail criteria. The qpu-kernel source uses the
same checks as the in-tree surface_code-4 tests — no decoder errors, a
residual logical-error ceiling of ``num_shots/50``, server-owned decoders,
and a server dispatch-count floor of ``num_shots * (num_rounds + 3)``. The
FPGA source verifies every shot's corrections against the expected values
computed at syndrome-generation time (the playback tool's per-shot
verification).


Decoders
--------

.. list-table::
   :header-rows: 1

   * - decoder
     - qpu-kernel (UDP)
     - fpga (real FPGA)
     - extra requirement
   * - ``pymatching``
     - CPU, no hardware
     - NIC
     - none
   * - ``trt_decoder``
     - GPU (server-side TensorRT)
     - NIC + GPU
     - python ``onnx`` + GPU
   * - ``nv-qldpc-decoder``
     - GPU (host-call path)
     - NIC + GPU (device path)
     - plugin + ``--gpu``

The ``trt_decoder`` profile generates a tiny identity-predecoder ONNX at
runtime (needs the python ``onnx`` module; override with ``--onnx``). The
``nv-qldpc-decoder`` profile needs the prebuilt plugin — auto-found in the
install prefix, else ``--nv-qldpc-plugin <path.so>`` — and a GPU; if the plugin
is unavailable the script exits ``77`` (skip).


The example source
------------------

.. literalinclude:: ../../examples/qec/realtime_decoding_demo/surface_code_realtime_decoding.cpp
   :language: cpp
