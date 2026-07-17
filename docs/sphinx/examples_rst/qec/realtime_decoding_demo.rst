Realtime Decoding with CUDA-Q Decoding Server
=============================================

.. note::

   This page describes a C++ example that you compile against the *installed*
   CUDA-QX SDK. The decoding server and the FPGA playback tool it drives are
   **deliverables** — prebuilt and installed alongside the SDK, not built from
   this example.

This example drives the delivered ``decoding_server`` from two different
syndrome sources, both decoding through the **same prebuilt server**:

- ``--source qpu-kernel`` — a lowered QEC kernel supplies syndromes itself over
  **UDP**, in software. No NIC, no FPGA. This is the portable, hardware-free
  path and is what CI exercises. With ``--wire cpu_roce`` the same kernel
  carries its syndromes over **real RDMA** between two RoCE-capable ports
  instead.
- ``--source fpga`` — the delivered ``hololink_fpga_syndrome_playback`` tool
  streams pre-generated syndromes over **RoCE from a real FPGA** into the
  server's RDMA ring. Requires a ConnectX NIC cabled to the FPGA. There is no
  emulator here.

Both sources decode with any of three decoders — ``pymatching``,
``multi_error_lut`` (CPU lookup table), and ``nv-qldpc-decoder`` (GPU
relay-BP). Runs are seeded (``--seed``, default 42), so the reported counts
are reproducible.

For the nv-qldpc Relay BP device-graph walkthrough see
:doc:`/examples_rst/qec/realtime_relay_bp`; for the software-only pymatching
benchmark see :doc:`/examples_rst/qec/realtime_predecoder_pymatching`.


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
  (``-frealtime-lowering -DQEC_APP_CQR``): the live
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
CUDA toolchain; the device-link architecture defaults to ``80`` (Ampere) —
override with ``-DCMAKE_CUDA_ARCHITECTURES=90`` for Hopper or
``-DCMAKE_CUDA_ARCHITECTURES=100`` for Blackwell (e.g. GB200).


Running
-------

``run_realtime_decoding.sh`` resolves the deliverables from ``--install-prefix``
and the two example binaries from ``--example-build-dir`` (default ``./build``).

QPU-kernel source (software, UDP, no hardware)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   ./run_realtime_decoding.sh --source qpu-kernel --decoder pymatching            --install-prefix <prefix>
   ./run_realtime_decoding.sh --source qpu-kernel --decoder multi_error_lut       --install-prefix <prefix>
   ./run_realtime_decoding.sh --source qpu-kernel --decoder nv-qldpc-decoder --gpu 0 --install-prefix <prefix>

The lowered kernel runs the surface-code memory experiment and streams each
shot's syndromes to the server over the ``udp`` wire; every decoder is served
on ``host`` dispatch (a server CPU thread calls the decoder — nv-qldpc still
decodes on its GPU) and the corrections come back before readout.

QPU-kernel source over real RDMA (``--wire cpu_roce``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The same lowered kernel can carry its syndromes over the ``cpu_roce`` wire
instead: its channel RDMA-writes each request straight into the server's ring.
This needs two RoCE-capable ports that can reach each other (kernel side =
*channel*, server side = *daemon*) — e.g. loopback-cabled ConnectX ports, or
SoftRoCE (``rdma_rxe``) devices. The topology comes from the same environment
variables as the in-tree cpu_roce tests; ``--setup-network`` assigns the IPs
and waits for the RoCE v2 GIDs:

.. code-block:: bash

   export CUDAQ_CPU_ROCE_TEST_CHANNEL_DEVICE=<kernel-side ibdev>   # e.g. mlx5_0
   export CUDAQ_CPU_ROCE_TEST_DAEMON_DEVICE=<server-side ibdev>    # e.g. mlx5_1
   # IPs default to 10.0.0.1 (channel) / 10.0.0.2 (daemon); override with
   # CUDAQ_CPU_ROCE_TEST_CHANNEL_IP / CUDAQ_CPU_ROCE_TEST_DAEMON_IP.

   ./run_realtime_decoding.sh --source qpu-kernel --decoder pymatching            --wire cpu_roce --setup-network --install-prefix <prefix>
   ./run_realtime_decoding.sh --source qpu-kernel --decoder multi_error_lut       --wire cpu_roce --setup-network --install-prefix <prefix>
   ./run_realtime_decoding.sh --source qpu-kernel --decoder nv-qldpc-decoder --gpu 0 --wire cpu_roce --setup-network --install-prefix <prefix>

The server's READY ``port=`` is then the TCP rendezvous port (the RDMA wire
itself is negotiated via the QP/rkey exchange); dispatch stays ``host``, and
the same pass/fail criteria apply unchanged.

FPGA source (real FPGA over RoCE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   ./run_realtime_decoding.sh --source fpga --decoder pymatching \
       --setup-network --device <nic> --bridge-ip <host-ip> --fpga-ip <fpga-ip> \
       --install-prefix <prefix>

   ./run_realtime_decoding.sh --source fpga --decoder multi_error_lut \
       --setup-network --device <nic> --bridge-ip <host-ip> --fpga-ip <fpga-ip> \
       --install-prefix <prefix>

   ./run_realtime_decoding.sh --source fpga --decoder nv-qldpc-decoder --gpu 0 \
       --setup-network --device <nic> --bridge-ip <host-ip> --fpga-ip <fpga-ip> \
       --install-prefix <prefix>

``--setup-network`` configures the ConnectX interface (needs ``sudo``);
``--spacing`` (default 10 µs) paces the playback so it does not overrun the
FPGA's fixed 64-slot RDMA RX ring.

The server has two independent knobs, both derived automatically (override
with ``--wire`` / ``--dispatch``): the **wire** is the bridge-provider library
that carries syndromes into the server, and the **dispatch** is the engine
consuming each decoder's ring. ``pymatching`` and ``multi_error_lut`` ride the
``cpu_roce`` wire on ``host`` dispatch (a server CPU thread calls the
decoder); ``nv-qldpc-decoder --gpu 0`` defaults to the ``hololink`` wire on
``device_graph`` dispatch (the GPU device-call scheduler fires the Relay BP
decode graph on-device) and can be forced onto the host path with
``--dispatch host``.

Both sources apply real pass/fail criteria. The qpu-kernel source uses the
same checks as the in-tree surface-code tests — no decoder errors, a residual
logical-error ceiling of ``num_shots/50``, an in-process dispatch count of 0
(proof the decode stayed in the external server), and a server dispatch-count
floor of ``num_shots * (num_rounds + 3)``. The FPGA source verifies every
shot's corrections against the expected values computed at
syndrome-generation time (the playback tool's per-shot verification).


Decoders
--------

.. list-table::
   :header-rows: 1

   * - decoder
     - qpu-kernel (udp / cpu_roce wire)
     - fpga (real FPGA)
     - extra requirement
   * - ``pymatching``
     - CPU (udp: no hardware)
     - NIC
     - none
   * - ``multi_error_lut``
     - CPU (udp: no hardware)
     - NIC
     - none
   * - ``nv-qldpc-decoder``
     - GPU (host dispatch)
     - NIC + GPU (device_graph dispatch)
     - plugin + ``--gpu``

The ``nv-qldpc-decoder`` profile needs the prebuilt plugin — auto-found in the
install prefix, else ``--nv-qldpc-plugin <path.so>`` — and a GPU; if the plugin
is unavailable the script exits ``77`` (skip).


The example source
------------------

.. literalinclude:: ../../examples/qec/realtime_decoding_demo/surface_code_realtime_decoding.cpp
   :language: cpp
