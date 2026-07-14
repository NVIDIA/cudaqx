Standalone Decoding Server
===========================

.. note::

   The following information is about a C++ server binary that must be built
   from source and is not part of any distributed CUDA-Q QEC binaries.

The ``decoding_server`` binary is a long-lived process that hosts one or more
decoder instances and serves the three realtime RPCs
(``enqueue_syndromes`` / ``get_corrections`` / ``reset_decoder``) over a
configurable transport.  It is the service-side counterpart to a
``cc.device_call``-lowered CUDA-Q kernel, and can also be driven directly by the
``hsb_fpga_decoding_server_test.sh`` orchestration script using real FPGA or
emulated HSB traffic.

Key design points:

- **Decoder-agnostic**: the decoder type and parameters are fully determined by a
  YAML config file.  Swapping decoders is a config change, not a rebuild.
- **Transport-selectable at launch**: ``--transport=udp`` (loopback, runs
  anywhere), ``--transport=cpu_roce`` (SoftRoCE / HSB FPGA HOST_CALL path),
  or ``--transport=gpu_roce`` (DOCA + Hololink device-graph scheduler, requires
  ConnectX and a CUDA-capable GPU).
- **Per-decoder GPU affinity**: the optional ``cuda_device_id`` YAML field pins
  decoder construction, graph capture, and worker-thread CUDA work to a specific
  GPU, which must agree with the NIC-affine GPU chosen for the RoCE transport.

Transports
----------

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Transport
     - Flag
     - When to use
   * - UDP (loopback)
     - ``--transport=udp``
     - Development, CI, no NIC required.  The two-process
       ``test_decoding_server_core`` test uses this.
   * - CPU RoCE
     - ``--transport=cpu_roce``
     - Real RDMA wire with HOST_CALL dispatch (CPU decoding).  Works with any
       CPU decoder.  Two QP-exchange modes: ``rendezvous`` (TCP swap with a
       ``CpuRoceChannel`` caller) and ``hsb_fpga`` (HSB FPGA/emulator QP
       printed to stdout, programmed by the playback tool).
   * - GPU RoCE
     - ``--transport=gpu_roce``
     - DOCA ring buffers DMA'd directly to GPU VRAM; the CUDAQ device-graph
       scheduler handles ``enqueue_syndromes`` / ``get_corrections`` /
       ``reset_decoder`` on-device with no CPU in the data path.  Requires
       ConnectX NIC + Hololink Sensor Bridge + DOCA + ``nv-qldpc-decoder``
       with graph-dispatch support.

YAML Configuration
------------------

The server is configured entirely by a YAML file passed via ``--config``.  The
top-level key is ``decoders``, containing a list of decoder entries.

Minimal example (pymatching, udp or cpu_roce path):

.. code-block:: yaml

   decoders:
     - id: 0
       type: pymatching
       block_size: 3
       syndrome_size: 3
       H_sparse: [0, -1, 1, -1, 2, -1]
       O_sparse: [0, -1, 1, -1, 2, -1]
       D_sparse: [0, -1, 1, -1, 2, -1]
       decoder_custom_args:
         merge_strategy: smallest_weight
         error_rate_vec: [0.1, 0.1, 0.1]

GPU RoCE example (nv-qldpc-decoder with Relay BP):

.. code-block:: yaml

   decoders:
     - id: 0
       type: nv-qldpc-decoder
       transport: gpu_roce
       cuda_device_id: 2
       block_size: ...
       syndrome_size: ...
       H_sparse: [...]
       O_sparse: [...]
       D_sparse: [...]
       decoder_custom_args:
         use_relay_bp: true
         ...

Per-decoder YAML fields
^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Field
     - Required
     - Description
   * - ``id``
     - Yes
     - Integer decoder identifier.  Referenced by ``decoder_id`` in RPC payloads.
   * - ``type``
     - Yes
     - Decoder plugin name: ``pymatching``, ``multi_error_lut``,
       ``nv-qldpc-decoder``, ``sliding_window``, ``trt_decoder``.
   * - ``transport``
     - No
     - Transport for this decoder's ring: ``cpu_roce`` (default) or
       ``gpu_roce``.  Must match the ``--transport`` CLI flag.
   * - ``cuda_device_id``
     - No
     - Integer CUDA device index this decoder is pinned to.  Decoder
       construction, graph capture, and all worker-thread CUDA calls use this
       device.  **Required for gpu_roce** to guarantee the decoder graph and
       the Hololink DOCA ring share the same GPU context.  If omitted the
       device is unpinned (whichever device is current at construction time).
   * - ``block_size``
     - Yes
     - Number of detector bits per decoding block.
   * - ``syndrome_size``
     - Yes
     - Number of syndrome bits per round.
   * - ``H_sparse``
     - Yes
     - Parity-check matrix in flat CSR-style: ``[col, -1, col, -1, ...]``,
       where ``-1`` is the row sentinel.
   * - ``O_sparse``
     - Yes
     - Observable matrix in the same format.
   * - ``D_sparse``
     - Yes
     - Detector matrix in the same format.
   * - ``decoder_custom_args``
     - No
     - Decoder-specific parameters (see decoder documentation for each type).

.. note::

   The ``transport`` and ``cuda_device_id`` fields are written into the YAML
   by the orchestration script (``--transport gpu_roce --gpu N``) when
   generating config from the ``surface_code-4-yaml`` binary.  Manually
   authored configs for gpu_roce must include both fields explicitly.

.. note::

   **Python bindings**: ``cuda_device_id`` is exposed on the
   ``cudaq.qec.DecoderConfig`` Python object (bound via
   ``py_decoding_config.cpp``).  Set it as an integer attribute before passing
   the config to ``cudaq.qec.configure_decoders``.

GPU device affinity and transport reconciliation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When ``--transport=gpu_roce`` is active, the server reconciles
``cuda_device_id`` (from the YAML) with ``HOLOLINK_GPU_ID`` (from the
environment) as follows:

- Both set and equal -> proceed.
- Only one set -> use the set value for both.
- Both set and **different** -> fatal error at startup.
- Neither set -> default to GPU 0.

This ensures decoder graph capture and the DOCA ring allocation always target
the same physical GPU.  Use a GPU that is NUMA-local to the ConnectX NIC for
lowest latency.

Building
--------

Prerequisites
^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Dependency
     - Notes
   * - CUDA Toolkit 12.6+
     - Required for all transports.
   * - ``cuda-quantum`` realtime libs
     - ``libcudaq-realtime``, ``libcudaq-realtime-udp-transport``,
       ``libcudaq-realtime-host-dispatch`` - always required.
       ``libcudaq-realtime-cpu-roce-transport`` enables ``cpu_roce``.
       ``libcudaq-realtime-dispatch`` enables the gpu_roce device-graph path.
   * - Holoscan Sensor Bridge (HSB)
     - Required for ``cpu_roce`` (hsb_fpga mode) and ``gpu_roce``.
       Tag ``2.6.0-EA2``.
   * - DOCA
     - Required for ``gpu_roce`` (installed at ``/opt/mellanox/doca``).
   * - ``libcudaq-qec-realtime-cudevice-proprietary.a``
     - Required for ``gpu_roce``.  Proprietary static archive providing the
       ``DEVICE_CALL`` handler shims and populate symbols.  Built from the
       private cuda-qx tree; pass via
       ``-DCUDAQ_QEC_REALTIME_CUDEVICE_PROPRIETARY_ARCHIVE=<path>``.
   * - ``libcudaq-qec-nv-qldpc-decoder.so``
     - Required at runtime for ``gpu_roce`` with Relay BP.  See
       :ref:`realtime_relay_bp` for how to obtain it.

CMake flags
^^^^^^^^^^^

.. code-block:: bash

   cmake -S cudaqx -B cudaqx/build \
     -DCMAKE_BUILD_TYPE=Release \
     -DCUDAQ_DIR=/path/to/cudaq-install/lib/cmake/cudaq \
     -DCUDAQ_REALTIME_ROOT=/path/to/cudaq-realtime-install \
     -DCUDAQ_REALTIME_INCLUDE_DIR=/path/to/cudaq-quantum/realtime/include \
     -DCUDAQ_REALTIME_LIBRARY=/path/to/cudaq-realtime/build/lib/libcudaq-realtime.so \
     -DCUDAQ_REALTIME_DISPATCH_LIBRARY=/path/to/cudaq-realtime/build/lib/libcudaq-realtime-dispatch.a \
     -DCUDAQ_REALTIME_HOST_DISPATCH_LIBRARY=/path/to/cudaq-realtime/build/lib/libcudaq-realtime-host-dispatch.a \
     -DQEC_UDP_TRANSPORT_LIBRARY=/path/to/cudaq-realtime/build/lib/libcudaq-realtime-udp-transport.a \
     -DQEC_CPU_ROCE_TRANSPORT_LIBRARY=/path/to/cudaq-realtime/build/lib/libcudaq-realtime-cpu-roce-transport.a \
     -DQEC_HOST_DISPATCH_LIBRARY=/path/to/cudaq-realtime/build/lib/libcudaq-realtime-host-dispatch.a \
     -DHOLOSCAN_SENSOR_BRIDGE_BUILD_DIR=/path/to/holoscan-sensor-bridge/build \
     -DHOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR=/path/to/holoscan-sensor-bridge \
     -DCUDAQ_QEC_REALTIME_CUDEVICE_PROPRIETARY_ARCHIVE=/path/to/libcudaq-qec-realtime-cudevice-proprietary.a

   cmake --build cudaqx/build --target decoding_server

Transport availability at build time is reported by CMake:

.. code-block:: text

   -- decoding_server: cpu_roce transport enabled
   -- decoding_server: gpu_roce transport enabled

If ``gpu_roce transport enabled`` does not appear, verify that
``HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR``, ``CUDAQ_REALTIME_INCLUDE_DIR``, and
all DOCA library paths are correct.

Private graph-dispatch symbols (private builds)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For ``gpu_roce``, the device-graph RPC dispatch logic lives in
``libs/qec/lib/realtime/decoder_rpc_dispatch.cu`` (part of the private
cuda-qx tree).  CMake builds the static library
``cudaq-qec-realtime-decoder-rpc-dispatch`` when that file is present in the
source tree, and links it whole-archive into ``decoding_server`` when the
resulting target exists.  The linker flag ``--export-dynamic`` (also added
by CMake) lets the running server call ``dlsym`` to locate the dispatch
entry points at startup.

In a public build (without the private tree) this file is absent and the
library target is never created.  The server still compiles; ``gpu_roce``
will fail at runtime with a missing symbol diagnostic rather than a link
error, so CI on public repos is not broken.  If you have access to the
private archive, pass:

.. code-block:: bash

   -DCUDAQ_QEC_REALTIME_CUDEVICE_PROPRIETARY_ARCHIVE=/path/to/libcudaq-qec-realtime-cudevice-proprietary.a

Running
-------

CLI reference
^^^^^^^^^^^^^

.. code-block:: text

   decoding_server --config=<decoders.yaml>
                   [--transport=udp|cpu_roce|gpu_roce]
                   [--port=N]          # udp: listen port (0 = ephemeral)
                   [--num-slots=N]     # ring depth (default 8; capped at 64 for hsb_fpga)
                   [--slot-size=N]     # ring slot stride bytes (default 256)
                   [--timeout=N]       # shutdown after N seconds (default 60)
                   # cpu_roce only:
                   [--device=NAME]     # ConnectX IB device (default mlx5_0)
                   [--local-ip=ADDR]   # local NIC IP (default 10.0.0.2)
                   [--qp_config=rendezvous|hsb_fpga]
                   [--peer-ip=ADDR]    # hsb_fpga: FPGA/emulator IPv4
                   [--remote-qp=N]     # hsb_fpga: peer QP number (hex or decimal)
                   [--frame-size=N]    # hsb_fpga: TX SGE bytes (default = slot-size)

.. note::

   For ``--transport=gpu_roce`` the Hololink ring geometry is configured
   entirely through environment variables (``HOLOLINK_DEVICE``,
   ``HOLOLINK_PEER_IP``, ``HOLOLINK_REMOTE_QP``, ``HOLOLINK_FRAME_SIZE``,
   ``HOLOLINK_NUM_PAGES``, ``HOLOLINK_GPU_ID``).  The CLI flags ``--device``,
   ``--peer-ip``, ``--num-slots``, and ``--slot-size`` are ignored on this
   path; pass them to the orchestration script instead (see below).

Readiness signals
^^^^^^^^^^^^^^^^^

The server writes to stdout once it is ready to accept traffic.  The
orchestration script and two-process test both wait for these lines:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Line
     - Transport / condition
   * - ``QEC_DECODING_SERVER_READY port=<P> transport=udp``
     - UDP transport bound and listening on port ``P``.
   * - ``QEC_DECODING_SERVER_READY port=<P> transport=cpu_roce roce_ip=<IP>``
     - CPU RoCE TCP rendezvous listening on port ``P`` (``rendezvous`` mode).
   * - ``QEC_DECODING_SERVER_READY port=0 transport=cpu_roce qp_config=hsb_fpga``
     - CPU RoCE HSB FPGA mode: QP / RKey / Buffer Addr already printed above.
   * - ``QEC_DECODING_SERVER_READY gpu_roce``
     - GPU RoCE: device-graph scheduler launched; QP / RKey / Buffer Addr
       printed by ``GpuRoceTransceiver`` before this line.

At shutdown the server also emits:

.. code-block:: text

   QEC_DECODING_SERVER_DISPATCHED count=<N>
   QEC_DECODING_SERVER_MAX_CONCURRENT_DECODERS count=<M>

UDP (development / CI)
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   ./build/bin/decoding_server \
     --config=decoding_server_config.yaml \
     --transport=udp \
     --timeout=60

The server prints its ephemeral port on stdout.  The two-process test
``test_decoding_server_core`` uses this mode; no NIC is required.

CPU RoCE with HSB FPGA / emulator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``hsb_fpga`` QP-exchange method is used when pairing with real FPGA
hardware or the software HSB emulator.  The server prints the canonical bridge
handshake (``QP Number`` / ``RKey`` / ``Buffer Addr``) and the orchestration
script relays these to the playback tool.

.. code-block:: bash

   LD_LIBRARY_PATH=/path/to/cudaq-realtime/build/lib:$LD_LIBRARY_PATH \
   ./build/bin/decoding_server \
     --config=config.yaml \
     --transport=cpu_roce \
     --qp_config=hsb_fpga \
     --device=mlx5_0 \
     --peer-ip=10.0.0.2 \
     --remote-qp=0x2 \
     --num-slots=64 \
     --slot-size=384 \
     --frame-size=64 \
     --timeout=60

GPU RoCE (device-graph scheduler)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The gpu_roce path wires the Hololink DOCA ring directly to the CUDAQ
device-graph scheduler.  Configure the ring geometry and GPU via environment
variables and pass ``--transport=gpu_roce`` with the YAML config.  The YAML
**must** include ``transport: gpu_roce`` and a matching ``cuda_device_id`` for
each decoder entry.

.. code-block:: bash

   CUDA_MODULE_LOADING=EAGER \
   LD_LIBRARY_PATH=/path/to/cudaq-realtime/build/lib:./build/lib:$LD_LIBRARY_PATH \
   HOLOLINK_DEVICE=mlx5_0 \
   HOLOLINK_PEER_IP=10.0.0.2 \
   HOLOLINK_REMOTE_QP=0xABCD \
   HOLOLINK_FRAME_SIZE=512 \
   HOLOLINK_NUM_PAGES=128 \
   HOLOLINK_GPU_ID=2 \
   ./build/bin/decoding_server \
     --config=config_gpu_roce.yaml \
     --transport=gpu_roce \
     --timeout=60

Environment variables for gpu_roce
"""""""""""""""""""""""""""""""""""

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Description
   * - ``HOLOLINK_DEVICE``
     - ConnectX IB device name (e.g. ``mlx5_0``).
   * - ``HOLOLINK_PEER_IP``
     - FPGA or emulator data-plane IPv4 address.
   * - ``HOLOLINK_REMOTE_QP``
     - Peer QP number in hex or decimal.
   * - ``HOLOLINK_FRAME_SIZE``
     - Ring slot stride in bytes.  Must be a multiple of 128.
   * - ``HOLOLINK_NUM_PAGES``
     - Server-side DOCA ring depth.  See GB200 note below.
   * - ``HOLOLINK_GPU_ID``
     - GPU device index.  Must match ``cuda_device_id`` in the YAML, or be
       the only GPU specifier set.
   * - ``CUDA_MODULE_LOADING``
     - Set to ``EAGER`` to avoid lazy-load stalls inside the persistent
       scheduler kernel.

GB200 / aarch64 page-size alignment
"""""""""""""""""""""""""""""""""""""

On GB200 (and other aarch64 kernels with 64 KiB host pages), DOCA rejects GPU
ring allocations whose total size is not host-page aligned.  With the FPGA/HSB
default slot size (512 bytes) and ring depth 64, the allocation is
``512 x 64 = 32768`` bytes = 32 KiB, which is below the 64 KiB page
boundary and fails.

The fix: keep ``HOLOLINK_FRAME_SIZE`` and the FPGA/playback ring depth (64)
unchanged, and grow only ``HOLOLINK_NUM_PAGES`` until
``HOLOLINK_FRAME_SIZE x HOLOLINK_NUM_PAGES`` is a multiple of the host page
size.  The orchestration script does this automatically when
``--gpu-roce-num-pages auto`` (the default) is passed:

.. code-block:: bash

   # Manual equivalent for 512-byte pages on a 64 KiB-page kernel:
   # 512 x 128 = 65536 bytes = 64 KiB -> aligned
   HOLOLINK_NUM_PAGES=128

The FPGA/emulator ring is always ``NUM_SLOTS=64``; only the server-side DOCA
allocation grows.

Orchestration Script
--------------------

``hsb_fpga_decoding_server_test.sh`` is the standard way to run end-to-end
tests against the decoding server with real FPGA hardware or the HSB software
emulator.  It handles building all tools, configuring the network, generating
test data, launching the server, running syndrome playback, and verifying
corrections.

.. code-block:: text

   hsb_fpga_decoding_server_test.sh [options]

Modes
^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Flag
     - Description
   * - ``--emulate``
     - Three-tool mode: HSB emulator + server + playback.  No real FPGA needed.
   * - *(default)*
     - Two-tool mode: server + playback against a real FPGA.

Actions
^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Flag
     - Description
   * - ``--build``
     - Build all required tools before running.
   * - ``--setup-network``
     - Configure ConnectX interfaces (run once per boot).
   * - ``--no-run``
     - Build only; skip the test run.

Decoder options
^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Flag
     - Description
   * - ``--decoder NAME``
     - Decoder profile: ``pymatching`` (default -> ``cpu_roce``) or
       ``nv-qldpc-decoder`` (Relay BP -> ``gpu_roce``).
   * - ``--transport T``
     - Override transport: ``cpu_roce`` or ``gpu_roce``.  Default is derived
       from the decoder profile.
   * - ``--config PATH``
     - Use a pre-made YAML config (skips auto-generation).
   * - ``--syndromes PATH``
     - Use a pre-made syndromes file (skips auto-generation; must pair with
       ``--config``).
   * - ``--data-dir DIR``
     - Use ``DIR/config_<decoder>.yml`` + ``DIR/syndromes_<decoder>.txt``.

Data-generation options
^^^^^^^^^^^^^^^^^^^^^^^

When no ``--config`` / ``--syndromes`` / ``--data-dir`` is given, the script
generates test data using the ``surface_code-4-yaml`` binary:

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Flag
     - Default
     - Description
   * - ``--distance N``
     - 3
     - Surface-code distance.
   * - ``--num-rounds N``
     - 4
     - Measurement rounds per shot.
   * - ``--p-spam F``
     - 0.01
     - SPAM error probability.
   * - ``--gen-shots N``
     - 100
     - Number of shots to generate.

Run options
^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Flag
     - Default
     - Description
   * - ``--gpu N``
     - 0
     - GPU device index for ``gpu_roce``.  Injected into ``cuda_device_id``
       in the generated YAML config.
   * - ``--gpu-roce-num-pages N``
     - ``auto``
     - Server-side gpu_roce ring depth.  ``auto`` grows the depth to satisfy
       host-page alignment (GB200 fix); FPGA/playback ring remains 64.
   * - ``--page-size N``
     - 384
     - Ring slot stride bytes (both FPGA/playback ring and server).
   * - ``--timeout N``
     - 60
     - Server timeout in seconds.
   * - ``--num-shots N``
     - all
     - Limit number of syndrome shots.
   * - ``--spacing N``
     - 10
     - Inter-shot spacing in microseconds (FPGA mode only).
   * - ``--no-verify``
     - *(verify)*
     - Skip correction verification.
   * - ``--frame-size N``
     - 64
     - Server TX SGE bytes (``cpu_roce`` only).
   * - ``--control-port N``
     - 8193
     - UDP control port for the emulator.

Network options
^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Flag
     - Default
     - Description
   * - ``--device DEV``
     - auto-detect
     - ConnectX IB device name.
   * - ``--bridge-ip ADDR``
     - ``10.0.0.1``
     - Server-side NIC IP.
   * - ``--emulator-ip ADDR``
     - ``10.0.0.2``
     - Emulator NIC IP (emulate mode).
   * - ``--fpga-ip ADDR``
     - ``192.168.0.2``
     - FPGA IP (non-emulate mode).
   * - ``--mtu N``
     - 4096
     - MTU size.

Build options
^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 45 25

   * - Flag
     - Description
     - Default
   * - ``--hsb-dir DIR``
     - holoscan-sensor-bridge source directory.
     - ``/workspaces/holoscan-sensor-bridge``
   * - ``--cuda-quantum-dir DIR``
     - cuda-quantum source directory (must match the ref in ``.cudaq_version``).
     - ``/workspaces/cuda-quantum``
   * - ``--cudaqx-dir DIR``
     - cudaqx source directory.
     - ``/workspaces/cudaqx``
   * - ``--proprietary-archive PATH``
     - Path to ``libcudaq-qec-realtime-cudevice-proprietary.a``.
     - cuda-qx build tree
   * - ``--nv-qldpc-plugin PATH``
     - Path to ``libcudaq-qec-nv-qldpc-decoder.so`` (symlinked into
       ``build/lib/decoder-plugins``).
     - cuda-qx build tree
   * - ``--jobs N``
     - Parallel build jobs.
     - ``nproc``

Quick-start examples
^^^^^^^^^^^^^^^^^^^^

CPU RoCE emulated run (pymatching, no FPGA):

.. code-block:: bash

   ./libs/qec/unittests/utils/hsb_fpga_decoding_server_test.sh \
     --emulate --build --setup-network

GPU RoCE emulated run (Relay BP / nv-qldpc-decoder, no FPGA):

.. code-block:: bash

   ./libs/qec/unittests/utils/hsb_fpga_decoding_server_test.sh \
     --emulate --build --setup-network \
     --decoder nv-qldpc-decoder \
     --gpu 2

Real FPGA run (cpu_roce, pymatching):

.. code-block:: bash

   ./libs/qec/unittests/utils/hsb_fpga_decoding_server_test.sh \
     --setup-network \
     --device mlx5_0 \
     --bridge-ip 192.168.0.1 \
     --fpga-ip 192.168.0.2 \
     --spacing 10

Ring buffer depth
^^^^^^^^^^^^^^^^^

The FPGA/playback ring depth is fixed at **64** (matching the HSB
``WQE_NUM = 64`` work-queue depth).  A ring deeper than ``WQE_NUM`` causes one
transceiver thread to service two ring slots at the same absolute positions
(``t`` and ``t + 64``), which produces rare duplicated-and-dropped frame pairs.
The server enforces the cap automatically for ``--qp_config=hsb_fpga``.

For ``gpu_roce``, only the server-side DOCA ring may be deeper when host-page
alignment requires it (``--gpu-roce-num-pages auto``); the FPGA ring remains
at 64.

Troubleshooting
---------------

``ERROR: gpu_roce startup failed: GPU RoCE support is not linked``
   The server binary was compiled with ``QEC_HAVE_GPU_ROCE_TRANSPORT``
   (``CUDAQ_GPU_ROCE_AVAILABLE`` was true) but the
   ``cudaq-qec-decoding-server-gpuroce`` component or the proprietary
   cudevice archive was not linked.  Rebuild with
   ``-DCUDAQ_QEC_REALTIME_CUDEVICE_PROPRIETARY_ARCHIVE`` pointing to the
   ``.a`` file.

``DOCA ring allocation failed`` / ``Cannot allocate memory``
   On GB200 (64 KiB host pages) the ring total ``HOLOLINK_FRAME_SIZE x
   HOLOLINK_NUM_PAGES`` must be a multiple of 65536.  Use
   ``--gpu-roce-num-pages auto`` (orchestration script) or compute the
   aligned depth manually.

``cuda_device_id`` / ``HOLOLINK_GPU_ID`` mismatch
   The server aborts if both are set to different values.  Set only one, or
   ensure both agree.

``QEC_DECODING_SERVER_READY gpu_roce`` never appears
   Check the server log for Hololink bring-up failures (device name, peer IP,
   QP number).  Ensure the ConnectX interface has the expected IP and is in
   RoCE v2 mode (``rdma link show``).

``cudaqx_qec_realtime_dispatch_populate_*_device_entry not found``
   The proprietary cudevice archive was not whole-archive linked into the
   binary.  The dlsym lookup for the populate shim failed.  Verify the build
   includes ``$<LINK_LIBRARY:WHOLE_ARCHIVE,cudaq-qec-realtime-cudevice-proprietary>``
   and ``--export-dynamic``.

Decoder worker thread stuck / ``NOT_READY`` responses
   The decoder's ``enqueue_syndrome`` did not trigger a decode (syndrome volume
   not yet complete).  Verify the syndrome count per round matches
   ``get_num_msyn_per_decode()`` for the configured decoder window.

ILA verification produces non-deterministic or flaky results
   Earlier versions of the orchestration script waited for ILA data to
   *stabilize* (no new samples for a timeout interval), which was racy under
   variable FPGA timing.  The current script waits for the **expected sample
   count** (``num_shots x frames_per_shot``) to appear in the ILA buffer
   before reading the capture.  If you see intermittent verification
   mismatches when using a custom ILA integration, ensure your wait logic
   polls on the sample count rather than a stabilization heuristic.

Multiple interfaces showing the same IP
   Remove the duplicate before starting the server:
   ``ip addr del <IP>/24 dev <other-iface>``.  A duplicate causes RDMA packet
   routing failures that are silent from the server's perspective.
