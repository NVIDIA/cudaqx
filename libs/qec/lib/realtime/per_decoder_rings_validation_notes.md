# Per-Decoder Rings / Mixed-Dispatch Server: Validation Notes

Campaign log for `per_decoder_rings_design.md` -- what was run, what it
proved, the dead ends, and the gotchas.  Read this before debugging the
decode-graph pipeline or reproducing the builds; read the design docs for
the architecture and contracts.

## 1. What is validated (2026-07-12, WSL2 laptop, sm_120 GPU)

- In-process per-decoder rings: `surface_code-5-per-decoder-rings` example
  + test script (two sessions/rings/dispatchers, per-device channel
  override routing).
- Two-process per-decoder rings over udp:
  `DecodingServerTwoProcess.TwoProcessPerDecoderRings` asserts >=3
  dispatches on EACH ring.
- Two-process CONTROL with rebuilt nv-qldpc (new ABI): full sc4 app, two
  host nv-qldpc rings, correct decode (`logical_errors=0/2` per decoder).
- Mixed-dispatch server bring-up: ring0 host + ring1 GPU scheduler over
  pinned-udp rings, one READY line, clean teardown when idle.
- The GPU dispatch graph SERVES DEVICE_CALLS OVER THE WIRE two-process:
  raw RPC datagram probes (hand-built RPCHeader, magic 0x43555152, fnv1a
  ids) round-trip reset_decoder and a full-window enqueue_syndromes with
  status=0 against the device ring.
- FULL mixed E2E incl. real device-ring decodes: see section 2 (the
  wedge resolution) -- decode graph fired 12/12 with rc=0 and correct
  RelayBP corrections.
- Device-side fire-and-forget graph launch works on this box (see probe
  notes below), native sm_120 SASS and forced-JIT compute_90 alike.

## 2. RESOLVED: the decode-graph wedge (2026-07-12, same day)

Root cause: **cooperative-launch co-residency starvation** -- one bug
class, two instances:

a. The decode graph was captured with `reserved_sms = 0`
   (DecodingSession::create called `capture_decode_graph()` with the
   default), sizing the cooperative grid for EVERY SM.  With the
   persistent dispatch graph resident on one SM, the fired decode graph
   could never become fully co-resident and would deadlock at
   grid.sync() -- the launch silently queues forever, stalling the tail
   self-relaunch and wedging the scheduler.  FIXED: DecodingSession now
   captures with `reserved_sms = 1` (override:
   `QEC_DEVICE_GRAPH_RESERVED_SMS`, e.g. higher on rigs where Hololink
   RX/TX kernels are also resident).
b. HOST-path nv-qldpc decode launches its own cooperative GPU kernels,
   also sized without reservation -- with a device-graph scheduler
   resident (mixed server), the host decode hangs the same way.  A udp
   proxy trace localized the app wedge to ring0's get_corrections (the
   HOST ring), not the device ring.  OPEN (plugin-side): plumb SM
   reservation into the host decode path; until then do not co-locate a
   host GPU-cooperative decoder with a resident scheduler (use a CPU
   decoder such as multi_error_lut on host rings in mixed deployments).

Positive end-to-end proof (WSL2 laptop, two-process, pinned-udp device
ring): sc4 app, 10 shots at p_spam=0.08, decoder 0 multi_error_lut
(host ring) + decoder 1 nv-qldpc RelayBP (device ring):
`trigger debug rc=0 fires=12 tail_relaunches=12`, both rings
dispatched=72, `decoder[1] corrections=2, logical_errors=0/10`, clean
teardown.  The rig is no longer required to exercise this path; the flow
is now a gated ctest, `app_examples.surface_code-4-yaml-mixed-dispatch`
(registered when the server links the device-graph component; skips
without a GPU or the nv-qldpc plugin), whose YAML names the wire via the
transport section only -- no --transport on the command line.

Diagnostics added along the way (kept):
- `cudaq_dispatch_get_trigger_debug()` (cudaq-realtime-dispatch):
  device-side trigger rc + fire and tail-relaunch counters; async-safe
  to call while the scheduler is live or wedged.  Healthy signature:
  rc=0, fires == tail_relaunches.  Logged by DeviceGraphRingConsumer at
  shutdown.
- DeviceGraphRingConsumer::dispatched() and the debug reader use async
  copies on non-blocking streams: a legacy default-stream cudaMemcpy
  SYNCHRONIZES with the persistent scheduler graph and self-deadlocks
  (this trap manufactured a fake teardown hang during the
  investigation).

## 3. Probe methodology (and the invalid-probe lesson)

- VALID probe for device graph launch: two-graph form.  Child graph
  (payload) instantiated `DeviceLaunch` + `cudaGraphUpload`ed; trigger
  kernel calling `cudaGraphLaunch(child, cudaStreamGraphFireAndForget)`
  placed INSIDE a device-launched parent graph; parent host-launched.
  Passes here (`payload ran: YES`), including with PTX-only compute_90
  compilation (JIT'd kernels are fine).
- INVALID probe (do not repeat): triggering fire-and-forget from a PLAIN
  kernel.  That is illegal per the CUDA contract (device graph launch is
  only legal from within a device-launched graph) and fails with
  `cudaErrorNotSupported` on EVERY platform -- including target rigs --
  visible only via `cudaGetLastError()` immediately after the trigger
  kernel launch (stream sync reports success).  This briefly produced a
  false "WSL2 platform limitation" diagnosis until a rig run showed the
  same failure.
- Also probed good on this box: GPU spin-polling of CPU-written
  pinned+mapped memory (`ack=1`), plain `DeviceLaunch` instantiation,
  `CUDA_MODULE_LOADING=EAGER` (no effect on any of the above).
- Artifact arch coverage on this box: GPU is sm_120; nv-qldpc plugin +
  cudevice archives + cudaq-realtime-dispatch carry <= sm_100 SASS (+
  PTX).  Ruled out as the wedge cause by the forced-JIT probe.

## 4. Rebuilding the proprietary artifacts against a cudaqx branch

Scratch worktree of the private cuda-qx repo (`bmh/private-device-ring`):
merge the cudaqx branch into the LOCAL `private/main` (the local branch
was AHEAD of the gitlab remote -- check before basing anything on the
remote tip).  The private realtime/pcm/decoder deltas were already
upstreamed to cudaqx main; the pieces that must survive the merge are the
`NV_PRIVATE` CMake hooks (`add_subdirectory(decoders/plugins/
nv-qldpc-decoder)` in libs/qec/lib/CMakeLists.txt, and the
`libs/qec/lib/realtime/private/` proprietary sources present on the local
branch).  Build with the same CUDAQ_DIR/CUDAQ_REALTIME_ROOT as the cudaqx
tree; pluck ONLY binaries (`libcudaq-qec-nv-qldpc-decoder.so`,
`libcudaq-qec-realtime-cudevice-proprietary.a`) back, and repoint
`CUDAQ_QEC_REALTIME_CUDEVICE_PROPRIETARY_ARCHIVE`.

## 5. Build/runtime gotchas (each cost real time)

- nvq++-compiled TUs (`cudaqx_add_device_code` custom commands) do NOT
  track header dependencies: after changing `decoding_config.h`, touch or
  clean those sources, or stale objects pass old-layout structs across
  the .so boundary (symptom: segfault inside `configure_decoders`).
- `libcudaq-device-call-runtime.so` must NOT re-export transport-archive
  symbols (`cpu_udp_*`): a provider .so carries its own copy and the
  dynamic linker binds across -- silent ODR violation corrupting teardown
  (fixed upstream with `--exclude-libs`).
- `cudaq_bridge_create`'s cached-provider path did not register new
  handles, so every SECOND bridge instance of one provider was unusable
  (fixed; surfaced by the N-ring server).
- Dispatcher stats flush when the loop EXITS: sample per-ring counters
  after `cudaq_dispatcher_stop`, never before.
- Shell hygiene: never `pkill -f <pattern>` where the pattern matches
  your own shell's command string.

## Post-merge revalidation (2026-07-15, origin/main with landed PR 670)

origin/main (15 commits: squashed 670 + #656/#673/#674/#678/#680/#683/#690/
#691/#692) merged into this branch; main's GpuRoce* fixes ported into the
DeviceGraph* files, cuda_device_id + session stats adopted, dispatch/
transport-section schema kept.  Revalidated on this box after rebuilding
nv-qldpc against the merged tree (#674 virtualized the realtime decoder API
-- an ABI break; a stale plugin .so manifests as malloc corruption or
segfaults in ANY test process, since the plugin loader scans the whole
decoder-plugins directory):

- Two-process suite 6/6 (incl. the transport-section and CLI-conflict
  tests), DecoderYAMLTest 18/18, per-decoder-rings app test, sc1 inproc-rpc
  with RelayBP, py-surface_code-1 (python bindings enabled;
  PYTHONPATH=/usr/local/cudaq:<build>/python), and main's new gpu-required
  surface_code-4-yaml-mixed-dispatch.
- Flagship mixed E2E re-passed: trigger rc=0, fires=10 == tail_relaunches,
  both rings dispatched=60, decoder[1] nv-qldpc corrections=2,
  logical_errors=0/10.
- QEC_DECODING_SERVER_STATS (adopted from main) works in the mixed server;
  note the counters are HOST-session counters, so a device_graph ring
  reports decodes=0 there -- its evidence is the trigger-debug fires and
  the RING dispatched line.
- Reminder that bit twice: nvq++ custom-command TUs do not track header
  deps; touch the qec sources after decoding_config.h / decoder.h changes
  or stale objects segfault.
