# Decoding-Server Transport-Provider Design

Status: prototype complete and validated on branch pair
`cuda-quantum:bmh/realtime-bridge-providers` +
`cudaqx:bmh/decoding-server-bridge-670` (2026-07).
This document is written so the work can be reproduced from scratch by
reading it: it records the before/after architecture, the API contracts, the
commit-by-commit change plan, and the validation gates.

Document family: this file owns the WIRE seam (provider ABI + all server
deployment contracts); `per_decoder_rings_design.md` owns the TOPOLOGY
(rings, consumers, dispatch shapes, mixed server);
`per_decoder_rings_validation_notes.md` is the campaign log.

## 1. Problem

Before this work, `libs/qec/tools/decoding-server/decoding_server.cpp` was a
catalog of transports with a decoder attached:

- Every wire (`udp`, `cpu_roce`, `gpu_roce`) was a compiled-in branch behind
  its own `#ifdef`, CLI flags, and CMake detection block.
- The CPU RoCE TCP rendezvous protocol (a byte-exact `RendezvousInfo` struct
  "must match CpuRoceChannel byte-for-byte") and the HSB FPGA QP handshake
  were duplicated inline in the server -- a silent cross-repo drift hazard.
- `GpuRoceTransceiver` fused the Hololink transport bring-up with the QEC
  device-graph dispatch engine, so building the gpu_roce path required
  Holoscan-Sensor-Bridge, DOCA, hololink, and libcuda at *link time*.
- The server hand-rolled a `std::thread` around
  `cudaq_host_ring_dispatch_loop`, an API surface cuda-quantum PR 4869
  retires in favor of the dispatcher object.
- Transport selection had two knobs that had to agree: `--transport` on the
  CLI and a per-decoder `transport: cpu_roce|gpu_roce` YAML key.  When they
  disagreed, a stub transceiver was silently selected and died with an
  unrelated error.
- A partner transport library could not be integrated without adding
  code to this repository.

## 2. Goal

The decoding server should treat the wire the way it already treats
decoders: **as configuration**.  A partner ships one shared library; the
server changes zero lines.

## 3. Architecture after

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    decoding_server (transport-blind)                     │
│  YAML `dispatch:` picks the ENGINE     YAML `transport:` picks the WIRE  │
│   (per decoder)                        (per deployment; --transport is   │
│                                         a fallback, conflict = error)    │
│  ONE RING PER DECODER, each with its own consumer:                       │
│   host ─► dispatcher object over        <name>  ► libcudaq-realtime-     │
│           its provider ring (4869 API)            bridge-<name>.so       │
│   device_graph ─► DeviceGraphRing-      /path.so ► partner library,      │
│           Consumer (GPU scheduler)                verbatim, zero changes │
│  geometry + READY ring tokens derived FROM the providers (v2 queries)    │
└──────────────────────────────────────────────────────────────────────────┘
     │ links                            │ links (weak factory, WHOLE_ARCHIVE)
┌────────────────────┐   ┌────────────────────────────────────────────────┐
│ CQR plugin         │   │ DecodingServer core · device-graph component:  │
│ (HOST_CALL table)  │   │ DeviceGraphTransceiver = dispatch ENGINE only  │
└────────────────────┘   └────────────────────────────────────────────────┘
     │ links (the ONLY realtime link dependency)
┌──────────────────────────────────────────────────────────────────────────┐
│  libcudaq-realtime.so — bridge loader (iface v2) + dispatcher object     │
└──────────────────────────────────────────────────────────────────────────┘
     ◌ dlopen ─────────────── runtime plug-in seam ────────────── ◌ dlopen
┌────────────┬──────────────────┬─────────────────┬────────────────────────┐
│ bridge-udp │ bridge-cpu-roce  │ bridge-hololink │ partner transport .so  │
│ .so        │ .so (rendezvous  │ .so (built on   │ (out of tree, ~9 C     │
│            │ + hsb_fpga)      │ the HSB rig)    │ functions)             │
└────────────┴──────────────────┴─────────────────┴────────────────────────┘
```

Design rules that fall out of the CUDA constraints:

- **Anything that must be device-linked or inlined stays in static archives
  inside the server's single device-link unit** (the dispatch kernel in
  `cudaq-realtime-dispatch.a`, the proprietary cudevice DEVICE_CALL entries,
  `CUDA_SEPARABLE_COMPILATION` + `CUDA_RESOLVE_DEVICE_SYMBOLS` +
  `--export-dynamic` + dlsym populate shims).  CUDA device function pointers
  are module-scoped; they cannot cross a dlopen boundary.
- **Anything that only moves bytes goes behind the provider `.so`.**  The
  bridge seam passes pointers (ring addresses, strides, QP identity) once at
  setup; the per-message data plane is copy-identical to the pre-refactor
  code (handlers still operate in place on RX/TX slots).

## 4. The contracts (API details)

### 4.1 Bridge provider interface (cuda-quantum, `bridge_interface.h`, v2)

A provider is one shared library exporting
`cudaq_realtime_bridge_interface_t *cudaq_realtime_get_bridge_interface()`
with:

```
version                (int; v2)
create(handle*, argc, argv)     bring-up to the point where the endpoint
                                identity is known (bind/QP), BEFORE any
                                blocking peer wait; parses its own args and
                                IGNORES unrecognized ones (callers forward
                                their whole arg list)
destroy(handle)
get_transport_context(handle, RING_BUFFER|UNIFIED, out)
connect(handle)                 may BLOCK on the peer (e.g. TCP rendezvous
                                accept); hsb_fpga-style no-peer methods make
                                this a no-op
launch(handle)                  start the provider's I/O loop/threads
disconnect(handle)
-- v2 fields (read only from providers reporting version >= 2) --
get_cpu_dataplane(handle, out)  unified single-thread loop shape; NULL if
                                unsupported
get_endpoint_info(handle, buf, len)   one-line `key=value ...` description
                                (port=, roce_ip=, qp=0x.., rkey=,
                                buffer_addr=0x..); valid after create()
get_ring_geometry(handle, num_slots*, slot_size*)
```

Loader rules (`bridge_interface_api.cpp`): accepts provider versions in
`[1, CURRENT]`; fields past `disconnect` are read only from v2+; missing
capability => `CUDAQ_ERR_UNSUPPORTED` (new status value).  Providers load
via `cudaq_bridge_create_from_library(<name-or-path>)`: libraries are
cached per process keyed by that string, so any number of distinct
provider libraries coexist in one process ("hololink" is just the
provider whose library name is `libcudaq-realtime-bridge-hololink.so`).
The enum API (`CUDAQ_PROVIDER_HOLOLINK` / `CUDAQ_PROVIDER_EXTERNAL` +
`CUDAQ_REALTIME_BRIDGE_LIB=<path>`) remains as a convenience wrapper over
the string-keyed path.  Version compatibility is
asymmetric BY DESIGN: new core runs old providers; an old core rejects a v2
provider loudly at load ("expected 1, got 2") -- core and providers upgrade
together across the v1->v2 boundary.

Providers implemented (cuda-quantum `realtime/lib/daemon/bridge/`):

- `udp/` -> `libcudaq-realtime-bridge-udp.so`: wraps the udp ring
  transceiver.  Args: `--port=` (0=ephemeral) `--num-slots=` `--slot-size=`.
  create=bind, connect=no-op, launch=start threads.
- `cpu_roce/` -> `libcudaq-realtime-bridge-cpu-roce.so`: wraps the CPU RoCE
  transceiver AND both QP-exchange methods.  `--qp_config=rendezvous`
  (default): create=setup()+TCP listen, connect=accept+byte-exact
  RendezvousInfo swap+connect (the wire struct now lives next to
  CpuRoceChannel -- one copy).  `--qp_config=hsb_fpga`: create=one-shot
  start + the canonical `=== Bridge Ready ===` banner (strict-regex format
  from hololink_bridge_common.h, printed BEFORE any consumer readiness
  line), connect=no-op.  Extra args: `--device= --local-ip= --peer-ip=
  --remote-qp= --frame-size=`.  Gated on libibverbs at build.
- `hololink/` (pre-existing provider, extended): v2 endpoint-info
  (qp/rkey/buffer_addr/peer_ip) + ring geometry.  Only compiles with
  HSB+DOCA; must be build-verified on the rig.

### 4.2 Decoding server deployment contracts (cudaqx)

```
decoding_server --config=<decoders.yaml>
                [--transport=<name|/path/to/lib.so>]   default udp
                [--timeout=N]
                [everything else forwarded verbatim to provider create()]
```

- `--transport` name -> `libcudaq-realtime-bridge-<name>.so` resolved in
  QEC_BRIDGE_PROVIDER_DIR (baked at CMake time: the dir of
  libcudaq-realtime.so); a value containing '/' is loaded verbatim (partner
  drop-in).  The server passes the resolved library to
  `cudaq_bridge_create_from_library` per ring; the loader caches libraries
  per name, so rings may mix provider libraries freely (`hololink`
  resolves like any other name).

**YAML transport section.**  The wire is deployment configuration and
lives at the TOP LEVEL of the config file, never inside decoder entries.
Rings differ by dispatch shape only, so the single override is shape-keyed:

```yaml
transport:                 # server-level deployment config
  provider: udp            # name or /path/to/lib.so
  args: [--slot-size=256]  # appended to forwarded CLI provider args
  device_graph:            # applied to dispatch: device_graph rings
    provider: udp          # "hololink" on an HSB rig
    args: [--pinned-rings]
decoders: ...              # pure decoding config; no transport keys
```

Per-ring resolution: shape override > section provider/args >
`--transport` CLI fallback.  The CLI flag is a default for configs that
intentionally leave the wire unspecified (one YAML reused across wires,
selected per launch); a YAML that names a provider plus an explicit
`--transport` is a startup ERROR, never a silent override.

**Server flow (one ring per decoder).**  CQR plugin session first
(decoders built BEFORE readiness), then per decoder: bridge create ->
geometry query; then ONE readiness line for all rings; then per ring:
connect (may block on peer) -> ring context -> consumer (host dispatcher
object, or the device-graph scheduler for `dispatch: device_graph`) ->
bridge launch.  Teardown stops consumers before bridges and samples
per-ring counters AFTER the consumer stops (dispatcher stats flush on
loop exit).

**Readiness / shutdown wire lines** (parsed by tests and orchestration;
changes here are contract changes):

```
QEC_DECODING_SERVER_READY port=<P0> <ring0 endpoint info> ring0=<P0> ring1=<P1> ...
QEC_DECODING_SERVER_RING decoder=<id> dispatched=<n>     (one per ring, at shutdown)
QEC_DECODING_SERVER_DISPATCHED count=<total>
QEC_DECODING_SERVER_MAX_CONCURRENT_DECODERS count=<n>
```

The leading `port=` token is ring 0's (single-ring consumers sscanf it
right after the prefix and keep working; their traffic payload-demuxes
over ring 0).  An all-device_graph config takes the standalone
DecodingServer path instead, with sentinel
`QEC_DECODING_SERVER_READY device_graph`.

**Caller-side per-ring endpoints (CUDA-Q runtime contract).**  External
channel arguments accept device scoping `<key>.<id>=<value>`, applied to
device id `<id>` only (overriding the plain key), e.g.
`udp-port=<P0> udp-port.1=<P1>`; a device with scoped args gets its own
channel/ring, others share device 0's channel.  The builtin channel spec
accepts per-device overrides too:
`CUDAQ_DEVICE_CALL_CHANNEL=<default>[,<id>=<channel>...]`.

### 4.3 Deployment cookbook

Complete minimal recipes for the deployment shapes covered by the
contracts in 4.2.  Each recipe is: YAML config + exact launch line +
matching caller-side configuration.  All YAML uses only keys the parser
accepts (top-level `transport:` with `provider:`/`args:`/`device_graph:`;
per-decoder `dispatch:`; decoder entries carry NO transport key).  The
3-bit identity pymatching entry stands in wherever the decoder is not the
point.

**Recipe 1 -- local dev: one host decoder over udp loopback.**
The udp provider binds an ephemeral port (`--port=0` is the default); the
caller reads the port from the READY line.

```yaml
# decoders.yaml
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
```

```
decoding_server --config=decoders.yaml
```

Caller (P = the `port=` value from `QEC_DECODING_SERVER_READY port=<P>`):

```
--cudaq-device-call=udp udp-host=127.0.0.1 udp-port=<P>
```

**Recipe 2 -- two host decoders, one udp ring per decoder.**
Each decoder gets its own ring/port; the READY line carries them as
`ring0=<P0> ring1=<P1>` (the leading `port=` token is ring 0's).

```yaml
decoders:
  - id: 0
    type: pymatching
    # ... as above
  - id: 1
    type: pymatching
    # ... as above
```

```
decoding_server --config=decoders.yaml
```

Parse `QEC_DECODING_SERVER_READY port=<P0> ... ring0=<P0> ring1=<P1>`;
the caller scopes ring 1's endpoint with `<key>.<id>=<value>`:

```
--cudaq-device-call=udp udp-host=127.0.0.1 udp-port=<P0> udp-port.1=<P1>
```

**Recipe 3 -- cpu_roce rendezvous on an RDMA rig.**
The provider listens on a TCP rendezvous port (published as the READY
port); the RDMA wire itself is negotiated by the QP/rkey exchange.

```yaml
transport:
  provider: cpu_roce
  args: [--device=mlx5_0, --local-ip=10.0.0.2]
decoders:
  - id: 0
    type: pymatching
    # ... as above
```

```
decoding_server --config=decoders.yaml
```

Caller (P = the READY port = the TCP rendezvous port):

```
--cudaq-device-call=cpu_roce ib-device=mlx5_0 local-ip=10.0.0.1 \
  rendezvous-host=10.0.0.2 rendezvous-port=<P>
```

**Recipe 4 -- cpu_roce hsb_fpga (FPGA peer).**
The peer is the FPGA, not a cudaq caller: there is no caller-side channel
config.  The provider's one-shot QP bring-up prints the canonical
`=== Bridge Ready ===` banner BEFORE the READY line; the FPGA-side flow
keys off the banner.

```yaml
transport:
  provider: cpu_roce
  args: [--qp_config=hsb_fpga, --device=mlx5_0, --local-ip=10.0.0.2,
         --peer-ip=10.0.0.3, --remote-qp=0x2,
         --slot-size=384, --num-slots=64]
decoders:
  - id: 0
    type: pymatching
    # ... as above
```

```
decoding_server --config=decoders.yaml
```

Caller: none (the FPGA drives the wire directly).

**Recipe 5 -- mixed dispatch locally (host + device_graph, both udp).**
Decoder 0 dispatches on the CPU, decoder 1 on the GPU device-graph
scheduler; the shape-keyed override tunes only the device_graph ring.
The device_graph decoder must be graph-capable (e.g. nv-qldpc-decoder
with RelayBP) -- pymatching is host-only.

```yaml
transport:
  provider: udp
  device_graph:
    args: [--pinned-rings]
decoders:
  - id: 0
    type: pymatching
    # ... as above
  - id: 1
    type: nv-qldpc-decoder
    dispatch: device_graph
    # block_size/syndrome_size/H_sparse/... per the decoder's code
```

```
decoding_server --config=decoders.yaml
```

Caller: as recipe 2 (`udp-port=<P0> udp-port.1=<P1>` from the READY
`ring0=`/`ring1=` tokens).

**Recipe 6 -- mixed dispatch on an HSB rig.**
Same YAML shape, but the device_graph ring rides the hololink provider --
just another provider library to the string-keyed loader, so it composes
with any provider on the host rings.

```yaml
transport:
  provider: udp
  device_graph:
    provider: hololink
decoders:
  - id: 0
    type: pymatching
    # ... as above
  - id: 1
    type: nv-qldpc-decoder
    dispatch: device_graph
    # ... per the decoder's code
```

```
decoding_server --config=decoders.yaml
```

Caller: host ring as recipe 1; the device_graph ring's peer is the HSB
wire (configured via the QEC_DEVICE_GRAPH_* env, see 4.4).

**Recipe 7 -- all-device_graph (standalone DecodingServer / HSB flow).**
When every decoder is `dispatch: device_graph` the server takes the
standalone DecodingServer path; the READY sentinel is the bare
`QEC_DECODING_SERVER_READY device_graph` (no port tokens).

```yaml
decoders:
  - id: 0
    type: nv-qldpc-decoder
    dispatch: device_graph
    # ... per the decoder's code
```

```
decoding_server --config=decoders.yaml
```

Orchestration waits for `QEC_DECODING_SERVER_READY device_graph`; the
peer is the HSB wire, no cudaq caller channel.

**Recipe 8 -- partner transport drop-in.**
A `--transport` value containing '/' is loaded verbatim; unrecognized
CLI args are forwarded to the provider's create() untouched.

```
decoding_server --config=decoders.yaml \
  --transport=/opt/partner/libpartner_bridge.so --lane=3
```

Or equivalently via the YAML transport section (in which case drop the
CLI `--transport` -- combining both is a startup error):

```yaml
transport:
  provider: /opt/partner/libpartner_bridge.so
  args: [--lane=3]
decoders:
  - id: 0
    type: pymatching
    # ... as above
```

Caller: whatever channel the partner wire terminates in (partner-defined).

**Recipe 9 -- in-process, no server (for completeness).**
No decoding_server, no YAML: the application process runs the dispatch
loop itself via the builtin channel spec:

```
CUDAQ_DEVICE_CALL_CHANNEL=host_dispatch
```

or per-device (decoder 0 host, decoder 1 device):

```
CUDAQ_DEVICE_CALL_CHANNEL=host_dispatch,1=device_dispatch
```

See `per_decoder_rings_design.md` for the in-process topology and the
per-device channel-spec semantics.

### 4.4 Dispatch shape vs wire (the two-axis split)

Per-decoder YAML key `dispatch: host | device_graph` (`DecoderDispatch`)
names HOW RPCs execute; `--transport` names the wire.  They are orthogonal:

- `host`: HOST_CALL through the CQR plugin dispatcher; any provider.
- `device_graph`: `DeviceGraphTransceiver` (decoding-server-cqr) -- the GPU
  device-graph dispatch ENGINE, transport-blind.  It consumes a bridge
  provider itself (hololink by default; the YAML transport section,
  `--transport` fallback, or CUDAQ_REALTIME_BRIDGE_LIB select another),
  REQUIRES v2 geometry+endpoint
  queries, and configures the provider generically: the named env knobs
  `QEC_DEVICE_GRAPH_{DEVICE,PEER_IP,REMOTE_QP,FRAME_SIZE,PAGE_SIZE,
  NUM_PAGES}` each become a `--<flag>=` argument ONLY when set (they match
  the hololink provider's flags); `QEC_DEVICE_GRAPH_PROVIDER_ARGS` (or the
  YAML transport section's args, which the server forwards through it) is
  the free-form pass-through for providers with a different argument
  surface; `--gpu=` is always passed and is the decoder's cuda_device_id
  (not an env knob); RESERVED_SMS is consumed by DecodingSession.
  Providers should ignore arguments they do not recognize.  After launch
  the transceiver publishes the provider's endpoint description VERBATIM
  as one line -- `QEC_DECODING_SERVER_ENDPOINT <key=value ...>` -- and
  never parses it; the orchestration layer scrapes whatever rendezvous
  tokens its wire uses (qp=/rkey=/buffer_addr= for RDMA playback).  It
  wires
  `cudaq_create/launch/destroy_dispatch_graph_regular` (dlsym) + the
  proprietary populate shims onto the provider's GPU rings.  Lives in the
  `cudaq-qec-decoding-server-device-graph` STATIC component behind PR 670's
  weak factory `cudaqx_qec_make_device_graph_transceiver` (WHOLE_ARCHIVE
  consumers; link canary target).  Gate: `CUDAQ_QEC_DEVICE_GRAPH_AVAILABLE`
  = realtime headers + libcudaq-realtime + cudaq-realtime-dispatch + CUDA
  toolkit.  NO HSB/DOCA/ibverbs at build time (verified: no such DT_NEEDED
  in the server binary).

There are deliberately NO legacy aliases (all-new component this cycle): no
`transport:` YAML key, no `--transport=gpu_roce`, no HOLOLINK_* env
fallback, no CpuRoceTransceiver stub.  YAML alias parsing note for anyone
re-adding compatibility keys: a second `mapOptional` onto the same field
must use the two-arg form, or an absent key resets the value to the default.

## 5. Reproduction plan (what an agent should do)

1. cuda-quantum worktree at PR 4869 head merged with origin/main (merge was
   clean).  Build realtime + full CUDA-Q to a FRESH prefix (e.g.
   /usr/local/cudaq-4869) via cudaqx scripts/build_cudaq_with_realtime.sh
   (CUDAQ_SRC must be a dir literally named `cudaq`).  Verify cudaqx main
   already contains the CUDA-Q pin bump compatible with 4869 (PR 654/675).
2. cuda-quantum commits, in order: (a) bridge interface v2 + loader rules +
   CUDAQ_ERR_UNSUPPORTED + fix caching-before-validation; (b) udp provider
   (+ reorder lib/ subdirs: cpu_transport before daemon); (c) cpu_roce
   provider (absorb rendezvous + hsb_fpga from the cudaqx server);
   (d) hololink provider v2 queries (syntax-check only without HSB/DOCA).
3. cudaqx branch off main: rewrite the server per 4.2 (drop all transport
   code + host-dispatch/ibverbs/transport link deps; dispatcher object;
   keep READY/DISPATCHED contracts byte-compatible).
4. Merge PR 670 (component split + weak factory + CMP0126 fix), then
   refactor GpuRoceTransceiver into DeviceGraphTransceiver as a bridge
   consumer (4.4), then the dispatch/transport split, then remove aliases.
5. Point the cudaqx build at the new prefix:
   `-DCUDAQ_DIR=<prefix>/lib/cmake/cudaq -DCUDAQ_REALTIME_ROOT=<prefix>`;
   pass `-DCUDAQ_QEC_REALTIME_CUDEVICE_PROPRIETARY_ARCHIVE=<...>.a` to
   enable the device-graph component end to end.

## 6. Validation gates

- Two-process udp suite (`test_decoding_server`): 5/5, including dual
  decoders over one ring.
- `test_decoding_server_core`: 5/5; pymatching realtime: 6/6.
- Negative: `--transport=/nonexistent.so` and `--transport=bogusname` fail
  with the resolved path in the message; device_graph without the hololink
  provider .so fails at provider load with an actionable error; missing
  QEC_DEVICE_GRAPH_* env means the flags are simply not passed (the
  provider applies its own defaults or rejects the bring-up itself).
- Dependency hygiene: `objdump -p decoding_server | grep NEEDED` shows no
  doca/hololink/ibverbs.
- Known pre-existing failure (NOT caused by this work):
  PyMatchingDeviceCallRealtime.HostDispatch "null dispatch session" fails
  identically on untouched main + the old install.

## 7. Follow-ups (recorded, not done)

- Rig validation: build cudaq-realtime-bridge-hololink from the branch
  (verifies the v2 edits compile against real HSB headers) and run the HSB
  script end to end.
- Upstream the interface-v2 commit to cuda-quantum with the version-pairing
  rule called out; propose HOST_CALL handler-context (`host_fn(ctx, ...)`)
  and multi-writer-ring slot discipline for the fan-in topology.
- SIGHUP/RPC session-epoch reload (the top-level `transport:` YAML section
  and one-ring-per-decoder are DONE -- see 4.2 and
  per_decoder_rings_design.md).
- Provider-internal QP fan-in for one-to-many decoder:QP topologies;
  `vp_id` in the enqueue_syndromes payload for the syndrome mapping table
  (cudaq-spec decoder_server_runtime.md).  (The bridge loader is now keyed
  by library name/path -- `cudaq_bridge_create_from_library` -- so one
  process can mix provider libraries; the enum API remains as a wrapper.)
- Public-surface cleanup in cudaq-realtime: deprecate the raw ring loop,
  graph_launch_engine surface, transport wrapper headers, and hololink
  headers; replace the `cudaq*` version-script wildcard with an explicit
  export list.
