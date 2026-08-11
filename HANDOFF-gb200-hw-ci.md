# Handoff: decoding-server hardware CI — GB200 validation

**TEMPORARY FILE — do not commit. DELETE THIS FILE once everything is
passing on the GB200** (Chuck's instruction: it exists only for this
bring-up).

You are a fresh Claude Code instance on the GB200 HOST. The hardware CI
(`docker/decoding-server/hw_ci/run_hw_ci.sh`) has been fully validated on a
DGX Spark in both of its cable configurations; your job is the GB200 leg:
verify prerequisites, run the suite in the GB200 configuration, and
report/fix what breaks.

## Ground rules (the user is Chuck; these are standing preferences)

- **Never run `git commit` or `git push`** (or `git add`/`git reset` that
  alters staging). Put code changes directly into the working tree of the
  clone you are told to use and present a proposed commit message in your
  reply; Chuck makes every commit himself.
- Work from a clone of **Chuck's fork** (`cketcham2333/cudaqx`), branch
  `decoding-server-hw-ci`.
- Cheap probe before long runs: `--list`, then a single udp/pymatching
  lane, then one FPGA lane, then the full set.
- Answer Chuck's clarifying questions standalone; don't bundle them with
  new question prompts.
- When Chuck rules something out about his own environment ("I never
  updated X"), take it as ground truth and redirect the investigation.
- Do not generalize hardware capability from one device in one boot state:
  sweep all devices and re-test across reboots before concluding "this
  platform can't do X". (This lesson was paid for on the Spark.)

## State of the work

Branch `decoding-server-hw-ci` on the fork. Expected tip: `c7c740d` ("Fix
decoding-server hw CI issues found in DGX Spark host validation") plus a
follow-up commit (BlueFlame sed removal, Spark FPGA port doc, bridge-lane
`--spacing 100`, build-message clarity). **Verify before starting** that
your checkout has the follow-up: the gpu-roce-qldpc-bridge lane in
`run_hw_ci.sh` must pass `--spacing 100`, and `dev.Dockerfile` must NOT
sed `DOCA_SEND_BLUE_FLAME`. If those are missing, ask Chuck whether the
follow-up commit was pushed.

## Already validated on the Spark (do not redo)

- Loopback config (`--no-fpga --roce-pair rocep1s0f0,rocep1s0f1`):
  13 passed / 0 failed / 9 named skips.
- FPGA config (`--fpga-device roceP2p1s0f0`): 15 passed / 0 failed /
  7 named skips — all four decoders over udp and the FPGA source,
  device_graph GPU dispatch, hsb-fpga-server on both wires, the
  gpu-roce-qldpc-bridge, ising-prepare (gated HF download + on-GPU
  export), and the qldpc-graph / mixed-dispatch ctests.
- Tokenless trt fallback (staged `ising-bundle/` in the artifacts dir)
  and `--hf-token-prompt` were added and validated for public-account use.

## Your job on the GB200 (cheapest first)

1. Prereqs: docker + nvidia-container-toolkit, user in `docker` group;
   `ghcr.io/nvidia/cudaqx-dev` pulls anonymously (verify with
   `docker manifest inspect` of the pin-matched tag: shortref from
   `jq -r .cudaq.ref .cudaq_version | head -c8`, tag
   `<shortref>-arm64-cu13.0`); ~60 GB free disk.
2. `sudo modprobe rdma_rxe` for the SoftRoCE mode (see Known gaps below
   before spending time here).
3. Proprietary artifacts staged (default `/opt/nvqlink-lab-artifacts`):
   `decoder-plugins/libcudaq-qec-nv-qldpc-decoder.so` and
   `cudevice/libcudaq-qec-realtime-cudevice-proprietary.a` — must cover
   sm_100 (GB200). Do NOT reuse the Spark's copies blindly (those came
   from a GB10 dev build; verify arch coverage or rebuild).
4. **Ask Chuck which IB device faces the FPGA** — do not guess from link
   state. On the Spark, a wrong-but-linked `--fpga-device` passed the HSB
   control plane and BRAM verification but failed with
   `ILA: captured 0 of N expected samples`. That symptom = wrong port.
5. Probe: `run_hw_ci.sh --list`, then
   `--only 'examples/qpu-kernel/udp/pymatching'`, then one FPGA lane
   (`--only 'examples/fpga/cpu_roce/pymatching'`), then the full run:

   ```
   ./docker/decoding-server/hw_ci/run_hw_ci.sh \
       --repo <clone-or-fork-url> --sha decoding-server-hw-ci \
       --roce-pair rxe --fpga-device <ibdev-from-Chuck> \
       --artifacts-dir /opt/nvqlink-lab-artifacts
   ```

   HF token: the GB200 runs under a PUBLIC account. Preferred: stage the
   pre-built Ising bundle at `<artifacts-dir>/ising-bundle/` (copy from
   the Spark: `~/nvqlink-lab-artifacts/ising-bundle/`) and run tokenless —
   the trt lanes then run and ising-prepare SKIPs with a named reason.
   If Chuck wants the HF download path exercised, use `--hf-token-prompt`
   (interactive, nothing written to disk); never store a token file on a
   shared account.

## Known gaps / expected results on GB200

- `--roce-pair rxe` SKIPs the two-process cpu_roce lanes: the image's
  Mellanox ibverbs-providers has no rxe userspace provider (README
  documents it; the skip is loud). Fixing it means building the rxe
  provider from Mellanox rdma-core 2601 source into the image — a known
  follow-up, not a bug you introduced.
- `extra/ctest/ai-decoder-fp8` reports "not registered" everywhere until
  `CUDAQX_QEC_ENABLE_REALTIME_PIPELINE` is re-enabled upstream.
- Multi-decoder two-process tests (DualDecoders, num-logical-2,
  PerDecoderRings) skip by name off udp — device-scoped cpu_roce endpoint
  args don't exist yet upstream.
- 64 KiB-page kernels (GB200 `-64k`): `derive_page_sizes` defaults the
  slot size to 512 and the device_graph lane rounds up to a
  host-page-compatible value. This logic has NEVER run on a real 64K
  host — watch the device-graph lane's geometry first.

## Hard-won lessons that may recur (from the Spark bring-up)

- **BlueFlame UAR failures are transient/stateful, not a platform
  property.** Symptom: gpu_roce lanes die at start with
  `gpu_roce_transceiver.cpp:314 ... Failed to create UAR: DOCA Driver
  call failure`. On the Spark this hit one ConnectX card for a whole
  afternoon and cleared on reboot, while NONCACHE always worked. Do NOT
  patch HSB (a NONCACHE sed was tried and deliberately removed). If it
  appears: sweep ALL devices with a minimal `doca_uar_create` reproducer
  (open device, try BLUEFLAME then NONCACHE), try another boot, and
  report. Upstream ask on record: runtime BF->NONCACHE fallback in HSB.
- NetworkManager (if the GB200 runs it) silently drops statically
  assigned pair addresses on its DHCP retry timer; the runner re-asserts
  addresses + waits for IPv4 RoCE GIDs before each cpu_roce lane, so this
  should be handled — the permanent fix is
  `sudo nmcli device set <netdev> managed no`.
- Root-owned clones: git needs `safe.directory` entries for BOTH the
  clone path and `<path>/.git` (local-path `--repo` accesses the repo by
  its `.git` path).
- `decoding_server` ignores SIGTERM while blocked in a cpu_roce
  rendezvous accept(); the test scripts carry TERM->KILL escalation and
  ctest lanes run with `--timeout 900`, so a wedge costs minutes, not
  hours. If you see a wedged server, that upstream bug is already on
  record.
- The runner's `ibdev2netdev` comes from `mlnx-ofed-kernel-utils` (modern
  `mlnx-tools` no longer ships it); the image installs it.

## Reporting

Summarize the PASS/FAIL/SKIP table per configuration, plus any fixes made
(working-tree diffs + a proposed commit message; never commit). Compare
against the Spark tables above: the GB200's expected deltas are the two
rxe skips (instead of cpu_roce passes) and FPGA lanes running on GB200's
port. When everything passes and Chuck confirms, **delete this file**.
