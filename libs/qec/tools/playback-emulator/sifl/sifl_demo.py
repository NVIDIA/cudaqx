#!/usr/bin/env python3
"""Minimal SIFL (Steady-state Inter-circuit Feed-forward Latency) demo: sweeps
dummy_sifl_decoder's `us_per_bit` timing knob and shows the closed loop
staying stable (bounded rounds_streamed) or running away (unbounded growth)
depending on whether decode time keeps up with the syndrome rate. Millisecond-
scale timing (TICK_NS/PERIOD_TICKS below) is required to see this cleanly --
microsecond-scale pacing is dominated by the emulator's own host-dependent
scheduling jitter rather than the decoder-vs-syndrome-rate race being tested.

The syndrome source is a real rotated surface-code memory circuit (stim). A
fixed 5-round bootstrap (`enqueue`) primes the source with stabilizer rounds
before the first data-qubit readout (`enqueue_data`); after that, each cycle's
readout fires a decode and the following `stream_until` streams more rounds
while it runs, pipelining decode and syndrome collection."""

import cudaq_qec as qec
import stim

CYCLES = 20
BOOTSTRAP_ROUNDS = 5
TICK_NS = 1_000_000
PERIOD_TICKS = 2
MAX_ROUNDS = 5000

def rounds_streamed(us_per_bit):
    pb = qec.playback
    circuit = stim.Circuit.generated("surface_code:rotated_memory_z", rounds=1_000_000,
                                     distance=3, after_clifford_depolarization=0.001,
                                     before_measure_flip_probability=0.001)
    syndrome_source = pb.stim_memory_source(str(circuit), 1)

    config = qec.multi_decoder_config.from_yaml_str(f"""
decoders:
  - id: 0
    type: dummy_sifl_decoder
    block_size: 10000000
    syndrome_size: 1
    H_sparse: [9999999, -1]
    O_sparse: [0, -1]
    D_sparse: [9999999, -1]
    decoder_custom_args:
      us_per_bit: {us_per_bit}
      num_obs: 1
      bits_per_shot: {syndrome_source.data_width()}
""")
    # Initial enqueues: prime the source with a fixed stabilizer-round bootstrap.
    sched_text = "".join(f"{i} 0 enqueue source=0\n" for i in range(BOOTSTRAP_ROUNDS))
    # SIFL rounds: each cycle's readout fires a decode; stream_until streams
    # more rounds while it runs and absorbs the correction once ready.
    sched_text += "".join(
        f"{BOOTSTRAP_ROUNDS + i * PERIOD_TICKS} 0 enqueue_data source=0\n"
        f"{BOOTSTRAP_ROUNDS + i * PERIOD_TICKS} 0 stream_until source=0 every=1 "
        f"max_rounds={MAX_ROUNDS} timeout=5s 0\n"
        for i in range(CYCLES))

    result = pb.run(sched_text, tick_ns=TICK_NS, sources={0: syndrome_source},
                    decoders=config)
    return [r.rounds_streamed for r in result.records if r.op == pb.operation.stream_until]

for us_per_bit in [10, 50, 100]:
    rounds = rounds_streamed(us_per_bit)
    verdict = "runaway" if rounds[-1] > 3 * rounds[0] else "stable"
    print(f"us_per_bit={us_per_bit:>4}  rounds_streamed={rounds}  [{verdict}]")
