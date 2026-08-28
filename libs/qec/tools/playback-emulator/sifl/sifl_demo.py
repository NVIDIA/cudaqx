#!/usr/bin/env python3
"""SIFL demo: decoding a rotated surface-code memory experiment whose per-shot round count
is only decided at run time, for any count up to max_rounds. Template building (a full DEM
analysed from scratch for each round count, no chunking or stitching) lives in
dem_templates.py.

Which decoder does the work is a runtime option, and it is the only thing that varies:
`--decoder pymatching` and `--decoder nv-qldpc` run the same circuit, schedule and
cadences, so the two are directly comparable and any difference between them belongs to
the decoder. BP runs all its iterations on every shot (it does not converge on this code)
and pays a fixed GPU dispatch per shot, so it is much the slower of the two.

Two identical decoding-server rings (decoder_id 0 and 1) with shots alternating between
them, which is what lets shot i's decode overlap shot i+1's rounds. The schedule is one
timeline addressing both rings, and each line is one RPC to a ring, in the order the ring
receives it. Shot i's rounds stream to ring i%2 until shot i-1's answer comes up -- that
is the whole SIFL shape, in one operand.

The read carries `signal=shot{i}`, which means it does not block: the request goes out
where the line sits and the signal comes up later, when the answer lands. That placement
is what matters, because a ring keeps a shot's result only until the next round arrives,
so the read has to be in front of shot i+2's rounds. Writing it in the same timeline as
the syndromes puts it there by construction, and not blocking is what keeps that timeline
running while the decode happens.

The decoding server is spawned for this process and killed when it exits.

Stim supplies the circuit and its noise. The two run() calls differ only in tick_ns,
sampling one decoder-latency curve at two syndrome cadences.

Usage:
  sifl_demo.py                       # pymatching, over UDP to a decoding server
  sifl_demo.py --decoder nv-qldpc    # GPU belief propagation instead
  sifl_demo.py --inproc              # no server: decoders realized in this process
"""
import argparse, cudaq_qec as qec, numpy as np, os, re, stim, subprocess, tempfile
from dem_templates import TEMPLATE_ROOT, build_templates

pb = qec.playback
SERVER_BIN = os.path.join(os.path.dirname(os.path.dirname(qec.__file__)), "bin", "decoding_server")

# The whole of the difference between the two decoders: a delegate name and its
# parameters, handed to multi_round_decoder's sub-decoders unchanged.
DELEGATES = {
    "pymatching": dict(delegate_type="pymatching"),
    "nv-qldpc": dict(delegate_type="nv-qldpc-decoder",
                     delegate_params=dict(use_sparsity=True, max_iterations=30,
                                          use_osd=False)),
}

def require_delegate(delegate_type):
    """nv-qldpc-decoder is a plugin with a CUDA GPU requirement; without this the failure
    surfaces as the decoding server dying during startup."""
    try:
        qec.get_decoder(delegate_type, np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8))
    except Exception as e:
        raise SystemExit(f"{delegate_type} unavailable: {e}")

def build_schedule(shots, stream_cap, preamble_rounds=3):
    """The syndrome timeline and its reads, in one file-ordered timeline.

    `stream_cap` is how long a stream is willing to wait, which is NOT the
    decoder's own max_rounds -- see run().
    """
    # Every line is an RPC to a ring, in the order that ring sees it: the
    # shot's rounds, its data readout, then the read of its result -- which
    # has to be ahead of the next shot's rounds on that same ring.
    lines = [f"0 stream source=0 rounds={preamble_rounds}",
             f"- enqueue_data source=0",
             f"- get_corrections return_size=1 signal=shot0"]
    for i in range(1, shots):
        ring = i % 2
        lines += [f"- stream session={ring} source=0 every=1 min_rounds=1 "
                  f"max_rounds={stream_cap} until=shot{i - 1}",
                  f"- enqueue_data session={ring} source=0",
                  f"- get_corrections session={ring} return_size=1 signal=shot{i}"]

    return "\n".join(lines) + "\n"

def make_multi_config(template_dir, round_width, terminal_width, max_rounds, delegate):
    def make_config(decoder_id):
        config = qec.decoder_config()
        config.id, config.type = decoder_id, "multi_round_decoder"
        config.block_size, config.syndrome_size = 10_000_000, 1
        config.H_sparse, config.O_sparse, config.D_sparse = [9999999, -1], [0, -1], [9999999, -1]
        config.decoder_custom_args = dict(
            template_dir=template_dir, round_width=round_width, terminal_width=terminal_width,
            max_rounds=max_rounds, num_obs=1, **delegate)
        return config
    multi = qec.multi_decoder_config()
    multi.decoders = [make_config(0), make_config(1)]
    return multi

def spawn_server(multi):
    """One decoding server for this process's lifetime, one ring per decoder."""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(multi.to_yaml_str()); cfg_path = f.name
    proc = subprocess.Popen([SERVER_BIN, f"--config={cfg_path}", "--port=0", "--timeout=600",
                             "--num-slots=32768"], stdout=subprocess.PIPE)
    ports = {}
    for line in iter(proc.stdout.readline, b""):
        if b"QEC_DECODING_SERVER_READY" not in line:
            continue
        for m in re.finditer(rb"ring(\d+)=(\d+)", line):
            ports[int(m.group(1))] = int(m.group(2))
        break
    os.unlink(cfg_path)
    if len(ports) != len(multi.decoders):
        proc.terminate()
        raise RuntimeError(f"decoding_server did not report all ring ports: got {ports}")
    return proc, ports

def run(label, backend, stim_params, max_rounds, tick_ns, shots=15):
    preamble_rounds = 10
    sched = build_schedule(shots, stream_cap=max_rounds - preamble_rounds,
                           preamble_rounds=preamble_rounds)
    print(sched)

    result = pb.run(sched, tick_ns=tick_ns, **backend,
                    sources={0: pb.stim_memory_source(1, **stim_params)})
    recs = [r for r in result.records if r.op == pb.operation.stream]
    rounds = [r.rounds_streamed for r in recs]
    gc = [r for r in result.records if r.op == pb.operation.get_corrections]
    time_ms = [(r.return_ns - r.call_ns)/1e6 for r in gc]
    status = [r.status for r in gc]
    decoded = sum(1 for r in gc if r.read_completed)
    print(f"{label} rounds_streamed={rounds} time={time_ms}\n"
          f"{' ' * (len(label) + 9)} status={status}  shots_decoded={decoded}/{len(gc)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--decoder", choices=sorted(DELEGATES), default="pymatching",
                        help="Which delegate multi_round_decoder hands each shot to. "
                             "Everything else about the run is identical.")
    parser.add_argument("--inproc", action="store_true",
                        help="Realize the decoders in this process instead of talking to a "
                             "decoding server. No sockets and no subprocess, so the same "
                             "schedule can be run as a test; the decoders are rebuilt on "
                             "every pb.run() call, which for max_rounds=150 is most of the "
                             "wall clock.")
    args = parser.parse_args()
    delegate = DELEGATES[args.decoder]
    require_delegate(delegate["delegate_type"])

    # A shot's whole round history is flushed as one unpaced burst of enqueue RPCs, and
    # decoding_server's receive socket silently drops them past roughly 220 frames, so
    # max_rounds is capped well short of that. See /workspaces/bug for the diagnosis.
    D, P, MAX_ROUNDS = 7, 0.01, 150
    gen = lambda rounds: stim.Circuit.generated("surface_code:rotated_memory_z", rounds=rounds,
                                                distance=D, before_measure_flip_probability=P,
                                                after_clifford_depolarization=P)
    round_width = gen(2).num_measurements - gen(1).num_measurements
    stim_params = {"code": "surface_code", "task": "rotated_memory_z",
                   "distance": D, "rounds": 10_000,
                   "before_measure_flip_probability": P,
                   "after_clifford_depolarization": P}
    template_dir = f"{TEMPLATE_ROOT}/d{D}_p{P}_scratch"
    build_templates(gen, template_dir, MAX_ROUNDS)
    multi = make_multi_config(template_dir, round_width,
                              gen(1).num_measurements - round_width, MAX_ROUNDS, delegate)

    proc = None
    if args.inproc:
        backend = dict(decoders=multi)
    else:
        proc, ports = spawn_server(multi)
        backend = dict(udp_endpoints={r: f"127.0.0.1:{p}" for r, p in ports.items()})
    try:
        run(f"slow cadence [{args.decoder}]", backend, stim_params, MAX_ROUNDS, tick_ns=200_000)
        run(f"fast cadence [{args.decoder}]", backend, stim_params, MAX_ROUNDS, tick_ns=5_000)
    finally:
        if proc:
            proc.terminate(); proc.wait()
