#!/usr/bin/env python3
"""SIFL demo: pymatching decoding of a rotated surface-code memory experiment whose
per-shot round count is only decided at run time, for any count up to max_rounds. Template
building (a full DEM analysed from scratch for each round count, no chunking or stitching)
lives in dem_templates.py, shared with sifl_demo_nv_qldpc.py.

Stim supplies the circuit and its noise. The two run() calls differ only in tick_ns,
sampling one decoder-latency curve at two syndrome cadences."""
import cudaq_qec as qec, os, re, stim, subprocess, tempfile
from dem_templates import TEMPLATE_ROOT, build_templates

pb = qec.playback
SERVER_BIN = os.path.join(os.path.dirname(os.path.dirname(qec.__file__)), "bin", "decoding_server")

def run(label, gen, template_dir, round_width, terminal_width, max_rounds, tick_ns, shots=15):
    # Build the multi_round_decoder config
    config = qec.decoder_config()
    config.id, config.type = 0, "multi_round_decoder"
    config.block_size, config.syndrome_size = 10_000_000, 1
    config.H_sparse, config.O_sparse, config.D_sparse = [9999999, -1], [0, -1], [9999999, -1]
    config.decoder_custom_args = dict(
        template_dir=template_dir, round_width=round_width, terminal_width=terminal_width,
        max_rounds=max_rounds, delegate_type="pymatching", num_obs=1)
    multi = qec.multi_decoder_config()
    multi.decoders = [config]
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(multi.to_yaml_str()); cfg_path = f.name

    # Start the decoding server and get its port
    proc = subprocess.Popen([SERVER_BIN, f"--config={cfg_path}", "--port=0", "--timeout=60"],
                            stdout=subprocess.PIPE)
    port = next(int(m.group(1)) for line in iter(proc.stdout.readline, b"")
                if (m := re.search(rb"port=(\d+)", line)))
    os.unlink(cfg_path)
    
    # Write the schedule
    sched = ("0 0 enqueue source=0\n"
            "1 0 enqueue source=0\n"
            "2 0 enqueue source=0\n"
            "3 0 enqueue_data source=0\n")
    for _ in range(shots):
        sched += f"+1 0 stream_until source=0 return_size=1 every=1 max_rounds={max_rounds}\n"
        sched += f"+1 0 enqueue_data source=0\n"
    print(sched)

    try:
        result = pb.run(sched, tick_ns=tick_ns, udp_endpoints={0: f"127.0.0.1:{port}"},
                        sources={0: pb.stim_memory_source(str(gen(10_000)), 1)})
        recs = [r for r in result.records if r.op == pb.operation.stream_until]
        rounds = [r.rounds_streamed for r in recs]
        ok = [r.status == "OK" for r in recs]
    finally:
        proc.terminate(); proc.wait()
    print(f"{label}  rounds_streamed={rounds}  ok={ok}")

if __name__ == "__main__":
    # A shot's whole round history is flushed as one unpaced burst of enqueue RPCs, and
    # decoding_server's receive socket silently drops them past roughly 220 frames, so
    # max_rounds is capped well short of that. See /workspaces/bug for the diagnosis.
    D, P, MAX_ROUNDS = 7, 0.01, 150
    gen = lambda rounds: stim.Circuit.generated("surface_code:rotated_memory_z", rounds=rounds,
                                                distance=D, before_measure_flip_probability=P,
                                                after_clifford_depolarization=P)
    round_width = gen(2).num_measurements - gen(1).num_measurements
    template_dir = f"{TEMPLATE_ROOT}/d{D}_p{P}_scratch"
    #build_templates(gen, template_dir, MAX_ROUNDS)
    args = (gen, template_dir, round_width, gen(1).num_measurements - round_width, MAX_ROUNDS)
    run("slow cadence    ", *args, tick_ns=200_000)
    run("fast cadence    ", *args, tick_ns=10_000)
