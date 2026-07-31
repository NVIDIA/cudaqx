# Decoding server live configuration

`decoding_server` reads its decoder and transport configuration from the YAML
passed with `--config`. On POSIX systems, send the process `SIGHUP` after
atomically replacing that file to apply decoder changes without rebinding its
transport providers, per-decoder rings, or published endpoints.

```bash
kill -HUP <decoding-server-pid>
```

Apply `SIGHUP` only between complete shots. The server stops admitting new
requests, drains work already queued in each affected host session, and stops
an affected device-graph scheduler before releasing its captured CUDA graph.
It then constructs the replacement decoder and relaunches the scheduler on the
same provider rings. Host requests during this interval fail with `NOT_READY`.
For device-graph reload, keep clients idle until the server reports
`QEC_DECODING_SERVER_CONFIG_APPLIED`.

Live apply intentionally preserves the ring topology. The following changes
require a process restart and are rejected while the old configuration remains
active:

- adding, removing, or renumbering decoder IDs;
- changing a decoder between `host` and `device_graph` dispatch;
- changing the transport provider or its arguments.

Decoder types, matrices, dimensions, and decoder custom arguments may change
while the decoder ID and dispatch shape remain stable. Host decoders may also
change CUDA placement; a device-graph decoder must stay on the GPU that owns
its preserved provider rings.
The standalone all-`device_graph` path supports live reload and currently
supports exactly one decoder. A mixed host/`device_graph` configuration still
requires a restart when its device-graph decoder changes.

Malformed YAML, schema failures, and topology changes print
`QEC_DECODING_SERVER_CONFIG_REJECTED old_config_active` and leave the old
sessions serving. To avoid a temporary double allocation for large decoders,
the old decoder resources are released before replacement construction. If
construction or device-scheduler relaunch fails, the affected IDs return
`NOT_READY` and the daemon prints
`QEC_DECODING_SERVER_CONFIG_FAILED awaiting_config`; correct the file and send
`SIGHUP` again. Applying an unchanged configuration is safe and reports
`QEC_DECODING_SERVER_CONFIG_APPLIED`.
