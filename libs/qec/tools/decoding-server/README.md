# Decoding server live configuration

`decoding_server` reads its decoder and transport configuration from the YAML
passed with `--config`. On POSIX systems, send the process `SIGHUP` after
atomically replacing that file to apply host-decoder changes without rebinding
its transport providers, per-decoder rings, or published endpoints.

```bash
kill -HUP <decoding-server-pid>
```

Apply `SIGHUP` only between complete shots. The server stops admitting new host
RPCs, drains work already queued in each affected session, releases the old host
decoders, constructs the replacements, and then resumes admission. Requests
that arrive during this interval fail with `NOT_READY`.

Live apply intentionally preserves the ring topology. The following changes
require a process restart and are rejected while the old configuration remains
active:

- adding, removing, or renumbering decoder IDs;
- changing a decoder between `host` and `device_graph` dispatch;
- changing the transport provider or its arguments; and
- changing any `device_graph` decoder entry, because its captured CUDA graph is
  bound to a live GPU ring consumer.

Host decoder types, matrices, dimensions, CUDA placement, and decoder custom
arguments may change while the decoder ID and dispatch shape remain stable.

Malformed YAML, schema failures, and topology changes print
`QEC_DECODING_SERVER_CONFIG_REJECTED old_config_active` and leave the old
sessions serving. To avoid a temporary double allocation for large decoders,
the old host resources are released before replacement construction. If that
construction fails, the affected host IDs return `NOT_READY` and the daemon
prints `QEC_DECODING_SERVER_CONFIG_FAILED awaiting_config`; correct the file and
send `SIGHUP` again. Applying an unchanged configuration is safe and reports
`QEC_DECODING_SERVER_CONFIG_APPLIED`.
