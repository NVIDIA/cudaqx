# PySCF REST server (`cudaq-pyscf`)

`solvers.create_molecule(...)` does not run PySCF in-process. It
spawns a long-running REST server (`cudaq-pyscf --server-mode`) on
`localhost:8000` and talks to it over HTTP. Understanding this is the
difference between "my second call hangs" and "fixed in 5 seconds."

## Source of truth

- Server script: `libs/solvers/tools/molecule/cudaq-pyscf.py`
- C++ client: `libs/solvers/lib/operators/molecule/drivers/pyscf_driver.cpp`
- Python entry point: `solvers.create_molecule` in
  `libs/solvers/python/cudaq_solvers/__init__.py`

## What spawns the server

The first `solvers.create_molecule(...)` call in a Python process
launches `cudaq-pyscf --server-mode` as a child process bound to
`127.0.0.1:8000`. Subsequent `create_molecule` calls reuse the same
server (HTTP keep-alive). The server exits when the parent Python
process exits — *usually*. Crashed parents can leave it orphaned.

## Anatomy of the server

```
Python (cudaq_solvers)
  │
  │ HTTP request (geometry, basis, options)
  ▼
localhost:8000   <—  cudaq-pyscf (FastAPI + uvicorn)
                       │
                       ▼
                     PySCF (in the server's Python)
                       │
                       ▼
                     Hamiltonian generators (UCCSD, GAS, etc.)
```

The server is **plugin-extensible**: any module under
`cudaq_solvers.tools.molecule.pyscf.generators` is auto-discovered as
a Hamiltonian generator. Adding a generator means adding a module
there; no server code changes.

## The three states port 8000 can be in

| State | Symptom | Fix |
|---|---|---|
| Free | First `create_molecule` succeeds | (nothing to do) |
| Owned by *this* process's server | Subsequent `create_molecule` calls are fast | (nothing to do) |
| Owned by an orphaned server from a *previous* crashed Python process | New `create_molecule` hangs forever | `lsof -n -i :8000 && kill -9 <pid>` |

```bash
# Always-safe pre-check before a new Python session:
lsof -n -i :8000     # should be empty
```

## When the server hangs

The standard hang pattern: previous Python session crashed (segfault,
`kill -9`, container restart) while the server was still alive. The
new session can't bind to 8000, but `create_molecule` doesn't check
ownership — it just waits.

Recovery:

```bash
lsof -n -i :8000
# COMMAND  PID  USER  ...  cudaq-py 12345  vsarav  ...
kill -9 12345
```

If `lsof` isn't installed: `ss -ltnp | grep 8000` works the same.

## Configuration

The server is normally not configured directly — `create_molecule`
options become HTTP request fields. The exceptions:

| Need | Parameter |
|---|---|
| Use a different port | currently hard-coded to 8000 in `pyscf_driver.cpp`; would need a code change |
| Use a Python other than `sys.executable` | the server child uses `sys.executable` from the calling process, so activate the right venv before importing `cudaq_solvers` |
| Verbose PySCF logs | `solvers.create_molecule(..., verbose=True)` → forwarded to PySCF |
| Big memory budget | `solvers.create_molecule(..., memory=8000.0)` (MB) — forwarded |

## Working with the server during development

Launch it manually for debugging:

```bash
python libs/solvers/tools/molecule/cudaq-pyscf.py --server-mode &
# now interrogate the API
curl http://localhost:8000/docs   # FastAPI auto-doc
```

Kill it cleanly:

```bash
curl -X POST http://localhost:8000/shutdown 2>/dev/null || pkill -f cudaq-pyscf
```

## Common pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `create_molecule` hangs forever | port 8000 occupied | `lsof -n -i :8000 && kill -9 <pid>` |
| Server starts but uses wrong Python | venv not activated before import | activate the env, then `import cudaq_solvers` |
| `ModuleNotFoundError: cudaq_solvers.tools.molecule.pyscf.generators.<name>` | a generator plugin missing its `__init__.py` | check the package layout |
| Coefficient signs change run-to-run | unrelated to the server — see Convention #1 (`OMP_NUM_THREADS=1`) |
| Server stays alive after parent exits | parent crashed before clean shutdown | `pkill -f cudaq-pyscf` |

## Self-check

```
[ ] lsof -n -i :8000 is empty before the first create_molecule in a fresh session.
[ ] PYTHONPATH / venv matches between the parent and the server.
[ ] Pre-existing server cleared with kill -9 after any crash.
[ ] `curl http://localhost:8000/docs` returns HTML after first create_molecule.
```

## Where next

- Build the molecule and get a Hamiltonian:
  `references/molecule-building.md`.
- Pick the fermion-to-qubit mapping: `references/fermion-mappings.md`.
- Choose an operator pool: `references/operator-pools-using.md`.
- Server hangs but it's something else: see Convention table in
  `cudaq-solvers-chemistry/SKILL.md`.
