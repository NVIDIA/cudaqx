# Extend cudaq_qec: define new codes and new decoders

For agents implementing new functionality, not running existing
workflows. For running, see `decode.md`.

## Define a new code

In Python for prototyping or C++ for production.

**Python template**: `docs/sphinx/examples/qec/python/my_steane.py` and
`my_steane_test.py`.

**Steps (Python)**

1. Define `@cudaq.kernel` functions for each operation you support
   (`prep0`, `stabilizer`, optionally `x`, `z`, `h`, ...). Each takes a
   `qec.patch` and acts on its `data`, `ancx`, `ancz` views.
2. Define a class decorated with `@qec.code("name")` that inherits from
   `qec.Code`. Set:

   - `self.stabilizers` — list of `cudaq.SpinOperator.from_word(...)`
   - `self.pauli_observables`
   - `self.operation_encodings` — mapping from `qec.operation.{...}` to
     your kernels

   Override `get_num_data_qubits`, `get_num_ancilla_x_qubits`,
   `get_num_ancilla_z_qubits`, `get_num_ancilla_qubits`,
   `get_num_x_stabilizers`, `get_num_z_stabilizers`.
3. Use it via `qec.get_code("name")` once the module is imported.

**C++**: read the "Implementing a New Code" section of
`docs/sphinx/components/qec/introduction.rst`. Subclass
`cudaq::qec::code`, register kernels in `operation_encodings`, set
`m_stabilizers`, then register the type with
`CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(name, ...)` and
`CUDAQ_REGISTER_TYPE(name)`. Reference implementation:
`libs/qec/include/cudaq/qec/codes/steane.h` together with the matching
`.cpp` in `libs/qec/lib/codes/`.

**Self-check**: `qec.get_code("name").get_stabilizers()` returns the
stabilizers you defined. Running the code-capacity workflow against
the new code shows zero logical errors at `p=0`.

## Define a new decoder

In Python or C++.

**Steps (Python)**

1. Decorate the class: `@qec.decoder("my_decoder")`.
2. `__init__(self, H, **kwargs)` must call `qec.Decoder.__init__(self, H)`.
3. `decode(self, syndrome)` returns a `qec.DecoderResult()` with
   `.converged: bool` and `.result: list[float]` of length
   `block_size`.

**C++**: subclass `cudaq::qec::decoder`. The full virtual surface lives
in `libs/qec/include/cudaq/qec/decoder.h`, including:

- `decode_async`, `decode_batch`
- the realtime API: `set_O_sparse`, `set_D_sparse`,
  `enqueue_syndrome`, `get_obs_corrections`, `reset_decoder`

Register the type with `CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION` and
`CUDAQ_REGISTER_TYPE`.

**Self-check**: `qec.get_decoder("my_decoder", H)` returns an instance.
Running the code-capacity workflow with this decoder produces a
sensible LER (zero at `p=0`).

## Where to look in the source

| Need                                  | File                                                                          |
|---------------------------------------|-------------------------------------------------------------------------------|
| Python decoder protocol               | `libs/qec/python/cudaq_qec/__init__.py` (search `Decoder`)                    |
| C++ decoder base + realtime API       | `libs/qec/include/cudaq/qec/decoder.h`                                        |
| C++ code base + `patch` type          | `libs/qec/include/cudaq/qec/code.h`, `libs/qec/include/cudaq/qec/patch.h`     |
| Built-in code reference               | `libs/qec/include/cudaq/qec/codes/steane.h`, `lib/codes/steane.cpp`           |
| Built-in decoder plugins (Python)     | `libs/qec/python/cudaq_qec/plugins/decoders/`                                 |
| Built-in decoder plugins (C++)        | `libs/qec/lib/decoders/`                                                      |
