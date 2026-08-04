# Decoder model inputs: a design walkthrough

Note to reviewers:

This draft is not in a mergeable state. The intent is to get the design discussion started and have some concrete examples to look at. The design was done with Tracy's dynamic DEM PR in mind and was meant to be extensible to support DEM chunks, though there are certainly still rough edges. 

## A quick note on H/O/D

`H` has shape `detectors x error mechanisms`. Column `e` says which detectors fire when error
mechanism `e` occurs; this is the model a matrix-based decoder decodes against.

`O` has shape `observables x error mechanisms`. The same column `e` says which logical observables
that error mechanism flips. If a decoder predicts an error frame `x`, the observable correction is
`O * x` over GF(2). A decoder such as Chromobius can instead predict those observable flips
directly, but the meaning of O does not change.

`D` has shape `detectors x raw measurements`. Hardware sends measurement bits; the decoder consumes
detectors. `D * m` over GF(2) is the bridge between those two bases.

So H and O describe the decoding model. D is not part of the noise model, but it is dimensionally
bound to H's detector basis, and the base class owns the buffers and preprocessing derived from it.


## A quick note about decoders

This design is based on the standing convention that a decoder is immutable once constructed. E.g., its H, error rate, return type
are set at construction time and are not meant to change during the lifetime of the decoder instance. 

## The current problem

All I wanted to do was to enable Chromobius on the decoding server path.

But the road to Chromobius is fraught with false leads. On baseline `main` the server always builds a plugin from H, while
Chromobius can only be constructed from a raw Stim DEM. So Chromobius works perfectly well through
the offline DEM factory, and is unreachable through the server's matrix-only construction
path. Effectively, it is blocked from the decoding server.

Concretely, the server configuration carries H, O and D, but only H reaches the factory:

```cpp
// realtime_decoding.cpp, baseline main
auto decoder = cudaq::qec::get_decoder(
    decoder_config.type, pcm, prepare_decoder_params(decoder_config));   // pcm is H
decoder->set_decoder_id(decoder_config.id);
decoder->set_O_sparse(decoder_config.O_sparse);                          // O arrives later
decoder->set_D_sparse(decoder_config.D_sparse);                          // D arrives later
```

Chromobius, meanwhile, accepts the other arm of `decoder_init` and rejects the matrix arm:

```cpp
// chromobius.cpp, baseline main
const auto *dem_text = std::get_if<std::string>(&init);
if (!dem_text)
  throw std::runtime_error(...);
```

The old construction pathway is:

```cpp
using decoder_init = std::variant<sparse_binary_matrix, std::string>;
```

This string variant allows Chromobius to be constructible offline. However, the baseline
YAML schema requires the matrix branch unconditionally:

```cpp
// config.cpp, baseline main
io.mapRequired("block_size", config.block_size);
io.mapRequired("syndrome_size", config.syndrome_size);
io.mapRequired("H_sparse", config.H_sparse);
io.mapRequired("O_sparse", config.O_sparse);
io.mapRequired("D_sparse", config.D_sparse);
```
This means that simply teaching the server to choose the string arm would leave two authorities. 
A DEM-backed configuration would therefore need to supply H and O even though the DEM already
defines them. H and O might contradict DEM. 

And then, it gets worse:

### O arrives by a different road for every decoder

The realtime path needs O so that error-frame decoders can produce observable corrections. On
baseline main, when and how that O arrives depends on which decoder is being used and which path
the decode is on. In other words, we have an "all roads lead to Rome" situation, with some very
precise "turn left, then right, then left" call-order implications embedded.

Drawn out, with the worst case at the bottom:

```
  baseline main - how O reaches a decoder

  offline, PyMatching ....  params["O"] ------------> ctor --> this->set_O_sparse()
  offline, Chromobius ....  (read out of the DEM text; no O argument at all)
  realtime, top level ....  get_decoder(H, params) --> ctor
                                                       `-- then: set_O_sparse()

  realtime, TensorRT with a PyMatching global decoder - the same matrix, three times:

      server --(1)-- params["O"] ---------------------------> TensorRT ctor
      server --(2)-- params["global_decoder_params"]["O"] ---> PyMatching global-decoder ctor
      server --(3)-- set_O_sparse() -------------------------> after construction
                        ^
                        `-- (1) and (2) are selected by hardcoded decoder names
```


That trt+pymatching is the one that should set off an alarm. Common server code knows both a wrapper's
internal parameter convention and a particular global decoder's name, *hardcoded*. This opens the gate that a third party decoder
author will need to modify our source code in order to plug in a different global decoder. Baseline `main` says so itself:

```cpp
// realtime_decoding.cpp, baseline main
// PyMatching consumes the observable matrix through its params; other global
// decoders receive only the top-level O until they define a matching contract.
if (has_pymatching_global) { ... global_decoder_params.insert("O", O); }
```

In addition, O arrives three different times and is stored three times. Both plugins convert `params["O"]` and call `set_O_sparse()` on
themselves, so the matrix ends up living in the server's `decoder_config`, in the TensorRT object's
base member, and in the global decoder's base member — having passed through two parameter maps as a *dense
tensor* to get there. The third delivery then overwrites the first copy with the same content!!! 

That's not all and I certainly contributed to this, because:

### O carries two meanings at once

In baseline `main`, supplying O means "this is the observable matrix" and "return observables
instead of errors." It can also select a matching strategy in the case of pymatching:

```cpp
// pymatching.cpp, baseline main — inside `if (params.contains("O"))`
this->set_O_sparse(O_sparse);
this->set_result_type(decode_result_type::decode_to_obs);
decode_to_observables = true;
if (!merge_strategy_explicit)
  merge_strategy_enum = pm::MERGE_STRATEGY::INDEPENDENT;   // surprise bonus
```

A caller cannot supply O as model data while asking for an error
frame, even though that is useful for a server that wants to perform the projection itself. Nor can
the caller discuss output shape without also discussing whether O happened to be present in the config.

## What we propose

This draft separates decoder model data (H/O/D/DEM/error rate) from decoder knobs (iterations to run, strategy to use etc),
makes O mean O and nothing else, teaches the decoding server to accept either a raw DEM or H/O/D but not both
at the same time, and removes the late O/D setters (this part is debatable but the intention of removal was to keep a single
source of truth). 

In order:

1. **Give stable construction input a typed home:** `decoder_inputs`.
2. **Every path resolves it before construction** — an offline caller, a decoder the server builds
   directly from a `decoder_config` entry, and a decoder that a wrapper builds internally (TensorRT's
   global decoder, sliding window's inner decoders) all resolve the same value.
3. **Output form becomes an explicit construction argument**, because O is now a field and can no
   longer double as the request.
4. **The base sizes realtime state at construction**, since it finally knows the inputs then; the
   sliding-window subclass hands over its own streaming geometry rather than being `dynamic_cast`
   to. (This part is debatable. I ported sliding window for completeness's sake)
5. **The setters are now unused: delete them**, or keep them as assertions.

We now go over the above statements in detail below:

### One construction input, distinct from the knobs

`decoder_inputs` is a small immutable handle to shared construction state. It owns:

- H in sparse CSC form;
- optional O in sparse CSR form;
- error rates and optional error IDs, indexed by H column;
- optional D in sparse CSR form;
- the authoritative source kind and, for a Stim source, the raw DEM text; and
- dimensions as metadata, so asking for a size does not force a future compact source to
  materialize a matrix.

The public surface, abridged:

```cpp
class decoder_inputs {
  // Build it from whichever source is authoritative. D is optional: it is only
  // meaningful for a decoder fed directly by the measurement transport.
  decoder_inputs(sparse_binary_matrix H,
                 std::optional<sparse_binary_matrix> O = std::nullopt,
                 std::vector<double> error_rates = {},
                 std::optional<sparse_binary_matrix> D = std::nullopt,
                 std::optional<std::vector<std::size_t>> error_ids = std::nullopt);

  static decoder_inputs from_stim_dem(std::string stim_dem_text,
                                      std::optional<sparse_binary_matrix> D = std::nullopt);

  decoder_model_source source() const noexcept;      // which one is authoritative

  // The common matrix view, available whatever the source was.
  const sparse_binary_matrix &detector_error_matrix() const;      // H
  const sparse_binary_matrix &observable_flips_matrix() const;    // O
  const std::vector<double>  &error_rates() const;
  const sparse_binary_matrix *measurement_to_detectors() const;   // D, or nullptr

  // The raw view, for decoders that want the source itself.
  bool has_stim_dem() const noexcept;
  const std::string &stim_dem() const;
};
```

A DEM-native decoder reads `stim_dem()`; a matrix decoder reads `detector_error_matrix()`. Both are
looking at one source, which is the point.

The distinction is: **stable construction input describes the decoding problem and this session's input
basis independently of one decoder's implementation; parameters choose how a particular decoder
solves it.** Not every decoder consumes every field. That is fine. `error_rate_vec` belongs here
because it has one entry per H column and comes from the same DEM as H and O. D belongs here because
it is fixed for the session and dimensionally bound to H's detector basis, even though it is not
part of the noise model. On the other hand, `max_iterations` and `merge_strategy` are *parameters*, specific to how one
particular decoder solves the problem.

`decoder_config` does not disappear and it does not magically become pure knobs. It remains the
server's serializable configuration form, including the selected model source. What changes is where
the YAML stops. Today the plugin sees fragments of the config: a flat `-1`-delimited sparse vector
here, a dense tensor in a parameter map there, a matrix arriving after construction. Under the
proposal the server converts the config once, into the same `decoder_inputs` an offline caller would
build by hand, and that is the only thing the factory ever sees. No plugin author needs to know that
`O_sparse` was a flat vector with sentinel values in a YAML file:

```cpp
// realtime_decoding.cpp, proposed — resolve first, construct second
auto D = canonical_measurement_to_detectors(decoder_config.D_sparse);

if (!decoder_config.stim_dem_path.empty()) {
  // stim_dem_path is mutually exclusive with H_sparse/O_sparse/error_rate_vec:
  // one authoritative source, not two representations of the same model.
  auto dem_text = read_file(resolve_against(base_dir, decoder_config.stim_dem_path));
  return decoder_inputs::from_stim_dem(std::move(dem_text), std::move(D));
}
return decoder_inputs::from_matrices(H, O, rates, std::move(D));   // matrix source

// ...and later, after every decoder's inputs have resolved successfully:
auto decoder = cudaq::qec::get_decoder(decoder_config.type, std::move(inputs),
                                       requested_output, params);
```

The resolve step is side-effect-free, so a bad configuration fails before any live decoder is
touched.

The handle uses a PIMPL/shared-state representation. That makes copies cheap and leaves room to add
a typed compact source later without changing the handle's object layout. 

### Output form is fixed at construction

O becomes data only. `decoder_output::{errors, observables}` is a separate factory argument and is
fixed for the lifetime of the instance.

The plugin validates the combination during construction:

- an error-producing decoder asked for observables requires O and may call the base projection
  helper before returning;
- Chromobius accepts observables and rejects an error-frame request;
- TensorRT validates the request against its engine output format; and
- PyMatching constructs the graph corresponding to the requested form.

A forward-looking note: this design does require the user to construct two decoder instances if they
want both errors and observables as the return type, though that can be expanded later.


### The decoding server resolves one authoritative model source

The server now accepts two source shapes:

- **matrix source:** H, O and optional rates, with sizes required to interpret the flat sparse
  encoding; or
- **Stim source:** `stim_dem_path`, mutually exclusive with H, O and rates. H, O, rates and sizes
  are derived once from the DEM.

D is orthogonal to that choice and remains required by the current realtime server because its
transport supplies raw measurements. For a DEM source, D's row count is checked against the
DEM-derived detector count. Optional `block_size` and `syndrome_size` values are assertions checked
against the DEM, not competing authorities.

The raw DEM path is resolved relative to the configuration document (or the working directory for
programmatic/raw-string configuration), made absolute, read, parsed and normalized *before* decoder
construction. The plugin receives the validated artifact; construction does not re-parse a second
copy.

This draft deliberately uses an operator-visible filesystem path rather than transporting an
839 KiB DEM in the published configuration payload. One caveat is that editing a
DEM in place without changing the path is invisible to the current reload comparison. 

### Wrapper decoders 

Two decoders wrap another decoder: TensorRT constructs a **global decoder** to run after
its engine, and sliding window constructs an **inner decoder** per window. Both deserve
a bit of special treatment:

Concretely, a wrapper may never hand the decoder it constructs a *different problem*. It hands
it the same problem — the same code, the same noise — possibly a slice of it, possibly at a later
stage of the pipeline. That is what `decoder_inputs` is for: the wrapper derives the decoder it
constructs from its own inputs, so there is nothing else it *could* hand over. *This is also where a conflict 
with the streaming DEM work is most likely to happen.* 

"Same problem" is not the same as "same bytes," and this is where the shapes differ. PyMatching as a
global decoder wants H and O; Chromobius as a global decoder wants the raw DEM text. They receive the
same `decoder_inputs`, which carries both views of one source, and each reads the representation it
needs. Nobody derives a second model.

Only two things can legitimately differ between a wrapper's inputs and its constructed decoder's:

1. **Who feeds it.** Only the decoder the realtime transport feeds directly needs D, because D is
   what the base applies to turn an arriving measurement stream into detectors. Anything a wrapper
   constructs is fed by that wrapper, never by the transport, so D never travels inward. This is not
   a per-wrapper judgement call — it is true of every wrapped decoder.
2. **Indexing.** Did the wrapper renumber detectors or error mechanisms? This decides whether the raw
   DEM *text* still reads correctly, because a DEM names its detectors by position.

TensorRT changes neither. It hands its global decoder the same inputs minus D, since the global
decoder is fed the engine's residual detectors rather than a measurement stream. The detector basis
and ordering are untouched, so the raw DEM still describes them exactly and Chromobius can be the
global decoder. That preservation is a **caller guarantee, not something the code proves**: declaring
`engine_output_format` as one of the residual forms is the caller asserting that the engine emits
residual detectors in exactly the H-row basis and order supplied at construction. The implementation
validates width only. A reordered engine would silently feed the global decoder a permuted syndrome,
and a raw-DEM global decoder would then decode it against the wrong detector identities. The source
says so at the declaration site, and supporting reordered residuals would need an explicit detector
mapping this contract does not provide.

Sliding window changes indexing. Each window is the same code and the same noise — just a subset of
detector rows and error columns, renumbered from zero. The matrices slice cleanly and carry the
problem faithfully. The raw DEM text does not: it names detectors by the outer numbering, so handing
it to a window would have the inner decoder reading `D17` as its own detector 17 rather than the
outer one. So sliding window slices its matrices and constructs each inner decoder's inputs from
them directly. Passing the outer DEM through that slice would be worse than losing provenance: it
would be confidently wrong.

That gives two cases, and only one of them needs an operation:

- **Same numbering, different feed.** `decoder_inputs_without_d()` returns the same inputs without D,
  for a decoder that receives detectors rather than a measurement stream. Everything else, the raw
  source included, is preserved. TensorRT uses this for its global decoder.
- **New numbering.** The wrapper builds fresh `decoder_inputs` from the matrices it computed. There
  is no operation for this and no need for one: a matrix-constructed handle carries no raw source, so
  the DEM text is dropped structurally rather than by a rule someone has to remember.

This also allows us to expand into more exotic wrapping schemes by utilizing `decoder_inputs`. 


### Who owns realtime allocation

Once O and D are construction inputs, the base owns everything whose size they determine:

- the measurement buffer from D's column count;
- D's measurement-to-detector mapping;
- detector and soft-detector buffers from H's row count; and
- observable corrections from O's row count.

Baseline main already sizes the detector buffers in the base constructor from H's row count, so this
is not a wholesale relocation — the point is that the *remaining* pieces stop depending on *setter
call order*.

Who owns what, before and after:

| state | derived from | baseline main | proposed |
|---|---|---|---|
| H | model | factory argument | factory argument, inside `decoder_inputs` |
| O | model | `params["O"]`, or `set_O_sparse()` after construction | `decoder_inputs` |
| D | session input basis | `set_D_sparse()` after construction | `decoder_inputs` |
| error rates | model | `params["error_rate_vec"]`, beside the knobs | `decoder_inputs` |
| detector buffers | H row count | base constructor | base constructor |
| measurement buffer | D column count | sized by `set_D_sparse()` | base constructor |
| corrections buffer | O row count | sized by `set_O_sparse()` | base constructor |
| streaming layer geometry | the decoder's own choice | `set_D_sparse()` resizes detector buffers after the base `dynamic_cast`s to `sliding_window` | subclass hands it over once |
| output form | the caller's request | `result_type_`, set by decoder-specific constructor behaviour when O is present | explicit factory argument |

And the lifecycle:

```
  baseline main

  get_decoder ---> ctor: H only ---> set_O_sparse ---> set_D_sparse ---> realtime-usable
                        ^                  ^                 ^
                        `----- decode(syndrome) already works here: H is all it
                               needs. enqueue_syndrome() works too -- but only
                               once both setters have run. Nothing states that
                               requirement or enforces it; you are expected to
                               know the call order.

  proposed

  resolve ---> ctor: inputs + output form + allocation ---> usable
     ^                                          ^
     |                                          `-- the server may still assign an ID, dry-run a
     |                                              decode, or let the decoder initialize GPU
     |                                              resources lazily. None of that changes what
     |                                              the decoder means.
     `-- every path produces the same decoder_inputs
```

One of the things that the proposal aims to remove is that the decoder's readiness depends on a call sequence the type system only implies.

Sliding window has one extra construction step: its subclass constructor calls
`initialize_streaming_layout()` with detector-layer offsets and the maximum layer width. That
geometry is not a property of H/O/D; it is how this decoder chooses to consume rounds. The base
cannot obtain it through a virtual call while the subclass is still constructing, so the subclass
hands it over through a one-shot, construction-only latch.

## What this buys a plugin author and user of the decoding server

Before, an ordinary H-based plugin creator is effectively written against this contract:

```cpp
create(const decoder_init &init, const heterogeneous_map &params) {
  // Extract H or reject the other variant arm.
  // O might be in params offline, or appear through a base setter online.
  // D arrives only after construction on the server path.
}
```

After:

```cpp
create(decoder_inputs inputs,
       std::optional<decoder_output> requested_output,
       const heterogeneous_map &params) {
  // All construction input is present now. Validate the fixed output request and build.
}
```

The concrete gains are:

- offline, top-level server and nested construction share one input value;
- stable construction data no longer travels in an untyped decoder-parameter bag;
- a plugin can reject unsupported model/output combinations before becoming live;
- common server code no longer knows decoder names in order to forward O; and
- wrappers have an explicit rule for retaining or dropping authoritative source data.

The costs are also concrete:

- Lots, I mean lots, of code change, not even counting the change needed for the private decoder;
- each plugin owns construction-time validation of the output forms it promises;


## Performance and memory

Measured against the merge base, `upstream/main` at `674cb8f2` — using Pymatching

### The realtime path, over UDP

Benchmarked using `surface_code-1-cqr`. The server is run
with `QEC_DECODING_SERVER_SPIN_US=0` so it blocks rather than busy-polls: the semantics are identical
either way, but its CPU time then measures decode and transport work instead of poll loops.

| distance 5, 5 rounds, 1000 shots (8000 decodes) | main | proposed |
|---|---:|---:|
| server CPU per decode | 262.5 / 265.0 / 263.8 µs | 256.3 / 257.5 / 256.3 µs |
| server peak RSS | 418.1 / 418.5 / 418.5 MiB | 417.9 / 417.9 / 418.1 MiB |
| app wall clock | 1.71 / 1.72 / 1.71 s | 1.68 / 1.67 / 1.67 s |

| distance 9, 9 rounds, 500 shots (6000 decodes) | main | proposed |
|---|---:|---:|
| server CPU per decode | 751.7 / 745.0 µs | 738.3 / 746.7 µs |
| server peak RSS | 421.7 / 421.5 MiB | 420.5 / 420.4 MiB |
| app wall clock | 4.03 / 3.98 s | 3.95 / 4.00 s |

**No regression.** At distance 5 the proposed branch is about 2.7% cheaper per decode and the
repetition ranges do not overlap; at distance 9 the two are indistinguishable. Peak RSS is the same
to within 0.3%, which is expected: this configuration carries its model as matrices, so it never
exercises the DEM parsing path below.

This is the right benchmark for the question "did lifting O and D out of the setters cost anything,"
because it is the one path that uses both, per shot. The generated configuration carries `H_sparse`,
`O_sparse` and `D_sparse` and no `stim_dem_path`, so on main it takes exactly the setter route —
`get_decoder(H)`, then `set_O_sparse()`, then `set_D_sparse()` — while this branch resolves the same
three into `decoder_inputs` before construction. Neither matrix is decoration at run time: D converts
every arriving measurement stream into detectors inside `enqueue_syndrome()`, and O turns the decoded
frame into the corrections the app counts. Both branches found the same number of corrections (50 at
distance 5, 71 at distance 9).

### Resolving a model from a DEM

This is where the branch is meaningfully different, and it is decoder-independent — it is the step
that turns DEM text into whatever the framework holds as the model. On the distance-13 surface code
DEM (`H = 2184 x 47129`):

| | main (`dem_from_stim_text`) | proposed (`decoder_inputs::from_stim_dem`) |
|---|---:|---:|
| parse | 75.5 / 97.0 / 75.2 ms | 11.7 / 11.2 / 11.3 ms |
| retained | 105.9 / 105.9 / 105.9 MiB | 6.0 / 5.8 / 5.8 MiB |
| peak | 106.7 / 106.7 / 106.9 MiB | 6.9 / 6.8 / 6.8 MiB |

About 7x faster and 17x smaller. The retained figures differ because the representations differ,
which is the point: main materializes dense `detectors x mechanisms` tensors for H and O, while this
branch builds the sparse arrays directly from the hit lists the parser already has. Nothing about
this required the lifecycle change — but the lifecycle change is what put a single, obvious
resolution step where the cost was visible.

This matters most for exactly the case that started all of this - a Chromobius-on-the-server
configuration is DEM-sourced by definition.

## Compatibility with incoming work

### Chunked or streaming DEM sources

PR #759 introduces a compact repeated-round description and points toward decoders consuming
chunks without flattening. `decoder_inputs` therefore stores source metadata separately from its
common matrix view and hides its representation behind a PIMPL. The current enum exposes only the
two implemented sources—matrices and raw Stim DEM. A chunked source should be added only with a
typed constructor/accessor and a real consumer, not as a decorative enum value.

The unresolved contract is what makes a transformation basis-preserving for a compact source,
including whether D still maps into the same detector basis. The current rule gives that future
work somewhere to attach: preserve the authoritative source when detector/error identity survives;
drop it when a transformation re-indexes either basis.

## Why not do a smaller fix by simply expanding what the decoding server accepts and leave O/D where they are?

The minimal fix is: accept `stim_dem_path` in the server config, pass the text through the existing
string arm of `decoder_init`, and keep the setters. It is much smaller and it does unblock
Chromobius. It also does not work on its own, and leaves the rest in place:

- **It still has to derive O and inject it.** The base sizes its corrections buffer from whatever
  `set_O_sparse()` hands it, so a DEM-only config reports zero observables. The server must derive O
  from the DEM and set it on a decoder that already read that same O out of its own DEM text. In the case
  of Chromobius, you need to pass an O in just so Chromobius can be constructed and then discard the O.
- **Two authorities for O, with no check.** The setter's O and the decoder's own O can disagree; the
  base validates row count, never content. We reproduced a silent inverted correction this way.
- **D still arrives after construction**, so the base still cannot size realtime state when the
  constructor returns, and readiness still depends on an unstated call order.
- **Output form stays coupled to O's presence**, so a caller cannot ask for an error frame while
  supplying O as data.
- **Name-based routing survives.** Common server code still forwards O by comparing against
  `"trt_decoder"` and `"pymatching"`, so a third-party global decoder still requires editing our
  source.
- **Nested construction stays special.** TensorRT and sliding window still receive their model by a
  different mechanism than a top-level decoder.




