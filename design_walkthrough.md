# Decoder model inputs: a design walkthrough

Note to reviewers:

This draft is not in a mergeable state. The intent is to get the design discussion started and have some concrete examples to look at. The design was done with Tracy's dynamic DEM PR in mind and was meant to be extensible to support dem chunks, though there certainly will be rough edges still. 

## A quick note on H/O/D

`H` has shape `detectors x error mechanisms`. Column `e` says which detectors fire when error
mechanism `e` occurs; this is the model a matrix-based decoder decodes against.

`O` has shape `observables x error mechanisms`. The same column `e` says which logical observables
that error mechanism flips. If a decoder predicts an error frame `x`, the observable correction is
`O * x` over GF(2). A decoder such as Chromobius can instead predict those observable flips
directly, but the meaning of O does not change.

`D` has shape `detectors x raw measurements`. Hardware sends measurement bits; the decoder consumes
detectors. `D * m` over GF(2) is the bridge between those two bases.

So H and O describe the decoding model. D is dimensionally
bound to H's detector basis, and the base class owns the buffers and preprocessing derived from it.


## A quick note about decoders

This design is based on the standing convention that a decoder is immutable once constructed. E.g., its H, error rate, return type
are set at construction time and are not meant to change during the lifetime of the decoder instantance. 

## The current problem

All I wanted to do was to enable Chromobius on the decoding server path.

But the road to Chromobius is fraught with false leads. On baseline `main` the server always builds a plugin from H, while
Chromobius can only be constructed from a raw Stim DEM. So Chromobius works perfectly well through
the offline DEM factory, and unreachable through the server's matrix-only construction
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

This string variant allows Chromobius to be constructable offline. However, the baseline
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
defines them. H and O might contradict DEM, and checking one against the other when both are supplied
can be expensive depending on the size of H/DEM. 

And then, it gets worse:

### O arrives by a different road for every decoder

The realtime path needs O so that error-frame decoders can produce observable corrections. On
baseline main, when and how that O arrives depends on which decoder is being used and which path
the decode is on. In other words, we have an "all roads lead to Rome" situation, with some very
precise "turn left, then right, then left" call-order implications embedded:

Drawn out, with the worst case at the bottom:

```
  baseline main - how O reaches a decoder

  offline, PyMatching ....  params["O"] ------------> ctor --> this->set_O_sparse()
  offline, Chromobius ....  (read out of the DEM text; no O argument at all)
  realtime, top level ....  get_decoder(H, params) --> ctor
                                                       `-- then: set_O_sparse()

  realtime, TensorRT with a PyMatching child - the same matrix, three times:

      server --(1)-- params["O"] ---------------------------> TensorRT ctor
      server --(2)-- params["global_decoder_params"]["O"] ---> PyMatching child ctor
      server --(3)-- set_O_sparse() -------------------------> after construction
                        ^
                        `-- (1) and (2) are selected by hardcoded decoder names
```


That trt+pymatching is the one that should set off an alarm. Common server code knows both a wrapper's
internal parameter convention and a particular child decoder's name, *hardcoded*. This opens the gate that a third party decoder 
author will need to modify our source code in order to plug in a different global decoder. {claude, double check this. also, the word "child decoder" is incorrect. did you mean global decoder?}. In addition, O arrives three different times and stored three times. Both plugins convert `params["O"]` and call `set_O_sparse()` on
themselves, so the matrix ends up living in the server's `decoder_config`, in the TensorRT object's
base member, and in the child's base member — having passed through two parameter maps as a *dense
tensor* to get there. The third delivery then overwrites the first copy with the same content!!! 

Under the proposal this collapses to a single road. `decoder_inputs` carries O, TensorRT hands its
child the same inputs through an explicit derivation, and nothing needs to know a decoder's name to
route a matrix. The routing function is the clearest before-and-after in the change:

```cpp
// prepare_decoder_params(), baseline main - roughly 60 lines, abridged
if (decoder_config.type != "trt_decoder")                          // decoder name #1
  return params;
...
const bool has_pymatching_global =
    params.get<std::string>("global_decoder") == "pymatching";     // decoder name #2
params.insert("O", O);
if (has_pymatching_global)
  global_decoder_params.insert("O", O);

// prepare_decoder_params(), proposed - this is the entire function
auto params = decoder_config.decoder_custom_args_to_heterogeneous_map();
if (decoder_config.cuda_device_id.has_value())
  params.insert("cuda_device_id", decoder_config.cuda_device_id.value());
return params;
```

{claude, your code example above shows something about cuda_device_id. how is that relevant to O at all?}

That's not all, because:

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








## The smallest rival design

There is a much smaller change that unblocks Chromobius, and it deserves to be stated accurately
rather than dismissed. A reviewer will propose it, and they will be right that it is cheaper.

It is *not* "add a DEM arm to `decoder_init`" — that arm already exists. The real minimal fix is:

1. add `stim_dem_path` to the server configuration and make the matrix fields optional;
2. read the DEM and pass its text through the existing string arm;
3. derive O from the DEM anyway, and still call `set_O_sparse()`, because the base sizes its
   corrections buffer from the O it was handed and would otherwise see zero observables; and
4. keep injecting D through `set_D_sparse()` afterwards, exactly as today.

Step 3 is the part that is easy to miss, and it is worth being precise about why it is unavoidable
on baseline main:

```cpp
// decoder.cpp, baseline main — the base's observable count is whatever the setter handed it
void decoder::set_O_sparse(const std::vector<std::vector<uint32_t>> &O_sparse) {
  this->O_sparse = O_sparse;
  ...
  this->pimpl->corrections.resize(O_sparse.size());
}
```

So "make O optional in the schema and select the string arm" does not by itself enable Chromobius on
the server. The decoder would be fine; the base around it would report zero observables. The minimal
fix has to derive O from the DEM and inject it back through the setter — into a decoder that already
read that same O out of the DEM text it was constructed from.

Which is where the joke writes itself:

> **The cheap fix has the server derive O from the DEM the decoder already read it from, and then
> inject it back through a setter.**

One matrix, two authorities, and the one that wins is the one that arrives last. We would be down to
praying that O does not get lost on its way to the decoder.

I am deliberately not claiming a road count here. Whether that is "a fifth road" or two existing
roads used at once is arguable, and arguing about it would be arguing about the wrong thing. The
uncontestable statement is the one that matters: **the minimal fix delivers O twice, through two
authorities, with no mechanism that checks they agree.** It removes no existing road.

What it leaves untouched:

- O still reaches a decoder by every convention it already did, two of them selected by comparing a
  decoder's name against a string literal in framework code;
- the same matrix is still delivered three times to a TensorRT decoder with a PyMatching child;
- stable construction data still travels in the untyped parameter bag beside `merge_strategy`, so a
  plugin author still has to work out which of their inputs are model and which are knobs;
- output form is still inferred from whether O happened to be present;
- D still arrives after construction, so the base still cannot size realtime state when the
  constructor returns; and
- a plugin still cannot rely on O or D during construction, and there is still nothing in the
  factory signature that says so.

Note what the minimal fix is *not*: it is not a Chromobius special case. Selecting a model source by
shape rather than by decoder name is generic, and any DEM-native decoder would benefit. It is a
smaller, legitimately general patch — it simply stops at unblocking a source shape instead of
producing a construction contract anyone else can build against.

That is the honest trade: a smaller source-shape fix against a larger lifecycle correction. The
smaller option is not absurd, and this proposal should not win on road-count rhetoric. It should win
because the dual authorities it preserves have already produced concrete divergence in this
codebase, twice — which is the next section.

### Two authorities have already disagreed here, twice

**First, D.** When the measurement-to-detector map was populated into `decoder_inputs` while
`set_D_sparse()` still fed the realtime path, the two representations disagreed on a repeated index:
one rasterized the row into a dense matrix and collapsed the duplicate, the other XORed it entry by
entry and cancelled it. Nothing observable end to end disagreed, because the realtime path drove
decoding and was self-consistent. Only a plugin reading its own construction inputs could see it.

**Second, O.** Rather than argue that the same hazard applies to O, we ran it — against intermediate
commit `95f18f09`, where both sources still existed. A model whose O says *error 0 flips observable
0*, and a `set_O_sparse()` call installing a same-shaped O that says the opposite:

```
set_O_sparse with a different (same-shaped) O: accepted, no error
enqueue_syndrome decoded: yes

  error frame predicted        : e0 = 1 (detector 0 fired)
  model O says obs correction  : (1, 0)   <- error 0 flips observable 0
  realtime corrections produced: (0, 1)

  DIVERGED: the realtime path used the setter's O, not the model's.
```

The realtime path emitted the **inverted logical correction**, silently. Nothing rejected the
contradicting matrix, because the guard on that path compares O's *row count* against the model and
never looks at its contents:

```cpp
// decoder.cpp at 95f18f09 — shape is checked, content is not
if (O_sparse.size() != num_observables)
  throw std::runtime_error("Observable matrix is not configured: ...");
...
for (auto col : O_sparse[i])          // the setter's O decides the correction
```

Same signature as the D bug: two representations agreeing on shape, disagreeing on meaning, with
only one of them consulted at the point that matters. For D the consequence was a detector value.
Here it is the logical correction an experiment applies, which is the last place we should be
relying on two copies happening to match.

> **[remove for PR]** Reproduce at `95f18f09`. Build `cudaq-qec-decoders`, then compile and link:
>
> ```cpp
> auto H = sparse_binary_matrix::from_nested_csc(2, 4, {{0}, {1}, {}, {}});
> auto O_model = sparse_binary_matrix::from_nested_csr(2, 4, {{0}, {1}});  // e0 -> obs0
> auto D = sparse_binary_matrix::from_nested_csr(2, 2, {{0}, {1}});
> cudaq::qec::decoder_inputs inputs(H, O_model, /*rates=*/{}, D);
> auto d = cudaq::qec::decoder::get("single_error_lut", inputs,
>                                   cudaq::qec::decoder_output::errors);
> d->set_O_sparse(std::vector<std::vector<std::uint32_t>>{{1}, {0}});      // contradiction
> d->set_D_sparse(D);
> d->enqueue_syndrome(std::vector<std::uint8_t>{1, 0});
> // d->get_obs_corrections() reports (0, 1); the model implies (1, 0).
> ```
>
> `g++ -std=c++20 probe.cpp -Ilibs/qec/include -Ilibs/core/include -Lbuild/lib -lcudaq-qec-decoders`

This is why the setters should eventually go — or, at minimum, become strict equality assertions
against the construction input. Two supported ways to supply the same matrix means the matrix can be
two things at once, and shape checks do not catch it.

That deletion is still the *last* step and a reversible one. It is not the argument. The argument is
the lift.

## What we propose

The draft turns stable construction data into a single resolved value that exists before the decoder
does, makes output form an independent construction-time choice, teaches the server to resolve a raw
DEM source generically, and removes the late O/D mutation path.

In order, with the one contested decision marked:

1. **Give stable construction input a typed home:** `decoder_inputs`.
2. **Every path resolves it before construction** — offline, top-level server, and nested children
   alike. ← *this is the decision being asked for.*
3. **Output form becomes an explicit construction argument**, because O is now a field and can no
   longer double as the request.
4. **The base sizes realtime state at construction**, since it finally knows the inputs then; the
   sliding-window subclass hands over its own streaming geometry rather than being `dynamic_cast`
   to.
5. **The setters are now unused: delete them**, or keep them as assertions.

Steps 3 through 5 are the coherent consequences this design selects, not deductive necessities, and
each has its own local contract worth examining. Explicit output selection could be built without
`decoder_inputs` at all. The setters could survive step 5 as equality assertions. Someone could
accept unified construction inputs and still dislike the exact streaming-layout handoff in step 4.
Those are all legitimate arguments to have.

What I am asking is that they be had *after* step 2, not instead of it. Reject step 2 and most of
the rest loses its motivation and the draft roughly halves; accept it and the remaining
disagreements are about API shape, which is a much better conversation than a file count.

### One construction input, distinct from the knobs

`decoder_inputs` is a small immutable handle to shared construction state. It owns:

- H in sparse CSC form;
- optional O in sparse CSR form;
- error rates and optional error IDs, indexed by H column;
- optional D in sparse CSR form;
- the authoritative source kind and, for a Stim source, the raw DEM text; and
- dimensions as metadata, so asking for a size does not force a future compact source to
  materialize a matrix.

The boundary is: **stable construction input describes the decoding problem and this session's input
basis independently of one decoder's implementation; parameters choose how a particular decoder
solves it.** Not every decoder consumes every field. That is fine. `error_rate_vec` belongs here
because it has one entry per H column and comes from the same DEM as H and O. D belongs here because
it is fixed for the session and dimensionally bound to H's detector basis, even though it is not
part of the noise model. `max_iterations` and `merge_strategy` are parameters.

`decoder_config` does not disappear and it does not magically become pure knobs. It remains the
server's serializable configuration form, including the selected model source. The server resolver
turns that configuration into `decoder_inputs`; the factory and plugin see the normalized runtime
inputs, not the YAML transport representation.

The handle uses a PIMPL/shared-state representation. That makes copies cheap and leaves room to add
a typed compact source later without changing the handle's object layout. It is not, by itself, a
promise of a versioned cross-release `.so` ABI; we explicitly deferred that problem.

### Output form is fixed at construction

O becomes data only. `decoder_output::{errors, observables}` is a separate factory argument and is
fixed for the lifetime of the instance. No output-selection bit is added to each decode call or to
the realtime wire message.

The plugin validates the combination during construction:

- an error-producing decoder asked for observables requires O and may call the base projection
  helper before returning;
- Chromobius accepts observables and rejects an error-frame request;
- TensorRT validates the request against its engine output format; and
- PyMatching constructs the graph corresponding to the requested form.

There is one required single-shot virtual, `decode()`. We deliberately removed the experimental
`decode_native`, capability-query and alternate-dispatch machinery that this design effort had
itself introduced: "native" was circular for some wrappers, nobody outside the base queried the
capabilities, and a fixed instance does not need to renegotiate its contract on every shot. Batch
and async keep their previous shape and inherit the instance's fixed output form.

This does have a cost: if a caller genuinely needs both errors and observables, this pass asks it to
construct two decoder instances. We found no current consumer requiring both, so we chose a simple
contract over a speculative multi-result API.

### The server resolves one authoritative model source

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
programmatic/raw-string configuration), made absolute, read, parsed and normalized before decoder
construction. The plugin receives the validated artifact; construction does not re-parse a second
copy.

This draft deliberately uses an operator-visible filesystem path rather than transporting an
839 KiB DEM in the published configuration payload. That is sufficient for the current
operator-hosted deployment. It does **not** provide remote configuration portability, and editing a
DEM in place without changing the path is invisible to the current reload comparison. Both are real
limitations, recorded here rather than solved.

### Wrapper decoders and provenance

Wrappers create child decoders, so they must answer a more interesting question: does the parent's
authoritative source still describe the child's detector and error bases?

`decoder_inputs` provides distinct operations for the distinct answers:

- `canonicalized()` preserves dimensions, row/column identity and source provenance;
- `without_measurement_to_detectors()` removes D when a child already receives detectors, while
  preserving the model source; and
- `derive_with_changed_basis(...)` accepts new matrices and drops the raw source because the old
  DEM indexes the parent's basis, not the child's.

TensorRT hands a global child the same inputs without D. The reason a raw DEM survives that hop is a
**caller guarantee, not something the code proves**: declaring `engine_output_format` as one of the
residual forms is the caller asserting that the engine emits residual detectors in exactly the H-row
basis and order supplied at construction. The implementation validates width only. A reordered
engine would silently feed the child a permuted syndrome, and a raw-DEM child would then decode it
against the wrong detector identities. The source says so at the declaration site, and supporting
reordered residuals would need an explicit detector mapping this contract does not provide. So the
nesting acceptance test proves provenance reaches the child for a conforming fixture; it does not
verify arbitrary engine orderings.

Sliding window slices detector rows and error columns for each child, so it uses
`derive_with_changed_basis()` and the raw DEM is dropped. Passing the parent's DEM through that
slice would be worse than losing provenance: it would be confidently wrong.

We removed the compulsory free-form `provenance_loss_reason` string. The invariant is that a
basis-changing derivation drops the source; no production consumer read the prose explaining why.
If a real diagnostic consumer appears later, it should drive a structured representation, rather
than requiring every wrapper author to write a justification that nothing reads.

### Who owns realtime allocation

Once O and D are construction inputs, the base owns everything whose size they determine:

- the measurement buffer from D's column count;
- D's measurement-to-detector mapping;
- detector and soft-detector buffers from H's row count; and
- observable corrections from O's row count.

Baseline main already sizes the detector buffers in the base constructor from H's row count, so this
is not a wholesale relocation — the point is that the *remaining* pieces stop depending on setter
call order.

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
                        `----- offline decode() works here; the measurement-stream
                               path does not, and nothing says so

  proposed

  resolve ---> ctor: inputs + output form + allocation ---> usable
     ^                                          ^
     |                                          `-- the server may still assign an ID, dry-run a
     |                                              decode, or let the decoder initialize GPU
     |                                              resources lazily. None of that changes what
     |                                              the decoder means.
     `-- every path produces the same decoder_inputs
```

Being precise about that interval, because I overstated it in an earlier draft: a baseline decoder
is perfectly usable for ordinary offline `decode()` the moment its constructor returns. What is
incomplete is the realtime measurement-stream path, which needs O for corrections and D for the
measurement-to-detector conversion. The defect is not that the object is broken — it is that its
readiness depends on a call sequence the type system never mentions.

Sliding window has one extra construction step: its subclass constructor calls
`initialize_streaming_layout()` with detector-layer offsets and the maximum layer width. That
geometry is not a property of H/O/D; it is how this decoder chooses to consume rounds. The base
cannot obtain it through a virtual call while the subclass is still constructing, so the subclass
hands it over through a one-shot, construction-only latch.

This deserves scrutiny, because a construction-only method can look a lot like a setter. The
distinction is ownership: it does not inject or replace construction input, it cannot be called
twice, and it exists only for subclass-specific streaming geometry. The payoff is removal of the
base's `dynamic_cast<sliding_window *>` and all decoder-name/concrete-type knowledge from common
decoder code.

## What this buys a plugin author

The `.so` discovery and schema-registration story already worked on main. We should not claim this
PR invented it. What changes is what the registered factory receives.

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

The plugin still registers its factory and custom-argument schema in its own library. It still has
to understand H, or raw DEM, or whichever source it supports. We have not made decoder development
free; we have made the bill itemized.

The concrete gains are:

- offline, top-level server and nested construction share one input value;
- stable construction data no longer travels in an untyped decoder-parameter bag;
- a plugin can reject unsupported model/output combinations before becoming live;
- common server code no longer knows decoder names in order to forward O; and
- wrappers have an explicit rule for retaining or dropping authoritative source data.

The costs are also concrete:

- the factory signature changes, so every in-tree and private plugin must be ported;
- private nv-qldpc currently calls `set_O_sparse()` and will break until it adopts the constructor
  input;
- each plugin owns construction-time validation of the output forms it promises;
- two instances are needed if a caller wants two output forms; and
- this is a large cross-cutting draft, not a merge-sized change.

We accepted the source and ABI break for current plugins. We did **not** take on a versioned `.so`
ABI or compatibility layer in this round.

## Performance and memory

### Per shot: no measured regression result yet

The code-path analysis is neutral-to-favorable, and it is worth stating what it does *not* claim.
Baseline main already reads its result form from an instance member (`result_type_`), so making it
an immutable construction-time field is a contract improvement rather than a hot-path saving. The
capability dispatch and general per-call validation we removed were introduced during this design
effort and never existed on main, so removing them is not subtractive relative to main either. The
accurate statement is that the final implementation leaves **no new** general validation or
capability dispatch on the hot path.

The one genuine final-versus-main difference is that projection walks contiguous sparse CSR storage
instead of nested vectors. The accidental full-vector copy introduced in the LUT wrapper was found
and removed.

But analysis is not a benchmark. A paired end-to-end latency comparison against main has **not**
been recorded for PyMatching, TensorRT and Chromobius. The earlier caller-buffer experiment measured
23,564.4 ns for allocating `decode()` versus 23,351.3 ns for the proposed buffered hook—a 213.2 ns,
0.90% difference—and we rejected the extra API machinery. That number answers the caller-buffer
question; it does not prove the whole branch is "within noise."

> **[remove for PR]** Before making a per-shot claim, add one reproducible benchmark that runs the
> same model, decoder, output form, warm-up and shot count on baseline main and this branch. Until
> then the honest sentence is: "no regression is expected from code-path analysis; not yet measured."

### Construction: one large transient was removed

The DEM parser already collects detector hits per error mechanism. Those hit lists are H's sparse
columns. The first implementation materialized a dense `detectors x mechanisms` tensor and scanned
it back into sparse form; the final path builds CSC/CSR arrays directly.

Measured on the distance-13 model (`H = 2184 x 47129`):

| | before | after direct sparse projection |
|---|---:|---:|
| retained model memory | ~4.6 MiB | ~4.6 MiB |
| transient above retained | ~99.3 MiB | ~2.1–2.3 MiB |

This is a real saving, but it belongs to DEM normalization, not to moving the setters. The lifecycle
redesign is what made the path visible; the sparse projection is what fixed it. Worth attributing
correctly.

Resolution retains every normalized input before construction. On the measured eight-decoder,
distance-13 configuration, retaining normalized inputs used about 23.1 MiB and resolved in about
1.06 s. Retaining only raw snapshots used about 14.1 MiB but required a second parse at construction,
taking about 2.1 s and violating "validate the artifact you construct." We chose the first strategy:
roughly 9 MiB more for eight decoders to avoid a second derivation and about a second of reload
latency.

We did **not** construct a complete replacement decoder set beside the live set. That would approach
2x peak decoder memory. Consequently, resolution failure preserves the active configuration, but a
plugin constructor failure can still leave the decoder set empty. The cached/published
configuration is not updated on that failure. This is an explicit memory-versus-transactionality
trade, not an accidental omission.

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

### Reload work

Model resolution is side-effect-free and happens before global decoder state is touched. Model
paths are written back as absolute only after construction and session initialization succeed. A
live realtime session rejects reconfiguration instead of destroying decoders that it still
references.

The remaining reload limitations are deliberately visible:

- constructor failure does not preserve the old live decoder set because we rejected overlapping
  allocations;
- an in-place edit to a DEM at the same path is invisible to bytewise config comparison; and
- a path-based DEM source assumes the server can read the operator's filesystem.

Those belong in the reload/transport design rather than in a decoder-specific parameter escape
hatch.

## Acceptance evidence

The useful acceptance tests are not "did this helper return the object it just built?" They cross
the boundaries where the old paths diverged:

1. An H-based, out-of-tree-style plugin constructs offline and through the server with no
   decoder-name framework branch.
2. A capture plugin observes the same canonical H, O, D and rates at construction offline and
   through the server.
3. A server configuration resolves a raw DEM and constructs standalone Chromobius without a
   Chromobius branch in server construction.
4. TensorRT nests Chromobius when its derivation preserves the detector/error basis; the converse
   matrix-only case fails, proving the test did not pass for an unrelated reason.

Step 4 adds the lifecycle evidence: the base sizes realtime state from construction inputs,
PyMatching's realtime path constructs through the new contract, and sliding-window streaming works
without a `dynamic_cast` in the base.

The clean verification built all QEC and core realtime targets from scratch. The focused
realtime set passed 18/18; the full C++ scope had no failures. The qLDPC device-graph test compiled
and linked but skipped execution because the available GPUs are compute capability 8.6 and the test
requires 9.0+. The proprietary nv-qldpc plugin itself is not covered by this public-tree result.

## Open questions for the draft discussion

1. **Names.** `get_default_output()` really means the instance's fixed requested/resolved output,
   not a default that can later be overridden. The behavior is settled; the name deserves reviewer
   input.
2. **Compact-source equivalence.** What exact detector/error-basis guarantees let a wrapper retain
   a future chunked source, and how is D's basis relationship represented?
3. **Remote model transport.** Is an operator-local path sufficient for the intended server
   deployment, or must a later contract carry content/hash/URI semantics?
4. **Reload transactionality.** Is preserving the old decoder set across constructor failure worth
   the 2x peak memory, or should a future reload mechanism construct and swap one decoder at a time?
5. **Private nv-qldpc migration.** Which constructor inputs and graph-dispatch ownership does the
   private decoder need once its late O setter is removed?

Settled questions should stay settled: no per-call output selection, no batch result redesign, no
capability apparatus, no caller-buffer hook, no versioned `.so` ABI in this proposal. There are
enough real questions here without inventing speculative ones.

## What would falsify this design

I would change or reject the proposal if review produces evidence that:

- a supported construction path genuinely cannot know O or D until after the decoder must become
  live;
- a current consumer needs one decoder instance to switch output basis per shot, and constructing
  two instances is materially unacceptable;
- normalized `decoder_inputs` cannot represent a model needed by an existing target decoder without
  eagerly materializing a compact source;
- a wrapper cannot state whether it preserves detector/error identity, making the provenance rule
  unusable rather than merely unfinished;
- the paired latency benchmark shows a meaningful per-shot regression attributable to the new
  contract; or
- the common factory contract forces decoder-specific knowledge back into the server or base.

Conversely, "this touches many files" is evidence that the draft needs careful review and likely PR
splitting; it is not by itself evidence that the lifecycle boundary is wrong. The proposal should
survive on whether its invariants are useful, not on how much work went into it.
