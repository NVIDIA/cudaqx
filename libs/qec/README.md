# CUDA-Q QEC Library

CUDA-Q QEC is a high-performance quantum error correction library
that leverages NVIDIA GPUs to accelerate classical decoding and
processing of quantum error correction codes. The library provides optimized
implementations of common QEC tasks including syndrome extraction,
decoding, and logical operation tracking.

**Note**: CUDA-Q QEC is currently only supported on Linux operating systems
using `x86_64` processors or `aarch64`/`arm64` processors. CUDA-Q QEC does
not require a GPU to use, but some components are GPU-accelerated.

## Features

- Fast syndrome extraction and processing on GPUs
- Common decoders for surface codes and other topological codes
- Real-time decoding capabilities for quantum feedback
- Integration with CUDA-Q quantum program execution
- Sparse `sparse_binary_matrix` / Python sparse-dict PCM representations to avoid allocating a dense `rows × cols` parity-check tensor where possible (large DEM-sized PCMs; see GitHub `#379`; `generate_random_pcm_sparse` for random-matrix demos without a dense tensor).

## PCM and matrix size notes (for PR / release text)

- Dense `generate_random_pcm` **throws** if `rows * cols` exceeds `cudaq::qec::k_max_dense_pcm_elements` (currently ~400 million elements); use `generate_random_pcm_sparse` for larger random PCMs instead (**behavior change** for previously "too large to allocate" failures).
- Parity check `H` is generally a `sparse_binary_matrix` (or Python sparse dict). Optional **observables** `O` in PyMatching / TensorRT decoders remain a modest-sized dense `cudaqx::tensor` parameter (`num_observables × block_size`); that API asymmetry is intentional for now.
- **Non-canonical sparse PCMs**: `sparse_binary_matrix::from_csc` / `from_csr` / `from_nested_*` accept duplicate indices within a column (CSC) or row (CSR) — this is legitimate output from DEM decompositions where `1 + 1 = 0` under GF(2). Consumers that dispatch on per-column nnz count (e.g. PyMatching) call `cudaq::qec::canonicalize_pcm(H)` internally to collapse duplicates under GF(2) parity before reading the column structure. If you write a new decoder that assumes uniqueness, call `canonicalize_pcm` on entry.

## Optional Dependencies

Some decoders require additional dependencies to operate. You can install them with

- `pip install cudaq-qec[tensor-network-decoder]` for the Tensor Network Decoder
- `pip install cudaq-qec[trt-decoder]` for the TensorRT Decoder

## Getting Started

For detailed documentation, tutorials, and API reference, visit the
[CUDA-Q QEC Documentation](https://nvidia.github.io/cudaqx/components/qec/introduction.html).

## License

Most components of CUDA-Q QEC are open source. The source code is available on
[GitHub][github_link] and licensed under [Apache License
2.0](https://github.com/NVIDIA/cudaqx/blob/main/LICENSE).

The `libcudaq-qec-nv-qldpc-decoder.so` library (distributed with CUDA-Q QEC) is
closed source and is subject to the [NVIDIA Software License Agreement][github_qec_license]

[github_link]: https://github.com/NVIDIA/cudaqx/tree/main/libs/qec
[github_qec_license]: https://github.com/NVIDIA/cudaqx/blob/main/libs/qec/LICENSE
