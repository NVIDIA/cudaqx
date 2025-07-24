# CUDA-Q Solvers Library

CUDA-Q Solvers provides GPU-accelerated implementations of common
quantum-classical hybrid algorithms and numerical routines frequently
used in quantum computing applications. The library is designed to
work seamlessly with CUDA-Q quantum programs.

**Note**: CUDA-Q Solvers is currently only supported on Linux operating systems
using `x86_64` processors or `aarch64`/`arm64` processors. CUDA-Q Solvers does
not require a GPU to use, but some components are GPU-accelerated.

**Note**: CUDA-Q Solvers will require the presence of `libgfortran`, which is not distributed with the Python wheel, for provided classical optimizers. If `libgfortran` is not installed, you will need to install it via your distribution's package manager. On debian based systems, you can install this with `apt-get install gfortran`.

## Features

- Variational quantum eigensolvers (VQE)
- ADAPT-VQE
- Quantum approximate optimization algorithm (QAOA)
- Hamiltonian simulation routines

Note: if you would like to use our Generative Quantum Eigensolver API, you will need
additional dependencies installed. You can install them with
`pip install cudaq-solvers[gqe]`.

## Getting Started

For detailed documentation, tutorials, and API reference,
visit the [CUDA-Q Solvers Documentation](https://nvidia.github.io/cudaqx/components/solvers/introduction.html).

## License

CUDA-Q Solvers is an open source project. The source code is available on
[GitHub][github_link] and licensed under [Apache License
2.0](https://github.com/NVIDIA/cudaqx/blob/main/LICENSE).

[github_link]: https://github.com/NVIDIA/cudaqx/tree/main/libs/solvers
