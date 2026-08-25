# CUDA-Q Solvers Library

> [!IMPORTANT]
> **CUDA-Q Solvers is deprecated.** Version 0.6.0 is the final planned
> release of the CUDA-Q Solvers library. Development continues in **CUDA-Q
> Algorithms**, which supersedes CUDA-Q Solvers and expands on it:
>
> ```bash
> pip install cudaq-algorithms
> ```
>
> * Documentation: https://nvidia.github.io/cudaq-algorithms/
> * PyPI: https://pypi.org/project/cudaq-algorithms/
> * GitHub: https://github.com/NVIDIA/cudaq-algorithms
>
> Existing code keeps working with CUDA-Q Solvers 0.6.0, but all new features
> and fixes land in CUDA-Q Algorithms. We encourage every CUDA-Q Solvers user to
> migrate.

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

New projects should start with the
[CUDA-Q Algorithms documentation](https://nvidia.github.io/cudaq-algorithms/),
which supersedes this library.

For documentation, tutorials, and API reference for CUDA-Q Solvers 0.6.0,
visit the [CUDA-Q Solvers Documentation](https://nvidia.github.io/cudaqx/components/solvers/introduction.html).

## License

CUDA-Q Solvers is an open source project. The source code is available on
[GitHub][github_link] and licensed under [Apache License
2.0](https://github.com/NVIDIA/cudaqx/blob/main/LICENSE).

[github_link]: https://github.com/NVIDIA/cudaqx/tree/main/libs/solvers
