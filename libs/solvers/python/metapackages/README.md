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
quantum-classical hybrid algorithms and numerical routines frequently used in
quantum computing applications. The library is designed to work seamlessly with
CUDA-Q quantum programs.

This is package is a meta-package that when installed, simply installs the
version of the package that is appropriate for your system, e.g.
[`cudaq-solvers-cu12`](https://pypi.org/project/cudaq-solvers-cu12/) or
[`cudaq-solvers-cu13`](https://pypi.org/project/cudaq-solvers-cu13/). Please
click those links for more details about the package.

The optional dependencies that work with the above sub-packages also work with
this meta-package.
