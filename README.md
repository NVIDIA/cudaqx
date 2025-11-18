# Welcome to the CUDA-QX repository

This repository contains a set of libraries that build on
NVIDIA CUDA-Q. These libraries enable the rapid development of hybrid quantum-classical
application code leveraging state-of-the-art CPUs, GPUs, and QPUs.

## Getting Started

To learn more about how to work with the CUDA-QX libraries, please take a look at the
[CUDA-QX Documentation][cudaqx_docs]. The page contains detailed
[installation instructions][official_install] for officially released packages.

[cudaqx_docs]: https://nvidia.github.io/cudaqx
[official_install]: https://nvidia.github.io/cudaqx/quickstart/installation.html

## Contributing

There are many ways in which you can get involved with CUDA-QX. If you are
interested in developing quantum applications with the CUDA-QX libraries,
this repository is a great place to get started! For more information about
contributing to the CUDA-QX platform, please take a look at
[Contributing.md](./Contributing.md).

## License

The code in this repository is licensed under [Apache License 2.0](./LICENSE).

When distributed via PyPI, GHCR, or NGC, the binaries generated from this source
code are also distributed under the Apache License 2.0; however, the
`libcudaq-qec-nv-qldpc-decoder.so` library is closed source and is subject to
the [NVIDIA Software License Agreement][github_qec_license]

[github_qec_license]: https://github.com/NVIDIA/cudaqx/blob/main/libs/qec/LICENSE

Contributing a pull request to this repository requires accepting the
Contributor License Agreement (CLA) declaring that you have the right to, and
actually do, grant us the rights to use your contribution. A CLA-bot will
automatically determine whether you need to provide a CLA and decorate the PR
appropriately. Simply follow the instructions provided by the bot. You will only
need to do this once.
