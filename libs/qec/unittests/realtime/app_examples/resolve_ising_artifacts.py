#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                           #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Resolve and validate prepared artifacts for surface_code-4-yaml.

The Hugging Face model repository contains weights, while the realtime example
consumes a locally exported ONNX model plus matching decoder data.  This helper
only validates that prepared, offline artifact directory; it never downloads
or converts model data.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

REQUIRED_FILES = (
    "model.onnx",
    "H_csr.bin",
    "O_csr.bin",
    "priors.bin",
    "metadata.txt",
    "D_sparse.txt",
)


def default_artifacts_dir() -> Path:
    cache_home = os.environ.get("XDG_CACHE_HOME")
    if cache_home:
        return Path(cache_home) / "cudaqx/ising/fast/d7_t7_z_xv"
    home = os.environ.get("HOME")
    if home:
        return Path(home) / ".cache/cudaqx/ising/fast/d7_t7_z_xv"
    raise ValueError(
        "cannot determine the default artifact directory: set XDG_CACHE_HOME "
        "or HOME, or pass --artifacts-dir")


def read_metadata(path: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for line_number, raw_line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{line_number}: expected key=value")
        key, value = (part.strip() for part in line.split("=", 1))
        if not key or not value:
            raise ValueError(
                f"{path}:{line_number}: expected non-empty key=value")
        if key in metadata:
            raise ValueError(f"{path}:{line_number}: duplicate key '{key}'")
        metadata[key] = value
    return metadata


def validate_artifacts(directory: Path, distance: int, num_rounds: int) -> Path:
    missing = [
        name for name in REQUIRED_FILES if not (directory / name).is_file()
    ]
    empty = [
        name for name in REQUIRED_FILES
        if (directory / name).is_file() and (directory /
                                             name).stat().st_size == 0
    ]
    if missing or empty:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if empty:
            details.append("empty: " + ", ".join(empty))
        raise ValueError(
            f"incomplete Ising artifact directory '{directory}' ({'; '.join(details)})"
        )

    metadata = read_metadata(directory / "metadata.txt")
    required_metadata = {
        "distance": str(distance),
        "n_rounds": str(num_rounds),
        "basis": "Z",
        "code_rotation": "XV",
    }
    for key, expected in required_metadata.items():
        actual = metadata.get(key)
        if actual is None:
            raise ValueError(
                f"{directory / 'metadata.txt'}: missing key '{key}'")
        if actual != expected:
            raise ValueError(
                f"{directory / 'metadata.txt'}: {key}='{actual}', expected '{expected}'"
            )
    return directory.resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=(
        "Resolve a prepared Ising directory and verify its required files "
        "and d/T/Z/XV metadata. This helper never downloads model data."))
    parser.add_argument("--artifacts-dir", type=Path)
    parser.add_argument("--distance", type=int, required=True)
    parser.add_argument("--num-rounds", type=int, required=True)
    args = parser.parse_args()

    try:
        directory = args.artifacts_dir or default_artifacts_dir()
        print(validate_artifacts(directory, args.distance, args.num_rounds))
    except (OSError, ValueError) as error:
        parser.exit(
            1,
            f"Error: {error}\n"
            "The Hugging Face repository provides SafeTensors weights, not a "
            "ready CUDA-QX artifact directory. Follow the Ising export recipe "
            "in surface_code-4-yaml-test.sh.\n",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
