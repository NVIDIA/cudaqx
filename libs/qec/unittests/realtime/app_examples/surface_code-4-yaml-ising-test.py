#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                           #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Offline tests for the surface_code-4-yaml Ising artifact workflow."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest


def write_csr(path: Path, rows: int, cols: int, indptr: list[int],
              indices: list[int]) -> None:
    path.write_bytes(
        struct.pack("=III", rows, cols, len(indices)) +
        struct.pack(f"={len(indptr)}i", *indptr) +
        struct.pack(f"={len(indices)}i", *indices))


def make_bundle(path: Path, distance: int = 3, num_rounds: int = 3) -> None:
    path.mkdir(parents=True)
    (path / "model.onnx").write_bytes(b"offline-test-placeholder")
    write_csr(path / "H_csr.bin", 24, 1, [0, 1] + [1] * 23, [0])
    write_csr(path / "O_csr.bin", 1, 1, [0, 1], [0])
    (path /
     "priors.bin").write_bytes(struct.pack("=I", 1) + struct.pack("=d", 0.01))
    (path / "metadata.txt").write_text(
        f"distance={distance}\n"
        f"n_rounds={num_rounds}\n"
        "basis=Z\n"
        "code_rotation=XV\n",
        encoding="utf-8",
    )
    (path / "D_sparse.txt").write_text(
        "\n".join([f"{row} -1" for row in range(23)] + ["23 32 -1"]) + "\n",
        encoding="utf-8",
    )


class IsingArtifactWorkflowTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = Path(ARGS.app).resolve()
        cls.driver = Path(ARGS.driver).resolve()
        cls.resolver = Path(ARGS.resolver).resolve()

    def setUp(self) -> None:
        self.temp = Path(tempfile.mkdtemp(prefix="cudaqx-ising-test-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.temp)

    def run_command(
            self,
            command: list[os.PathLike[str] | str],
            *,
            env: dict[str, str] | None = None,
            expected: int = 0,
            diagnostic: str | None = None) -> subprocess.CompletedProcess[str]:
        full_env = os.environ.copy()
        if env:
            full_env.update(env)
        result = subprocess.run(
            [str(value) for value in command],
            cwd=self.temp,
            env=full_env,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if expected == 0:
            self.assertEqual(result.returncode, 0, result.stdout)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout)
        if diagnostic:
            self.assertIn(diagnostic, result.stdout)
        return result

    def app_command(self, bundle: Path) -> list[Path | str]:
        return [
            self.app,
            "--distance",
            "3",
            "--num_rounds",
            "3",
            "--decoder_type",
            "trt_decoder",
            "--use-ising",
            "--ising-artifacts-dir",
            bundle,
            "--save_dem",
            self.temp / "decoder.yml",
        ]

    def test_driver_requires_an_explicit_model_source(self) -> None:
        cases = (
            (
                ["trt_decoder", "1"],
                "trt_decoder requires one model source",
            ),
            (
                [
                    "trt_decoder", "1", "--generate-identity-onnx",
                    "--onnx-path", "model.onnx"
                ],
                "model sources are mutually exclusive",
            ),
            (
                ["trt_decoder", "1", "--ising-artifacts-dir", "bundle"],
                "does not enable Ising",
            ),
            (
                ["pymatching", "1", "--generate-identity-onnx"],
                "decoder_type has no trt_decoder entry",
            ),
        )
        for arguments, diagnostic in cases:
            with self.subTest(diagnostic=diagnostic):
                self.run_command(
                    ["bash", self.driver, "/bin/false", "3", "3", *arguments],
                    expected=1,
                    diagnostic=diagnostic,
                )

    def test_builtin_preset_rejects_geometry_overrides(self) -> None:
        self.run_command(
            [
                self.app,
                "--distance",
                "5",
                "--num_rounds",
                "5",
                "--decoder_type",
                "trt_decoder",
                "--use-ising",
                "--save_dem",
                self.temp / "decoder.yml",
            ],
            expected=1,
            diagnostic="built-in Ising example only supports distance=7",
        )

    def test_driver_preserves_builtin_preset_validation(self) -> None:
        default = self.temp / "cache/cudaqx/ising/fast/d7_t7_z_xv"
        make_bundle(default, distance=7, num_rounds=7)
        self.run_command(
            [
                "bash",
                self.driver,
                self.app,
                "7",
                "7",
                "trt_decoder",
                "1",
                "--use-ising",
                "--p_spam",
                "0.02",
            ],
            env={"XDG_CACHE_HOME": str(self.temp / "cache")},
            expected=1,
            diagnostic="built-in Ising example only supports distance=7",
        )

    def test_resolver_handles_default_and_custom_directories(self) -> None:
        custom = self.temp / "custom"
        make_bundle(custom)
        result = self.run_command([
            os.environ.get("PYTHON", "python3"),
            self.resolver,
            "--artifacts-dir",
            custom,
            "--distance",
            "3",
            "--num-rounds",
            "3",
        ])
        self.assertEqual(result.stdout.strip(), str(custom.resolve()))

        default = self.temp / "cache/cudaqx/ising/fast/d7_t7_z_xv"
        make_bundle(default, distance=7, num_rounds=7)
        result = self.run_command(
            [
                os.environ.get("PYTHON", "python3"),
                self.resolver,
                "--distance",
                "7",
                "--num-rounds",
                "7",
            ],
            env={"XDG_CACHE_HOME": str(self.temp / "cache")},
        )
        self.assertEqual(result.stdout.strip(), str(default.resolve()))

    def test_resolver_rejects_incomplete_and_ambiguous_metadata(self) -> None:
        incomplete = self.temp / "incomplete"
        make_bundle(incomplete)
        (incomplete / "model.onnx").unlink()
        self.run_command(
            [
                os.environ.get("PYTHON", "python3"),
                self.resolver,
                "--artifacts-dir",
                incomplete,
                "--distance",
                "3",
                "--num-rounds",
                "3",
            ],
            expected=1,
            diagnostic="missing: model.onnx",
        )

        duplicate = self.temp / "duplicate"
        make_bundle(duplicate)
        with (duplicate / "metadata.txt").open("a", encoding="utf-8") as stream:
            stream.write("basis=Z\n")
        self.run_command(
            [
                os.environ.get("PYTHON", "python3"),
                self.resolver,
                "--artifacts-dir",
                duplicate,
                "--distance",
                "3",
                "--num-rounds",
                "3",
            ],
            expected=1,
            diagnostic="duplicate key 'basis'",
        )

    def test_cpp_loader_accepts_compatible_custom_bundle(self) -> None:
        bundle = self.temp / "valid"
        make_bundle(bundle)
        result = self.run_command(
            self.app_command(bundle),
            diagnostic="trt+Ising: loaded Ising bundle",
        )
        config = (self.temp / "decoder.yml").read_text(encoding="utf-8")
        self.assertIn(str(bundle / "model.onnx"), config)
        self.assertRegex(config, r"syndrome_size:\s+24")
        self.assertRegex(config, r"block_size:\s+1")
        self.assertIn("Running with p_spam_per_patch = [0.01]", result.stdout)

    def test_cpp_loader_rejects_malformed_bundle_data(self) -> None:
        cases = {
            "csr-count-overflow": (
                lambda bundle: (bundle / "H_csr.bin").write_bytes(
                    struct.pack("=III", 0xFFFFFFFF, 1, 0)),
                "Invalid Ising CSR dimensions",
            ),
            "csr-column": (
                lambda bundle: write_csr(bundle / "H_csr.bin", 24, 1, [0, 1] +
                                         [1] * 23, [1]),
                "Invalid Ising CSR column index",
            ),
            "csr-trailing-data": (
                lambda bundle: (bundle / "H_csr.bin").write_bytes(
                    (bundle / "H_csr.bin").read_bytes() + b"\x00"),
                "Ising CSR file size does not match header",
            ),
            "priors": (
                lambda bundle: (bundle / "priors.bin").write_bytes(
                    struct.pack("=I", 1) + struct.pack("=d", 1.5)),
                "Invalid probability in Ising priors file",
            ),
            "priors-count-overflow": (
                lambda bundle: (bundle / "priors.bin").write_bytes(
                    struct.pack("=I", 0xFFFFFFFF)),
                "Ising priors file size does not match header",
            ),
            "priors-trailing-data": (
                lambda bundle: (bundle / "priors.bin").write_bytes(
                    (bundle / "priors.bin").read_bytes() + b"\x00"),
                "Ising priors file size does not match header",
            ),
            "mapping": (
                lambda bundle:
                (bundle / "D_sparse.txt").write_text("33 -1\n" + "0 -1\n" * 23,
                                                     encoding="utf-8"),
                "Invalid measurement index in Ising D_sparse",
            ),
            "mapping-span": (
                lambda bundle: (bundle / "D_sparse.txt").write_text(
                    "\n".join(f"{row} -1" for row in range(24)) + "\n",
                    encoding="utf-8"),
                "Ising D_sparse measurement span (24)",
            ),
            "observable-rows": (
                lambda bundle: write_csr(bundle / "O_csr.bin", 2, 1, [0, 1, 1],
                                         [0]),
                "expected one logical observable",
            ),
        }
        for name, (mutate, diagnostic) in cases.items():
            with self.subTest(case=name):
                bundle = self.temp / name
                make_bundle(bundle)
                mutate(bundle)
                self.run_command(
                    self.app_command(bundle),
                    expected=1,
                    diagnostic=diagnostic,
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", required=True)
    parser.add_argument("--driver", required=True)
    parser.add_argument("--resolver", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    ARGS = parse_args()
    unittest.main(argv=[__file__])
