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
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest


def csr(rows: int, cols: int, indptr: list[int], indices: list[int]) -> bytes:
    return (struct.pack("=III", rows, cols, len(indices)) +
            struct.pack(f"={len(indptr)}i", *indptr) +
            struct.pack(f"={len(indices)}i", *indices))


def make_bundle(path: Path, distance: int = 3, num_rounds: int = 3) -> None:
    path.mkdir(parents=True)
    (path / "model.onnx").write_bytes(b"offline-test-placeholder")
    (path / "H_csr.bin").write_bytes(csr(24, 1, [0, 1] + [1] * 23, [0]))
    (path / "O_csr.bin").write_bytes(csr(1, 1, [0, 1], [0]))
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

    def setUp(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory(
            prefix="cudaqx-ising-test-")
        self.addCleanup(temporary_directory.cleanup)
        self.temp = Path(temporary_directory.name)

    def run_command(
            self,
            command: list[Path | str],
            *,
            expected: int = 0,
            diagnostic: str | None = None) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [str(value) for value in command],
            cwd=self.temp,
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

    def test_builtin_preset_rejects_overrides(self) -> None:
        cases = (
            [
                self.app, "--distance", "5", "--num_rounds", "5",
                "--decoder_type", "trt_decoder", "--use-ising", "--save_dem",
                self.temp / "decoder.yml"
            ],
            [
                "bash", self.driver, self.app, "7", "7", "trt_decoder", "1",
                "--use-ising", "--p_spam", "0.02"
            ],
        )
        for command in cases:
            with self.subTest(command=command):
                self.run_command(
                    command,
                    expected=1,
                    diagnostic="built-in Ising example only supports distance=7",
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
        metadata = "distance=3\nn_rounds=3\nbasis=Z\ncode_rotation=XV\n"
        h_csr = csr(24, 1, [0, 1] + [1] * 23, [0])
        cases = (
            ("missing-model", "model.onnx", None,
             "missing or empty: model.onnx"),
            ("duplicate-metadata", "metadata.txt", metadata + "basis=Z\n",
             "duplicate metadata key 'basis'"),
            ("wrong-basis", "metadata.txt",
             metadata.replace("basis=Z", "basis=X"), "basis='X'"),
            ("csr-count-overflow", "H_csr.bin",
             struct.pack("=III", 0xFFFFFFFF, 1,
                         0), "Invalid Ising CSR dimensions"),
            ("csr-column", "H_csr.bin", csr(24, 1, [0, 1] + [1] * 23, [1]),
             "Invalid Ising CSR column index"),
            ("csr-trailing-data", "H_csr.bin", h_csr + b"\x00",
             "Ising CSR file size does not match header"),
            ("priors", "priors.bin",
             struct.pack("=I", 1) + struct.pack("=d", 1.5),
             "Invalid probability in Ising priors file"),
            ("priors-count-overflow", "priors.bin",
             struct.pack("=I", 0xFFFFFFFF),
             "Ising priors file size does not match header"),
            ("priors-trailing-data", "priors.bin",
             struct.pack("=I", 1) + struct.pack("=d", 0.01) + b"\x00",
             "Ising priors file size does not match header"),
            ("mapping", "D_sparse.txt", "33 -1\n" + "0 -1\n" * 23,
             "Invalid measurement index in Ising D_sparse"),
            ("mapping-span", "D_sparse.txt",
             "\n".join(f"{row} -1" for row in range(24)) + "\n",
             "Ising D_sparse measurement span (24)"),
            ("observable-rows", "O_csr.bin", csr(2, 1, [0, 1, 1], [0]),
             "expected one logical observable"),
        )
        for name, filename, contents, diagnostic in cases:
            with self.subTest(case=name):
                bundle = self.temp / name
                make_bundle(bundle)
                path = bundle / filename
                if contents is None:
                    path.unlink()
                elif isinstance(contents, bytes):
                    path.write_bytes(contents)
                else:
                    path.write_text(contents, encoding="utf-8")
                self.run_command(
                    self.app_command(bundle),
                    expected=1,
                    diagnostic=diagnostic,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", required=True)
    parser.add_argument("--driver", required=True)
    ARGS = parser.parse_args()
    unittest.main(argv=[__file__])
