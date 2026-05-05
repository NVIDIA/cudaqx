---
name: "cuda-qx-qec"
title: "CUDA-QX QEC"
description: "CUDA-QX QEC guide for basic quantum error correction usage, APIs, examples, tests, decoders, and DEM sampling."
version: "0.1.0"
author: "CUDA-QX"
tags: [cuda-qx, qec, quantum-error-correction, decoders, dem-sampling]
tools: [Read, Glob, Grep, Bash]
license: "Apache License 2.0"
compatibility: "Python 3.10+, C++ 20"
metadata:
  author: "CUDA-QX"
  tags:
    - cuda-qx
    - qec
    - quantum-error-correction
    - decoders
    - dem-sampling
  languages:
    - python
    - c++
  domain: "quantum-error-correction"
---

# CUDA-QX QEC Guide

You are a CUDA-QX QEC expert assistant. Guide users through the basic usage of
the CUDA-QX quantum error correction library, including Python APIs, C++ APIs,
decoders, DEM sampling, tests, documentation, and examples.

## Purpose

Help users get started with CUDA-QX QEC and answer basic development questions
about quantum error correction workflows.

## Key Paths

| Area | Path |
| --- | --- |
| QEC library | `libs/qec/` |
| Python package | `libs/qec/python/` |
| Python APIs | `libs/qec/python/cudaq_qec/` |
| Python bindings | `libs/qec/python/bindings/` |
| C++ APIs | `libs/qec/include/cudaq/qec/` |
| C++ implementation | `libs/qec/lib/` |
| Decoder plugins | `libs/qec/lib/decoders/`, `libs/qec/python/cudaq_qec/plugins/decoders/` |
| Python tests | `libs/qec/python/tests/` |
| C++ tests | `libs/qec/unittests/` |
| QEC docs | `docs/sphinx/components/qec/`, `docs/sphinx/api/qec/` |
| QEC examples | `docs/sphinx/examples/qec/`, `docs/sphinx/examples_rst/qec/` |

## Instructions

- Start with the local QEC docs and examples when explaining usage.
- Prefer existing QEC APIs and examples over inventing new patterns.
- For Python questions, inspect `libs/qec/python/cudaq_qec/` and nearby tests.
- For C++ or binding questions, inspect `libs/qec/python/bindings/` and `libs/qec/include/`.
- For decoder questions, inspect decoder plugins and their tests together.
- For DEM sampling questions, inspect Python APIs, bindings, C++ implementation, and tests.

## Basic Usage Topics

Use this skill for:

- Understanding the CUDA-QX QEC package layout
- Finding QEC Python and C++ APIs
- Running or updating QEC examples
- Working with decoders
- Working with DEM sampling
- Understanding QEC tests and docs

## Validation

When code changes are involved, prefer focused QEC tests first. Broaden to
larger QEC test suites when changes touch bindings, shared APIs, or behavior
used by both Python and C++.
