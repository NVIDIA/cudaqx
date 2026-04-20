# ============================================================================ #
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  #
# All rights reserved.                                                        #
#                                                                             #
# This source code and the accompanying materials are made available under    #
# the terms of the Apache License 2.0 which accompanies this distribution.   #
# ============================================================================ #

# [Begin Documentation]
import numpy as np
import cudaq_qec as qec

# Define a check matrix for a [3,1] repetition code.
# Rows = checks (stabilizers), columns = error mechanisms.
H = np.array([[1, 1, 0],
              [0, 1, 1]], dtype=np.uint8)

# Independent error probability for each mechanism.
error_probs = np.array([0.05, 0.10, 0.05])

num_shots = 10

# Sample syndromes and errors from the detector error model.
# backend="auto" (default) tries GPU first, then falls back to CPU.
syndromes, errors = qec.dem_sampling(H, num_shots, error_probs, seed=42)

print(f"Check matrix H ({H.shape[0]} checks x {H.shape[1]} mechanisms):")
print(H)
print(f"\nError probabilities: {error_probs}")
print(f"\nSampled errors  ({errors.shape}):\n{errors}")
print(f"\nSampled syndromes ({syndromes.shape}):\n{syndromes}")

# Verify: syndromes should equal (errors @ H^T) mod 2.
expected = (errors @ H.T) % 2
assert np.array_equal(syndromes, expected), "Mismatch!"
print("\nVerification passed: syndromes == (errors @ H^T) mod 2")

# Reproducibility: the same seed yields the same results.
s1, e1 = qec.dem_sampling(H, num_shots, error_probs, seed=123)
s2, e2 = qec.dem_sampling(H, num_shots, error_probs, seed=123)
assert np.array_equal(e1, e2)
print("Reproducibility check passed: same seed -> same output")

# Force a specific backend (cpu or gpu).
syndromes_cpu, errors_cpu = qec.dem_sampling(
    H, num_shots, error_probs, seed=42, backend="cpu")
print(f"\nCPU backend result shapes: syndromes {syndromes_cpu.shape}, "
      f"errors {errors_cpu.shape}")
