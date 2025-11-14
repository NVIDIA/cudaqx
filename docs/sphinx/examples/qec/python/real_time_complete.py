# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# [Begin Documentation]

#!/usr/bin/env python3
"""
Simple 3-qubit repetition code with real-time decoding.
This is the most basic QEC example possible.
"""

import os
os.environ["CUDAQ_DEFAULT_SIMULATOR"] = "stim"

import cudaq
import cudaq_qec as qec

# Prepare logical |0⟩
@cudaq.kernel
def prep0(logical: qec.patch):
    for i in range(logical.data.size()):
        cudaq.reset(logical.data[i])

# Measure ZZ stabilizers for 3-qubit repetition code
@cudaq.kernel
def measure_stabilizers(logical: qec.patch):
    for i in range(logical.ancz.size()):
        cudaq.reset(logical.ancz[i])
    
    # Z0Z1 stabilizer
    cudaq.cx(logical.data[0], logical.ancz[0])
    cudaq.cx(logical.data[1], logical.ancz[0])
    
    # Z1Z2 stabilizer
    cudaq.cx(logical.data[1], logical.ancz[1])
    cudaq.cx(logical.data[2], logical.ancz[1])
    
    return [cudaq.mz(logical.ancz[0]), cudaq.mz(logical.ancz[1])]

# QEC circuit with real-time decoding
@cudaq.kernel
def qec_circuit():
    qec.reset_decoder(0)
    
    data = cudaq.qvector(3)
    ancz = cudaq.qvector(2)
    ancx = cudaq.qvector(0)
    logical = qec.patch(data, ancx, ancz)
    
    prep0(logical)
    
    # 3 rounds of syndrome measurement
    for _ in range(3):
        syndromes = measure_stabilizers(logical)
        qec.enqueue_syndromes(0, syndromes)
    
    # Get corrections and apply them
    corrections = qec.get_corrections(0, 3, False)
    for i in range(3):
        if corrections[i]:
            cudaq.x(data[i])
    
    cudaq.mz(data)

def main():
    # Get 3-qubit repetition code
    code = qec.get_code("repetition", distance=3)
    
    # Step 1: Generate detector error model
    print("Step 1: Generating DEM...")
    cudaq.set_target("stim")
    
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.Depolarization2(0.01), 1)
    
    dem = qec.z_dem_from_memory_circuit(
        code, qec.operation.prep0, 3, noise)
    
    # Save decoder config
    config = qec.DecoderConfig()
    config.id = 0
    config.type = "multi_error_lut"
    config.block_size = dem.detector_error_matrix.shape[1]
    config.syndrome_size = dem.detector_error_matrix.shape[0]
    config.H_sparse = qec.pcm_to_sparse_vec(dem.detector_error_matrix)
    config.O_sparse = qec.pcm_to_sparse_vec(dem.observables_flips_matrix)
    
    # Calculate numRounds from DEM (we send 1 additional round, so add 1)
    num_syndromes_per_round = 2  # Z0Z1 and Z1Z2
    num_rounds = dem.detector_error_matrix.shape[0] // num_syndromes_per_round + 1
    config.D_sparse = qec.generate_timelike_sparse_detector_matrix(
        num_syndromes_per_round, num_rounds, False)
    config.decoder_custom_args = {"lut_error_depth": 2}
    
    multi_config = qec.MultiDecoderConfig()
    multi_config.decoders = [config]
    
    with open("config.yaml", 'w') as f:
        f.write(multi_config.to_yaml_str(200))
    print("Saved config to config.yaml")
    
    # Step 2: Load config and run circuit
    print("\nStep 2: Running circuit with decoding...")
    qec.configure_decoders_from_file("config.yaml")
    
    cudaq.sample(qec_circuit, shots_count=10)
    print("Ran 10 shots")
    
    qec.finalize_decoders()
    print("\nDone!")

if __name__ == "__main__":
    main()
