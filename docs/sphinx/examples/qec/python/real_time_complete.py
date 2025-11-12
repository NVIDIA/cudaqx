# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# [Begin Documentation]

import cudaq
import cudaq_qec as qec
from cudaq_qec import patch

# Configure decoder before circuit execution (host-side)
qec.configure_decoders_from_file("decoder_config.yaml")


# Define helper operations
@cudaq.kernel
def prep0(logical: patch):
    # Your state preparation logic
    return


@cudaq.kernel
def measure_stabilizers(logical: patch) -> list[bool]:
    # Your stabilizer measurement logic
    return [False] * 12  # 12 stabilizers for the surface code


# Quantum kernel with real-time decoding (device-side)
@cudaq.kernel
def qec_circuit(decoder_id: int, num_rounds: int):
    # Reset decoder state
    qec.reset_decoder(decoder_id)

    # Allocate qubits
    data = cudaq.qvector(25)
    ancx = cudaq.qvector(12)
    ancz = cudaq.qvector(12)
    logical = patch(data, ancx, ancz)

    # Prepare logical state
    prep0(logical)

    # Syndrome extraction with real-time decoding
    for round in range(num_rounds):
        syndromes = measure_stabilizers(logical)
        qec.enqueue_syndromes(decoder_id, syndromes)

    # Get and apply corrections
    corrections = qec.get_corrections(decoder_id, 1, False)
    if corrections[0]:
        x(data)  # Apply correction

    # Measure logical observable
    result = mz(data)


# Execute
qec_circuit(0, 10)

# Clean up (host-side)
qec.finalize_decoders()
