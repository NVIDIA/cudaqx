#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Usage:
# bash scripts/run_yapf_format.sh
#
# This script will use the yapf executable in your PATH.

cd $(git rev-parse --show-toplevel)

# Run Clang Format
git ls-files -- '*.py' | xargs yapf -i

# Take us back to where we were
cd -
