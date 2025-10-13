#!/bin/sh

# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Abort on error
set -e

# Note, you must run this script from the root of the repository.
TOP_DIR=$(pwd)

if [ ! -e Contributor_License_Agreement.md ]; then
  echo "You must run this script from the root of the repository."
  exit 1
fi

# There should be one command line argument, the version to use for the metapackages.
if [ $# -ne 1 ]; then
  echo "Usage: $0 <version>"
  exit 1
fi
VERSION=$1

# Validate that the version is a valid version for a Python package.
if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+ ]]; then
  echo "Error: Version $VERSION is not a valid version for a Python package."
  exit 1
fi

FILES_TO_COPY=(LICENSE NOTICE CITATION.cff)

for package in qec solvers; do
  echo "Building $package metapackage..."
  cd $TOP_DIR/libs/$package/python/metapackages

  # Copy the appropriate LICENSE, NOTICE, and CITATION.cff files to the metapackage directory.
  rm -rf dist *.egg-info _version.txt $FILES_TO_COPY
  for file in ${FILES_TO_COPY[@]}; do
    cp $TOP_DIR/$file .
  done

  # Create a version.txt file in the metapackage directory.
  echo $VERSION > _version.txt

  CUDAQ_META_WHEEL_BUILD=1 python3 -m build . --sdist

done
