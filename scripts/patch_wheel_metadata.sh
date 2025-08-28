#!/bin/bash
# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This script is used to patch the wheel metadata for a PyPI package. For now,
# it is anticipated that one would use this script as a reference and update the
# MODIFY_ME1 and MODIFY_ME2 sections before using it.

# Note: you may need to run "python3 -m pip install -U wheel" first.

# MODIFY_ME1 - review and modify the following variables
PACKAGE_NAME_DASH=cudaq-qec
PACKAGE_NAME_UNDER=cudaq_qec
ORIG_VER=0.4.0
NEW_VER=0.4.0.post1

# Make sure that curl, jq, python3, and wget are installed.
if ! command -v curl &> /dev/null; then
  echo "curl could not be found"
  exit 1
fi
if ! command -v jq &> /dev/null; then
  echo "jq could not be found"
  exit 1
fi
if ! command -v python3 &> /dev/null; then
  echo "python3 could not be found"
  exit 1
fi
if ! command -v wget &> /dev/null; then
  echo "wget could not be found"
  exit 1
fi

# Make a temporary directory to work in
TMP_DIR=$(mktemp -d)
echo "Using temporary directory: $TMP_DIR"

# Be sure to clean up the temporary directory on exit
trap "rm -rf $TMP_DIR" EXIT

echo "Downloading the original wheels into wheels_orig..."
mkdir -p wheels_orig && \
curl -fsSL "https://pypi.org/pypi/${PACKAGE_NAME_DASH}/${ORIG_VER}/json" \
| jq -r '.urls[] | select(.packagetype=="bdist_wheel") | .url' \
| xargs -n1 -P4 -I{} wget -c -P wheels_orig {}

mkdir -p wheels_new

echo "Placing the patched wheels into wheels_new..."
for f in wheels_orig/*.whl; do
  python3 -m wheel unpack $f -d $TMP_DIR

  # --- Begin modifications
  # Update the version
  sed -i "s/^Version: ${ORIG_VER}/Version: ${NEW_VER}/" $TMP_DIR/${PACKAGE_NAME_UNDER}-${ORIG_VER}/${PACKAGE_NAME_UNDER}-${ORIG_VER}.dist-info/METADATA
  # MODIFY_ME2 - review and modify the METADATA file here
  sed -i "s/tensor_network_decoder/tensor-network-decoder/" $TMP_DIR/${PACKAGE_NAME_UNDER}-${ORIG_VER}/${PACKAGE_NAME_UNDER}-${ORIG_VER}.dist-info/METADATA 
  # --- End modifications

  # Re-package into a new whl file now
  cd $TMP_DIR
  mv ${PACKAGE_NAME_UNDER}-${ORIG_VER}/${PACKAGE_NAME_UNDER}-${ORIG_VER}.dist-info ${PACKAGE_NAME_UNDER}-${ORIG_VER}/${PACKAGE_NAME_UNDER}-${NEW_VER}.dist-info
  python3 -m wheel pack ${PACKAGE_NAME_UNDER}-${ORIG_VER} -d .
  cd -
  mv $TMP_DIR/${PACKAGE_NAME_UNDER}-${NEW_VER}*.whl wheels_new
  rm -rf $TMP_DIR
done

echo "Done!"
echo "Your original wheels are in wheels_orig, and your patched wheels are in wheels_new."
echo "You can now upload the patched wheels to PyPI."
