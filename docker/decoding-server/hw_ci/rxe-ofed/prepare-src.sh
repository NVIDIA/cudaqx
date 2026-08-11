#!/usr/bin/env bash
# Stage the pinned upstream rxe (SoftRoCE) driver source with the
# OFED-compat patch applied, ready for a per-host `make` at container setup
# time (see the sibling makefile and README.md for why).
#
# Runs at dev-image build time (dev.Dockerfile); can also be run manually.
#
#   prepare-src.sh [kernel-ref] [dest-dir]
set -euo pipefail

ref=${1:-v6.17}
dest=${2:-/opt/rxe-ofed/src}
here=$(cd "$(dirname "$0")" && pwd)

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# Sparse partial clone: only the rxe directory's blobs are fetched.
git clone --depth 1 --branch "$ref" --filter=tree:0 --sparse --quiet \
    https://github.com/torvalds/linux.git "$tmp/linux"
git -C "$tmp/linux" sparse-checkout set drivers/infiniband/sw/rxe

mkdir -p "$dest"
cp "$tmp/linux/drivers/infiniband/sw/rxe/"* "$dest/"
test -f "$dest/rxe.c"   # sparse checkout sanity

# --fuzz=0: any drift between the pinned ref and the patch must fail the
# image build loudly rather than half-apply.
patch -d "$dest" -p1 --fuzz=0 < "$here/ofed-compat.patch"

cp "$here/rxe_ofed_compat.h" "$here/makefile" "$dest/"
echo "rxe-ofed: staged $ref + ofed-compat.patch at $dest"
