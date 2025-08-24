# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

#!/usr/bin/env bash
set -euo pipefail

JSON_FILE="versions.json"
ORG="NVIDIA"
PKG="cudaqx-dev"
FORCE=false
SKIP_CLEANUP=false

usage() {
  cat <<EOF
Usage: $0 [--force --skip-cleanup]

- Reads GitHub Packages API for $ORG/$PKG/versions
- Shows all version IDs and tags
- Prompts for which 8-char SHA to delete
- Blocks if any selected versions include a semver tag (x.y.z) unless --force
- Confirms before deleting

Examples:
  $0
  $0 --force
  $0 --skip-cleanup
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE=true; shift ;;
    --skip-cleanup) SKIP_CLEANUP=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

# Verify that the token is set
if [[ -z "$TOKEN" ]]; then
  echo "ERROR: TOKEN is not set. You must set the TOKEN environment variable, and"
  echo "it must have read:packages and delete:packages scopes."
  exit 1
fi

# Create a temporary directory and cd into it.
TMP_DIR=$(mktemp -d)
echo "TMP_DIR: $TMP_DIR"
cd "$TMP_DIR" || exit 1

if [[ "$SKIP_CLEANUP" != "true" ]]; then
  # Remove the temporary directory on exit.
  trap 'rm -rf "$TMP_DIR"' EXIT
fi

API_BASE="https://api.github.com/orgs/${ORG}/packages/container/${PKG}/versions"

echo "Downloading $JSON_FILE from GitHub Packages..."
curl -H "Authorization: Bearer $TOKEN" -H "Accept: application/vnd.github.v3+json" "$API_BASE?per_page=100" > "$JSON_FILE"

if [[ ! -s "$JSON_FILE" ]]; then
  echo "ERROR: JSON file '$JSON_FILE' not found or empty."
  exit 1
fi

# 1) Pretty list: <id>: <tags...>
echo "Current versions:"
jq -r '
  map(select(.metadata? and .metadata.container? and (.metadata.container.tags? | type=="array")))
  | .[]
  | "\(.id): \(.metadata.container.tags | join(" "))"
' "$JSON_FILE" > ids.txt
cat ids.txt | column -t
echo

read -rp "Enter the 8-char SHA to delete (e.g., 5fad63e4): " SHORTSHA
if [[ ! "$SHORTSHA" =~ ^[0-9a-f]{8}$ ]]; then
  echo "ERROR: '$SHORTSHA' is not an 8-char lowercase hex SHA."
  exit 1
fi

# 3) Compute candidate IDs & show what would be deleted
echo
echo "Matching versions for SHA '$SHORTSHA':"
cat ids.txt | grep "$SHORTSHA" > ids_to_delete.txt
cat ids_to_delete.txt | column -t
IDS=$(cat ids_to_delete.txt | awk -F: '{gsub(/^[ \t]+/, "", $1); print $1}')
# Convert IDS to array
mapfile -t IDS < <(echo "$IDS")

# Check for semver tags (x.y.z) among candidates
HAS_SEMVER=$(cat ids_to_delete.txt | grep -q "[0-9]\+\.[0-9]\+\.[0-9]\+" && echo "true" || echo "false")

if [[ "$HAS_SEMVER" == "true" && "$FORCE" != "true" ]]; then
  echo "ABORT: One or more matching versions include a semantic version tag (e.g., 0.4.0)."
  echo "       Re-run with --force if you are REALLY sure you want to delete them."
  exit 1
fi

# 5) Final confirmation
echo "About to delete ${#IDS[@]} version(s) from org '$ORG' package '$PKG'."
read -rp "Type 'delete' to proceed: " CONFIRM
if [[ "$CONFIRM" != "delete" ]]; then
  echo "Canceled."
  exit 0
fi

for id in "${IDS[@]}"; do
  # Sanity check: make sure that the id is exactly 9 digits and a number
  if [[ ! "$id" =~ ^[0-9]+$ ]]; then
    echo "ERROR: ID '$id' is not a number. Exiting just to be safe."
    exit 1
  fi
  if [[ ${#id} -ne 9 ]]; then
    echo "ERROR: ID '$id' is not 9 digits. Exiting just to be safe."
    exit 1
  fi
  echo "Deleting version id $id ..."
  curl -sS -X DELETE \
    -H "Authorization: Bearer $TOKEN" \
    -H "Accept: application/vnd.github.v3+json" \
    "${API_BASE}/${id}" \
    | jq -r 'if .message? then "  -> " + .message else "  -> deleted" end'
done

echo "Done."
