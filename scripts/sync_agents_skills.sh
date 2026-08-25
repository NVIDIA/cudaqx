#!/usr/bin/env bash

# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Mirror .agents/skills/ -> per-agent skill discovery directories.
#
# Different agent tools discover skills under different paths, even though
# the SKILL.md format and directory layout are identical across all of them:
#
#   * Codex       : .agents/skills/<name>/SKILL.md  (canonical, open AGENTS.md convention)
#   * Claude Code : .claude/skills/<name>/SKILL.md  (mirror)
#   * Cursor      : .cursor/skills/<name>/SKILL.md  (mirror)
#
# To avoid maintaining N copies by hand, the canonical copy lives under
# .agents/skills/ and this script regenerates every other tree from it.
# To add a new agent, append one "<dir>:<name>" entry to TARGETS below.
#
# Symlinks are not used because Claude Code's symlink behavior is undocumented
# and Windows requires admin rights for symlink creation. A copy-based mirror
# is more portable. cp -r is used (rather than rsync) so Windows Git Bash
# works out of the box.
#
# Usage:
#   bash scripts/sync_agents_skills.sh          # mirror once
#   bash scripts/sync_agents_skills.sh --check  # exit non-zero if stale
#
# --check is for CI: it builds the mirror in a temp directory and diffs
# against the committed mirror. If they differ the contributor forgot to
# re-run the sync after editing skills.

set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CANONICAL_REL=".agents/skills"

# Per-agent mirror destinations. Each entry is "<relative-path>:<agent-name>".
# To add a new agent (Copilot, Gemini, Windsurf, etc.), append one entry.
TARGETS=(
  ".claude/skills:Claude Code"
  ".cursor/skills:Cursor"
)

# Files / directories we never want to copy into a mirror.
IGNORE_NAMES=(".DS_Store" "__pycache__" ".GENERATED")

# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

CHECK=0
case "${1:-}" in
  --check) CHECK=1 ;;
  -h|--help)
    sed -n '11,33p' "$0"
    exit 0
    ;;
  "") ;;
  *)
    echo "unknown argument: $1" >&2
    echo "usage: bash scripts/sync_agents_skills.sh [--check]" >&2
    exit 2
    ;;
esac

# ---------------------------------------------------------------------------
# Locate repo root
# ---------------------------------------------------------------------------

# Prefer git so moving this script up/down a directory does not silently
# break path arithmetic. Fall back to the script's parent when not in a
# git checkout (e.g. extracted tarball).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if command -v git >/dev/null 2>&1 \
    && REPO_ROOT="$(git -C "$HERE" rev-parse --show-toplevel 2>/dev/null)" \
    && [[ -n "$REPO_ROOT" ]]; then
  :
else
  REPO_ROOT="$(cd "$HERE/.." && pwd)"
fi

CANONICAL="$REPO_ROOT/$CANONICAL_REL"

if [[ ! -d "$CANONICAL" ]]; then
  echo "error: $CANONICAL does not exist" >&2
  exit 1
fi

# Single temp dir for --check mode; cleaned up on any exit. The `if` form
# is intentional: `[[ ]] && rm` would return 1 when TMP is empty (sync
# mode), and that would bubble up as the script's exit code.
TMP=""
_cleanup() {
  if [[ -n "$TMP" && -d "$TMP" ]]; then
    rm -rf "$TMP"
  fi
}
trap _cleanup EXIT

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Copy CANONICAL into $1, drop a .GENERATED sentinel naming agent $2,
# then strip ignored files from the destination.
_sync_one() {
  local dst="$1"
  local agent="$2"
  rm -rf "$dst"
  mkdir -p "$(dirname "$dst")"
  cp -r "$CANONICAL" "$dst"
  for pat in "${IGNORE_NAMES[@]}"; do
    find "$dst" -name "$pat" -prune -exec rm -rf {} + 2>/dev/null || true
  done
  cat > "$dst/.GENERATED" <<EOF
This directory is a generated mirror of $CANONICAL_REL for $agent.

DO NOT EDIT FILES HERE. Edit the corresponding file under $CANONICAL_REL
and re-run scripts/sync_agents_skills.sh.

A CI check fails the build if this directory ever drifts out of sync.
EOF
}

# Returns 0 if $1 matches what a fresh sync would produce for agent $2.
# Writes a STALE diagnostic to stderr otherwise.
_check_one() {
  local dst="$1"
  local agent="$2"
  local staged
  staged="$TMP/$(basename "$dst")"
  _sync_one "$staged" "$agent" >/dev/null
  if [[ ! -d "$dst" ]]; then
    echo "STALE: $dst does not exist" >&2
    return 1
  fi
  if diff -r --brief --exclude='.GENERATED' "$dst" "$staged" >/dev/null 2>&1; then
    return 0
  fi
  echo "STALE: $dst does not match $CANONICAL_REL" >&2
  return 1
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if [[ $CHECK -eq 1 ]]; then
  TMP="$(mktemp -d)"
  stale=0
  for entry in "${TARGETS[@]}"; do
    rel="${entry%%:*}"
    agent="${entry#*:}"
    if ! _check_one "$REPO_ROOT/$rel" "$agent"; then
      stale=1
    fi
  done
  if [[ $stale -eq 1 ]]; then
    echo "Run: bash scripts/sync_agents_skills.sh" >&2
    exit 1
  fi
  echo "OK: all mirrors in sync with $CANONICAL_REL"
  exit 0
fi

for entry in "${TARGETS[@]}"; do
  rel="${entry%%:*}"
  agent="${entry#*:}"
  _sync_one "$REPO_ROOT/$rel" "$agent"
  echo "Synced $CANONICAL_REL -> $rel  ($agent)"
done
