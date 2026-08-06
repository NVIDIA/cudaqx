# Formatting (clang-format and yapf)

Code style in CUDA-Q Libraries is enforced by `clang-format` for C++ /
CUDA and `yapf` for Python. Local format must match what CI does;
failing this is the most common reason for review churn.

## Scripts

| Script | Use |
|--------|-----|
| `scripts/run_clang_format.sh` | format all C++ / CUDA files in-tree |
| `scripts/run_clang_format.sh --check` | exit non-zero if anything needs reformatting (CI mode) |
| `scripts/run_yapf_format.sh` | format all Python files in-tree |
| `scripts/run_yapf_format.sh --check` | check-only (CI mode) |

Always run *without* `--check` before pushing; CI runs *with*
`--check`.

## Pre-commit hook (recommended)

```bash
cat > .git/hooks/pre-commit <<'EOF'
#!/usr/bin/env bash
set -e
bash scripts/run_clang_format.sh --check
bash scripts/run_yapf_format.sh --check
bash scripts/sync_agents_skills.sh --check
EOF
chmod +x .git/hooks/pre-commit
```

This blocks commits that would fail CI. Override with `--no-verify`
if you must (rare; e.g. when intentionally committing
formatting-only changes).

## clang-format

The repo's `.clang-format` defines the style. Common things to know:

- Column limit: see `.clang-format` (typically 80 or 100).
- Tabs: no — spaces only.
- `#include` ordering enforced; CI catches misordered includes.
- Long parameter lists are wrapped after the opening paren.

If your editor (VSCode, CLion) is configured to use the project's
`.clang-format`, edits are auto-formatted as you type.

## yapf

The repo's `.style.yapf` (or `pyproject.toml`) sets:

- Indent width: 4 spaces.
- Quotes: double or single depending on the file's existing pattern.
- Max line length: see config.

Imports are *not* reordered by `yapf`. If you reorganize imports,
do it manually and don't rely on autoformatting.

## Editor integration

### VSCode

`settings.json` snippet:

```json
{
  "editor.formatOnSave": true,
  "python.formatting.provider": "yapf",
  "python.formatting.yapfArgs": ["--style=pep8"],
  "C_Cpp.clang_format_style": "file",
  "C_Cpp.formatting": "clangFormat"
}
```

For a project-specific yapf config, the project's `.style.yapf`
(or `pyproject.toml`) is picked up automatically.

### CLion / IntelliJ

Enable "Use the project's `.clang-format` file" in
Settings → Editor → Code Style → C/C++.

### Vim / Neovim

```vim
autocmd FileType cpp,cuda autocmd BufWritePre <buffer> silent! ClangFormat
autocmd FileType python autocmd BufWritePre <buffer> silent! YAPF
```

## Common pitfalls

| Symptom | Cause |
|---------|-------|
| `run_clang_format.sh` fails: "binary not found" | install `clang-format` (`apt install clang-format` or `brew install clang-format`) |
| Mass diff after running locally | your `clang-format` version differs from CI's |
| `run_yapf_format.sh` produces a different output than VSCode | yapf version mismatch; use the version from `pip install yapf==<version>` if pinned |
| Mixed tabs/spaces | open the file in a tab-visible editor and fix; clang-format/yapf will then keep it clean |

For clang-format version skew, the CI script pins a specific version;
match locally:

```bash
clang-format --version          # check
pip install clang-format==<ver> # if pinned via Python wrapper
```

## What clang-format / yapf do *not* enforce

- Naming conventions — manual.
- Comment style — manual (one-line trivia is fine; multi-paragraph
  docstrings discouraged per general guidelines).
- License header — copy from a neighbor file; not enforced by
  formatters.
- Unused imports / variables — `flake8` / `clang-tidy` find these,
  not the formatters.

## Self-check

```
[ ] `bash scripts/run_clang_format.sh --check` exits 0
[ ] `bash scripts/run_yapf_format.sh --check` exits 0
[ ] License header present on every new file
[ ] Imports manually ordered if you added or removed any
```

## Where next

- File layout: `file-layout.md`.
- Submit the PR: `pr-workflow.md`.
- Test before pushing: `tests-pytest.md` / `tests-googletest.md` / `tests-run-locally.md`.
