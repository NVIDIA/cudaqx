# Build the docs and serve them offline

Doxygen → Sphinx + Breathe pipeline. The doc tree imports the Python
package to render API tables, so the build must run **after** `ninja
install` (or against an installed wheel).

## One-shot build

```bash
export CUDAQX_INSTALL_PREFIX=$HOME/.cudaqx
export PYTHONPATH=$CUDAQX_INSTALL_PREFIX:$PYTHONPATH
bash scripts/build_docs.sh        # honors CUDAQX_DOCS_GEN_IMPORT_CUDAQ=ON
```

What the script does:

1. Substitutes `@VAR@` placeholders in `docs/sphinx/conf.py.in` →
   `docs/sphinx/conf.py`.
2. Generates `docs/Doxyfile` from `docs/Doxyfile.in`, feeding it the
   headers from `libs/{core,qec,solvers}/include/`.
3. Runs Doxygen → `build/docs/doxygen/`.
4. Copies `build/docs/doxygen/` to `docs/sphinx/_doxygen/` (Breathe input).
5. Runs `sphinx-build -b html -j auto sphinx build/docs/sphinx`.
6. Copies `build/docs/sphinx/` to `$CUDAQX_INSTALL_PREFIX/docs/`.

**`ninja docs` target**: when the top-level CMake configures with
`CUDAQX_INCLUDE_DOCS=ON` (the default), `ninja docs` is a thin wrapper
that calls `build_docs.sh`. Output ends up in `build/docs/build/`.

## Serve the docs offline

After the build finishes, the rendered HTML lives at
`$CUDAQX_INSTALL_PREFIX/docs/` (and at `build/docs/sphinx/` in-tree).
Two ways to read it:

**Option A: simple HTTP server (recommended for browsing across pages)**

```bash
python3 -m http.server --directory $CUDAQX_INSTALL_PREFIX/docs 8000
```

Then open <http://localhost:8000>. Use this when navigating between API
pages, components, and examples — relative links and search work
correctly only when served, not when opened from `file://`.

**Option B: open the HTML directly (single-page reads)**

```bash
xdg-open $CUDAQX_INSTALL_PREFIX/docs/index.html
# macOS:  open ...
# In VS Code / Cursor:  right-click index.html in Explorer → "Open with Live Server"
```

Fast, but cross-page links and the search box may not work due to
browser `file://` restrictions.

## Verify the API tables are non-empty

```bash
python3 -c "
import urllib.request
html = urllib.request.urlopen('http://localhost:8000/api/qec/python_api.html').read().decode()
assert 'class' in html.lower() and len(html) > 5000, 'API page suspiciously empty'
print('OK: cudaq_qec API page rendered with classes')
"
```

If the API page is blank, the import-time autodoc rendered nothing —
almost always a `PYTHONPATH` issue. See **Troubleshooting** below.

## Troubleshooting (docs-specific)

| Symptom                                                      | Cause                                                        | Fix                                                       |
|--------------------------------------------------------------|--------------------------------------------------------------|-----------------------------------------------------------|
| API tables blank                                             | `CUDAQX_DOCS_GEN_IMPORT_CUDAQ` unset, or `PYTHONPATH` wrong | Use `scripts/build_docs.sh`; ensure `$CUDAQX_INSTALL_PREFIX` is on `PYTHONPATH` |
| `breathe_projects.cudaqx` empty / Doxygen XML missing        | Doxygen failed                                               | Check `$logs_dir/doxygen_error.txt`, fix header parse error, rerun `build_docs.sh` |
| Stale rendered content after a failed build                  | `_doxygen/` and `_mdgen/` left over                          | `rm -rf docs/sphinx/_doxygen docs/sphinx/_mdgen` and rerun |
| `sphinx-build` fails: `cudaq_qec` not importable             | install prefix not on `PYTHONPATH`                           | `export PYTHONPATH=$CUDAQX_INSTALL_PREFIX:$PYTHONPATH` and rerun |
| Search box returns nothing in the browser                    | Opened via `file://`; Sphinx search index is fetched async   | Serve via `python3 -m http.server` (Option A above)       |

## Where things live

| Output                          | Path                                       |
|---------------------------------|--------------------------------------------|
| Doxygen XML (intermediate)      | `build/docs/doxygen/`                      |
| Sphinx HTML (in-tree)           | `build/docs/sphinx/`                       |
| Sphinx HTML (installed)         | `$CUDAQX_INSTALL_PREFIX/docs/`             |
| `ninja docs` final destination  | `build/docs/build/`                        |
| Breathe input (copied Doxygen)  | `docs/sphinx/_doxygen/`                    |
