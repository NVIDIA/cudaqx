# CUDA-QX Documentation

This folder contains tools and content to build the CUDA-QX documentation.
The [script for building docs](../scripts/build_docs.sh) can be used to build
the complete CUDA-QX documentation. Please see the comment in that script
for more detail.

We use [Sphinx](https://www.sphinx-doc.org/) to produce documentation in HTML
format. This documentation includes conceptual documentation in the form of
Markdown or reStructuredText format, API documentation generated based on doc
comments in the source code, as well as potentially source code examples and
snippets.

## API Documentation

We use [Sphinx](https://www.sphinx-doc.org/) to include documentation defined in
the form of doc comments in the source code for all of our APIs. The build is
configured by the settings in the [sphinx/conf.py](./sphinx/conf.py) file.

- **C++ source code**: <br/>
  As part of the build [Doxygen](https://www.doxygen.org/) is used to generated
  documentation based on doc comments. The documentation generation is
  configured in the [Doxyfile.in](./Doxyfile.in) file - see the manual for
  [possible configurations](https://www.doxygen.nl/manual/config.html). Our
  build replaces the environment variables used in that file to produce the
  final `Doxyfile` with which `doxygen` is invoked. We use the [Breathe
  extension](https://breathe.readthedocs.io/) for Sphinx to incorporate content
  from the generated XML files in our docs.

- **Python bindings**: <br/>
  We use [pybind11](https://github.com/pybind/pybind11) to define Python
  bindings for the CUDA-QX API. Doc comments are defined as part of
  defining these bindings in C++. To incorporate the API documentation, the
  `cudaqx` Python package needs to be built and installed prior to generating the
  CUDA-QX documentation. This project generates
  ReStructuredText documentation from Python `docstrings` using `autodoc`.
  The generated `.rst` files are in `docs/source/apidoc`. The files are
  generated automatically by `doc-build` and `doc-clean` will remove them.

## Sphinx Extensions

The extensions we use to generate API docs are outlined and linked in the
section above. The full list of built-in Sphinx tensions can be found
[here](https://www.sphinx-doc.org/en/master/usage/extensions/index.html). The
list of extensions that are enabled for building CUDA-QX documentation is
defined by the value of the `extensions` configuration in
[conf.py](./sphinx/conf.py).

## References

Additional links that may be helpful that are not listed above:

- [References and automatic link generation in
  Doxygen](https://www.star.bnl.gov/public/comp/sofi/doxygen/autolink.html)
- [Using Napoleon style for Python doc
  comments](https://docs.softwareheritage.org/devel/contributing/sphinx.html)
- [Cross-referencing Python
  objects](https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#cross-referencing-python-objects)
- [Cross-referencing C++
  objects](https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#cross-referencing)
- [Sphinx configuration
  options](https://www.sphinx-doc.org/en/master/usage/configuration.html)
- [Syntax highlighting in inline
  code](https://sphinxawesome.xyz/demo/inline-code/#syntax-highlighting-in-inline-code)
- [Test examples in Python
  documentation](https://docs.python.org/3/library/doctest.html)
