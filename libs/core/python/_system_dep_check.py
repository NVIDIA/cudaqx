# ============================================================================ #
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Known system libraries not bundled with wheels, mapped to install instructions.
_KNOWN_SYSTEM_DEPS = {
    "libgfortran": {
        "package":
            "libgfortran5",
        "instructions":
            ("  Debian/Ubuntu:  apt-get install -y libgfortran5\n"
             "  RHEL/Fedora:    dnf install libgfortran\n"
             "  Conda:          conda install -c conda-forge libgfortran-ng"),
    },
    "libquadmath": {
        "package":
            "libquadmath0",
        "instructions":
            ("  Debian/Ubuntu:  apt-get install -y libquadmath0\n"
             "  RHEL/Fedora:    dnf install libquadmath\n"
             "  Conda:          conda install -c conda-forge libgcc-ng"),
    },
}


def raise_if_missing_system_dep(error: ImportError, package_name: str):
    """Re-raise *error* with an actionable message when a known system
    library is missing.  If the error does not match any known library,
    this function does nothing (caller should ``raise`` the original)."""
    msg = str(error)
    for lib_key, info in _KNOWN_SYSTEM_DEPS.items():
        if lib_key in msg:
            raise ImportError(
                f"{error}\n\n"
                f"{package_name} requires {info['package']}, which is not "
                f"bundled with the Python wheel. "
                f"Install it with your system package manager:\n"
                f"{info['instructions']}\n") from error
