/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "common/ObserveResult.h"
#include "cudaq/qis/pauli_word.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace nanobind {
namespace detail {

template <>
struct type_caster<cudaq::pauli_word> {
  NB_TYPE_CASTER(cudaq::pauli_word, const_name("pauli_word"))

  bool from_python(handle src, uint8_t, cleanup_list *) noexcept {
    if (!src)
      return false;
    try {
      if (nb::hasattr(src, "str")) {
        auto str_val = nb::cast<std::string>(src.attr("str")());
        value = cudaq::pauli_word(str_val);
        return true;
      } else if (nb::isinstance<nb::str>(src)) {
        value = cudaq::pauli_word(nb::cast<std::string>(src));
        return true;
      }
    } catch (...) {
    }
    return false;
  }

  static handle from_cpp(const cudaq::pauli_word &v, rv_policy,
                         cleanup_list *) noexcept {
    try {
      nb::object pauli_word_class =
          nb::module_::import_("cudaq").attr("pauli_word");
      nb::object pauli_word_obj = pauli_word_class(v.str());
      return pauli_word_obj.release();
    } catch (...) {
      return handle();
    }
  }
};

} // namespace detail
} // namespace nanobind
