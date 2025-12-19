#pragma once
#include <string_view>

namespace cudaq::solvers::adapt {

enum class AdaptVariant : int {
  VQE  = 0,
  QAOA = 1,
};

inline constexpr std::string_view to_string(AdaptVariant v) {
  switch (v) {
    case AdaptVariant::VQE:  return "vqe";
    case AdaptVariant::QAOA: return "qaoa";
    default:                 return "unknown";
  }
}

// ASCII helpers (locale-independent)
constexpr unsigned char ascii_lower(unsigned char c) {
  return (c >= 'A' && c <= 'Z') ? static_cast<unsigned char>(c + ('a' - 'A')) : c;
}
constexpr bool ascii_isspace(unsigned char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

inline constexpr std::string_view trim_ascii(std::string_view s) {
  size_t b = 0, e = s.size();
  while (b < e && ascii_isspace(static_cast<unsigned char>(s[b]))) ++b;
  while (e > b && ascii_isspace(static_cast<unsigned char>(s[e - 1]))) --e;
  return s.substr(b, e - b);
}

// ASCII-only case-insensitive equals
inline constexpr bool iequals_ascii(std::string_view a, std::string_view b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (ascii_lower(static_cast<unsigned char>(a[i])) !=
        ascii_lower(static_cast<unsigned char>(b[i])))
      return false;
  }
  return true;
}

inline bool parse_variant(std::string_view s, AdaptVariant &out) {
  s = trim_ascii(s);
  if (iequals_ascii(s, "vqe"))  { out = AdaptVariant::VQE;  return true; }
  if (iequals_ascii(s, "qaoa")) { out = AdaptVariant::QAOA; return true; }
  return false;
}

} // namespace cudaq::solvers::adapt
