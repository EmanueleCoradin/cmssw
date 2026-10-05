#ifndef L1TriggerScouting_Phase2_plugins_alpaka_L1TScPhase2BitsEncoding_h
#define L1TriggerScouting_Phase2_plugins_alpaka_L1TScPhase2BitsEncoding_h

#include "L1TriggerScouting/Phase2/interface/L1TScPhase2Common.h"

#include <cstdint>
#include <alpaka/alpaka.hpp>

namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels {

  // decode unsigned bit window from 64b word
  template <typename T, unsigned int start, unsigned int width>
  ALPAKA_FN_ACC T decodeBits(uint64_t word) {
    static_assert(std::is_integral<T>::value, "extract_unsigned_bits expects integral types");
    constexpr uint64_t mask = (width < 64) ? ((1ULL << width) - 1) : ~0ULL;
    return static_cast<T>((word >> start) & mask);
  }

  // decode unsigned bit window from 128b word
  template <typename T, unsigned int start, unsigned int width>
  ALPAKA_FN_ACC T decodeBits(uint128_t word) {
    static_assert(std::is_integral<T>::value, "extract_unsigned_bits expects integral types");
    constexpr uint128_t mask = (width < 128) ? ((1ULL << width) - 1) : ~0ULL;
    return static_cast<T>((word >> start) & mask);
  }

  // decode signed bit window from 64b word
  template <typename T, unsigned int start, unsigned int width>
  ALPAKA_FN_ACC T decodeBitsSigned(uint64_t word) {
    static_assert(std::is_integral<T>::value && std::is_signed<T>::value,
                  "extract_signed_bits expects signed integral types");
    auto sdata = static_cast<int64_t>(word << (64 - width - start));
    return static_cast<T>(sdata >> (64 - width));  // arithmetic right shift
  }

  // decode signed bit window from 128b word
  template <typename T, unsigned int start, unsigned int width>
  ALPAKA_FN_ACC T decodeBitsSigned(uint128_t word) {
    static_assert(std::is_integral<T>::value && std::is_signed<T>::value,
                  "extract_signed_bits expects signed integral types");
    auto sdata = static_cast<int128_t>(word << (128 - width - start));
    return static_cast<T>(sdata >> (128 - width));  // arithmetic right shift
  }

  // test bit inside 64b word
  template <unsigned int bit>
  ALPAKA_FN_ACC bool testBit(uint64_t word) {
    static_assert(bit < 64, "bit position must be smaller than 64");
    return ((word >> bit) & 1ULL) != 0;
  }

  // test bit inside 128b word
  template <unsigned int bit>
  ALPAKA_FN_ACC bool testBit(uint128_t word) {
    static_assert(bit < 128, "bit position must be smaller than 128");
    return ((word >> bit) & static_cast<uint128_t>(1)) != 0;
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::kernels

#endif  // L1TriggerScouting_Phase2_plugins_alpaka_L1TScPhase2BitsEncoding_h