#ifndef L1TriggerScouting_Phase2_interface_L1TScPhase2Common_h
#define L1TriggerScouting_Phase2_interface_L1TScPhase2Common_h

#include <compare>
#include <cstdint>

namespace l1sc {

  // typedefs
  typedef          __int128         int128_t;
  typedef unsigned __int128         uint128_t;

  // aliases
  using            data_t         = uint64_t;
  using            BxVec          = std::vector<uint16_t>;
  using            CountVec       = std::vector<uint16_t>;
  using            TkEmPayloadVec = std::vector<uint128_t>;

  // constants
  const size_t numTkEm = 12u;
  const size_t numTkEmWords = 18u; // (kNumTkEm * 96) / 64 = 18

  enum class Environment : int { kProduction = 0, kDevelopment = 1, kTest = 2, kDebug = 3 };

  constexpr std::strong_ordering operator<=>(Environment t, Environment u) {
    return static_cast<int>(t) <=> static_cast<int>(u);
  }

}  // namespace l1sc

#endif  // L1TriggerScouting_Phase2_interface_L1TScPhase2Common_h