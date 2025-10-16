#ifndef DataFormats_L1ScoutingSoA_interface_PFCandidateSoA_h
#define DataFormats_L1ScoutingSoA_interface_PFCandidateSoA_h

#include "DataFormats/L1ScoutingSoA/interface/PuppiSoA.h"

namespace l1sc {

  // alias to simplify the logic and intuition what is the base for analysis
  using PFCandidateSoA = PuppiLayout<>;

}  // namespace l1sc

#endif  // DataFormats_L1ScoutingSoA_interface_PFCandidateSoA_h