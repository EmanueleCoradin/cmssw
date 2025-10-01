#ifndef DataFormats_L1ScoutingSoA_interface_PFCandidateSoA_h
#define DataFormats_L1ScoutingSoA_interface_PFCandidateSoA_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace l1sc {

  GENERATE_SOA_LAYOUT(PFCandidateLayout,
                      SOA_COLUMN(float, eta),
                      SOA_COLUMN(float, phi),
                      SOA_COLUMN(float, pt),
                      SOA_COLUMN(float, z0),
                      SOA_COLUMN(int16_t, pdgid))

  using PFCandidateSoA = PFCandidateLayout<>;
  using PFCandidateSoAView = PFCandidateSoA::View;
  using PFCandidateSoAConstView = PFCandidateSoA::ConstView;

}  // namespace l1sc

#endif  // DataFormats_L1ScoutingSoA_interface_PFCandidateSoA_h