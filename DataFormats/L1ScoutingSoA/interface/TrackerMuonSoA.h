#ifndef DataFormats_L1ScoutingSoA_interface_TrackerMuonSoA_h
#define DataFormats_L1ScoutingSoA_interface_TrackerMuonSoA_h

#include <cstdint>

#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace l1sc {

  GENERATE_SOA_LAYOUT(TrackerMuonLayout,
                      SOA_COLUMN(float, pt),
                      SOA_COLUMN(float, eta),
                      SOA_COLUMN(float, phi),
                      SOA_COLUMN(float, z0),
                      SOA_COLUMN(float, d0),
                      SOA_COLUMN(float, beta),
                      SOA_COLUMN(int8_t, charge),
                      SOA_COLUMN(uint8_t, quality),
                      SOA_COLUMN(uint8_t, isolation));

  using TrackerMuonSoA = TrackerMuonLayout<>;

}  // namespace l1sc

#endif  // DataFormats_L1ScoutingSoA_interface_TrackerMuonSoA_h
