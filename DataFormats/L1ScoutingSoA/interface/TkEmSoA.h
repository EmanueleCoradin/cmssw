#ifndef DataFormats_L1ScoutingSoA_interface_TkEmSoA_h
#define DataFormats_L1ScoutingSoA_interface_TkEmSoA_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace l1sc {

  GENERATE_SOA_LAYOUT(TkEmLayout,
                      SOA_COLUMN(float, pt),
                      SOA_COLUMN(float, eta),
                      SOA_COLUMN(float, phi),
                      SOA_COLUMN(float, isolation),
                      SOA_COLUMN(uint8_t, quality));

  using TkEmSoA = TkEmLayout<>;

}  // namespace l1sc

#endif  // DataFormats_L1ScoutingSoA_interface_TkEmSoA_h