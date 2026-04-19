#ifndef DataFormats_L1ScoutingSoA_interface_BxLookupSoA_h
#define DataFormats_L1ScoutingSoA_interface_BxLookupSoA_h

#include "DataFormats/SoATemplate/interface/SoABlocks.h"

namespace l1sc {

    GENERATE_SOA_LAYOUT(BxLookupBxIndexLayout, SOA_COLUMN(uint16_t, bx))
    GENERATE_SOA_LAYOUT(BxLookupOffsetLayout, SOA_COLUMN(uint32_t, offset))

    GENERATE_SOA_BLOCKS(BxLookupLayout, 
        SOA_BLOCK(bx, BxLookupBxIndexLayout), 
        SOA_BLOCK(offset, BxLookupOffsetLayout)
    )

    using BxLookupSoA = BxLookupLayout<>;

}

# endif // DataFormats_L1ScoutingSoA_interface_BxLookupSoA_