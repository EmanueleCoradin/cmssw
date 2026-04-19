#ifndef DataFormats_L1ScoutingSoA_interface_AssociationMapSoA_h
#define DataFormats_L1ScoutingSoA_interface_AssociationMapSoA_h

#include "DataFormats/SoATemplate/interface/SoABlocks.h"

namespace l1sc {

    GENERATE_SOA_LAYOUT(AssociationMapIndexLayout, SOA_COLUMN(uint32_t, index))
    GENERATE_SOA_LAYOUT(AssociationMapOffsetLayout, SOA_COLUMN(uint32_t, offset))

    GENERATE_SOA_BLOCKS(AssociationMapLayout, 
        SOA_BLOCK(index, AssociationMapIndexLayout), 
        SOA_BLOCK(offset, AssociationMapOffsetLayout)
    )

    using AssociationMapSoA = AssociationMapLayout<>;
    
}

#endif // DataFormats_L1ScoutingSoA_interface_AssociationMapSoA_h