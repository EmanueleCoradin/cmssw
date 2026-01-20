#ifndef RecoTracker_FinalTrackSelectors_PixelTrackFeaturesSoA_h
#define RecoTracker_FinalTrackSelectors_PixelTrackFeaturesSoA_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"

GENERATE_SOA_LAYOUT(PixelTrackFeaturesSoALayout,
                    SOA_COLUMN(float, chi2), 
                    SOA_COLUMN(float, dzError),
                    SOA_COLUMN(float, dxyError),
                    SOA_COLUMN(float, eta),
                    SOA_COLUMN(float, ndof),
                    SOA_COLUMN(float, phi),
                    SOA_COLUMN(float, phiError),
                    SOA_COLUMN(float, pt),
                    SOA_COLUMN(float, ptError),
                    SOA_COLUMN(float, qoverp),
                    SOA_COLUMN(float, dzBS),
                    SOA_COLUMN(float, dxyBS)
);

using PixelTrackFeaturesSoA = PixelTrackFeaturesSoALayout<>;

// Define the SoA layout for track scores (output)
GENERATE_SOA_LAYOUT(PixelTrackScoresSoALayout, SOA_COLUMN(float, score))

using PixelTrackScoresSoA = PixelTrackScoresSoALayout<>;

#endif
