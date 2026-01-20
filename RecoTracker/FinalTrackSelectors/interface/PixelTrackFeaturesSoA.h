#ifndef RecoTracker_FinalTrackSelectors_PixelTrackFeaturesSoA_h
#define RecoTracker_FinalTrackSelectors_PixelTrackFeaturesSoA_h

#include "DataFormats/Common/interface/FPX.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

GENERATE_SOA_LAYOUT(PixelTrackFeaturesSoALayout,
                    SOA_COLUMN(FPX, chi2),
                    SOA_COLUMN(FPX, dzError),
                    SOA_COLUMN(FPX, dxyError),
                    SOA_COLUMN(FPX, eta),
                    SOA_COLUMN(FPX, nHits),
                    SOA_COLUMN(FPX, phi),
                    SOA_COLUMN(FPX, phiError),
                    SOA_COLUMN(FPX, pt),
                    SOA_COLUMN(FPX, qOverPtError),
                    SOA_COLUMN(FPX, dzBS),
                    SOA_COLUMN(FPX, dxyBS),
                    SOA_COLUMN(FPX, nLayers),
                    SOA_COLUMN(FPX, cotThetaError),
                    SOA_COLUMN(FPX, covCotThetaDz),
                    SOA_COLUMN(FPX, covDxyQOverPt),
                    SOA_COLUMN(FPX, covPhiDxy),
                    SOA_COLUMN(FPX, covPhiQOverPt));

using PixelTrackFeaturesSoA = PixelTrackFeaturesSoALayout<>;

// Define the SoA layout for track scores (output)
GENERATE_SOA_LAYOUT(PixelTrackScoresSoALayout, SOA_COLUMN(FPX, score))

using PixelTrackScoresSoA = PixelTrackScoresSoALayout<>;

#endif
