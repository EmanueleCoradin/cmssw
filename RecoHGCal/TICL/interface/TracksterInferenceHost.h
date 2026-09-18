#ifndef RecoHGCal_TICL_TracksterInferenceHost_h
#define RecoHGCal_TICL_TracksterInferenceHost_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "RecoHGCal/TICL/interface/TracksterInferenceSoA.h"

namespace ticl {

  using TracksterInferenceHost = PortableHostCollection<TracksterInferenceSoA>;
  using TracksterInferencePIDScoresHost = PortableHostCollection<TracksterInferencePIDScoresSoA>;

}  // namespace ticl

#endif
