#ifndef RecoHGCal_TICL_alpaka_TracksterInferenceDevice_h
#define RecoHGCal_TICL_alpaka_TracksterInferenceDevice_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTICL/Inference/interface/TracksterInferenceHost.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl {

  using TracksterInferenceDevice = PortableCollection<::ticl::TracksterInferenceSoA>;
  using TracksterInferencePIDScoresDevice = PortableCollection<::ticl::TracksterInferencePIDScoresSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl

#endif
