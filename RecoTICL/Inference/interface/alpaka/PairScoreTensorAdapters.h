#ifndef RecoTICL_Inference_alpaka_PairScoreTensorAdapters_h
#define RecoTICL_Inference_alpaka_PairScoreTensorAdapters_h

#include "RecoTICL/Inference/interface/alpaka/BatchedInferenceRunner.h"
#include "RecoTICL/Inference/interface/TracksterInferenceSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  // A flat scalar buffer is sufficient for tabular [B,F] input and [B,1]
  // output. Reuse the existing scalar layout without an EDM type/dictionary.
  struct PairScoreTensorAdapter {
    using ScalarSoA = ::ticl::TracksterInferenceEnergySoA;
    template <typename Records>
    static void registerInputs(InferenceTensors& tensors, Records const& records,
                               unsigned int batchSize, unsigned int features) {
      tensors.addBatched<ScalarSoA>("input", batchSize * features, records.energy());
    }
    template <typename Records>
    static void registerOutputs(InferenceTensors& tensors, Records const& records, unsigned int batchSize) {
      tensors.addBatched<ScalarSoA>("score", batchSize, records.energy());
    }
  };
}
#endif
