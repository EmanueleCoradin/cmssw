#ifndef RecoTICL_Inference_PairScoreInference_h
#define RecoTICL_Inference_PairScoreInference_h

#include <span>
#include <vector>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

namespace ticl {
  class PairScoreInference {
  public:
    virtual ~PairScoreInference() = default;
    // Row-major [rows, features]; exactly one score per row in the same order.
    // Synchronous boundary for CPU linking: no EDM product or Trackster mutation.
    virtual std::vector<float> evaluate(std::span<float const> input, unsigned int rows,
                                         unsigned int features) = 0;
  };
}
using PairScoreInferenceFactory = edmplugin::PluginFactory<ticl::PairScoreInference*(edm::ParameterSet const&)>;
#endif
