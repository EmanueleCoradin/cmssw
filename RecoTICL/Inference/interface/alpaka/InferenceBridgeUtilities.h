#ifndef RecoTICL_Inference_alpaka_InferenceBridgeUtilities_h
#define RecoTICL_Inference_alpaka_InferenceBridgeUtilities_h

#include <cstdint>
#include <limits>
#include <string>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/FileInPath.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  // These synchronous bridges are owned by CPU stream modules, not global caches.

  inline unsigned int inferenceBatchSize(edm::ParameterSet const& config) {
    auto size = config.getParameter<unsigned int>("batchSize");
    if (size == 0 || size > static_cast<unsigned int>(std::numeric_limits<int32_t>::max()) ||
        config.getParameter<int>("warmupIterations") < 0)
      throw cms::Exception("Configuration") << "Require 0 < batchSize <= INT32_MAX and warmupIterations >= 0";
    return size;
  }
  inline std::string inferenceModelPath(edm::ParameterSet const& config, std::string const& parameter) {
    auto path = config.getParameter<std::string>(parameter);
    if (path.empty())
      throw cms::Exception("Configuration") << "Enabled TICL inference task requires " << parameter;
    try {
      return edm::FileInPath(path).fullPath();
    } catch (std::exception const& error) {
      throw cms::Exception("Configuration") << "Supply the production TorchScript model " << parameter << "='" << path
                                            << "', or disable this task. " << error.what();
    }
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
#endif
