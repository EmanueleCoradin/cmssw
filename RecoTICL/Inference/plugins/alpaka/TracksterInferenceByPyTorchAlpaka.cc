#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include "RecoTICL/Inference/interface/TracksterInferenceAlgoFactory.h"
#include "RecoTICL/Inference/interface/alpaka/InferenceBridgeUtilities.h"
#include "RecoTICL/Inference/interface/alpaka/LegacyTracksterInferenceInput.h"
#include "RecoTICL/Inference/interface/alpaka/TracksterFeatureExtractionRunner.h"
#include "RecoTICL/Inference/interface/alpaka/TracksterInferenceRunner.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  template <typename Extractor, typename PIDAdapter, typename RegressionAdapter>
  class LegacyTracksterInference : public ::ticl::TracksterInferenceAlgoBase {
  public:
    LegacyTracksterInference(edm::ParameterSet const& config, ::ticl::TICLONNXGlobalCache const* cache)
        : TracksterInferenceAlgoBase(config, cache), queue_(inferenceDevice(config)), extractor_(config),
          minEnergy_(config.getParameter<double>("minClusterEnergy")) {
      auto batchSize = inferenceBatchSize(config);
      if (!std::isfinite(minEnergy_))
        throw cms::Exception("Configuration") << "minClusterEnergy must be finite";
      if constexpr (!PIDAdapter::channelsLast) {
        if (config.getParameter<bool>("doRegression"))
          throw cms::Exception("Configuration") << "CNN supports PID only";
      }
      auto fp16 = config.getParameter<bool>("convertToFP16");
      auto warmup = config.getParameter<int>("warmupIterations");
      if (config.getParameter<bool>("doPID"))
        pid_ = std::make_unique<TracksterInferenceRunner<PIDAdapter>>(
            inferenceModelPath(config, "pidModel"), batchSize, fp16, warmup);
      if (config.getParameter<bool>("doRegression"))
        regression_ = std::make_unique<TracksterInferenceRunner<RegressionAdapter>>(
            inferenceModelPath(config, "regressionModel"), batchSize, fp16, warmup);
      if (pid_) pid_->warmup(queue_);
      if (regression_) regression_->warmup(queue_);
      alpaka::wait(queue_);
    }

    void runInference(std::vector<::reco::CaloCluster> const& clusters,
                      std::vector<::ticl::Trackster>& tracksters,
                      ticlgeom::Tools const& geometry) const override {
      if (tracksters.empty() || (!pid_ && !regression_)) return;
      LegacyTracksterInferenceInput input(queue_, clusters, tracksters, [&](auto const& cluster) {
        // Unused/empty clusters have no image slot. Barrel clusters are excluded
        // by the same seed-ID policy used by the device feature extractor.
        return cluster.hitsAndFractions().empty() ? -1 :
            static_cast<int>(geometry.getLayerWithOffset(cluster.hitsAndFractions()[0].first)) - 1;
      });
      LayerClusterEnergyMultiView energies;
      LayerClusterPositionMultiView positions;
      LayerClusterIndexMultiView indexes;
      auto const view = input.clusters().const_view();
      auto n = detail::checkedSoASize(clusters.size(), "layer clusters");
      energies.addView(view.energy(), n);
      positions.addView(view.position(), n);
      indexes.addView(view.indexes(), n);
      extractor_.prepare(queue_, input.tracksters().const_view(), energies, indexes);
      // CPU interpretation requires the result before publishing either product.
      alpaka::wait(queue_);
      // Host geometry layers are already endcap-local; do not fold invalid or
      // beyond-image layers into a valid image layer with a detector-size modulo.
      auto features = extractor_.run(queue_, input.tracksters().const_view(), energies, positions,
                                      std::numeric_limits<uint32_t>::max(), indexes);
      auto total = extractor_.selectedCount();
      if constexpr (PIDAdapter::channelsLast)
        resetTracksterPredictions(queue_,
            std::get<0>(features.const_view().records().selectedTracksterIndex()).data(),
            input.mutableTracksters(), total);
      if (regression_) regression_->runInto(queue_, features, input.mutableTracksters(), minEnergy_);
      if (pid_) pid_->runInto(queue_, features, input.mutableTracksters());
      input.apply(tracksters);
    }

  private:
    mutable Queue queue_;
    mutable TracksterFeatureExtractionRunner<Extractor> extractor_;
    float const minEnergy_;
    std::unique_ptr<TracksterInferenceRunner<PIDAdapter>> pid_;
    std::unique_ptr<TracksterInferenceRunner<RegressionAdapter>> regression_;
  };

  class TracksterInferenceByPyTorchAlpaka : public ::ticl::TracksterInferenceAlgoBase {
  public:
    TracksterInferenceByPyTorchAlpaka(edm::ParameterSet const& config, ::ticl::TICLONNXGlobalCache const* cache)
        : TracksterInferenceAlgoBase(config, cache) {
      auto family = config.getParameter<std::string>("modelFamily");
      if (family == "CNN")
        implementation_ = std::make_unique<LegacyTracksterInference<CNNFeatureExtractor, CNNPIDAdapter, DNNEnergyAdapter>>(config, cache);
      else if (family == "DNN")
        implementation_ = std::make_unique<LegacyTracksterInference<DNNFeatureExtractor, DNNPIDAdapter, DNNEnergyAdapter>>(config, cache);
      else if (family == "PFN")
        implementation_ = std::make_unique<LegacyTracksterInference<PFNFeatureExtractor, PFNPIDAdapter, PFNEnergyAdapter>>(config, cache);
      else
        throw cms::Exception("Configuration") << "Unknown candidate inference modelFamily '" << family << "'";
    }
    void runInference(std::vector<::reco::CaloCluster> const& clusters, std::vector<::ticl::Trackster>& tracksters,
                      ticlgeom::Tools const& geometry) const override {
      implementation_->runInference(clusters, tracksters, geometry);
    }
  private:
    std::unique_ptr<::ticl::TracksterInferenceAlgoBase> implementation_;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#define TICL_STRINGIFY_IMPL(x) #x
#define TICL_STRINGIFY(x) TICL_STRINGIFY_IMPL(x)
DEFINE_EDM_PLUGIN(TracksterInferenceAlgoFactory, ALPAKA_ACCELERATOR_NAMESPACE::TracksterInferenceByPyTorchAlpaka,
                  TICL_STRINGIFY(ALPAKA_ACCELERATOR_NAMESPACE::TracksterInferenceByPyTorchAlpaka));
#undef TICL_STRINGIFY
#undef TICL_STRINGIFY_IMPL
