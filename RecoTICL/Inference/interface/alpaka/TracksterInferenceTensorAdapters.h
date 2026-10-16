#ifndef RecoTICL_Inference_alpaka_TracksterInferenceTensorAdapters_h
#define RecoTICL_Inference_alpaka_TracksterInferenceTensorAdapters_h

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>
#include <string_view>

#include <alpaka/alpaka.hpp>

#include "DataFormats/HGCalReco/interface/TracksterSoA.h"
#include "DataFormats/HGCalReco/interface/alpaka/TracksterDevice.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "PhysicsTools/PyTorchAlpaka/interface/BatchedTensorCollection.h"
#include "RecoTICL/Inference/interface/TracksterInferenceSoA.h"
#include "RecoTICL/Inference/interface/alpaka/TracksterInferenceDevice.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using InferenceTensors = cms::torch::alpakatools::BatchedTensorCollection<Queue>;

  namespace detail {

    inline int32_t checkedSoASize(std::size_t size, std::string_view name) {
      if (size > static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
        throw cms::Exception("TracksterInference") << name << " has " << size << " entries (exceeds int32_t).";
      }
      return static_cast<int32_t>(size);
    }

    template <typename View>
    std::array<int32_t, ::ticl::TracksterSoA::blocksNumber> tracksterBlockSizes(View const& view) {
      return {checkedSoASize(view.tracksters().metadata().size(), "tracksters"),
              checkedSoASize(view.vertices().keys(), "vertex offsets"),
              checkedSoASize(view.vertices().content().metadata().size(), "vertices"),
              checkedSoASize(view.multiplicity().keys(), "multiplicity offsets"),
              checkedSoASize(view.multiplicity().content().metadata().size(), "multiplicities"),
              checkedSoASize(view.edges().keys(), "edge offsets"),
              checkedSoASize(view.edges().content().metadata().size(), "edges"),
              checkedSoASize(view.tracks().keys(), "track offsets"),
              checkedSoASize(view.tracks().content().metadata().size(), "tracks"),
              checkedSoASize(view.tracksterGsfTrack().keys(), "GSF offsets"),
              checkedSoASize(view.tracksterGsfTrack().content().metadata().size(), "GSF tracks"),
              checkedSoASize(view.globalSeedingTracks().keys(), "global-seeding offsets"),
              checkedSoASize(view.globalSeedingTracks().content().metadata().size(), "global-seeding tracks")};
    }
  }  // namespace detail

  // Input schema: one CNN tensor [B, 50, 10, 3].
  struct CNNInputAdapter {
    using InputDevice = ticl::TracksterInferenceDevice;
    using InputSoA = ::ticl::TracksterInferenceSoA;

    static InputDevice makeWarmupInput(Queue& queue, uint32_t batchSize) {
      InputDevice input(queue, batchSize);
      input.zeroInitialise(queue);
      return input;
    }

    template <typename Records>
    static uint32_t const* selectedIndices(Records const& records) {
      return std::get<0>(records.selectedTracksterIndex()).data();
    }

    template <typename Records>
    static void registerTensors(InferenceTensors& inputs, Records const& records, std::optional<uint32_t> batchSize) {
      if (batchSize)
        inputs.addBatched<InputSoA>("input", *batchSize, records.energy(), records.absEta(), records.phi());
      else
        inputs.add<InputSoA>("input", records.energy(), records.absEta(), records.phi());
    }
  };

  // Model output: PID tensor [B, 8].
  // Published output: a copy of the original TracksterDevice with updated PID scores.
  struct PIDOutputAdapter {
    using ModelOutputDevice = ticl::TracksterInferencePIDScoresDevice;
    using ModelOutputSoA = ::ticl::TracksterInferencePIDScoresSoA;
    using PublishedSourceDevice = ticl::TracksterDevice;
    using PublishedOutputDevice = ticl::TracksterDevice;

    static ModelOutputDevice makeModelOutput(Queue& queue, int total) {
      ModelOutputDevice output(queue, total);
      output.zeroInitialise(queue);
      return output;
    }

    static void validateSampleCount(PublishedSourceDevice const& original, int total) {
      auto nTracksters = detail::checkedSoASize(original.const_view().tracksters().metadata().size(), "tracksters");
      if (total > nTracksters) {
        throw cms::Exception("TracksterInference")
            << "Selected " << total << " tracksters, but the source has only " << nTracksters << ".";
      }
    }

    static PublishedOutputDevice makePublishedOutput(Queue& queue, PublishedSourceDevice const& original) {
      PublishedOutputDevice output(queue, detail::tracksterBlockSizes(original.const_view()));
      alpaka::memcpy(queue, output.buffer(), original.buffer());
      return output;
    }

    template <typename Records>
    static void registerTensors(InferenceTensors& outputs, Records const& records, std::optional<uint32_t> batchSize) {
      if (batchSize)
        outputs.addBatched<ModelOutputSoA>("pid_output",
                                           *batchSize,
                                           records.id_probabilities0(),
                                           records.id_probabilities1(),
                                           records.id_probabilities2(),
                                           records.id_probabilities3(),
                                           records.id_probabilities4(),
                                           records.id_probabilities5(),
                                           records.id_probabilities6(),
                                           records.id_probabilities7());
      else
        outputs.add<ModelOutputSoA>("pid_output",
                                    records.id_probabilities0(),
                                    records.id_probabilities1(),
                                    records.id_probabilities2(),
                                    records.id_probabilities3(),
                                    records.id_probabilities4(),
                                    records.id_probabilities5(),
                                    records.id_probabilities6(),
                                    records.id_probabilities7());
    }

    static void applyResults(Queue& queue,
                             uint32_t const* selectedTracksters,
                             ModelOutputDevice const& predictions,
                             PublishedOutputDevice& published,
                             uint32_t total);
  };

  // The inference producer deals only with this interface, independent of
  // which concrete SoA columns the adapters register.
  template <typename InputAdapter, typename OutputAdapter>
  struct TracksterInferenceTensorAdapter {
    using InputDevice = typename InputAdapter::InputDevice;
    using InputSoA = typename InputAdapter::InputSoA;
    using ModelOutputDevice = typename OutputAdapter::ModelOutputDevice;
    using ModelOutputSoA = typename OutputAdapter::ModelOutputSoA;
    using PublishedSourceDevice = typename OutputAdapter::PublishedSourceDevice;
    using PublishedOutputDevice = typename OutputAdapter::PublishedOutputDevice;

    static InputDevice makeWarmupInput(Queue& queue, uint32_t batchSize) {
      return InputAdapter::makeWarmupInput(queue, batchSize);
    }

    static ModelOutputDevice makeModelOutput(Queue& queue, int total) {
      return OutputAdapter::makeModelOutput(queue, total);
    }

    static void validateSampleCount(PublishedSourceDevice const& original, int total) {
      OutputAdapter::validateSampleCount(original, total);
    }

    static PublishedOutputDevice makePublishedOutput(Queue& queue, PublishedSourceDevice const& original) {
      return OutputAdapter::makePublishedOutput(queue, original);
    }

    template <typename Records>
    static void registerInputs(InferenceTensors& tensors, Records const& records, std::optional<uint32_t> batchSize) {
      InputAdapter::registerTensors(tensors, records, batchSize);
    }

    template <typename Records>
    static void registerOutputs(InferenceTensors& tensors, Records const& records, std::optional<uint32_t> batchSize) {
      OutputAdapter::registerTensors(tensors, records, batchSize);
    }

    template <typename FeatureRecords>
    static void applyResults(Queue& queue,
                             FeatureRecords const& featureRecords,
                             ModelOutputDevice const& predictions,
                             PublishedOutputDevice& published,
                             uint32_t total) {
      OutputAdapter::applyResults(queue, InputAdapter::selectedIndices(featureRecords), predictions, published, total);
    }
  };

  using CNNPIDAdapter = TracksterInferenceTensorAdapter<CNNInputAdapter, PIDOutputAdapter>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTICL_Inference_alpaka_TracksterInferenceTensorAdapters_h
