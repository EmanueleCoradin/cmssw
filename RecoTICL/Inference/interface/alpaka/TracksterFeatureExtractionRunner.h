#ifndef RecoTICL_Inference_alpaka_TracksterFeatureExtractionRunner_h
#define RecoTICL_Inference_alpaka_TracksterFeatureExtractionRunner_h

#include <cstdint>
#include <optional>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoTICL/Inference/interface/alpaka/TracksterFeatureExtractors.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Owns the per-framework-stream GPU selection buffers and selected count.
  // There is no EDM Event/ES dependency here; the producer supplies the
  // collections and geometry parameters explicitly.
  //
  // Lifecycle: prepare() in SynchronizingEDProducer::acquire(), wait for
  // acquire() to complete via the framework, then run() in produce().
  template <typename Extractor>
  class TracksterFeatureExtractionRunner {
  public:
    using FeaturesDevice = typename Extractor::FeaturesDevice;

    explicit TracksterFeatureExtractionRunner(edm::ParameterSet const& config);

    // Enqueues selection, inclusive prefix sum and the final device-to-host
    // count transfer. clusterEnergies may be empty for an empty collection.
    void prepare(Queue& queue, ::ticl::TracksterSoA::ConstView tracksters, LayerClusterEnergyMultiView clusterEnergies);

    // Only call after the framework has synchronized the prepare() work.
    [[nodiscard]] uint32_t selectedCount() const;

    // Enqueues feature construction and releases the temporary selection
    // buffers, as in the original feature producer.
    [[nodiscard]] FeaturesDevice run(Queue& queue,
                                     ::ticl::TracksterSoA::ConstView tracksters,
                                     LayerClusterEnergyMultiView clusterEnergies,
                                     LayerClusterPositionMultiView clusterPositions,
                                     uint32_t layersPerEndcap);

  private:
    Extractor extractor_;
    std::optional<cms::alpakatools::device_buffer<Device, uint32_t[]>> selectionMaskDevice_;
    std::optional<cms::alpakatools::device_buffer<Device, uint32_t[]>> selectionOffsetsDevice_;
    std::optional<cms::alpakatools::host_buffer<uint32_t>> selectedCountHost_;
  };

  // The implementation and its explicit instantiation live in src/alpaka.
  extern template class TracksterFeatureExtractionRunner<CNNFeatureExtractor>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTICL_Inference_alpaka_TracksterFeatureExtractionRunner_h
