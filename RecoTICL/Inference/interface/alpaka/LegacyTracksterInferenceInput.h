#ifndef RecoTICL_Inference_alpaka_LegacyTracksterInferenceInput_h
#define RecoTICL_Inference_alpaka_LegacyTracksterInferenceInput_h

#include <array>
#include <vector>
#include "DataFormats/CaloRecHit/interface/CaloClusterHostCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "DataFormats/CaloRecHit/interface/alpaka/CaloClusterDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/HGCalReco/interface/TracksterHost.h"
#include "RecoTICL/Inference/interface/alpaka/TracksterInferenceTensorAdapters.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  // Transport only: the existing extraction kernels still define selection,
  // ordering, padding and features. The supplied Tracksters are the reconstructed
  // objects AFTER PCA, not the original inputs to interpretation/linking.
  class LegacyTracksterInferenceInput {
  public:
    template <typename LayerOfCluster>
    LegacyTracksterInferenceInput(Queue& queue, std::vector<::reco::CaloCluster> const& clusters,
                                  std::vector<::ticl::Trackster> const& tracksters, LayerOfCluster layer)
        : queue_(queue), host_(queue, blockSizes(tracksters)), device_(queue, blockSizes(tracksters)),
          clusterHost_(queue, clusterSize(clusters), clusterSize(clusters), clusterSize(clusters), 0),
          clusterDevice_(queue, clusterSize(clusters), clusterSize(clusters), clusterSize(clusters), 0) {
      host_.zeroInitialise();
      clusterHost_.zeroInitialise();
      auto view = host_.view();
      uint32_t offset = 0;
      for (unsigned int i = 0; i < tracksters.size(); ++i) {
        auto const& ts = tracksters[i];
        auto row = view.tracksters()[i];
        row.raw_energy() = ts.raw_energy();
        row.raw_em_energy() = ts.raw_em_energy();
        row.barycenterX() = ts.barycenter().x();
        row.barycenterY() = ts.barycenter().y();
        row.barycenterZ() = ts.barycenter().z();
        row.regressed_energy() = ts.regressed_energy();
        row.id_probabilities0() = ts.id_probabilities(0);
        row.id_probabilities1() = ts.id_probabilities(1);
        row.id_probabilities2() = ts.id_probabilities(2);
        row.id_probabilities3() = ts.id_probabilities(3);
        row.id_probabilities4() = ts.id_probabilities(4);
        row.id_probabilities5() = ts.id_probabilities(5);
        row.id_probabilities6() = ts.id_probabilities(6);
        row.id_probabilities7() = ts.id_probabilities(7);
        if (ts.vertices().size() != ts.vertex_multiplicity().size())
          throw cms::Exception("InvalidTrackster") << "Vertex/multiplicity sizes differ";
        for (unsigned int k = 0; k < ts.vertices().size(); ++k) {
          if (ts.vertices(k) >= clusters.size())
            throw cms::Exception("InvalidTrackster") << "Layer cluster index out of range";
          view.vertices().content()[offset].values() = ts.vertices(k);
          view.multiplicity().content()[offset++].values() = ts.vertex_multiplicity(k);
        }
        view.vertices().offsets()[i + 1].keys_offsets() = offset;
        view.multiplicity().offsets()[i + 1].keys_offsets() = offset;
      }
      for (unsigned int i = 0; i < clusters.size(); ++i) {
        auto const& cl = clusters[i];
        auto pos = clusterHost_.view().position()[i];
        pos.x() = cl.x(); pos.y() = cl.y(); pos.z() = cl.z();
        pos.layer() = layer(cl);
        pos.cells() = detail::checkedSoASize(cl.hitsAndFractions().size(), "cluster hits");
        clusterHost_.view().energy()[i].energy() = cl.energy();
        clusterHost_.view().indexes()[i].seedID() = cl.seed();
      }
      alpaka::memcpy(queue_, device_.buffer(), host_.buffer());
      alpaka::memcpy(queue_, clusterDevice_.buffer(), clusterHost_.buffer());
    }

    ~LegacyTracksterInferenceInput() { alpaka::wait(queue_); }
    auto const& tracksters() const { return device_; }
    auto& mutableTracksters() { return device_; }
    auto const& clusters() const { return clusterDevice_; }

    void apply(std::vector<::ticl::Trackster>& tracksters) {
      if (tracksters.size() != static_cast<std::size_t>(host_.const_view().tracksters().metadata().size()))
        throw cms::Exception("InvalidTrackster") << "Prediction destination size changed during inference";
      if (tracksters.empty()) return;
      // Transfer only predictions, not vertices, multiplicities or unused PCA
      // columns. Equal-sized portable layouts have identical column padding.
      auto hostRecords = host_.view().tracksters().records();
      auto deviceRecords = device_.const_view().tracksters().records();
      auto* firstPID = std::get<0>(hostRecords.id_probabilities0()).data();
      auto pidSize = std::get<0>(hostRecords.id_probabilities7()).data() - firstPID + tracksters.size();
      auto hostPID = cms::alpakatools::make_host_view(firstPID, pidSize);
      auto devicePID = cms::alpakatools::make_device_view(
          queue_, std::get<0>(deviceRecords.id_probabilities0()).data(), pidSize);
      alpaka::memcpy(queue_, hostPID, devicePID);
      auto hostEnergy = cms::alpakatools::make_host_view(
          std::get<0>(hostRecords.regressed_energy()).data(), tracksters.size());
      auto deviceEnergy = cms::alpakatools::make_device_view(
          queue_, std::get<0>(deviceRecords.regressed_energy()).data(), tracksters.size());
      alpaka::memcpy(queue_, hostEnergy, deviceEnergy);
      alpaka::wait(queue_);
      for (unsigned int i = 0; i < tracksters.size(); ++i) {
        auto row = host_.const_view().tracksters()[i];
        float probabilities[]{row.id_probabilities0(), row.id_probabilities1(), row.id_probabilities2(),
                              row.id_probabilities3(), row.id_probabilities4(), row.id_probabilities5(),
                              row.id_probabilities6(), row.id_probabilities7()};
        // Do not reconstruct Tracksters from this deliberately partial SoA:
        // all PCA, timing, edges and original-object associations stay intact.
        tracksters[i].setProbabilities(probabilities);
        tracksters[i].setRegressedEnergy(row.regressed_energy());
      }
    }

  private:
    static int32_t clusterSize(std::vector<::reco::CaloCluster> const& clusters) {
      return detail::checkedSoASize(clusters.size(), "layer clusters");
    }
    static std::array<int32_t, ::ticl::TracksterSoA::blocksNumber> blockSizes(std::vector<::ticl::Trackster> const& tracksters) {
      auto n = detail::checkedSoASize(tracksters.size(), "tracksters");
      std::size_t vertices = 0;
      for (auto const& ts : tracksters) vertices += ts.vertices().size();
      auto nv = detail::checkedSoASize(vertices, "vertices");
      return std::array<int32_t, ::ticl::TracksterSoA::blocksNumber>{n,n,nv,n,nv,n,0,n,0,n,0,n,0};
    }
    Queue& queue_;
    ::ticl::TracksterHost host_;
    ticl::TracksterDevice device_;
    ::reco::CaloClusterHostCollection clusterHost_;
    reco::CaloClusterDeviceCollection clusterDevice_;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
#endif
