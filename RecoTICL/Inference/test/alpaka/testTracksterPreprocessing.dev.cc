#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <Eigen/Core>
#include <alpaka/alpaka.hpp>

#include "DataFormats/CaloRecHit/interface/CaloClusterHostCollection.h"
#include "DataFormats/CaloRecHit/interface/alpaka/CaloClusterDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/TracksterHost.h"
#include "DataFormats/HGCalReco/interface/alpaka/TracksterDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "RecoTICL/Inference/interface/alpaka/TracksterFeatureExtractionRunner.h"
#include "RecoTICL/Inference/interface/alpaka/TracksterInferenceTensorAdapters.h"
#include "RecoTICL/Inference/test/PreprocessingReference.h"
#include "RecoTICL/Inference/interface/alpaka/LegacyTracksterInferenceInput.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;
namespace {
  void check(float actual, float expected, std::string const& where) {
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        std::abs(actual - expected) > 2.e-5f + 2.e-5f * std::abs(expected))
      throw std::runtime_error(where + ": actual=" + std::to_string(actual) + ", expected=" + std::to_string(expected));
  }

  template <typename Extractor, typename InputAdapter>
  void run(Queue& queue, trackster_test::Family family,
           std::vector<trackster_test::Cluster> const& clusters,
           std::vector<trackster_test::Trackster> const& tracksters, float threshold, bool legacy = false) {
    int n = tracksters.size(), nv = 0, nc = clusters.size();
    for (auto const& ts : tracksters)
      nv += ts.vertices.size();
    std::array<int32_t, ::ticl::TracksterSoA::blocksNumber> sizes{n,n,nv,n,nv,n,0,n,0,n,0,n,0};
    ::ticl::TracksterHost host(queue, sizes);
    host.zeroInitialise();
    auto view = host.view();
    uint32_t offset = 0;
    for (int i = 0; i < n; ++i) {
      auto const& ts = tracksters[i];
      auto row = view.tracksters()[i];
      row.raw_energy() = ts.energy;
      row.raw_em_energy() = ts.emEnergy;
      row.barycenterX() = ts.x;
      row.barycenterY() = ts.y;
      row.barycenterZ() = ts.z;
      for (unsigned k = 0; k < ts.vertices.size(); ++k) {
        view.vertices().content()[offset].values() = ts.vertices[k];
        view.multiplicity().content()[offset++].values() = ts.multiplicity[k];
      }
      view.vertices().offsets()[i + 1].keys_offsets() = offset;
      view.multiplicity().offsets()[i + 1].keys_offsets() = offset;
    }
    ::reco::CaloClusterHostCollection clusterHost(queue, nc, nc, nc, nc);
    clusterHost.zeroInitialise();
    for (int i = 0; i < nc; ++i) {
      auto const& cl = clusters[i];
      auto pos = clusterHost.view().position()[i];
      pos.x() = cl.x; pos.y() = cl.y; pos.z() = cl.z;
      // 0-based layer with side offset, as produced by HGCalSoARecHitsProducer.
      pos.layer() = cl.layer < 0 ? cl.layer : cl.layer + (cl.z > 0 ? 60 : 0);
      pos.cells() = cl.cells;
      clusterHost.view().energy()[i].energy() = cl.energy;
      clusterHost.view().indexes()[i].seedID() = DetId(cl.barrel ? DetId::Ecal : DetId::HGCalEE, 1);
    }
    ALPAKA_ACCELERATOR_NAMESPACE::ticl::TracksterDevice device(queue, sizes);
    ALPAKA_ACCELERATOR_NAMESPACE::reco::CaloClusterDeviceCollection clusterDevice(queue, nc, nc, nc, nc);
    alpaka::memcpy(queue, device.buffer(), host.buffer());
    alpaka::memcpy(queue, clusterDevice.buffer(), clusterHost.buffer());
    // Exercise the production candidate bridge with real legacy Tracksters.
    // Geometry is replaced only by a deterministic layer lookup; feature
    // extraction and tensor registration are the production implementations.
    std::vector<::reco::CaloCluster> legacyClusters;
    std::vector<::ticl::Trackster> legacyTracksters;
    std::unique_ptr<LegacyTracksterInferenceInput> bridge;
    if (legacy) {
      for (auto const& cl : clusters) {
        DetId seed(cl.barrel ? DetId::Ecal : DetId::HGCalEE, 1);
        legacyClusters.emplace_back(cl.energy, math::XYZPoint(cl.x, cl.y, cl.z), ::reco::CaloID(),
            std::vector<std::pair<DetId, float>>(cl.cells, {seed, 1.f}), ::reco::CaloCluster::hgcal_em, seed);
      }
      for (auto const& ts : tracksters) {
        auto& out = legacyTracksters.emplace_back();
        out.vertices() = ts.vertices;
        out.vertex_multiplicity() = ts.multiplicity;
        out.setRawEnergy(ts.energy);
        out.setRawEmEnergy(ts.emEnergy);
        out.setBarycenter({ts.x, ts.y, ts.z});
        out.setTimeAndError(7.f, 0.25f);
        out.addTrackIdx(42); out.addGSFTrackIdx(17);
        out.fillPCAVariables(Eigen::Vector3f(9,1,0), Eigen::Matrix3f::Identity(),
                            Eigen::Vector3f::Ones(), Eigen::Vector3f::Ones(), 3,
                            ::ticl::Trackster::PCAOrdering::descending);
        out.setRegressedEnergy(123.f);
        for (int j = 0; j < 8; ++j)
          out.setIdProbability(static_cast<::ticl::Trackster::ParticleType>(j), 0.125f);
      }
      bridge = std::make_unique<LegacyTracksterInferenceInput>(queue, legacyClusters, legacyTracksters,
          [&](auto const& cl) { return clusters[&cl - legacyClusters.data()].layer; });
    }
    auto tracksterView = legacy ? bridge->tracksters().const_view() : device.const_view();
    auto clusterView = legacy ? bridge->clusters().const_view() : clusterDevice.const_view();
    LayerClusterEnergyMultiView energies;
    LayerClusterPositionMultiView positions;
    LayerClusterIndexMultiView indexes;
    // The production multi-view is also exercised for empty collections.
    energies.addView(clusterView.energy(), nc);
    positions.addView(clusterView.position(), nc);
    indexes.addView(clusterView.indexes(), nc);
    edm::ParameterSet config;
    config.addParameter<double>("minClusterEnergy", threshold);
    TracksterFeatureExtractionRunner<Extractor> runner(config);
    runner.prepare(queue, tracksterView, energies, indexes);
    alpaka::wait(queue);  // test stands in for SynchronizingEDProducer
    auto expected = trackster_test::reference(family, clusters, tracksters, threshold);
    if (runner.selectedCount() != expected.size())
      throw std::runtime_error("Selection count differs from CPU reference");
    auto features = runner.run(queue, tracksterView, energies, positions, 60, indexes);
    using Host = PortableHostCollection<typename InputAdapter::InputSoA>;
    Host result(queue, expected.size());
    alpaka::memcpy(queue, result.buffer(), features.buffer());
    alpaka::wait(queue);
    for (unsigned i = 0; i < expected.size(); ++i) {
      auto row = result.const_view()[i];
      if (row.selectedTracksterIndex() != expected[i].index)
        throw std::runtime_error("Selected-index ordering mismatch at row " + std::to_string(i));
      for (int layer = 0; layer < 50; ++layer) {
        for (int slot = 0; slot < 10; ++slot) {
          std::vector<float> actual{row.energy()(layer, slot), row.absEta()(layer, slot), row.phi()(layer, slot)};
          if constexpr (std::is_same_v<InputAdapter, PFNInputAdapter>) {
            actual.insert(actual.end(), {row.x()(layer, slot), row.y()(layer, slot), row.absZ()(layer, slot), row.cells()(layer, slot)});
          }
          for (unsigned f = 0; f < actual.size(); ++f)
            check(actual[f], expected[i].clusters[(layer * 10 + slot) * actual.size() + f],
                  "row/layer/slot/feature=" + std::to_string(i) + "/" + std::to_string(layer) + "/" +
                  std::to_string(slot) + "/" + std::to_string(f));
        }
      }
      if constexpr (std::is_same_v<InputAdapter, PFNInputAdapter>) {
        std::array<float,7> actual{row.rawEnergy(), row.rawEmEnergy(), row.barycenterX(), row.barycenterY(),
                                  row.barycenterAbsZ(), row.barycenterAbsEta(), row.barycenterPhi()};
        for (int f = 0; f < 7; ++f)
          check(actual[f], expected[i].trackster[f], "Trackster feature " + std::to_string(f));
      }
    }
    // Validate actual registered tensors, including minibatch tails and strides.
    InferenceTensors tensors;
    InputAdapter::registerTensors(tensors, features.const_view().records(), 2u);
    ALPAKA_ACCELERATOR_NAMESPACE::torch::QueueGuard guard(queue);
    for (unsigned batch = 0; batch < tensors.batchCount(); ++batch) {
      cms::torch::alpakatools::TensorCollection<Queue> input;
      tensors.materializeBatch(batch, input);
      input.copy(queue);
      auto args = cms::torch::alpakatools::detail::convertInput(input, cms::torch::alpakatools::getDevice(queue));
      auto lc = args[0].toTensor().permute({0, 2, 3, 1}).to(::torch::kCPU).contiguous();
      int nf = family == trackster_test::Family::PFN ? 7 : 3;
      if (lc.dim() != 4 || lc.size(1) != 50 || lc.size(2) != 10 || lc.size(3) != nf)
        throw std::runtime_error("Layer-cluster tensor shape mismatch");
      for (int i = 0; i < lc.size(0); ++i)
        for (int j = 0; j < 50 * 10 * nf; ++j)
          check(lc.data_ptr<float>()[i * 50 * 10 * nf + j], expected[batch * 2 + i].clusters[j], "Tensor layout");
      if constexpr (std::is_same_v<InputAdapter, PFNInputAdapter>) {
        auto tr = args[1].toTensor().to(::torch::kCPU).contiguous();
        if (tr.dim() != 2 || tr.size(0) != lc.size(0) || tr.size(1) != 7)
          throw std::runtime_error("Trackster tensor shape mismatch");
        for (int i = 0; i < tr.size(0); ++i)
          for (int j = 0; j < 7; ++j)
            check(tr.data_ptr<float>()[i * 7 + j], expected[batch * 2 + i].trackster[j], "Trackster tensor layout");
      }
    }
    if (legacy && !legacyTracksters.empty()) {
      resetTracksterPredictions(queue,
          std::get<0>(features.const_view().records().selectedTracksterIndex()).data(),
          bridge->mutableTracksters(), expected.size());
      bridge->apply(legacyTracksters);
      for (unsigned i = 0; i < legacyTracksters.size(); ++i) {
        bool selected = std::any_of(expected.begin(), expected.end(),
                                     [i](auto const& sample) { return sample.index == i; });
        auto const& ts = legacyTracksters[i];
        check(ts.regressed_energy(), selected ? 0.f : 123.f, "Bridge prediction mapping");
        for (auto value : ts.id_probabilities()) check(value, selected ? 0.f : 0.125f, "Bridge PID mapping");
        check(ts.time(), 7.f, "Preserved time");
        check(ts.eigenvalues()[0], 9.f, "Preserved PCA eigenvalues");
        if (ts.trackIdx() != 42 || ts.gsftrackIdx() != 17)
          throw std::runtime_error("Bridge changed original track associations");
        check(ts.timeError(), 0.25f, "Preserved time error");
        check(ts.barycenter().z(), tracksters[i].z, "Preserved PCA barycenter");
        if (ts.vertices() != tracksters[i].vertices || ts.vertex_multiplicity() != tracksters[i].multiplicity)
          throw std::runtime_error("Bridge changed original-object associations");
      }
    }
    alpaka::wait(queue);
  }

  void allFamilies(Queue& queue, std::vector<trackster_test::Cluster> const& clusters,
                   std::vector<trackster_test::Trackster> const& tracksters, float threshold) {
    for (bool legacy : {false, true}) {
      run<CNNFeatureExtractor, CNNInputAdapter>(queue, trackster_test::Family::CNN, clusters, tracksters, threshold, legacy);
      run<DNNFeatureExtractor, DNNInputAdapter>(queue, trackster_test::Family::DNN, clusters, tracksters, threshold, legacy);
      run<PFNFeatureExtractor, PFNInputAdapter>(queue, trackster_test::Family::PFN, clusters, tracksters, threshold, legacy);
    }
  }
}

int main() {
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    std::cerr << "No device for this backend; preprocessing tests NOT executed\n";
    return 77;
  }
  try {
    for (auto const& dev : devices) {
      Queue queue(dev);
      allFamilies(queue, {}, {}, 1.f);
      allFamilies(queue, {}, {trackster_test::Trackster{}}, 1.f);
      std::vector<trackster_test::Cluster> clusters;
      trackster_test::Trackster ts;
      for (uint32_t i = 0; i < 25; ++i) {
        clusters.push_back({float((i * 7) % 25 + 1), float(i + 1), float(i + 2), i % 2 ? 300.f : -300.f,
                            i < 15 ? 0 : (i < 24 ? 49 : -1), int(i + 3)});
        ts.vertices.push_back(i);
        ts.multiplicity.push_back(float(i % 3 + 1));
      }
      auto low = ts; low.vertices = {0}; low.multiplicity = {2};
      auto barrel = low;
      clusters.push_back({100.f, 10.f, 20.f, 30.f, 2, 5, true});
      barrel.vertices = {25};
      auto mixed = low; mixed.vertices = {25, 0}; mixed.multiplicity = {1, 2};
      auto origin = low;
      origin.vertices = {26}; origin.x = origin.y = origin.z = 0.f;
      clusters.push_back({4.f, 0.f, 0.f, 0.f, 5, 0});
      auto axis = low;
      axis.vertices = {27}; axis.x = axis.y = 0.f;
      clusters.push_back({3.f, 0.f, 0.f, -300.f, 6, 1});
      allFamilies(queue, clusters, {origin, axis}, 1.f);
      auto prefix = low;
      prefix.vertices = {28, 29}; prefix.multiplicity = {1, 1};
      clusters.push_back({2.f, 10.f, 20.f, 300.f, 50, 2}); // beyond image, inside detector
      clusters.push_back({-5.f, 10.f, 20.f, -300.f, 1, 2});
      allFamilies(queue, clusters, {prefix}, 1.f);
      allFamilies(queue, clusters, {trackster_test::Trackster{}}, 0.f);
      // More than ten clusters/layer, sparse layers, both endcaps, invalid layer,
      // barrel-only and mixed Tracksters, exact threshold, partial minibatches.
      allFamilies(queue, clusters, {ts, low, barrel, mixed, {}}, 1.f);
      allFamilies(queue, clusters, {low}, 2.f);
      // Exact equal-energy ordering is part of the CPU preprocessing contract.
      {
        for (auto& cl : clusters) cl.energy = 1.f;
        allFamilies(queue, clusters, {ts}, 1.f);
      }
    }
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  std::cout << "CNN, DNN and PFN preprocessing and tensor-layout checks passed\n";
}
