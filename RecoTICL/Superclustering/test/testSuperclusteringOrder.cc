#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <Eigen/Core>
#include "FWCore/Framework/interface/EDConsumerBase.h"
// Compile the production algorithm into this standalone test. The scorer is
// injected; no plugin discovery, Event, geometry or production model is needed.
#include "RecoTICL/Superclustering/plugins/TracksterLinkingbySuperClusteringDNN.cc"

namespace {
  class Consumer : public edm::EDConsumerBase {
  public:
    using edm::EDConsumerBase::consumesCollector;
  };
  class RecordingScorer : public ticl::PairScoreInference {
  public:
    std::vector<std::pair<int, int>> pairs;
    std::vector<float> evaluate(std::span<float const> input, unsigned int rows, unsigned int features) override {
      if (features != 18 || input.size() != rows * features)
        throw std::runtime_error("Wrong feature dimensions");
      std::vector<float> result;
      for (unsigned int i = 0; i < rows; ++i) {
        int seed = std::lround(input[i * features + 7]);
        int cand = std::lround(input[i * features + 2]);
        pairs.emplace_back(seed, cand);
        result.push_back(cand == 80 ? 0.4f : cand == 40 ? 0.8f : seed == 40 ? 0.99f : seed == 80 ? 0.7f : 0.6f);
      }
      return result;
    }
  };
  ticl::Trackster trackster(
      float energy, float pt, unsigned int vertex, float eta = 2.f, float phi = 0.2f, float ratio = 0.96f) {
    ticl::Trackster ts;
    ts.setRawEnergy(energy);
    ts.setRawPt(pt);
    ts.setBarycenter({100.f * std::cos(phi), 100.f * std::sin(phi), 100.f * std::sinh(eta)});
    ts.vertices() = {vertex};
    ts.vertex_multiplicity() = {1.f};
    ts.setIdProbability(ticl::Trackster::ParticleType::photon, 1.f);
    ts.fillPCAVariables(Eigen::Vector3f(100 * ratio, 100 * (1 - ratio), 0),
                        Eigen::Matrix3f::Identity(),
                        Eigen::Vector3f::Zero(),
                        Eigen::Vector3f::Zero(),
                        3,
                        ticl::Trackster::PCAOrdering::descending);
    return ts;
  }
  edm::ParameterSet config(unsigned int batch) {
    edm::ParameterSet ps;
    ps.addParameter<std::string>("onnxModelPath", "");
    ps.addParameter<float>("nnWorkingPoint", 0.5f);
    ps.addParameter<unsigned int>("inferenceBatchSize", 18 * batch);
    edm::ParameterSetDescription desc;
    ticl::TracksterLinkingbySuperClusteringDNN::fillPSetDescription(desc);
    desc.validate(ps);
    return ps;
  }
  struct Result {
    std::vector<std::pair<int, int>> pairs;
    std::vector<std::vector<unsigned int>> groups;
    std::vector<ticl::Trackster> tracksters;
  };
  Result run(std::vector<ticl::Trackster> const& input, unsigned int batch, bool split = false) {
    Consumer consumer;
    auto recorder = std::make_unique<RecordingScorer>();
    auto* recorded = recorder.get();
    ticl::TracksterLinkingbySuperClusteringDNN algo(
        config(batch), consumer.consumesCollector(), nullptr, std::move(recorder));
    edm::MultiSpan<ticl::Trackster> span;
    if (split && input.size() > 1) {
      span.add(std::span(input).first(1));
      span.add(std::span(input).subspan(1));
    } else
      span.add(input);
    Result result;
    std::vector<std::vector<unsigned int>> associations;
    algo.linkTracksters(span, result.tracksters, result.groups, associations);
    if (associations != result.groups)
      throw std::runtime_error("Linked input-index mapping differs");
    result.pairs = recorded->pairs;
    return result;
  }
  void selection(ticl::Trackster seed, ticl::Trackster cand, bool expected) {
    auto result = run({seed, cand}, 1);
    if (result.pairs.size() != unsigned(expected))
      throw std::runtime_error("Pair-selection mismatch");
  }
}  // namespace
int main() {
  try {
    std::vector<ticl::Trackster> input{
        trackster(40, 8, 0), trackster(100, 10, 1), trackster(80, 10, 2), trackster(20, 5, 3)};
    std::vector<std::pair<int, int>> expectedPairs{{100, 80}, {100, 40}, {80, 40}, {100, 20}, {80, 20}, {40, 20}};
    std::vector<std::vector<unsigned int>> expectedGroups{{1, 0}, {2, 3}};
    for (auto batch : {1u, 2u, 4u, 64u}) {
      for (bool split : {false, true}) {
        auto result = run(input, batch, split);
        if (result.pairs != expectedPairs || result.groups != expectedGroups)
          throw std::runtime_error("Stable pT ordering, ties, masking or cross-batch candidate transition changed");
        if (result.tracksters.size() != 2 || result.tracksters[0].raw_energy() != 140.f ||
            result.tracksters[1].raw_energy() != 100.f)
          throw std::runtime_error("Merged energies differ");
      }
      if (!run({}, batch).pairs.empty() || !run({}, batch).groups.empty())
        throw std::runtime_error("Empty input produced pairs/groups");
      auto singleton = run({trackster(100, 10, 0)}, batch);
      if (!singleton.pairs.empty() || singleton.groups != std::vector<std::vector<unsigned int>>{{0}})
        throw std::runtime_error("Singleton handling changed");
    }
    auto seed = trackster(100, 10, 0), cand = trackster(40, 8, 1);
    selection(seed, cand, true);
    auto lowPID = seed;
    lowPID.zeroProbabilities();
    selection(lowPID, cand, false);
    lowPID = cand;
    lowPID.zeroProbabilities();
    selection(seed, lowPID, true);  // PID cut applies only to seed
    selection(seed, trackster(1.9f, 8, 1), false);
    selection(trackster(100, 3.9f, 0), trackster(40, 2, 1), false);
    selection(seed, trackster(40, 8, 1, 2.f, 0.2f, 0.91f), false);
    selection(seed, trackster(50, 8, 1, 2.f, 0.2f, 0.93f), true);
    selection(seed, trackster(50.1f, 8, 1, 2.f, 0.2f, 0.93f), false);
    selection(seed, trackster(40, 8, 1, 2.11f), false);
    selection(seed, trackster(40, 8, 1, 2.09f), true);
    selection(seed, trackster(40, 8, 1, 2.f, 0.8f), false);
    selection(seed, trackster(40, 8, 1, -2.f), false);
    selection(trackster(100, 10, 0, 2.f, 3.13f), trackster(40, 8, 1, 2.f, -3.13f), true);
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  std::cout << "Superclustering selection, ordering, merging and minibatch transitions passed\n";
}
