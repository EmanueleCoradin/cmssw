#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <vector>
#include <Eigen/Core>
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "RecoTICL/Inference/interface/TracksterInferenceHost.h"
#include "RecoTICL/Inference/interface/alpaka/TracksterInferenceDevice.h"
#include "RecoTICL/Inference/interface/alpaka/PairScoreTensorAdapters.h"
#include "RecoTICL/Superclustering/interface/SuperclusteringDNNInputs.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;
namespace {
  void check(float actual, float expected) {
    if (std::isnan(expected)
            ? !std::isnan(actual)
            : (!std::isfinite(actual) || std::abs(actual - expected) > 2.e-5f * (1.f + std::abs(expected))))
      throw std::runtime_error("Pair feature/tensor mismatch: " + std::to_string(actual) + " vs " +
                               std::to_string(expected));
  }
  ::ticl::Trackster makeTrackster(float energy, bool candidate) {
    ::ticl::Trackster ts;
    ts.setRawEnergy(energy);
    ts.setBarycenter(candidate ? ::ticl::Trackster::Vector(0, 10, 100) : ::ticl::Trackster::Vector(10, 0, 100));
    ts.calculateRawPt();
    Eigen::Matrix3f vectors = Eigen::Matrix3f::Identity();
    if (candidate)
      vectors.col(0) = Eigen::Vector3f(0, 1, 0);
    ts.fillPCAVariables(Eigen::Vector3f(9, 1, 0),
                        vectors,
                        Eigen::Vector3f::Zero(),
                        Eigen::Vector3f::Zero(),
                        3,
                        ::ticl::Trackster::PCAOrdering::descending);
    ts.setTimeAndError(candidate ? 5.f : 2.f, 0.1f);
    return ts;
  }
  std::vector<float> golden() {
    float halfpi = std::numbers::pi_v<float> / 2.f;
    return {0.f,
            halfpi,
            40.f,
            std::asinh(10.f),
            40.f / std::sqrt(101.f),
            std::asinh(10.f),
            0.f,
            100.f,
            100.f / std::sqrt(101.f),
            halfpi,
            0.f,
            0.f,
            halfpi,
            halfpi,
            halfpi,
            9.f,
            0.9f,
            3.f};
  }
  void tensors(Queue& queue, std::vector<float> const& featureRow, unsigned int rows, unsigned int batchSize) {
    unsigned int nf = featureRow.size();
    ::ticl::TracksterInferenceEnergyHost host(queue, rows * nf), scoreHost(queue, rows);
    ALPAKA_ACCELERATOR_NAMESPACE::ticl::TracksterInferenceEnergyDevice device(queue, rows * nf),
        scoreDevice(queue, rows);
    for (unsigned int i = 0; i < rows * nf; ++i)
      host.view()[i].energy() = featureRow[i % nf] + (i / nf);
    alpaka::memcpy(queue, device.buffer(), host.buffer());
    InferenceTensors inputs, outputs;
    PairScoreTensorAdapter::registerInputs(inputs, device.view().records(), batchSize, nf);
    PairScoreTensorAdapter::registerOutputs(outputs, scoreDevice.view().records(), batchSize);
    if (inputs.batchCount() != (rows + batchSize - 1) / batchSize || inputs.batchCount() != outputs.batchCount())
      throw std::runtime_error("Pair minibatch count mismatch");
    ALPAKA_ACCELERATOR_NAMESPACE::torch::QueueGuard guard(queue);
    for (unsigned int batch = 0; batch < inputs.batchCount(); ++batch) {
      cms::torch::alpakatools::TensorCollection<Queue> input, output;
      inputs.materializeBatch(batch, input);
      outputs.materializeBatch(batch, output);
      input.copy(queue);
      auto args = cms::torch::alpakatools::detail::convertInput(input, cms::torch::alpakatools::getDevice(queue));
      adaptInferenceInputs(args, InputTransform::FlatRows, nf);
      auto flat = args.at(0).toTensor();
      auto actual = flat.to(::torch::kCPU).contiguous();
      auto count = std::min(batchSize, rows - batch * batchSize);
      if (actual.dim() != 2 || actual.size(0) != count || actual.size(1) != nf)
        throw std::runtime_error("Pair tensor shape mismatch");
      for (unsigned int i = 0; i < count * nf; ++i)
        check(actual.data_ptr<float>()[i], featureRow[i % nf] + batch * batchSize + (i / nf));
      auto destination =
          cms::torch::alpakatools::detail::convertOutput(output, cms::torch::alpakatools::getDevice(queue));
      if (destination.dim() != 2 || destination.size(0) != count || destination.size(1) != 1)
        throw std::runtime_error("Score tensor shape mismatch");
      // Synthetic result tests output indexing only; no model/equivalence claim.
      destination.copy_(flat.select(1, 2).unsqueeze(1));
    }
    alpaka::memcpy(queue, scoreHost.buffer(), scoreDevice.buffer());
    alpaka::wait(queue);
    for (unsigned int i = 0; i < rows; ++i)
      check(scoreHost.const_view()[i].energy(), featureRow[2] + i);
  }
}  // namespace
int main() {
  try {
    auto seed = makeTrackster(100.f, false), cand = makeTrackster(40.f, true);
    auto expected = golden();
    auto const& devices = cms::alpakatools::devices<Platform>();
    if (devices.empty())
      return 77;
    for (auto const& dev : devices) {
      Queue queue(dev);
      for (auto version : {"v1", "v2", "v3"}) {
        auto features = ::ticl::makeSuperclusteringDNNInputFromString(version);
        std::vector<float> row(features->featureCount());
        features->computeInto(seed, cand, row);
        for (unsigned int i = 0; i < row.size(); ++i)
          check(row[i], expected[i]);
        for (auto rows : {0u, 1u, 3u, 5u})
          for (auto batch : {1u, 2u, 4u})
            tensors(queue, row, rows, batch);
      }
      ::ticl::SuperclusteringDNNInputV3 v3;
      std::array<float, 18> row;
      for (float dt :
           {-51.f, -50.f, 50.f, 51.f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
        seed.setTimeAndError(0.f, 1.f);
        cand.setTimeAndError(dt, 1.f);
        v3.computeInto(seed, cand, row);
        check(row[17], (dt < -50 || dt > 50) ? -5.f : dt);
      }
      seed.setBarycenter({-10.f, 0.001f, 100.f});
      cand.setBarycenter({-10.f, -0.001f, 100.f});
      v3.computeInto(seed, cand, row);
      check(row[1], -2.f * std::numbers::pi_v<float> + 0.0002f);  // deliberately NOT wrapped
      cand.fillPCAVariables(Eigen::Vector3f::Zero(),
                            Eigen::Matrix3f::Identity(),
                            Eigen::Vector3f::Zero(),
                            Eigen::Vector3f::Zero(),
                            3,
                            ::ticl::Trackster::PCAOrdering::descending);
      v3.computeInto(seed, cand, row);
      check(row[16], 0.f);  // zero eigenvalue denominator
      // Zero projected vectors have angle pi/2, not NaN, in the CPU convention.
      check(row[13], std::numbers::pi_v<float> / 2.f);
      // Restore the analytic fixture before testing the next device.
      seed = makeTrackster(100.f, false);
      cand = makeTrackster(40.f, true);
    }
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  std::cout << "Superclustering v1/v2/v3 features, tensor rows and partial minibatches passed\n";
}
