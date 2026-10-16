#ifndef RecoTICL_Inference_test_PreprocessingReference_h
#define RecoTICL_Inference_test_PreprocessingReference_h

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

// Independent host oracle transcribed from the three CPU ONNX preprocessing loops.
// No model is needed. Keep selection, full std::sort and sparse filling separate
// from the device top-k implementation so this can detect algorithm differences.
namespace trackster_test {
  enum class Family { CNN, DNN, PFN };
  struct Cluster {
    float energy, x, y, z;
    int layer, cells;
    bool barrel = false;
  };
  struct Trackster {
    std::vector<uint32_t> vertices;
    std::vector<float> multiplicity;
    float energy = 5.f, emEnergy = 2.f, x = 3.f, y = 4.f, z = -300.f;
  };
  struct Sample {
    uint32_t index;
    std::vector<float> clusters;
    std::array<float, 7> trackster;
  };
  inline float absEta(float x, float y, float z) {
    double r = std::hypot(double(x), double(y));
    return r > 0 ? std::asinh(std::abs(double(z)) / r) : (z == 0 ? 0.f : std::abs(z) + 22756.f);
  }
  inline std::vector<Sample> reference(Family family, std::vector<Cluster> const& clusters,
                                      std::vector<Trackster> const& tracksters, float threshold) {
    std::vector<Sample> result;
    int features = family == Family::PFN ? 7 : 3;
    for (uint32_t i = 0; i < tracksters.size(); ++i) {
      auto const& ts = tracksters[i];
      bool selected = family == Family::PFN;
      float sum = 0;
      for (auto v : ts.vertices) {
        auto const& cl = clusters.at(v);
        if (family == Family::PFN && cl.barrel) {
          selected = false;
          break;
        }
        if (family == Family::DNN && cl.barrel)
          continue;
        if (family != Family::PFN) {
          sum += cl.energy;
          if (sum >= threshold) {
            selected = true;
            break;
          }
        }
      }
      if (!selected)
        continue;
      Sample sample{i, std::vector<float>(50 * 10 * features, 0.f),
                    {ts.energy, ts.emEnergy, ts.x, ts.y, std::abs(ts.z), absEta(ts.x, ts.y, ts.z),
                     std::atan2(ts.y, ts.x)}};
      std::vector<int> order(ts.vertices.size());
      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(), [&](int a, int b) {
        return clusters[ts.vertices[a]].energy > clusters[ts.vertices[b]].energy;
      });
      std::array<int, 50> seen{};
      for (auto k : order) {
        auto const& cl = clusters[ts.vertices[k]];
        if (family == Family::DNN && cl.barrel)
          continue;
        int layer = cl.layer;
        if (layer < 0 || layer >= 50 || seen[layer] >= 10)
          continue;
        auto base = (layer * 10 + seen[layer]++) * features;
        sample.clusters[base] = cl.energy / ts.multiplicity[k];
        sample.clusters[base + 1] = absEta(cl.x, cl.y, cl.z);
        sample.clusters[base + 2] = std::atan2(cl.y, cl.x);
        if (family == Family::PFN) {
          sample.clusters[base + 3] = cl.x;
          sample.clusters[base + 4] = cl.y;
          sample.clusters[base + 5] = std::abs(cl.z);
          sample.clusters[base + 6] = cl.cells;
        }
      }
      result.push_back(std::move(sample));
    }
    return result;
  }
}
#endif
