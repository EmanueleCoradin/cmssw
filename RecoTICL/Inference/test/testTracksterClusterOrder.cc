#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include "RecoTICL/Inference/interface/TracksterClusterOrder.h"

int main() {
  std::mt19937 random(42);
  for (int size = 0; size < 2048; ++size) {
    for (int variant = 0; variant < 8; ++variant) {
      std::vector<float> energy(size);
      for (int i = 0; i < size; ++i)
        energy[i] = variant == 0   ? 1.f
                    : variant == 1 ? float(i)
                    : variant == 2 ? float(size - i)
                    : variant == 3 ? float(std::min(i, size - i))
                                   : float(random() % (variant * 3));
      auto before = [&](uint32_t a, uint32_t b) { return energy[a] > energy[b]; };
      std::vector<uint32_t> reference(size), actual(size);
      std::iota(reference.begin(), reference.end(), 0);
      actual = reference;
      std::sort(reference.begin(), reference.end(), before);
      ticl::detail::TracksterClusterOrder<decltype(before)>{before}(actual.data(), size);
      if (reference != actual) {
        std::cerr << "Cluster ordering differs from std::sort: size=" << size << " variant=" << variant << '\n';
        return 1;
      }
    }
  }
  std::cout << "16384 index-order comparisons with std::sort passed (including ties)\n";
}
