#ifndef RecoTICL_Inference_TracksterClusterOrder_h
#define RecoTICL_Inference_TracksterClusterOrder_h

#include <cstdint>
#if defined(__CUDACC__) || defined(__HIPCC__)
#define TICL_ORDER_HOST_DEVICE __host__ __device__
#else
#define TICL_ORDER_HOST_DEVICE
#endif

namespace ticl::detail {
  // The CPU references use libstdc++ std::sort with an energy-only comparator.
  // Equal energies therefore cannot be reproduced by a stable top-k algorithm.
  // This iterative introsort follows its median partition / 16-element insertion
  // strategy, including heap fallback. Keep the host parity test when changing
  // compiler/STL versions: ordering of equivalent elements is not a C++ guarantee.
  template <typename Before>
  struct TracksterClusterOrder {
    Before before;

    TICL_ORDER_HOST_DEVICE void swap(uint32_t& a, uint32_t& b) const {
      auto value = a;
      a = b;
      b = value;
    }

    TICL_ORDER_HOST_DEVICE void adjustHeap(uint32_t* a, int hole, int size, uint32_t value) const {
      int top = hole;
      int child = hole;
      while (child < (size - 1) / 2) {
        child = 2 * child + 2;
        if (before(a[child], a[child - 1]))
          --child;
        a[hole] = a[child];
        hole = child;
      }
      if (size % 2 == 0 && child == (size - 2) / 2) {
        child = 2 * child + 1;
        a[hole] = a[child];
        hole = child;
      }
      while (hole > top) {
        int parent = (hole - 1) / 2;
        if (!before(a[parent], value))
          break;
        a[hole] = a[parent];
        hole = parent;
      }
      a[hole] = value;
    }

    TICL_ORDER_HOST_DEVICE void heapSort(uint32_t* a, int size) const {
      if (size < 2)
        return;
      for (int parent = (size - 2) / 2; parent >= 0; --parent)
        adjustHeap(a, parent, size, a[parent]);
      for (int end = size - 1; end > 0; --end) {
        auto value = a[end];
        a[end] = a[0];
        adjustHeap(a, 0, end, value);
      }
    }

    TICL_ORDER_HOST_DEVICE int partition(uint32_t* a, int begin, int end) const {
      int first = begin + 1, middle = begin + (end - begin) / 2, last = end - 1;
      int pivot;
      if (before(a[first], a[middle]))
        pivot = before(a[middle], a[last]) ? middle : (before(a[first], a[last]) ? last : first);
      else
        pivot = before(a[first], a[last]) ? first : (before(a[middle], a[last]) ? last : middle);
      swap(a[begin], a[pivot]);
      int left = begin + 1, right = end;
      while (true) {
        while (before(a[left], a[begin]))
          ++left;
        do {
          --right;
        } while (before(a[begin], a[right]));
        if (left >= right)
          return left;
        swap(a[left++], a[right]);
      }
    }

    TICL_ORDER_HOST_DEVICE void operator()(uint32_t* a, int size) const {
      if (size < 2)
        return;
      int depth = 0;
      for (int n = size; n > 1; n /= 2)
        depth += 2;
      // At most 2*floor(log2(INT_MAX)) pending partitions, bounded by depth.
      struct Range {
        int begin, end, depth;
      };
      Range pending[64];
      int count = 1;
      pending[0] = {0, size, depth};
      while (count) {
        auto range = pending[--count];
        if (range.end - range.begin <= 16)
          continue;
        if (range.depth == 0) {
          heapSort(a + range.begin, range.end - range.begin);
          continue;
        }
        int cut = partition(a, range.begin, range.end);
        pending[count++] = {range.begin, cut, range.depth - 1};
        pending[count++] = {cut, range.end, range.depth - 1};
      }
      // Guarded insertion is equivalent to the final guarded/unguarded passes
      // after introsort has placed the global minimum in the first partition.
      for (int i = 1; i < size; ++i) {
        auto value = a[i];
        int j = i;
        while (j > 0 && before(value, a[j - 1])) {
          a[j] = a[j - 1];
          --j;
        }
        a[j] = value;
      }
    }
  };
}  // namespace ticl::detail
#undef TICL_ORDER_HOST_DEVICE
#endif
