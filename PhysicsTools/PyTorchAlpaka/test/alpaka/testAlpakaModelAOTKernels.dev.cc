#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableCollection.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "PhysicsTools/PyTorchAlpaka/test/alpaka/TestAlpakaModelAOTKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest {

  class FillKernel {
  public:
    template <alpaka::concepts::Acc TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                   InputDeviceCollectionView view) const {
      for (auto i : cms::alpakatools::uniform_elements(
               acc, view.metadata().size())) {
        view.x()[i] = static_cast<float>(i + 1);
        view.y()[i] = static_cast<float>(i + 2);
        view.z()[i] = static_cast<float>(i + 3);
        view.a()[i] = static_cast<float>(i + 4);
        view.b()[i] = static_cast<float>(i + 5);
        view.c()[i] = static_cast<float>(i + 6);
        view.d()[i] = static_cast<float>(i + 7);
        view.e()[i] = static_cast<float>(i + 8);
        view.f()[i] = static_cast<float>(i + 9);
        view.g()[i] = static_cast<float>(i + 10);
      }
    }
  };

  void fill(Queue& queue, InputDeviceCollection& collection) {
    constexpr uint32_t items = 64;

    auto groups =
        cms::alpakatools::divide_up_by(collection->metadata().size(), items);

    auto workDiv =
        cms::alpakatools::make_workdiv<Acc1D>(groups, items);

    alpaka::exec<Acc1D>(
        queue,
        workDiv,
        FillKernel{},
        collection.view());
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest