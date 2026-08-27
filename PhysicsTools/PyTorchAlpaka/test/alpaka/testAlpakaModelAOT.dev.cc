#include <alpaka/alpaka.hpp>
#include <cppunit/extensions/HelperMacros.h>

#include "DataFormats/Portable/interface/PortableCollection.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "PhysicsTools/PyTorchAlpaka/interface/GetDevice.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/AlpakaModelAOT.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest {

  constexpr auto modelPath = "PhysicsTools/PyTorchAlpaka/models/cpp-lxplus.pt2";

  using namespace ALPAKA_ACCELERATOR_NAMESPACE::torch;

  // Input SOA
  GENERATE_SOA_LAYOUT(SoAInputTemplate,
                    SOA_COLUMN(float, x),
                    SOA_COLUMN(float, y),
                    SOA_COLUMN(float, z),
                    SOA_COLUMN(float, a),
                    SOA_COLUMN(float, b),
                    SOA_COLUMN(float, c),
                    SOA_COLUMN(float, d),
                    SOA_COLUMN(float, e),
                    SOA_COLUMN(float, f),
                    SOA_COLUMN(float, g))
  
  using SoAInput = SoAInputTemplate<>;
  using InputDeviceCollection = PortableCollection<Device, SoAInput>;
  using InputDeviceCollectionView = PortableCollection<Device, SoAInput>::View;
  
  // Output SOA 
  GENERATE_SOA_LAYOUT(SoAOutputTemplate, SOA_COLUMN(float, output))

  using SoAOutput = SoAOutputTemplate<>;
  using OutputDeviceCollection = PortableCollection<Device, SoAOutput>;
  using OutputDeviceCollectionView = PortableCollection<Device, SoAOutput>::View;

  // Kernel

class FillKernel {
public:
  template <alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC void operator()(TAcc const& acc, InputDeviceCollectionView view) const {
    for (auto i : cms::alpakatools::uniform_elements(acc, view.metadata().size())) {
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

  class TestAlpakaModelAOT : public CppUnit::TestFixture {
  public:
    void testCtorFromDevice();
    void testCtorFromQueue();
    void testMoveToDeviceFromAlpakaDevice();
    void testMoveToDeviceFromAlpakaQueue();
    void testAsyncExecution();

  private:
    CPPUNIT_TEST_SUITE(TestAlpakaModelAOT);
    CPPUNIT_TEST(testCtorFromQueue);
    //CPPUNIT_TEST(testAsyncExecution);
    CPPUNIT_TEST_SUITE_END();

    const int64_t batch_size_ = 8;

    template <typename Fn>
    void forEachAlpakaDevice(Fn&& fn) {
      auto m_path = edm::FileInPath(modelPath).fullPath();
      const auto& devices = cms::alpakatools::devices<Platform>();
      CPPUNIT_ASSERT(!devices.empty());
      for (auto& dev : devices) {
        std::cout << "Running test on device " << cms::torch::alpakatools::getDevice(dev) << std::endl;
        fn(dev, m_path);
      }
    }
  };

  CPPUNIT_TEST_SUITE_REGISTRATION(TestAlpakaModelAOT);

  void TestAlpakaModelAOT::testCtorFromQueue() {
    forEachAlpakaDevice([&](auto dev, auto m_path) {
      Queue queue{dev};
      auto m = AlpakaModelAOT(m_path, queue);
      CPPUNIT_ASSERT_EQUAL(cms::torch::alpakatools::getDevice(queue), m.device());
    });
  }

  void fill(Queue& queue, InputDeviceCollection& collection) {
    uint32_t items = 64;
    auto groups = cms::alpakatools::divide_up_by(collection->metadata().size(), items);
    auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(groups, items);
    alpaka::exec<Acc1D>(queue, workDiv, FillKernel{}, collection.view());
  }

  void TestAlpakaModelAOT::testAsyncExecution() {
    // load model
    forEachAlpakaDevice([&](auto dev, auto m_path) {
      Queue queue{dev};
      
      // Create input and output SoAs
      InputDeviceCollection inputCollection(dev, batch_size_);
      OutputDeviceCollection outputCollection(dev, batch_size_);
      fill(queue, inputCollection);

      // Create Tensor Collection
      cms::torch::alpakatools::TensorCollection<Queue> input(batch_size_);
      cms::torch::alpakatools::TensorCollection<Queue> output(batch_size_);
      auto inputRecords = inputCollection.const_view().records();
      auto outputRecords = outputCollection.const_view().records();

      input.add<SoAInput>("input", 
        inputRecords.x(), 
        inputRecords.y(), 
        inputRecords.z(), 
        inputRecords.a(), 
        inputRecords.b(), 
        inputRecords.c(), 
        inputRecords.d(), 
        inputRecords.e(), 
        inputRecords.f(), 
        inputRecords.g());

      output.add<SoAOutput>("output", outputRecords.output());
      
      // Load the model
      auto m = AlpakaModelAOT(m_path, queue);

      m.forward(queue, input, output);
      alpaka::wait(queue);
    });
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest
