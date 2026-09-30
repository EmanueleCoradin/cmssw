#include <alpaka/alpaka.hpp>
#include <cppunit/extensions/HelperMacros.h>

#include "DataFormats/Portable/interface/PortableCollection.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "PhysicsTools/PyTorchAlpaka/interface/GetDevice.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/AlpakaModelAOT.h"

#include "PhysicsTools/PyTorchAlpaka/test/alpaka/TestAlpakaModelAOTKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest {

  constexpr auto modelPath =
      "PhysicsTools/PyTorchAlpaka/models/model.pt2";

  using namespace ALPAKA_ACCELERATOR_NAMESPACE::torch;

  // ----------------------------------------------------------------------
  // Output SOA
  // ----------------------------------------------------------------------

  GENERATE_SOA_LAYOUT(
      SoAOutputTemplate,
      SOA_COLUMN(float, output))

  using SoAOutput = SoAOutputTemplate<>;
  using OutputDeviceCollection =
      PortableCollection<Device, SoAOutput>;
  using OutputDeviceCollectionView =
      OutputDeviceCollection::View;

  // ----------------------------------------------------------------------
  // Test
  // ----------------------------------------------------------------------

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

    // Enable this again once the model execution test is ready.
    // CPPUNIT_TEST(testAsyncExecution);

    CPPUNIT_TEST_SUITE_END();

    const int64_t batch_size_ = 8;

    template <typename Fn>
    void forEachAlpakaDevice(Fn&& fn) {
      auto m_path = edm::FileInPath(modelPath).fullPath();

      const auto& devices =
          cms::alpakatools::devices<Platform>();

      CPPUNIT_ASSERT(!devices.empty());

      for (auto& dev : devices) {
        std::cout
            << "Running test on device "
            << cms::torch::alpakatools::getDevice(dev)
            << std::endl;

        fn(dev, m_path);
      }
    }
  };

  CPPUNIT_TEST_SUITE_REGISTRATION(TestAlpakaModelAOT);

  // ----------------------------------------------------------------------
  // Constructor test
  // ----------------------------------------------------------------------

  void TestAlpakaModelAOT::testCtorFromQueue() {
    forEachAlpakaDevice([&](auto dev, auto m_path) {
      Queue queue{dev};

      auto m = AlpakaModelAOT(m_path, queue);

      CPPUNIT_ASSERT_EQUAL(
          cms::torch::alpakatools::getDevice(queue),
          m.device());
    });
  }

  // ----------------------------------------------------------------------
  // Async execution test
  // ----------------------------------------------------------------------

  void TestAlpakaModelAOT::testAsyncExecution() {
    forEachAlpakaDevice([&](auto dev, auto m_path) {
      Queue queue{dev};

      // ------------------------------------------------------------
      // Create input/output collections
      // ------------------------------------------------------------

      InputDeviceCollection inputCollection(
          dev,
          batch_size_);

      OutputDeviceCollection outputCollection(
          dev,
          batch_size_);

      // ------------------------------------------------------------
      // Fill input on the device
      // ------------------------------------------------------------

      fill(queue, inputCollection);

      // ------------------------------------------------------------
      // Create TensorCollections
      // ------------------------------------------------------------

      cms::torch::alpakatools::TensorCollection<Queue>
          input(batch_size_);

      cms::torch::alpakatools::TensorCollection<Queue>
          output(batch_size_);

      auto inputRecords =
          inputCollection.const_view().records();

      auto outputRecords =
          outputCollection.const_view().records();

      input.add<SoAInput>(
          "input",
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

      output.add<SoAOutput>(
          "output",
          outputRecords.output());

      // ------------------------------------------------------------
      // Load model
      // ------------------------------------------------------------

      auto m = AlpakaModelAOT(m_path, queue);

      // ------------------------------------------------------------
      // Execute asynchronously
      // ------------------------------------------------------------

      m.forward(queue, input, output);

      alpaka::wait(queue);
    });
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest
