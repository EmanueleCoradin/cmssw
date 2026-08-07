#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/PyTorchAllocatorBridge.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/PyTorchAlpakaService.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  PyTorchAlpakaService::PyTorchAlpakaService(edm::ParameterSet const&, edm::ActivityRegistry&) {
    edm::LogInfo("PyTorchAlpakaService") << "Plugging in CMSSW's allocator in PyTorch." << std::endl;
    PyTorchAllocatorBridge<Queue>::install();
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_SERVICE(ALPAKA_TYPE_ALIAS(PyTorchAlpakaService));
