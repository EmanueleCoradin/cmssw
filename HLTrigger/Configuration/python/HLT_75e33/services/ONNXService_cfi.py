import FWCore.ParameterSet.Config as cms
import os

ONNXService = cms.Service(
    "ONNXService",
    tensorRTLibrary=cms.untracked.string(
        os.path.join(
            os.environ["CMSSW_BASE"],
            "external",
            os.environ["SCRAM_ARCH"],
            "lib",
            "libonnxruntime_ep_tensorrt.so",
        )
    ),
)
