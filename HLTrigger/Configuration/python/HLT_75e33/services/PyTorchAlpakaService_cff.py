import FWCore.ParameterSet.Config as cms

def loadPyTorchAlpakaService(process):
    # Load the appropriate PyTorchAlpakaService configuration based on the available AlpakaService in the process.
    if hasattr(process, "AlpakaServiceCudaAsync"):
        process.load("HLTrigger/Configuration/HLT_75e33/services/PyTorchAlpakaServiceCudaAsync_cfi")
    elif hasattr(process, "AlpakaServiceROCmAsync"):
        process.load("HLTrigger/Configuration/HLT_75e33/services/PyTorchAlpakaServiceROCmAsync_cfi")
    else:
        process.load("HLTrigger/Configuration/HLT_75e33/services/PyTorchAlpakaServiceSerialSync_cfi")
    return process

