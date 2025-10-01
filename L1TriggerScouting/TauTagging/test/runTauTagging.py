import FWCore.ParameterSet.Config as cms
from IOPool.Input.modules import PoolSource
from L1TriggerScouting.TauTagging.options_cff import args
from L1TriggerScouting.TauTagging.modules import (
    l1sc_PFCandidatesAoSToSoA_alpaka,
    l1sc_CLUETaus_alpaka,
    l1sc_TauTaggingSink,
)


args.parseArguments()
process = cms.Process("L1TScPhase2TauTagging")

# enable multithreading
process.options.numberOfThreads = args.numberOfThreads if args.numberOfThreads > 1 else 1 
process.options.numberOfStreams = args.numberOfStreams if args.numberOfStreams > 1 else 1 
process.maxEvents.input = args.numberOfEvents if args.numberOfEvents > 1 else 1 

# enable alpaka and GPU support
process.load("Configuration.StandardSequences.Accelerators_cff")

# logging configuration
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100

# process a limited number of events
process.source = PoolSource(
    fileNames = [
      "file:" + cms.FileInPath("L1TriggerScouting/TauTagging/data/pfOnly.root").value()
    ]
)

# setup chain configs
# PFCandidates
process.PFCandidatesAoSToSoA = l1sc_PFCandidatesAoSToSoA_alpaka(
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string(args.backend)
    ),
    src = cms.InputTag("l1tLayer1Extended", "PF", "L1Dump"),
    environment = cms.untracked.int32(args.environment),
)
process.CLUETaus = l1sc_CLUETaus_alpaka(
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string(args.backend)
    ),
    pf = 'PFCandidatesAoSToSoA',
    dc = cms.double(args.dc),
    rhoc = cms.double(args.rhoc),
    dm = cms.double(args.dm),
    wrapCoords = cms.bool(args.wrapCoords),
    environment = cms.untracked.int32(args.environment),
)
# debug sink
process.TauTaggingSink = l1sc_TauTaggingSink(
    pf = 'PFCandidatesAoSToSoA',
    clusters = 'CLUETaus',
    environment = cms.untracked.int32(args.environment),
)

# schedule the modules
process.path = cms.Path(
    process.PFCandidatesAoSToSoA +
    process.CLUETaus +
    process.TauTaggingSink
)