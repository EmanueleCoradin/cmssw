# define process
import FWCore.ParameterSet.Config as cms
process = cms.Process("ScoutingPhase2Unpackers")

# import L1 scouting options
from L1TriggerScouting.Phase2.options_cff import options, VarParsing

# extra options
options.register ("reportEvery",
    100, # default value
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.int, 
    "Print message after the specified number of events (proper events in this case)"
)

options.register ("unpackerType",
    "legacy", # "legacy" (default) or "alpaka"
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.string, 
    "Specify which flavor of unpacker to use"
)

# parse options
options.parseArguments()

# check options
if options.unpackerType not in ["legacy", "alpaka"]:
    raise ValueError("unpackerType must be either legacy or alpaka")

if options.unpackerType == "legacy" and options.backend == "cuda_async":
    raise ValueError("Legacy unpacker can be run only on serial_sync backend")

# setup DAQ director and DAQ source
from L1TriggerScouting.Phase2.L1TScPhase2RunScouting_cff import setupPhase2Scouting
setupPhase2Scouting(process, options)

# timing
process.load( "HLTrigger.Timer.FastTimerService_cfi" )
process.FastTimerService.printEventSummary = True
process.FastTimerService.writeJSONSummary = cms.untracked.bool(True)
process.FastTimerService.jsonFileName = cms.untracked.string(f"Phase2Timing_resources.json")
process.FastTimerService.enableTimingPaths = cms.untracked.bool(True)
process.FastTimerService.enableTimingModules = cms.untracked.bool(True)
process.FastTimerService.useRealTimeClock = cms.untracked.bool(True)

# define pipeline path
process.p_unpacking = cms.Path()

# define unpacker
"""
Pay attention that the splitFactor parameter here has a different meaning in the legacy unpacker wrt the alpaka unpacker.
I think that the correct implementation of the split factor is on the alpaka unpacker, thus the correct value should be 1.
"""
if options.unpackerType == "legacy":
    process.unpacker = cms.EDProducer("ScPhase2PuppiRawToDigi", 
        src = cms.InputTag("rawDataCollector"),
        fedIDs = cms.vuint32(
            *list(range(options.buNumStreams[0])) # here we assume that buNumStreams[0] corresponds to the number of streams of PF candidates
            if options.pfBarrelStreamIDs == [] else options.pfBarrelStreamIDs + options.pfEndcapStreamIDs
        ), 
        splitFactor = cms.uint32(2) 
    )

elif options.unpackerType == "alpaka":
    process.unpacker = cms.EDProducer("l1sc::L1TScPhase2PuppiRawToDigi@alpaka",
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string(options.backend)
        ),
        src = cms.InputTag("rawDataCollector"),
        streams = cms.vuint32(
            *list(range(options.buNumStreams[0])) # here we assume that buNumStreams[0] corresponds to the number of streams of PF candidates
            if options.pfBarrelStreamIDs == [] else options.pfBarrelStreamIDs + options.pfEndcapStreamIDs
        ),
        splitFactor = cms.uint32(1)
    )

process.p_unpacking += process.unpacker

process.schedule = cms.Schedule(
    process.p_unpacking
)

