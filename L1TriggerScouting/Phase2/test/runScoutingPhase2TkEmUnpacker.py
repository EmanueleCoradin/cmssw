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
if options.unpackerType not in ["legacy", "alpaka", "alpaka-v2"]:
    raise ValueError("unpackerType must be either legacy, alpaka or alpaka-v2")

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
if options.unpackerType == "legacy":
    process.unpacker = cms.EDProducer("ScPhase2TkEmRawToDigi", 
        src = cms.InputTag("rawDataCollector"),
        fedIDs = cms.vuint32(
            *list(range(options.buNumStreams[0])) # here we assume that buNumStreams[0] corresponds to the number of streams of PF candidates
            if options.tkEmStreamIDs == [] else options.tkEmStreamIDs
        )
    )

elif options.unpackerType == "alpaka":
    process.unpacker = cms.EDProducer("l1sc::L1TScPhase2TkEmRawToDigi@alpaka",
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string(options.backend)
        ),
        src = cms.InputTag("rawDataCollector"),
        streams = cms.vuint32(
            *list(range(options.buNumStreams[0])) # here we assume that buNumStreams[0] corresponds to the number of streams of PF candidates
            if options.tkEmStreamIDs == [] else options.tkEmStreamIDs
        ),
        splitFactor = cms.uint32(1)
    )

elif options.unpackerType == "alpaka-v2":
    process.unpacker = cms.EDProducer("l1sc::L1TScPhase2TkEmRawToDigiV2@alpaka",
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string(options.backend)
        ),
        src = cms.InputTag("rawDataCollector"),
        streams = cms.vuint32(
            *list(range(options.buNumStreams[0])) # here we assume that buNumStreams[0] corresponds to the number of streams of PF candidates
            if options.tkEmStreamIDs == [] else options.tkEmStreamIDs
        ),
        splitFactor = cms.uint32(1)
    )

process.p_unpacking += process.unpacker

process.schedule = cms.Schedule(
    process.p_unpacking
)

