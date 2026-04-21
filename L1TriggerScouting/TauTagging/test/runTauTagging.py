import os
import FWCore.ParameterSet.Config as cms
from IOPool.Input.modules import PoolSource
from L1TriggerScouting.TauTagging.options_cff import *

args = parse_args()
process = cms.Process("L1TScPhase2TauTagging")

# summary
process.options.wantSummary = cms.untracked.bool(args.wantSummary)

# enable multithreading
process.options.numberOfThreads = args.numberOfThreads if args.numberOfThreads > 1 else 1 
process.options.numberOfStreams = args.numberOfStreams if args.numberOfStreams > 1 else 1 
process.maxEvents.input = args.numberOfEvents if args.numberOfEvents > 1 else 1 

# timing
process.load( "HLTrigger.Timer.FastTimerService_cfi" )
process.FastTimerService.printEventSummary = True
process.FastTimerService.printJobSummary = True
process.FastTimerService.writeJSONSummary = cms.untracked.bool(args.timer)
streams = list(range(sum(args.buNumStreams))) if args.streams == [] else args.streams
process.FastTimerService.jsonFileName = cms.untracked.string(f"resources.json")
process.FastTimerService.enableTimingPaths = cms.untracked.bool(True)
process.FastTimerService.enableTimingModules = cms.untracked.bool(True)
process.FastTimerService.useRealTimeClock = cms.untracked.bool(True)

# enable alpaka and GPU support
process.load("Configuration.StandardSequences.Accelerators_cff")

# logging configuration
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 10

# define path
process.path = cms.Path()

if args.runScouting:
    if len(args.buNumStreams) != len(args.buBaseDir):
        raise RuntimeError("Mismatch between buNumStreams (%d) and buBaseDirs (%d)" % (len(args.buNumStreams), len(args.buBaseDir)))

    process.EvFDaqDirector = cms.Service("EvFDaqDirector",
        useFileBroker = cms.untracked.bool(args.broker != "none"),
        fileBrokerHostFromCfg = cms.untracked.bool(False),
        fileBrokerHost = cms.untracked.string(args.broker.split(":")[0] if args.broker != "none" else "htcp40.cern.ch"),
        fileBrokerPort = cms.untracked.string(args.broker.split(":")[1] if args.broker != "none" else "8080"),
        runNumber = cms.untracked.uint32(args.runNumber),
        baseDir = cms.untracked.string(args.fuBaseDir),
        buBaseDir = cms.untracked.string(args.buBaseDir[0]),
        buBaseDirsAll = cms.untracked.vstring(*args.buBaseDir),
        buBaseDirsNumStreams = cms.untracked.vint32(*args.buNumStreams),
        directorIsBU = cms.untracked.bool(False),
    )

    fuDir = args.fuBaseDir+("/run%06d" % args.runNumber)
    buDirs = [b+("/run%06d" % args.runNumber) for b in args.buBaseDir]
    for d in [fuDir, args.fuBaseDir] + buDirs + args.buBaseDir:
        if not os.path.isdir(d):
            os.makedirs(d)

    process.source = cms.Source("DAQSource",
        testing = cms.untracked.bool(True),
        dataMode = cms.untracked.string(args.daqSourceMode),
        verifyChecksum = cms.untracked.bool(True),
        useL1EventID = cms.untracked.bool(False),
        eventChunkBlock = cms.untracked.uint32(8 * 1024),
        eventChunkSize = cms.untracked.uint32(8 * 1024),
        maxChunkSize = cms.untracked.uint32(16 * 1024),
        numBuffers = cms.untracked.uint32(4),
        maxBufferedFiles = cms.untracked.uint32(4),
        fileListMode = cms.untracked.bool(args.broker == "none"),
        fileNames = cms.untracked.vstring(
            buDirs[0] + "/" + "run%06d_ls%04d_index%06d_stream00.raw" % (args.runNumber, args.lumiNumber, 1),
        )
    )
    os.system("touch " + buDirs[0] + "/" + "fu.lock")
else:
    raise RuntimeError("Currently only runScoutng is supported")

if args.runScouting and args.step >= Step.UNPACKING:
    from L1TriggerScouting.Phase2.modules import l1sc_L1TScPhase2PuppiRawToDigi_alpaka
    process.PFCandidatesProducer = l1sc_L1TScPhase2PuppiRawToDigi_alpaka(
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string(args.backend)
        ),
        streams = cms.vuint32(*list(range(sum(args.buNumStreams))) if args.streams == [] else args.streams),
        splitFactor = cms.uint32(args.splitFactor),
        src = cms.InputTag('rawDataCollector')
    )
    process.path += process.PFCandidatesProducer

    if args.dump >= Dump.CANDIDATES:
        from L1TriggerScouting.Phase2.modules import PFCandidateSoAToOrbitFlatTable
        process.DumpCandidates = PFCandidateSoAToOrbitFlatTable(
            srcBx = cms.InputTag("PFCandidatesProducer", "bxLookup"), 
            srcPF = cms.InputTag("PFCandidatesProducer", "candidates"),
            name = "L1PF"
        )
        process.path += process.DumpCandidates

# CLUEstering
if args.runScouting and args.step >= Step.CLUSTERING:
    from L1TriggerScouting.TauTagging.modules import l1sc_CLUETaus_alpaka
    process.CLUETaus = l1sc_CLUETaus_alpaka(
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string(args.backend)
        ),
        candidates = cms.InputTag("PFCandidatesProducer", "candidates"),
        bxSizes = cms.InputTag("PFCandidatesProducer", "bxSizes"),
        dc = cms.double(args.dc),
        rhoc = cms.double(args.rhoc),
        dm = cms.double(args.dm),
        wrapCoords = cms.bool(args.wrapCoords)
    )
    process.path += process.CLUETaus

    if args.dump >= Dump.CLUSTERS:
        from L1TriggerScouting.Phase2.modules import ClusterSoAToOrbitFlatTable
        process.DumpClusters = ClusterSoAToOrbitFlatTable(
            srcBx = cms.InputTag("PFCandidatesProducer", "bxLookup"), 
            srcClusters = cms.InputTag("CLUETaus", "clusters"), 
            name = "CLUE", # so that the column in the root file is CLUE_cluster
            doc = ""
        )
        process.path += process.DumpClusters

        from L1TriggerScouting.Phase2.modules import ScPhase2ClusterMapsToOrbitFlatTable
        process.DumpClusterObj = ScPhase2ClusterMapsToOrbitFlatTable(
            srcCandidates = cms.InputTag("PFCandidatesProducer", "candidates"), 
            srcBxClustersMap = cms.InputTag("CLUETaus", "bxClustersMap"), 
            srcClustersCandsMap = cms.InputTag("CLUETaus", "clustersCandsMap"), 
            srcClusterIndexes = cms.InputTag("CLUETaus", "clusterIndexes"), 
            nameClusters = "CLUEClusters", 
            doc = ""
        )
        process.path += process.DumpClusterObj

# Tagging
if args.runScouting and args.step >= Step.SORTING: 
    substep = 0

    if args.step >= Step.RESHAPING:
            substep = 1
    if args.step >= Step.TAGGING:
            substep = 2

    from L1TriggerScouting.TauTagging.modules import l1sc_SoftTauIdML_alpaka
    process.SoftTauId = l1sc_SoftTauIdML_alpaka(
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string(args.backend)
        ),
        srcCandidates = cms.InputTag("PFCandidatesProducer", "candidates"),
        srcBxClustersMap = cms.InputTag("CLUETaus", "bxClustersMap"),
        srcClustersCandsMap = cms.InputTag("CLUETaus", "clustersCandsMap"),
        srcClusters = cms.InputTag("CLUETaus", "clusters"),
        model = cms.FileInPath(args.model),
        step = cms.uint32(substep),
        maxBatchSize = cms.uint32(10000)
    )
    process.path += process.SoftTauId

    # if args.dump >= Dump.CLUSTERS:
    #     from L1TriggerScouting.Phase2.modules import ScPhase2ClusterMapsToOrbitFlatTable
    #     process.DumpClusterObj = ScPhase2ClusterMapsToOrbitFlatTable(
    #         srcCandidates = cms.InputTag("PFCandidatesProducer", "candidates"), 
    #         srcBxClustersMap = cms.InputTag("CLUETaus", "bxClustersMap"), 
    #         srcClustersCandsMap = cms.InputTag("SoftTauId", "clusterCandsMapSorted"), 
    #         srcClusterIndexes = cms.InputTag("CLUETaus", "clusterIndexes"), 
    #         nameClusters = "CLUEClusters", 
    #         doc = ""
    #     )
    #     process.path += process.DumpClusterObj



if args.dump > Dump.NONE:
    if args.runScouting:
        process.out = cms.OutputModule("OrbitNanoAODOutputModule",
            fileName = cms.untracked.string(f"ScoutCLUETaus_new_sorted.root"),
            SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring()),  # keep all events
            outputCommands = cms.untracked.vstring(
                "drop *",
                "keep l1ScoutingRun3OrbitFlatTable_*_*_*",
                "keep nanoaodFlatTable_*Table_*_*"  
            )
        )   
        process.end = cms.EndPath(process.out)