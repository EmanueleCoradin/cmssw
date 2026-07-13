import os
import FWCore.ParameterSet.Config as cms
process = cms.Process("ScoutingPhase2TauTagging")

# enable alpaka and GPU support
process.load("Configuration.StandardSequences.Accelerators_cff")

# import l1scouting options
from L1TriggerScouting.Phase2.options_cff import options, VarParsing

# extra options
options.register ("splitFactor", 
    1, 
    VarParsing.VarParsing.multiplicity.singleton, 
    VarParsing.VarParsing.multiplicity.int, 
    "Number of sub-streams in which a single data stream is divided into."
)

options.register ("pipelineStep", 
    "tagging",
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.string,
    "Step up to which run the pipeline (unpacking, clustering, sorting, reshaping, tagging)."
)

options.register ("model", 
    "L1TriggerScouting/TauTagging/data/softtauid_sigmoid_col.pt", 
    VarParsing.VarParsing.multiplicity.singleton, 
    VarParsing.VarParsing.varType.string,
    "PyTorch model to be used."
)

options.register ("batchSize", 
    8192, 
    VarParsing.VarParsing.multiplicity.singleton, 
    VarParsing.VarParsing.multiplicity.int, 
    "Batch size (in terms of clusters) for model inference."
)

options.register ("dump", 
    [],
    VarParsing.VarParsing.multiplicity.list,
    VarParsing.VarParsing.varType.string,
    "List of objects to dump (candidates, cluster_indexes, soft_tau_inputs, soft_tau_outputs), compatible with the selected pipelineStep."
)

options.register ("synchronize", 
    False, 
    VarParsing.VarParsing.multiplicity.singleton, 
    VarParsing.VarParsing.multiplicity.bool, 
    "Force synchronization after each module, must be used when benchmarking"
)

options.register ("reportEvery",
    10, 
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.int, 
    "Print message after the specified number of events (proper events in this case)"
)

# parse arguments
options.parseArguments()

# check arguments consistency
if options.pipelineStep not in ["unpacking", "clustering", "sorting", "reshaping", "tagging"]:
    raise ValueError("pipelineStep must be one among unpacking, clustering, sorting, reshaping, tagging")

for d in options.dump:
    if d not in ["candidates", "cluster_indexes", "soft_tau_inputs", "soft_tau_outputs"]:
        raise ValueError("dump must be one among candidates, cluster_indexes, soft_tau_inputs, soft_tau_outputs")

step_mapper = {
    "unpacking": 0, 
    "clustering": 1, 
    "sorting": 2, 
    "reshaping": 3, 
    "tagging": 4, 
    "candidates": 0, 
    "cluster_indexes": 1, 
    "soft_tau_inputs": 3, 
    "soft_tau_outputs": 4
}

for d in options.dump:
    if step_mapper[d] > step_mapper[options.pipelineStep]:
        raise ValueError(f"Requested object dump ({d}) is not compatible with specified pipelineStep ({options.pipelineStep})")

if len(buNumStreams) != len(buBaseDir):
        raise RuntimeError("Mismatch between buNumStreams (%d) and buBaseDirs (%d)" % (len(buNumStreams), len(buBaseDir)))

# timing
process.load( "HLTrigger.Timer.FastTimerService_cfi" )
process.FastTimerService.printEventSummary = True
process.FastTimerService.writeJSONSummary = cms.untracked.bool(True)
process.FastTimerService.jsonFileName = cms.untracked.string(f"Phase2Timing_resources.json")
process.FastTimerService.enableTimingPaths = cms.untracked.bool(True)
process.FastTimerService.enableTimingModules = cms.untracked.bool(True)
process.FastTimerService.useRealTimeClock = cms.untracked.bool(True)

# define process and its options
process.options = cms.untracked.PSet(
    numberOfThreads = cms.untracked.uint32(options.numThreads),
    numberOfStreams = cms.untracked.uint32(options.numFwkStreams),
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1),
    wantSummary = cms.untracked.bool(True)
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.maxEvents)
)
process.MessageLogger.cerr.FwkReport.reportEvery = options.reportEvery

# data source
process.EvFDaqDirector = cms.Service("EvFDaqDirector",
    useFileBroker = cms.untracked.bool(options.broker != "none"),
    fileBrokerHostFromCfg = cms.untracked.bool(False),
    fileBrokerHost = cms.untracked.string(broker.split(":")[0] if options.broker != "none" else "htcp40.cern.ch"),
    fileBrokerPort = cms.untracked.string(broker.split(":")[1] if options.broker != "none" else "8080"),
    runNumber = cms.untracked.uint32(options.runNumber),
    baseDir = cms.untracked.string(options.fuBaseDir),
    buBaseDir = cms.untracked.string(options.buBaseDir[0]),
    buBaseDirsAll = cms.untracked.vstring(*options.buBaseDir),
    buBaseDirsNumStreams = cms.untracked.vint32(*options.buNumStreams),
    directorIsBU = cms.untracked.bool(False),
)

fuDir = options.fuBaseDir + ("/run%06d" % options.runNumber)
buDirs = [b + ("/run%06d" % options.runNumber) for b in options.buBaseDir]
for d in [fuDir, fuBaseDir] + buDirs + buBaseDir:
    if not os.path.isdir(d):
        os.makedirs(d)

process.source = cms.Source("DAQSource",
    testing = cms.untracked.bool(True),
    dataMode = cms.untracked.string(options.daqSourceMode),
    verifyChecksum = cms.untracked.bool(True),
    useL1EventID = cms.untracked.bool(False),
    eventChunkBlock = cms.untracked.uint32(8 * 1024),
    eventChunkSize = cms.untracked.uint32(8 * 1024),
    maxChunkSize = cms.untracked.uint32(16 * 1024),
    numBuffers = cms.untracked.uint32(4),
    maxBufferedFiles = cms.untracked.uint32(4),
    fileListMode = cms.untracked.bool(options.broker == "none"),
    fileNames = cms.untracked.vstring(
        buDirs[0] + "/" + "run%06d_ls%04d_index%06d_stream00.raw" % (options.runNumber, options.lumiNumber, 1),
    )
)
os.system("touch " + buDirs[0] + "/" + "fu.lock")

# define pipeline path
process.p_pipeline = cms.Path()

# unpacking
process.unpacker = cms.EDProducer("l1sc::L1TScPhase2PuppiRawToDigi@alpaka",
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string(options.backend)
    ),
    src = cms.InputTag("rawDataCollector"),
    streams = cms.vuint32(*list(range(sum(options.buNumStreams))) if option.streams == [] else options.streams),
    splitFactor = options.splitFactor
)
process.p_pipeline += process.unpacker

# clustering
process.load(
    "L1TriggerScouting.Phase2.L1TScPhase2CLUEJets_cff"
)
process.L1TScPhase2CLUEJetsProducer.alpaka = cms.untracked.PSet(
    backend = cms.untracked.string(options.backend)
)
process.L1TScPhase2CLUEJetsProducer.candidates = cms.InputTag("unpacker", "candidates")
process.L1TScPhase2CLUEJetsProducer.bxSizes = cms.InputTag("unpacker", "bxSizes")

# sorting, reshaping, tagging
process.softTaus = cms.EDProducer("l1sc::SoftTauIdML@alpaka", 
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string(options.backend)
    ),
    srcCandidates = cms.InputTag("unpacker", "candidates"), 
    srcBxClustersMap = cms.InputTag("L1TScPhase2CLUEJetsProducer", "bxClustersMap"), 
    srcClustersCandsMap = cms.InputTag("L1TScPhase2CLUEJetsProducer", "clustersCandsMap"), 
    srcClusters = cms.InputTag("L1TScPhase2CLUEJetsProducer", "clusters"),
    model = cms.FileInPath(option.model), 
    substep = cms.string(options.pipelineStep), 
    batchSize = cms.uint32(options.batchSize)
)

if step_mapper[options.pipelineStep] >= step_mapper["clustering"]:
    process.p_pipeline += process.L1TScPhase2CLUEJetsProducer
if step_mapper[options.pipelineStep] >= step_mapper["sorting"]:
    process.p_pipeline += p.softTaus

# define table path
process.p_tables = cms.Path()

if "candidates" in options.dump:
    process.candOrbitTable = cms.EDProducer("PFCandidateSoAToOrbitFlatTable", 
        srcBx = cms.InputTag("unpacker", "bxLookup"), 
        srcCandidates = cms.InputTag("unpacker", "candidates"), 
        clustering_name = cms.string("CLUEstering")
        name = cms.string("L1PF")
    )
    process.p_tables += process.candOrbitTable
