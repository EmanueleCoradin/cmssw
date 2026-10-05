import FWCore.ParameterSet.Config as cms
process = cms.Process("ScoutingPhase2TauTagging")

# import TauTagging options
from L1TriggerScouting.TauTagging.options_cff import options, VarParsing

# parse arguments
options.parseArguments()

# check arguments consistency
if options.pipelineStep not in ["unpacking", "clustering", "sorting", "reshaping", "tagging"]:
    raise ValueError("pipelineStep must be one among unpacking, clustering, sorting, reshaping, tagging")

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

dump = [] # keep only dump requests compatible with the pipeline step chosen
if options.dump != []:
    for d in options.dump:
        if d not in ["candidates", "cluster_indexes", "soft_tau_inputs", "soft_tau_outputs"]:
            raise ValueError("dump must be one among candidates, cluster_indexes, soft_tau_inputs, soft_tau_outputs")
        if step_mapper[d] > step_mapper[options.pipelineStep]:
            print(f"Requested object dump ({d}) is not compatible with specified pipelineStep ({options.pipelineStep}) and will be removed.")
            continue
        dump.append(d)
    
    print("Dumps requested and compatible with pipeline step: ", dump)

# common framework options, accelerator support, and DAQ input
from L1TriggerScouting.Phase2.L1TScPhase2RunScouting_cff import setupPhase2Scouting
setupPhase2Scouting(process, options)

# disable pytorch inner threading
process.PyTorchService = cms.Service("PyTorchService")

# timing
process.load( "HLTrigger.Timer.FastTimerService_cfi" )
process.FastTimerService.printEventSummary = True
process.FastTimerService.writeJSONSummary = cms.untracked.bool(True)
process.FastTimerService.jsonFileName = cms.untracked.string(f"Phase2Timing_resources.json")
process.FastTimerService.enableTimingPaths = cms.untracked.bool(True)
process.FastTimerService.enableTimingModules = cms.untracked.bool(True)
process.FastTimerService.useRealTimeClock = cms.untracked.bool(True)

# define pipeline path
process.p_pipeline = cms.Path()

# unpacking
"""
Remarks on streams unpacker input parameter:
streams should be a list with each entry is the stream ID of stream files inside one or multiple buBaseDir(s). 
Since we are working with PF candidates, we care only about PF candidates stream files. The current working setup is one 
stream file for PF barrel and one stream file for PF endcap. Hence:
- either one specifies pfBarrelStreamIDs=0 and pfEndcapStreamIDs=1 to get streams = [0, 1]
- or one just specifies the total number of streams contained in each buBaseDir, which is expressed by buNumStreams. 
  Usually there is only one buBaseDir containing the stream files for PF barrel and PF endcap. Thus buNumStreams = [2] and it is
  convenient to then automatically generate the range of stream IDs directly from it. This is convenient also if PF barrel and PF
  endcap are going to be split among multiple streams in the same buBaseDir.
"""
process.unpacker = cms.EDProducer("l1sc::L1TScPhase2PuppiRawToDigi@alpaka",
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string(options.backend)
    ),
    src = cms.InputTag("rawDataCollector"),
    streams = cms.vuint32(
        *list(range(options.buNumStreams[0])) # here we assume that buNumStreams[0] corresponds to the number of streams of PF candidates
        if options.pfBarrelStreamIDs == [] else options.pfBarrelStreamIDs + options.pfEndcapStreamIDs
    ),
    splitFactor = cms.uint32(options.splitFactor)
)
process.p_pipeline += process.unpacker

process.unpackerLegacy = cms.EDProducer("ScPhase2PuppiRawToDigi", 
    src = cms.InputTag("rawDataCollector"),
    fedIDs = cms.vuint32(
        *list(range(options.buNumStreams[0])) # here we assume that buNumStreams[0] corresponds to the number of streams of PF candidates
        if options.pfBarrelStreamIDs == [] else options.pfBarrelStreamIDs + options.pfEndcapStreamIDs
    ), 
    splitFactor = cms.uint32(2)
)
process.p_pipeline += process.unpackerLegacy

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
    model = cms.FileInPath(options.model), 
    substep = cms.string(options.pipelineStep), 
    batchSize = cms.uint32(options.batchSize)
)

if step_mapper[options.pipelineStep] >= step_mapper["clustering"]:
    process.p_pipeline += process.L1TScPhase2CLUEJetsProducer
if step_mapper[options.pipelineStep] >= step_mapper["sorting"]:
    process.p_pipeline += process.softTaus

# define table path
process.p_tables = cms.Path()

process.candOrbitTable = cms.EDProducer("PFCandidateSoAToOrbitFlatTable", 
    srcBx = cms.InputTag("unpacker", "bxLookup"), 
    srcCandidates = cms.InputTag("unpacker", "candidates"), 
    name = cms.string("L1PF")
)
process.candOrbitTableLegacy = cms.EDProducer("ScPuppiToOrbitFlatTable", 
    src = cms.InputTag("unpackerLegacy"), 
    name = cms.string("L1PF_legacy"), 
    doc = cms.string("")
)
process.clusterOrbitTable = cms.EDProducer("ClusterSoAToOrbitFlatTable", 
    srcBx = cms.InputTag("unpacker", "bxLookup"), 
    srcClusters = cms.InputTag("L1TScPhase2CLUEJetsProducer", "clusters"), 
    clustering_name = cms.string("CLUEstering"),
    name = cms.string("L1PF"), 
    extension = cms.bool(True) # extends candOrbitTable, set same name as candOrbitTable
)
process.softTauInputsOrbitTable = cms.EDProducer("SoftTauInputTensorToOrbitFlatTable", 
    srcBxClustersMap = cms.InputTag("L1TScPhase2CLUEJetsProducer", "bxClustersMap"), 
    srcInputs = cms.InputTag("softTaus", "softTauInputs"), 
    name = cms.string("SoftTauInputs")
)
process.softTauOutputsOrbitTable = cms.EDProducer("SoftTauOutputTensorToOrbitFlatTable", 
    srcBxClustersMap = cms.InputTag("L1TScPhase2CLUEJetsProducer", "bxClustersMap"), 
    srcOutputs = cms.InputTag("softTaus", "softTauOutputs"), 
    name = cms.string("SoftTauOutputs")
)

if "candidates" in dump:
    process.p_tables += process.candOrbitTable
    process.p_tables += process.candOrbitTableLegacy
    if "cluster_indexes" in dump:
        process.p_tables += process.clusterOrbitTable

if "soft_tau_inputs" in dump:
    process.p_tables += process.softTauInputsOrbitTable

if "soft_tau_outputs" in dump:
    process.p_tables += process.softTauOutputsOrbitTable

# output
process.out = cms.OutputModule("OrbitNanoAODOutputModule",
    fileName = cms.untracked.string("softTauOrbitNano-L1PF_17_0_X_on_pre2-17_0_X.root"),
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring()),
    outputCommands = cms.untracked.vstring("drop *", "keep l1ScoutingRun3OrbitFlatTable_*_*_*"),
)
process.end = cms.EndPath(process.out)

# schedule
process.schedule = cms.Schedule(
    process.p_pipeline, 
    process.p_tables,
    process.end
)