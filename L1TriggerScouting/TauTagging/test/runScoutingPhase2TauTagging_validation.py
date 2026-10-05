import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras
process = cms.Process("ScoutingPhase2ClusteringValidation", eras.Phase2C17I13M9)

# nanoaod config
from PhysicsTools.NanoAOD.common_cff import Var, ExtVar
def LazyVar(expr, valtype, doc=None, precision=-1):
    return Var(expr, valtype, doc, precision, lazyEval=True)

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

# common framework options, PoolSource, L1 emulation, and PUPPI packing
from L1TriggerScouting.Phase2.L1TScPhase2RunValidation_cff import setupPhase2Validation
setupPhase2Validation(process, options)

# disable pytorch inner threading
process.PyTorchService = cms.Service("PyTorchService")

# define pipeline path, packing before unpacking
process.p_pipeline = cms.Path(process.packer)

process.unpacker = cms.EDProducer("l1sc::L1TScPhase2PuppiRawToDigi@alpaka",
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string(options.backend)
    ),
    src = cms.InputTag("packer"),
    streams = process.packer.fedIDs,
    splitFactor = process.packer.splitFactor
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
    model = cms.FileInPath(options.model), 
    substep = cms.string(options.pipelineStep), 
    batchSize = cms.uint32(options.batchSize)
)

if step_mapper[options.pipelineStep] >= step_mapper["clustering"]:
    process.p_pipeline += process.L1TScPhase2CLUEJetsProducer
if step_mapper[options.pipelineStep] >= step_mapper["sorting"]:
    process.p_pipeline += process.softTaus

# associate pipeline path to l1tEmulation task
process.p_pipeline.associate(process.l1tEmulation)

# define table path
process.p_tables = cms.Path()

# attention here that I should check what is being dumped
process.candNanoAODTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
    name = cms.string("L1PF"),
    src = cms.InputTag("l1tLayer1:PF"), # l1tLayer1Extended:PF or l1tLayer1:PF
    cut = cms.string(""),
    doc = cms.string(""),
    singleton = cms.bool(False), # the number of entries is variable
    extension = cms.bool(False), # this is the main table
    variables = cms.PSet(
        pt  = Var("pt",  float),
        eta  = Var("eta", float),
        phi = Var("phi", float),
        mass = Var("mass", float),
        z0 = LazyVar("vz", float),
        dxy = LazyVar("dxy", float),
        charge = Var("charge", int),
        pdgId = Var("pdgId", int),
        quality = LazyVar("hwQual", int),
        puppiWeight = LazyVar("puppiWeight", float),
    )
)
process.clusterNanoAODTable = cms.EDProducer("ClusterSoAToNanoAODFlatTable", 
    srcClusters = cms.InputTag("L1TScPhase2CLUEJetsProducer", "clusters"), 
    clustering_name = cms.string("CLUEstering"),
    name = cms.string("L1PF"), 
    extension = cms.bool(True) # extends candNanoAODTable, set same name as clusterNanoAODTable
)
process.softTauInputsNanoAODTable = cms.EDProducer("SoftTauInputTensorToNanoAODFlatTable", 
    srcInputs = cms.InputTag("softTaus", "softTauInputs"), 
    name = cms.string("SoftTauInputs")
)
process.softTauOutputsNanoAODTable = cms.EDProducer("SoftTauOutputTensorToNanoAODFlatTable", 
    srcOutputs = cms.InputTag("softTaus", "softTauOutputs"), 
    name = cms.string("SoftTauOutputs")
)

if "candidates" in dump:
    process.p_tables += process.candNanoAODTable
    if "cluster_indexes" in dump:
        process.p_tables += process.clusterNanoAODTable

if "soft_tau_inputs" in dump:
    process.p_tables += process.softTauInputsNanoAODTable

if "soft_tau_outputs" in dump:
    process.p_tables += process.softTauOutputsNanoAODTable

# output
process.out = cms.OutputModule("NanoAODOutputModule",
    fileName = cms.untracked.string("softTauNano-L1PF_17_0_X_on_pre2.root"),
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring()),
    outputCommands = cms.untracked.vstring("drop *", "keep nanoaodFlatTable_*Table_*_*"),
    compressionLevel = cms.untracked.int32(4),
    compressionAlgorithm = cms.untracked.string("ZLIB"),
)
process.end = cms.EndPath(process.out)

# schedule
process.schedule = cms.Schedule(
    process.p_pipeline, 
    process.p_tables,
    process.end
)