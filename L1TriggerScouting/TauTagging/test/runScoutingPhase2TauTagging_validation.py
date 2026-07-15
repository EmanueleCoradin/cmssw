import os
import FWCore.ParameterSet.Config as cms
process = cms.Process("ScoutingPhase2TauTagging")

# enable alpaka and GPU support
process.load("Configuration.StandardSequences.Accelerators_cff")

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

if len(options.buNumStreams) != len(options.buBaseDir):
        raise RuntimeError("Mismatch between buNumStreams (%d) and buBaseDirs (%d)" % (len(options.buNumStreams), len(options.buBaseDir)))


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

# trigger emulation
process.load('Configuration.Geometry.GeometryExtendedRun4D110Reco_cff')
process.load('Configuration.Geometry.GeometryExtendedRun4D110_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff') # needed to read HCal TPs
process.load('SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi') # needed for HGCAL_noise_fC
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '141X_mcRun4_realistic_v3', '')

process.l1tTrackSelectionProducer.processSimulatedTracks = False # these would need stubs, and are not used anyway

process.l1tEmulation = cms.Task(
    process.l1tTkMuonsGmt,
    process.l1tSAMuonsGmt,
    process.l1tGTTInputProducer,
    process.l1tTrackSelectionProducer,
    process.l1tVertexFinderEmulator,
    process.l1tPhase2L1CaloEGammaEmulator,
    process.l1tPhase2CaloPFClusterEmulator,
    process.l1tPhase2GCTBarrelToCorrelatorLayer1Emulator,    
    process.L1TLayer1TaskInputsTask,
    process.L1TLayer1Task,
    process.l1tLayer2EG,
    process.L1TPFJetsEmulationTask,
    process.L1TPFJetsExtendedTask,
    process.L1TBJetsTask, 
)

# Pool source 
process.source = cms.Source("PoolSource",
    # fileNames = cms.untracked.vstring("file:/eos/cms/store/cmst3/group/l1tr/vcamagni/"
    #                                 "L1TauID/DATA/FPinputs/m90/4STEPS/"
    #                                 f"142Xv0/inputs140X_7099351_{i}.root" for i in range(4000))
    # fileNames = cms.untracked.vstring([
    #     '/store/cmst3/group/l1tr/FastPUPPI/15_1_X/fpinputs_140X/v1/caseC_m220_67/4STEPS/151Xv0/inputs151X_14682953_1699.root'
    # ])
    fileNames = cms.untracked.vstring([
        f"file:/eos/cms/store/cmst3/group/l1tr/FastPUPPI/15_1_X/fpinputs_140X/v1/caseC_m220_67/4STEPS/151Xv3_pu200/inputs151X_15019206_{i}.root" for i in range(1000)
    ])
)

# define pipeline path
process.p_pipeline = cms.Path()

# FP inputs packing/unpacking
process.packer = cms.EDProducer("ScPhase2PuppiPacker",
    src = cms.InputTag("l1tLayer1Extended:PF"),
    fedIDs = cms.vuint32(0),
    splitFactor = cms.uint32(1),
    scoutingHeader = cms.bool(True)
)

process.unpacker = cms.EDProducer("l1sc::L1TScPhase2PuppiRawToDigi@alpaka",
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string(options.backend)
    ),
    src = cms.InputTag("packer"),
    streams = cms.vuint32(*[process.packer.fedIDs]),
    splitFactor = process.packer.splitFactor
)

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

process.candNanoAODTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
    name = cms.string("L1PF"),
    src = cms.InputTag("l1tLayer1Extended:PF"),
    cut = cms.string(""),
    doc = cms.string(""),
    singleton = cms.bool(False), # the number of entries is variable
    extension = cms.bool(False), # this is the main table
    variables = cms.PSet(
        pt  = Var("pt",  float, precision=16),
        eta  = Var("eta", float, precision=16),
        phi = Var("phi", float, precision=16),
        pdgId = Var("pdgId", int),
        z0 = Var("vz", float, precision=16),
        dxy = LazyVar("dxy", float, precision=16),
        quality = LazyVar("hwQual", int),
        puppiw = LazyVar("puppiWeight", float, precision=16),
    )
)
process.clusterNanoAODTable = cms.EDProducer("ClusterSoAToOrbitFlatTable", 
    srcBx = cms.InputTag("unpacker", "bxLookup"), 
    srcClusters = cms.InputTag("L1TScPhase2CLUEJetsProducer", "clusters"), 
    clustering_name = cms.string("CLUEstering"),
    name = cms.string("L1PF"), 
    extension = cms.bool(True) # extends candOrbitTable, set same name as candOrbitTable
)
process.softTauInputsNanoAODTable = cms.EDProducer("SoftTauInputTensorToOrbitFlatTable", 
    srcBxClustersMap = cms.InputTag("L1TScPhase2CLUEJetsProducer", "bxClustersMap"), 
    srcInputs = cms.InputTag("softTaus", "softTauInputs"), 
    name = cms.string("SoftTauInputs")
)
process.softTauOutputsNanoAODTable = cms.EDProducer("SoftTauOutputTensorToOrbitFlatTable", 
    srcBxClustersMap = cms.InputTag("L1TScPhase2CLUEJetsProducer", "bxClustersMap"), 
    srcOutputs = cms.InputTag("softTaus", "softTauOutputs"), 
    name = cms.string("SoftTauOutputs")
)

if "candidates" in dump:
    process.p_tables += process.candOrbitTable
    if "cluster_indexes" in dump:
        process.p_tables += process.clusterOrbitTable

if "soft_tau_inputs" in dump:
    process.p_tables += process.softTauInputsOrbitTable

if "soft_tau_outputs" in dump:
    process.p_tables += process.softTauOutputsOrbitTable

# output
process.out = cms.OutputModule("OrbitNanoAODOutputModule",
    fileName = cms.untracked.string("orbitNano.root"),
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