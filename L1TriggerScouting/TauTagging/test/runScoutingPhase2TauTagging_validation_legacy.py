import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras
process = cms.Process("ScoutingPhase2ClusteringValidation", eras.Phase2C17I13M9)

# nanoaod config
from PhysicsTools.NanoAOD.common_cff import Var, ExtVar
def LazyVar(expr, valtype, doc=None, precision=-1):
    return Var(expr, valtype, doc, precision, lazyEval=True)

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

process.load('L1Trigger.Phase2L1ParticleFlow.l1ctLayer1_cff')
process.load('L1Trigger.L1TTrackMatch.l1tGTTInputProducer_cfi')
process.load('L1Trigger.L1TTrackMatch.l1tTrackSelectionProducer_cfi')
process.load('L1Trigger.VertexFinder.l1tVertexProducer_cfi')
from L1Trigger.Configuration.SimL1Emulator_cff import l1tSAMuonsGmt
process.l1tSAMuonsGmt = l1tSAMuonsGmt.clone()
from L1Trigger.L1CaloTrigger.l1tPhase2L1CaloEGammaEmulator_cfi import l1tPhase2L1CaloEGammaEmulator
process.l1tPhase2L1CaloEGammaEmulator = l1tPhase2L1CaloEGammaEmulator.clone()
from L1Trigger.L1CaloTrigger.l1tPhase2CaloPFClusterEmulator_cfi import l1tPhase2CaloPFClusterEmulator
process.l1tPhase2CaloPFClusterEmulator = l1tPhase2CaloPFClusterEmulator.clone()
from L1Trigger.L1CaloTrigger.l1tPhase2GCTBarrelToCorrelatorLayer1Emulator_cfi import l1tPhase2GCTBarrelToCorrelatorLayer1Emulator
process.l1tPhase2GCTBarrelToCorrelatorLayer1Emulator = l1tPhase2GCTBarrelToCorrelatorLayer1Emulator.clone()

process.L1TInputTask = cms.Task(
    process.l1tSAMuonsGmt,
    process.l1tPhase2L1CaloEGammaEmulator,
    process.l1tPhase2CaloPFClusterEmulator,
    process.l1tPhase2GCTBarrelToCorrelatorLayer1Emulator
)
process.runPF = cms.Path( 
        process.l1tGTTInputProducer +
        process.l1tTrackSelectionProducer +
        process.l1tVertexFinderEmulator +
        process.l1tLayer1BarrelExtended + 
        process.l1tLayer1HGCalExtended +
        process.l1tLayer1HGCalNoTK +
        process.l1tLayer1HF +
        process.l1tLayer1Extended 
)
process.runPF.associate(process.L1TLayer1TaskInputsTask)
process.runPF.associate(process.L1TInputTask)

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
    src = cms.InputTag("l1tLayer1Extended:PF"), # l1tLayer1Extended:PF or l1tLayer1:PF
    fedIDs = cms.vuint32(0),
    splitFactor = cms.uint32(1),
    scoutingHeader = cms.bool(True)
)
process.p_pipeline += process.packer

process.unpacker = cms.EDProducer("l1sc::L1TScPhase2PuppiRawToDigi@alpaka",
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string(options.backend)
    ),
    src = cms.InputTag("packer"),
    streams = cms.vuint32(*[process.packer.fedIDs]),
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

# define table path
process.p_tables = cms.Path()

# attention here that I should check what is being dumped
process.candNanoAODTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
    name = cms.string("L1PF"),
    src = cms.InputTag("l1tLayer1Extended:PF"), # l1tLayer1Extended:PF or l1tLayer1:PF
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
    fileName = cms.untracked.string("msjCaseCNano_legacy.root"),
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring()),
    outputCommands = cms.untracked.vstring("drop *", "keep nanoaodFlatTable_*Table_*_*"),
    compressionLevel = cms.untracked.int32(4),
    compressionAlgorithm = cms.untracked.string("ZLIB"),
)
process.end = cms.EndPath(process.out)

# schedule
process.schedule = cms.Schedule(
    process.runPF,
    process.p_pipeline, 
    process.p_tables,
    process.end
)