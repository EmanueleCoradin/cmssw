import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras
process = cms.Process("ScoutingPhase2ClusteringValidation", eras.Phase2C17I13M9)

# enable alpaka and GPU support
process.load("Configuration.StandardSequences.Accelerators_cff")

from PhysicsTools.NanoAOD.common_cff import Var, ExtVar
def LazyVar(expr, valtype, doc=None, precision=-1):
    return Var(expr, valtype, doc, precision, lazyEval=True)

# import l1 scouting options
from L1TriggerScouting.Phase2.options_cff import options, VarParsing

# extra options
options.register ("reportEvery",
    10, # default value
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.int, 
    "Print message after the specified number of events (proper events in this case)"
)

# parse arguments
options.parseArguments()

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

# FP inputs packing/unpacking
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

# CLUEstering producer
process.load(
    "L1TriggerScouting.Phase2.L1TScPhase2CLUEJets_cff"
)
process.L1TScPhase2CLUEJetsProducer.alpaka = cms.untracked.PSet(
    backend = cms.untracked.string(options.backend)
)
process.L1TScPhase2CLUEJetsProducer.candidates = cms.InputTag("unpacker", "candidates")
process.L1TScPhase2CLUEJetsProducer.bxSizes = cms.InputTag("unpacker", "bxSizes")

# define clustering path
process.p_clustering = cms.Path(
    process.packer +
    process.unpacker +
    process.L1TScPhase2CLUEJetsProducer
)
process.p_clustering.associate(process.l1tEmulation)

# NanoAOD flat table producers
process.candTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
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
process.clusterTable = cms.EDProducer("ClusterSoAToNanoAODFlatTable",
    srcClusters = cms.InputTag("L1TScPhase2CLUEJetsProducer", "clusters"), 
    clustering_name = cms.string("CLUEstering"),
    name = cms.string("L1PF"), 
    extension = cms.bool(True) # extends candTable, set same name as candTable
)

process.p_tables = cms.Path(
    process.candTable + 
    process.clusterTable
)

# output
process.out = cms.OutputModule("NanoAODOutputModule",
    fileName = cms.untracked.string("plainNano.root"),
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring()),
    outputCommands = cms.untracked.vstring("drop *", "keep nanoaodFlatTable_*Table_*_*"),
    compressionLevel = cms.untracked.int32(4),
    compressionAlgorithm = cms.untracked.string("ZLIB"),
)

process.end = cms.EndPath(process.out)

# schedule
process.schedule = cms.Schedule(
    process.p_clustering, 
    process.p_tables,
    process.end
)