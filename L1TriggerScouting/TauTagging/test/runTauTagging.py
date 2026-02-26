import os
import FWCore.ParameterSet.Config as cms
from IOPool.Input.modules import PoolSource
from L1TriggerScouting.TauTagging.options_cff import parse_args
from L1TriggerScouting.TauTagging.modules import l1sc_TauTaggingSink

args = parse_args()
process = cms.Process("L1TScPhase2TauTagging")

# summary
process.options.wantSummary = cms.untracked.bool(args.wantSummary)

# enable multithreading
process.options.numberOfThreads = args.numberOfThreads if args.numberOfThreads > 1 else 1 
process.options.numberOfStreams = args.numberOfStreams if args.numberOfStreams > 1 else 1 
process.maxEvents.input = args.numberOfEvents if args.numberOfEvents > 1 else 1 

# enable alpaka and GPU support
process.load("Configuration.StandardSequences.Accelerators_cff")

# logging configuration
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 10

# define path
process.path = cms.Path()

# data source both for scouting and non-scouting scenarios
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
        eventChunkBlock = cms.untracked.uint32(2 * 1024),
        eventChunkSize = cms.untracked.uint32(2 * 1024),
        maxChunkSize = cms.untracked.uint32(4 * 1024),
        numBuffers = cms.untracked.uint32(4),
        maxBufferedFiles = cms.untracked.uint32(4),
        fileListMode = cms.untracked.bool(args.broker == "none"),
        fileNames = cms.untracked.vstring(
            buDirs[0] + "/" + "run%06d_ls%04d_index%06d_stream00.raw" % (args.runNumber, args.lumiNumber, 1),
        )
    )
    os.system("touch " + buDirs[0] + "/" + "fu.lock")

else:
    # pool source
    process.source = cms.Source("PoolSource",
        fileNames = cms.untracked.vstring(f'file:/eos/cms/store/cmst3/group/l1tr/vcamagni/L1TauID/DATA/FPinputs/m90/4STEPS/142Xv0/inputs140X_7099351_{i}.root' for i in range(4000)),
    )

    # extra configs
    process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
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

    # run correlator trigger
    process.deps = cms.Task(
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
        process.L1TBJetsTask
    )    

    process.l1tLayer1HGCalAll = cms.EDProducer("L1TPFCandMultiMerger",
        pfProducers = cms.VInputTag(
            cms.InputTag("l1tLayer1HGCal"),
            cms.InputTag("l1tLayer1HGCalNoTK"),
        )
    )
    process.deps.add(process.l1tLayer1HGCalAll)

    # associate dependencies with main path
    process.path.associate(process.deps)

# PFCandidates
if args.runScouting:
    from L1TriggerScouting.Phase2.modules import l1sc_L1TScPhase2PuppiRawToDigi_alpaka
    process.PFCandidatesProducer = l1sc_L1TScPhase2PuppiRawToDigi_alpaka(
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string(args.backend)
        ),
        streams = cms.vuint32(*list(range(sum(args.buNumStreams))) if args.streams == [] else args.streams),
        splitFactor = cms.uint32(sum(args.buNumStreams) if args.streams == [] else len(args.streams)),
        src = cms.InputTag('rawDataCollector'),
        environment = cms.untracked.int32(args.environment),
    )
    process.path += process.PFCandidatesProducer

    if "candidates" in args.dump in args.dump:
        from L1TriggerScouting.Phase2.modules import PFSoAToOrbitFlatTable
        process.PFToOrbit = PFSoAToOrbitFlatTable(
            srcBx = cms.InputTag("PFCandidatesProducer", "bxLookup"), 
            srcPF = cms.InputTag("PFCandidatesProducer", "candidates"),
            name = "L1PF"
        )
        process.path += process.PFToOrbit

else:
    from L1TriggerScouting.TauTagging.modules import l1sc_PFCandidateAoSToSoA_alpaka
    process.PFCandidatesProducer = l1sc_PFCandidateAoSToSoA_alpaka(
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string(args.backend)
        ),
        src = cms.InputTag("l1tLayer1Extended", "PF")
    )
    process.path += process.PFCandidatesProducer

# CLUEstering
if args.runScouting:
    if "clustering" in args.only or "tagging_pre" in args.only or "tagging_inf" in args.only:
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

        if "clusters" in args.dump or "all" in args.dump:
            from L1TriggerScouting.Phase2.modules import ClusterToOrbitFlatTable
            process.CLUEToOrbitTable = ClusterToOrbitFlatTable(
                srcCandidates = cms.InputTag("PFCandidatesProducer", "candidates"),
                srcBxCandidatesMap = cms.InputTag("PFCandidatesProducer", "bxLookup"),
                srcBxClustersMap = cms.InputTag("CLUETaus", "bxClustersMap"),
                srcClustersCandsMap = cms.InputTag("CLUETaus", "clustersCandsMap"), 
                nameCandidates = "L1PF", 
                nameClusters = "CLUEClusters",
                doc = ""
            )
            process.path += process.CLUEToOrbitTable
else:
    if "clustering" in args.only:
        from L1TriggerScouting.TauTagging.modules import l1sc_CLUEJetsProducer_alpaka
        process.CLUETaus = l1sc_CLUEJetsProducer_alpaka(
            alpaka = cms.untracked.PSet(
                backend = cms.untracked.string(args.backend)
            ),
            candidates = cms.InputTag("PFCandidatesProducer", "candidates"),
            dc = cms.double(args.dc),
            rhoc = cms.double(args.rhoc),
            dm = cms.double(args.dm),
            wrapCoords = cms.bool(args.wrapCoords)
        )
        process.path += process.CLUETaus

        if "clusters" in args.dump or "all" in args.dump:
            from L1TriggerScouting.Phase2.modules import ClusterToFlatTable
            process.CLUEToTable = ClusterToFlatTable(
                srcCandidates = cms.InputTag("PFCandidatesProducer", "candidates"),
                srcClustersCandsMap = cms.InputTag("CLUETaus", "clustersCandsMap"), 
                nameCandidates = "L1PF", 
                nameClusters = "CLUEClusters",
                doc = ""
            )
            process.path += process.CLUEToTable

# Tagging
if "tagging_pre" in args.only or "tagging_inf" in args.only:
    if "tagging_pre" in args.only:
        do_inference = cms.bool(False)
    
    if "tagging_inf" in args.only:
        do_inference = cms.bool(True)

    from L1TriggerScouting.TauTagging.modules import l1sc_SoftTauIdML_alpaka
    process.SoftTauId = l1sc_SoftTauIdML_alpaka(
        alpaka = cms.untracked.PSet(
            backend = cms.untracked.string(args.backend)
        ),
        pf = 'PFCandidatesProducer',
        clusters = 'CLUETaus',
        model = cms.FileInPath(args.model),
        do_inference = do_inference,
        maxBatchSize = cms.uint32(150)
    )
    process.path += process.SoftTauId

    if "tagging_inf" in args.dump or "all" in args.dump or "all_nocands" in args.dump:
        from L1TriggerScouting.Phase2.modules import TaggerOutToOrbitFlatTable
        process.TaggerOutToOrbit = TaggerOutToOrbitFlatTable(
            srcClusters = "CLUETaus", 
            srcCandidates = "PFCandidatesProducer", 
            srcOut = "SoftTauId", 
            name = "TaggerOut", 
            doc = ""
        )
        process.path += process.TaggerOutToOrbit

if args.dump != "none":
    if args.runScouting:
        process.out = cms.OutputModule("OrbitNanoAODOutputModule",
            fileName = cms.untracked.string("ScoutCLUETaus_run000043.root"),
            SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring()),  # keep all events
            outputCommands = cms.untracked.vstring(
                "drop *",
                "keep l1ScoutingRun3OrbitFlatTable_*_*_*",
                "keep nanoaodFlatTable_*Table_*_*"  
            )
        )   
        process.end = cms.EndPath(process.out)
    else:
        process.out = cms.OutputModule("NanoAODOutputModule",
            fileName = cms.untracked.string("ScoutCLUETaus_debug.root"),
            outputCommands = cms.untracked.vstring("drop *", "keep nanoaodFlatTable_*Table_*_*"),
            compressionLevel = cms.untracked.int32(4),
            compressionAlgorithm = cms.untracked.string("ZLIB"),
        )
        process.end = cms.EndPath(process.out)