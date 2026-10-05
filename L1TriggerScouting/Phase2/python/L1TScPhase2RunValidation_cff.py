"""Common PoolSource, L1 emulation, and PUPPI packing setup for validation.

Create the process with eras.Phase2C17I13M9 and parse Phase2.options_cff (or
options derived from it), then call::

    from L1TriggerScouting.Phase2.L1TScPhase2RunValidation_cff import setupPhase2Validation
    setupPhase2Validation(process, options)

Configure the unpacker with src=cms.InputTag("packer"), streams equal to
process.packer.fedIDs, and splitFactor equal to process.packer.splitFactor.
Schedule process.packer before the unpacker and associate process.l1tEmulation
with their path. For example, once process.unpacker has been defined::

    process.p_pipeline = cms.Path(process.packer + process.unpacker)
    process.p_pipeline.associate(process.l1tEmulation)

options.inputFiles overrides the reference FP-input dataset. The source, packer,
and GlobalTag can also be customized on the process after setup. Importing this
module does not parse arguments or create a process.
"""

import FWCore.ParameterSet.Config as cms


def setupPhase2Validation(process, options):
    """Configure a validation process from parsed scouting options and return it."""
    process.load("Configuration.StandardSequences.Accelerators_cff")
    process.load("FWCore.MessageService.MessageLogger_cfi")
    process.MessageLogger.cerr.FwkReport.reportEvery = getattr(options, "reportEvery", 100)
    process.options.numberOfThreads = cms.untracked.uint32(options.numThreads)
    process.options.numberOfStreams = cms.untracked.uint32(options.numFwkStreams)
    process.options.numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1)
    process.options.wantSummary = cms.untracked.bool(True)
    process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(options.maxEvents))

    # Reconstruct the PUPPI candidates consumed by ScPhase2PuppiPacker.
    process.load("Configuration.Geometry.GeometryExtendedRun4D110Reco_cff")
    process.load("Configuration.Geometry.GeometryExtendedRun4D110_cff")
    process.load("Configuration.StandardSequences.MagneticField_cff")
    process.load("Configuration.StandardSequences.SimL1Emulator_cff")
    process.load("SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff")
    process.load("SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi")
    process.load("SimGeneral.MixingModule.mixNoPU_cfi")
    process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

    from Configuration.AlCa.GlobalTag import GlobalTag
    process.GlobalTag = GlobalTag(process.GlobalTag, "141X_mcRun4_realistic_v3", "")
    process.l1tTrackSelectionProducer.processSimulatedTracks = False

    process.l1tEmulation = cms.Task(
        process.l1tSAMuonsGmt,
        process.l1tPhase2L1CaloEGammaEmulator,
        process.l1tPhase2CaloPFClusterEmulator,
        process.l1tPhase2GCTBarrelToCorrelatorLayer1Emulator,
        process.L1TLayer1TaskInputsTask,
        process.L1TLayer1Task,
        process.l1tLayer2EG,
        process.L1TPFJetsEmulationTask,
        process.L1TPFJetsExtendedTask,
        process.L1TBJetsTask,
        process.l1tPhase2CaloToCorrelatorTM18,
    )

    fileNames = options.inputFiles or [
        "file:/eos/cms/store/cmst3/group/l1tr/vcamagni/L1TauID/DATA/FPinputs/m90/4STEPS/"
        f"142Xv0/inputs140X_7099351_{i}.root" for i in range(4000)
    ]
    process.source = cms.Source("PoolSource",
        fileNames = cms.untracked.vstring(*fileNames),
    )
    process.packer = cms.EDProducer("ScPhase2PuppiPacker",
        src = cms.InputTag("l1tLayer1", "PF"),
        fedIDs = cms.vuint32(0),
        splitFactor = cms.uint32(1),
        scoutingHeader = cms.bool(True),
    )
    return process
