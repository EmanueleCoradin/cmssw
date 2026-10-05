"""Common framework and DAQ input setup for Phase-2 scouting jobs.

Call after parsing Phase2.options_cff (or options derived from it)::

    from L1TriggerScouting.Phase2.L1TScPhase2RunScouting_cff import setupPhase2Scouting
    setupPhase2Scouting(process, options)

Then configure and schedule the desired unpacker with src="rawDataCollector".
Importing this module does not parse arguments or create directories. The setup
call creates the FU/BU run directories and the first BU's fu.lock file, following
the standalone scouting test configurations. Source settings can be overridden
on process.source afterwards.
"""

import os
import FWCore.ParameterSet.Config as cms

def setupPhase2Scouting(process, options):
    """Configure a process from parsed scouting options and return it."""
    if not options.buBaseDir or len(options.buNumStreams) != len(options.buBaseDir):
        raise ValueError("Specify one buNumStreams entry per buBaseDir (at least one BU directory)")
    if any(n <= 0 for n in options.buNumStreams):
        raise ValueError("Each buNumStreams entry must be positive")

    useFileBroker = options.broker != "none"
    brokerHost, brokerPort = "htcp40.cern.ch", "8080"
    if useFileBroker:
        brokerHost, separator, brokerPort = options.broker.rpartition(":")
        if not separator or not brokerHost or not brokerPort.isdecimal() or not 0 < int(brokerPort) < 65536:
            raise ValueError("broker must be 'none' or 'hostname:port' with a valid port")

    process.load("Configuration.StandardSequences.Accelerators_cff")
    process.load("FWCore.MessageService.MessageLogger_cfi")
    process.MessageLogger.cerr.FwkReport.reportEvery = getattr(options, "reportEvery", 100)
    process.options.numberOfThreads = cms.untracked.uint32(options.numThreads)
    process.options.numberOfStreams = cms.untracked.uint32(options.numFwkStreams)
    process.options.numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1)
    process.options.wantSummary = cms.untracked.bool(True)
    process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(options.maxEvents))

    process.EvFDaqDirector = cms.Service("EvFDaqDirector",
        useFileBroker = cms.untracked.bool(useFileBroker),
        fileBrokerHostFromCfg = cms.untracked.bool(False),
        fileBrokerHost = cms.untracked.string(brokerHost),
        fileBrokerPort = cms.untracked.string(brokerPort),
        runNumber = cms.untracked.uint32(options.runNumber),
        baseDir = cms.untracked.string(options.fuBaseDir),
        buBaseDir = cms.untracked.string(options.buBaseDir[0]),
        buBaseDirsAll = cms.untracked.vstring(*options.buBaseDir),
        buBaseDirsNumStreams = cms.untracked.vint32(*options.buNumStreams),
        directorIsBU = cms.untracked.bool(False),
    )
    # DAQSource requires monitoring when it obtains files through the broker.
    if useFileBroker and not hasattr(process, "FastMonitoringService"):
        process.FastMonitoringService = cms.Service("FastMonitoringService")

    runDir = "run%06d" % options.runNumber
    fuDir = os.path.join(options.fuBaseDir, runDir)
    buDirs = [os.path.join(baseDir, runDir) for baseDir in options.buBaseDir]
    for directory in [fuDir] + buDirs:
        os.makedirs(directory, exist_ok=True)
    with open(os.path.join(buDirs[0], "fu.lock"), "a"):
        pass

    process.source = cms.Source("DAQSource",
        testing = cms.untracked.bool(True),
        dataMode = cms.untracked.string(options.daqSourceMode),
        verifyChecksum = cms.untracked.bool(True),
        useL1EventID = cms.untracked.bool(False),
        # Keep each read(2) request well below Linux's per-call transfer limit.
        eventChunkBlock = cms.untracked.uint32(64),
        eventChunkSize = cms.untracked.uint32(8 * 1024),
        maxChunkSize = cms.untracked.uint32(16 * 1024),
        numBuffers = cms.untracked.uint32(4),
        maxBufferedFiles = cms.untracked.uint32(4),
        fileListMode = cms.untracked.bool(not useFileBroker),
        fileNames = cms.untracked.vstring(
            os.path.join(buDirs[0], "run%06d_ls%04d_index%06d_stream00.raw" %
                         (options.runNumber, options.lumiNumber, 1)),
        ),
    )
    return process
