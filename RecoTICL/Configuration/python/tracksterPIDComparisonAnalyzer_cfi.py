import FWCore.ParameterSet.Config as cms


tracksterPIDComparisonAnalyzer = cms.EDAnalyzer(
    "TracksterPIDComparisonAnalyzer",
    reference=cms.InputTag(""),
    candidate=cms.InputTag(""),
    absoluteTolerance=cms.double(1.e-5),
    relativeTolerance=cms.double(1.e-4),
    maxMismatchesToPrint=cms.uint32(10),
    failOnMismatch=cms.bool(False),
    reportEveryEvent=cms.bool(True),
)
