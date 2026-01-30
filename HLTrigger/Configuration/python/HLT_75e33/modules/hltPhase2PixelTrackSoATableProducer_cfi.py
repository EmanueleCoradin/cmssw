import FWCore.ParameterSet.Config as cms

hltPhase2PixelTrackSoATableProducer = cms.EDProducer("HLTPixelTrackSoATableProducer",
    #trackSrc = cms.InputTag("hltPhase2PixelTracksSoA"),
    trackSrc = cms.InputTag("hltPhase2PixelTrackFeatureExtractor"),
)
