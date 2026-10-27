import FWCore.ParameterSet.Config as cms

hltPhase2PixelTrackONNXHighPuritySelector = cms.EDProducer('PixelTrackONNXHighPuritySelector@alpaka',
    pixelTrackSrc = cms.InputTag('hltPhase2PixelTracksSoA'),
    maxNumberOfTracks = cms.int32(2*60*1024),
    maxPreselectedTracks = cms.int32(9_984),
    minNumberOfHits = cms.int32(0),
    avgHitsPerTrack = cms.int32(8),
    minimumTrackQuality = cms.string('tight'),
    model = cms.FileInPath('RecoTracker/FinalTrackSelectors/data/pixel_track_classifier_pf16_fallback.onnx'),
    backend = cms.string("tensorRT"),
    scoreThreshold = cms.double(0.4),
    #featureMajor=False,
    batchSize = cms.int32(4_992),
)

