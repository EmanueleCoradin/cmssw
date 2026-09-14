import FWCore.ParameterSet.Config as cms

hltPhase2PixelTrackTorchHighPuritySelector = cms.EDProducer('PixelTrackTorchHighPuritySelector@alpaka',
    pixelTrackSrc = cms.InputTag('hltPhase2PixelTracksSoA'),
    maxNumberOfTracks = cms.uint32(2*60*1024),
    maxPreselectedTracks = cms.uint32(9_984),
    minNumberOfHits = cms.uint32(0),
    avgHitsPerTrack = cms.uint32(8),
    minimumTrackQuality = cms.string('tight'),
    model = cms.FileInPath('RecoTracker/FinalTrackSelectors/data/PixelTrackTorchHighPuritySelector/pixel_track_classifier_FP16.pt'),
    scoreThreshold = cms.double(0.4),
    batchSize = cms.uint32(4_992)
)
