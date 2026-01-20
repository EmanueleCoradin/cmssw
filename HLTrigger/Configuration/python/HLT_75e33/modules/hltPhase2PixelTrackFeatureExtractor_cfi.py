import FWCore.ParameterSet.Config as cms

hltPhase2PixelTrackFeatureExtractor = cms.EDProducer('PixelTrackFeaturesExtractor@alpaka',
    pixelRecHitSrc = cms.InputTag('hltPhase2PixelRecHitsExtendedSoA'),
    pixelTrackSrc = cms.InputTag('hltPhase2PixelTracksSoA'),
    maxTracks = cms.int32(2*60*1024),
    maxTracksPreselection = cms.int32(10_000),
    minNumberOfHits = cms.int32(0),
    avgHitsPerTrack = cms.int32(8),
    minQuality = cms.string('tight'),
    model = cms.FileInPath('RecoTracker/FinalTrackSelectors/data/track_classifier_GPU.pt'),
    scoreThreshold = cms.double(0.65)
)
