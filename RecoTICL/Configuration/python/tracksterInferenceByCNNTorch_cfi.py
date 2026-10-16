import FWCore.ParameterSet.Config as cms

tracksterInferenceByCNNTorch = cms.EDProducer(
    "TracksterInferenceByCNNTorch@alpaka",
    tracksters=cms.InputTag("ticlTrackstersToSoAProducer"),
    features=cms.InputTag("ticlTracksterInferenceFeaturesProducer", "features"),
    model=cms.FileInPath("RecoTICL/Inference/data/trackster_pid_cnn.pt"),
    batchSize=cms.uint32(64),
    convertToFP16=cms.bool(False),
    warmupIterations=cms.int32(3),
)
