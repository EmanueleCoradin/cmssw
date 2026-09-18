import FWCore.ParameterSet.Config as cms

tracksterInferenceFeaturesProducer = cms.EDProducer(
    "TracksterInferenceFeaturesProducer@alpaka",
    tracksters = cms.InputTag("ticlTrackstersToSoAProducer"),
    layerClusters = cms.VInputTag(
        "hgcalSoALayerClustersEE",
        "hgcalSoALayerClustersHSi",
        "hgcalSoALayerClustersHSci"),
    detector = cms.string("HGCAL"),
    minClusterEnergy = cms.double(1.0),
)
