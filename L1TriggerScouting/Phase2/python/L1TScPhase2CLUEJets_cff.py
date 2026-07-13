import FWCore.ParameterSet.Config as cms

L1TScPhase2CLUEJetsProducer = cms.EDProducer(
    "l1sc::L1TScPhase2CLUEJets@alpaka",
    density_radius = cms.double(0.2), 
    min_density = cms.double(5.0), 
    outlier_distance = cms.double(0.4), 
    wrapCoords = cms.bool(True)
)