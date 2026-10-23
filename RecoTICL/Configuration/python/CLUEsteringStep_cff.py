import FWCore.ParameterSet.Config as cms

from RecoTICL.PatternRecognition.hgCalCLUEsteringTrackstersProducer_cfi import hgCalCLUEsteringTrackstersProducer as _hgCalCLUEsteringTrackstersProducer
from RecoTICL.PatternRecognition.hgCalSoATrackstersProducer_cfi import hgCalSoATrackstersProducer as _hgCalSoATrackstersProducer
from RecoTICL.PatternRecognition.hgCalTrackstersFromSoAProducer_cfi import hgCalTrackstersFromSoAProducer as _hgCalTrackstersFromSoAProducer

# The layer clusters of all the HGCAL components, merged on the device by hgcalSoAMergeLayerClusters, are clustered
# into tracksters with CLUEstering, and the tracksters are built on the device

ticlCLUEstering = _hgCalCLUEsteringTrackstersProducer.clone(
    hgcalLayerClustersSoA = 'hgcalSoAMergeLayerClusters'
)

ticlSoATracksters = _hgCalSoATrackstersProducer.clone(
    hgcalLayerClustersSoA = 'hgcalSoAMergeLayerClusters',
    hgcalLayerClustersTrackstersSoA = 'ticlCLUEstering'
)

# Conversion to legacy tracksters
ticlTrackstersCLUEstering = _hgCalTrackstersFromSoAProducer.clone(
    src = 'ticlSoATracksters',
    originalMask = 'hgcalMergeLayerClusters:InitialLayerClustersMask'
)

ticlCLUEsteringStepTask = cms.Task(ticlCLUEstering
    ,ticlSoATracksters
    ,ticlTrackstersCLUEstering)
