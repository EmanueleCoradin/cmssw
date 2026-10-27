import FWCore.ParameterSet.Config as cms

from RecoTICL.PatternRecognition.hgCalCLUEsteringTrackstersProducer_cfi import hgCalCLUEsteringTrackstersProducer as _hgCalCLUEsteringTrackstersProducer
from RecoTICL.PatternRecognition.hgCalSoATrackstersProducer_cfi import hgCalSoATrackstersProducer as _hgCalSoATrackstersProducer
from RecoTICL.PatternRecognition.hgCalTrackstersFromSoAProducer_cfi import hgCalTrackstersFromSoAProducer as _hgCalTrackstersFromSoAProducer
from RecoTICL.Configuration.tracksterInference_cff import makeTracksterInference

# The layer clusters of all the HGCAL components, merged on the device by hgcalSoAMergeLayerClusters, are clustered
# into tracksters with CLUEstering, and the tracksters are built on the device

ticlCLUEstering = _hgCalCLUEsteringTrackstersProducer.clone(
    hgcalLayerClustersSoA = 'hgcalSoAMergeLayerClusters'
)

ticlSoATracksters = _hgCalSoATrackstersProducer.clone(
    hgcalLayerClustersSoA = 'hgcalSoAMergeLayerClusters',
    hgcalLayerClustersTrackstersSoA = 'ticlCLUEstering'
)

# CNN PID inference on the device, using features extracted from CLUEstering Tracksters.
ticlCLUEsteringCNNFeatures, ticlCLUEsteringCNNInference = makeTracksterInference(
    model="CNN", tracksters="ticlSoATracksters"
)
ticlCLUEsteringCNNInference.features = cms.InputTag("ticlCLUEsteringCNNFeatures", "features")

# Convert Tracksters with CNN PID predictions to the legacy format.
ticlTrackstersCLUEstering = _hgCalTrackstersFromSoAProducer.clone(
    src = 'ticlCLUEsteringCNNInference',
    originalMask = 'hgcalMergeLayerClusters:InitialLayerClustersMask'
)

ticlCLUEsteringStepTask = cms.Task(ticlCLUEstering
    ,ticlSoATracksters
    ,ticlCLUEsteringCNNFeatures
    ,ticlCLUEsteringCNNInference
    ,ticlTrackstersCLUEstering)
