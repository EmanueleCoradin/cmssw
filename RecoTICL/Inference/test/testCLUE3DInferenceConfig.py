"""Run in CMSSW, once with the default configuration and once with ticl_dev."""
import sys
import FWCore.ParameterSet.Config as cms
from Configuration.ProcessModifiers.ticl_dev import ticl_dev

process = cms.Process("TEST", *([ticl_dev] if "--ticl-dev" in sys.argv else []))
process.load("RecoTICL.Configuration.CLUE3DHighStep_cff")
expected = {
    "ticlSeedingGlobal", "filteredLayerClustersCLUE3DHigh", "ticlTrackstersCLUE3DHigh",
    "ticlTrackstersToSoAProducer", "ticlTracksterInferenceFeaturesProducer",
    "ticlTracksterInferenceProducer",
}
if "--ticl-dev" in sys.argv:
    expected.add("ticlTrackstersCLUEsteringAssignment")
assert process.ticlCLUE3DHighStepTask.moduleNames() == expected
producer = process.ticlTracksterInferenceProducer
assert producer.type_() == "TracksterCNNInferenceProducer@alpaka"
assert producer.features == cms.InputTag("ticlTracksterInferenceFeaturesProducer", "features")
assert producer.tracksters == cms.InputTag("ticlTrackstersToSoAProducer")
assert producer.doPID.value() and not producer.doRegression.value()
assert producer.pidModel.value() == "RecoTICL/Inference/data/trackster_pid_cnn.pt"
assert producer.batchSize.value() == 64
assert producer.warmupIterations.value() == 3
assert not producer.convertToFP16.value()
assert process.ticlTracksterInferenceFeaturesProducer.minClusterEnergy.value() == 1.0
assert process.ticlTrackstersToSoAProducer.src == cms.InputTag("ticlTrackstersCLUE3DHigh")
assert process.ticlTrackstersCLUE3DHigh.inferenceAlgo.value() == "TracksterInferenceByCNN"
assert not hasattr(process, "ticlTracksterPIDComparison")
process.path = cms.Path(process.ticlCLUE3DHighStepTask)
namespace = {}
exec(process.dumpPython(), namespace)
assert namespace["process"].ticlCLUE3DHighStepTask.moduleNames() == expected
process.load("RecoTICL.Configuration.tracksterPIDComparisonAnalyzer_cfi")
assert process.ticlTracksterPIDComparison.candidate == cms.InputTag("ticlTracksterInferenceProducer")
print("CLUE3D inference configuration and serialization OK", sys.argv[1:])
