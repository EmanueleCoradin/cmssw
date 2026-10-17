import unittest
import FWCore.ParameterSet.Config as cms
from RecoTICL.Configuration.stageInference_cff import (
    makeCandidateInference, withCandidateInference, withSuperclusteringInference,
)


class StageInferenceConfigTest(unittest.TestCase):
    def test_candidate_models_and_independent_tasks(self):
        for family in ("CNN", "DNN", "PFN"):
            for pid, regression in ((True, False), (False, False), (False, True), (True, True)):
                if family == "CNN" and regression:
                    with self.assertRaises(ValueError):
                        makeCandidateInference(model=family, doPID=pid, doRegression=regression)
                    continue
                config = makeCandidateInference(model=family, doPID=pid, doRegression=regression)
                self.assertEqual(config.modelFamily.value(), family)
                self.assertEqual(config.doPID.value(), pid)
                self.assertEqual(config.doRegression.value(), regression)
                self.assertIn("candidate_pid_" + family.lower(), config.pidModel.value())
                self.assertEqual(config.minClusterEnergy.value(), 2.5)

    def test_candidate_clone_keeps_products_and_inputs(self):
        original = cms.EDProducer("TICLCandidateProducer", inferenceAlgo=cms.string("TracksterInferenceByPFN"),
                                  regressionAndPid=cms.bool(True), layer_clusters=cms.InputTag("clusters"),
                                  original_masks=cms.VInputTag("mask"))
        result = withCandidateInference(original, backend="cuda_async", pidModel="pid.pt", regressionModel="energy.pt")
        self.assertEqual(result.type_(), original.type_())
        self.assertEqual(result.layer_clusters, original.layer_clusters)
        self.assertEqual(result.original_masks, original.original_masks)
        self.assertEqual(result.inferenceAlgo.value(), "PyTorchAlpaka")
        self.assertEqual(original.inferenceAlgo.value(), "TracksterInferenceByPFN")
        self.assertEqual(result.pluginInferenceAlgoPyTorchAlpaka.pidModel.value(), "pid.pt")
        self.assertEqual(result.pluginInferenceAlgoPyTorchAlpaka.regressionModel.value(), "energy.pt")

    @staticmethod
    def linking():
        return cms.EDProducer("TracksterLinksProducer", linkingPSet=cms.PSet(
            type=cms.string("SuperClusteringDNN"), onnxModelPath=cms.string("reference.onnx"),
            nnWorkingPoint=cms.double(0.57247), deltaEtaWindow=cms.double(0.1), filterByTracksterPID=cms.bool(True)))

    def test_pair_versions_and_batch_units(self):
        original = self.linking()
        for version, features in (("v1", 9), ("v2", 17), ("v3", 18)):
            result = withSuperclusteringInference(original, version=version, batchSize=7, backend="rocm_async")
            self.assertEqual(result.type_(), original.type_())
            self.assertEqual(result.linkingPSet.inferenceBatchSize.value(), 7 * features)
            self.assertEqual(result.linkingPSet.pytorchInference.batchSize.value(), 7)
            for key in ("nnWorkingPoint", "deltaEtaWindow", "filterByTracksterPID", "onnxModelPath"):
                self.assertEqual(getattr(result.linkingPSet, key), getattr(original.linkingPSet, key))
            self.assertIn(version, result.linkingPSet.pytorchInference.modelPath.value())
        self.assertFalse(hasattr(original.linkingPSet, "inferenceBackend"))

    def test_restore_onnx(self):
        torch = withSuperclusteringInference(self.linking(), modelPath="pair.pt")
        onnx = withSuperclusteringInference(torch, engine="ONNX")
        self.assertEqual(onnx.linkingPSet.inferenceBackend.value(), "ONNX")
        self.assertEqual(onnx.linkingPSet.onnxModelPath.value(), "reference.onnx")

    def test_disabled_models_need_no_files(self):
        config = makeCandidateInference(doPID=False, doRegression=False, pidModel="", regressionModel="")
        self.assertEqual(config.pidModel.value(), "")
        self.assertEqual(config.regressionModel.value(), "")

    def test_validation(self):
        for options in (dict(batchSize=0), dict(batchSize=True), dict(backend="automatic"),
                        dict(deviceIndex=-1), dict(warmupIterations=-1), dict(minClusterEnergy=float("nan")),
                        dict(pidModel=""), dict(regressionModel=""), dict(model="unknown")):
            with self.assertRaises((ValueError, TypeError)):
                makeCandidateInference(**options)
        for options in (dict(version="v4"), dict(batchSize=2147483647), dict(engine="unknown"), dict(modelPath="")):
            with self.assertRaises(ValueError):
                withSuperclusteringInference(self.linking(), **options)


if __name__ == "__main__":
    unittest.main()
