import unittest
import re
from pathlib import Path
import FWCore.ParameterSet.Config as cms
from RecoTICL.Configuration.tracksterInference_cff import makeTracksterInference


class ConfigurationTest(unittest.TestCase):
    def test_families_and_tasks(self):
        for family in ("CNN", "DNN", "PFN"):
            for pid, regression in ((True, False), (False, False), (False, True), (True, True)):
                if family == "CNN" and regression:
                    with self.assertRaisesRegex(ValueError, "CNN supports PID only"):
                        makeTracksterInference(model=family, doPID=pid, doRegression=regression)
                    continue
                process = cms.Process("TEST")
                config = makeTracksterInference(model=family, doPID=pid, doRegression=regression)
                config.install(process, "inference")
                self.assertEqual(config.output.getModuleLabel(), "inference")
                self.assertEqual(process.inference.features.getModuleLabel(), "inferenceFeatures")
                self.assertEqual(process.inference.features.getProductInstanceLabel(), "features")
                self.assertEqual(process.inference.doPID.value(), pid)
                self.assertEqual(process.inference.doRegression.value(), regression)
                process.path = cms.Path(config.task)
                dumped = process.dumpPython()
                namespace = {}
                exec(dumped, namespace)
                self.assertEqual(namespace["process"].inference.type_(), "Trackster" + family + "InferenceProducer@alpaka")

    def test_factory_plugins_are_registered(self):
        # Source check: runtime plugin discovery additionally requires a CMSSW build.
        plugins = Path(__file__).resolve().parents[1] / "plugins" / "alpaka"
        sources = "\n".join(p.read_text() for p in plugins.glob("*.cc"))
        registrations = re.findall(r"DEFINE_FWK_ALPAKA_MODULE\((\w+)\)", sources)
        for family in ("CNN", "DNN", "PFN"):
            config = makeTracksterInference(model=family)
            for module in (config.features, config.producer):
                self.assertEqual(registrations.count(module.type_().removesuffix("@alpaka")), 1)
        self.assertNotIn("TracksterInferenceProducer", registrations)

    def test_independent_instances(self):
        process = cms.Process("TEST")
        a = makeTracksterInference(model="DNN").install(process, "a")
        b = makeTracksterInference(model="PFN", batchSize=17).install(process, "b")
        self.assertEqual(a.producer.features.getModuleLabel(), "aFeatures")
        self.assertEqual(b.producer.features.getModuleLabel(), "bFeatures")
        self.assertEqual(a.producer.batchSize.value(), 64)
        with self.assertRaises(ValueError):
            a.install(process, "c")
        with self.assertRaises(ValueError):
            makeTracksterInference().install(process, "a")

    def test_install_clones_templates(self):
        config = makeTracksterInference(model="PFN")
        template = config.producer
        process = cms.Process("TEST")
        config.install(process, "inference")
        self.assertEqual(template.features.getModuleLabel(), "")
        self.assertEqual(config.producer.features.getModuleLabel(), "inferenceFeatures")

    def test_errors_and_overrides(self):
        for kwargs in ({"model": "bad"}, {"batchSize": 0}, {"batchSize": True},
                       {"warmupIterations": -1}, {"layerClusters": []}, {"pidModel": ""}):
            with self.assertRaises((ValueError, TypeError)):
                makeTracksterInference(**kwargs)
        config = makeTracksterInference(model="PFN", pidModel="future/pid.pt", regressionModel="future/energy.pt")
        self.assertEqual(config.producer.pidModel.value(), "future/pid.pt")
        with self.assertRaises(ValueError):
            _ = config.output


if __name__ == "__main__":
    unittest.main()
