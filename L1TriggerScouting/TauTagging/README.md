# How to run?

```
cmsrel CMSSW_16_0_0_pre1
cd CMSSW_16_0_0_pre1/src
cmsenv

git cms-checkout-topic -u lukaszmichalskii:tau-tagging/16_0_0_pre1/dev
git clone git@github.com:cms-patatrack/CLUEstering.git
cd CLUEstering && git submodule update --init && cd ..

scram b -j

# run non-scouting analysis
cmsRun L1TriggerScouting/TauTagging/test/runTauTagging.py -b cuda_async -e 1 -ne 1

# run scouting clustering
cmsRun L1TriggerScouting/TauTagging/test/runTauTagging.py -b cuda_async -e 1 -ne 1 -o clustering -bbd <path_to_data_dir> -bns 1 -scout
```