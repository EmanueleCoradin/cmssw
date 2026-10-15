#!/bin/bash
# to streamline debugging and testing of SoftTauTagging
# usage: bash L1TriggerScouting/TauTagging/test/runScoutingPhase2TauTagging.sh <backend(-dump)> <num_events> <run_number> <step>
# example usage: bash L1TriggerScouting/Phase2/test/runScoutingPhase2TauTagging.sh cuda 10 40 clustering
function die { echo Failed $1: status $2 ; exit $2 ; }

SCRIPT="L1TriggerScouting/TauTagging/test/runScoutingPhase2TauTagging.py"
DATA="/dev/shm"

if [ "$#" != "6" ]; then
    die "Need exactly 6 arguments: 1st ('cpu', 'cpu-dump', 'cuda', 'cuda-dump'), 2nd ('num_events'), 3rd ('num_threads'), 4th ('num_streams'), 5rd ('run_number'), 6th ('unpacking', 'clustering', 'sorting', 'reshaping', 'tagging') got $#" 1
fi
if [[ "$1" =~ ^(cpu|cpu-dump|cuda|cuda-dump|rocm)$ ]]; then
    TARGET=$1
else
    die "Argument needs to be 'cpu', 'cpu-dump', 'cuda', 'cuda-dump' or 'rocm'; got '$1'" 1
fi
if [[ "$6" =~ ^(unpacking|clustering|sorting|reshaping|tagging)$ ]]; then
    TARGET=$1
else
    die "Argument needs to be 'unpacking', 'clustering', 'sorting', 'reshaping' or 'tagging'; got '$4'" 1
fi

if [ "${TARGET}" == "cpu" ]; then
  echo "Running CPU-only test"
  cmsRun "${SCRIPT}" backend=serial_sync buBaseDir=${DATA} fuBaseDir=${DATA} buNumStreams=2 maxEvents=$2 numThreads=$3 numFwkStreams=$4 runNumber=$5 pipelineStep=$6
elif [ "${TARGET}" == "cpu-dump" ]; then
  echo "Running CPU-only test with dumps"
  cmsRun "${SCRIPT}" backend=serial_sync buBaseDir=${DATA} fuBaseDir=${DATA} dump=candidates,cluster_indexes,soft_tau_inputs,soft_tau_outputs buNumStreams=2 maxEvents=$2 numThreads=$3 numFwkStreams=$4 runNumber=$5 pipelineStep=$6
elif [ "${TARGET}" == "cuda" ]; then
  echo "Running CUDA test"
  cmsRun "${SCRIPT}" backend=cuda_async buBaseDir=${DATA} fuBaseDir=${DATA} buNumStreams=2 maxEvents=$2 numThreads=$3 numFwkStreams=$4 runNumber=$5 pipelineStep=$6 
elif [ "${TARGET}" == "cuda-dump" ]; then
  echo "Running CUDA test with dumps"
  cmsRun "${SCRIPT}" backend=cuda_async buBaseDir=${DATA} fuBaseDir=${DATA} dump=candidates,cluster_indexes,soft_tau_inputs,soft_tau_outputs buNumStreams=2 maxEvents=$2 numThreads=$3 numFwkStreams=$4 runNumber=$5 pipelineStep=$6
elif [ "${TARGET}" == "rocm" ]; then
  die "rocm not curently supported" 1 
fi