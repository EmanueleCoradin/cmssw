#!/bin/bash
# to streamline debugging and testing of SoftTauTagging
# usage: bash L1TriggerScouting/TauTagging/test/runScoutingPhase2TauTagging.sh <backend> <num_events> <run_number>
# example usage: bash L1TriggerScouting/Phase2/test/runScoutingPhase2HeterogeneousW3Pi.sh cuda 10 37
function die { echo Failed $1: status $2 ; exit $2 ; }

SCRIPT="L1TriggerScouting/TauTagging/test/runScoutingPhase2TauTagging_validation_legacy.py"

if [ "$#" != "3" ]; then
    die "Need exactly 3 arguments: 1st ('cpu', 'cpu-dump', 'cuda', 'cuda-dump', 'rocm'), 2nd ('num_events'), 3rd ('unpacking', 'clustering', 'sorting', 'reshaping', 'tagging') got $#" 1
fi
if [[ "$1" =~ ^(cpu|cpu-dump|cuda|cuda-dump|rocm)$ ]]; then
    TARGET=$1
else
    die "Argument needs to be 'cpu', 'cpu-dump', 'cuda', 'cuda-dump' or 'rocm'; got '$1'" 1
fi
if [[ "$3" =~ ^(unpacking|clustering|sorting|reshaping|tagging)$ ]]; then
    TARGET=$1
else
    die "Argument needs to be 'unpacking', 'clustering', 'sorting', 'reshaping' or 'tagging'; got '$4'" 1
fi

if [ "${TARGET}" == "cpu" ]; then
  echo "Running CPU-only test"
  cmsRun "${SCRIPT}" reportEvery=1 maxEvents=$2 pipelineStep=$3 backend=serial_sync
elif [ "${TARGET}" == "cpu-dump" ]; then
  echo "Running CPU-only test with dumps"
  cmsRun "${SCRIPT}" reportEvery=1 maxEvents=$2 pipelineStep=$3 dump=candidates,cluster_indexes,soft_tau_inputs,soft_tau_outputs backend=serial_sync
elif [ "${TARGET}" == "cuda" ]; then
  echo "Running CUDA test"
  cmsRun "${SCRIPT}" reportEvery=1 maxEvents=$2 pipelineStep=$3 backend=cuda_async
elif [ "${TARGET}" == "cuda-dump" ]; then
  echo "Running CUDA test with dumps"
  cmsRun "${SCRIPT}" reportEvery=1 maxEvents=$2 pipelineStep=$3 dump=candidates,cluster_indexes,soft_tau_inputs,soft_tau_outputs backend=cuda_async
elif [ "${TARGET}" == "rocm" ]; then
  die "rocm not curently supported" 1 
fi