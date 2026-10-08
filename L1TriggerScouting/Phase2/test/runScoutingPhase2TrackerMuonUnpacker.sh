#!/bin/bash
# to streamline debugging and testing Unpackers
# usage: bash L1TriggerScouting/Phase2/test/runScoutingPhase2TrackerMuonUnpacker.sh <backend> <num_events> <run_number> <unpacker_type>
# example usage: bash L1TriggerScouting/Phase2/test/runScoutingPhase2TrackerMuonUnpacker.sh cuda 10000 70 legacy
function die { echo Failed $1: status $2 ; exit $2 ; }

SCRIPT="L1TriggerScouting/Phase2/test/runScoutingPhase2TrackerMuonUnpacker.py"
DATA="/mnt/ramdisk/gizago/raw"

if [ "$#" != "4" ]; then
    die "Need exactly 3 arguments: 1st ('cpu', 'cuda', or 'rocm'), 2nd ('num_events'), 3rd ('run_number'), 4th ('legacy', 'alpaka') got $#" 1
fi
if [[ "$1" =~ ^(cpu|cuda)$ ]]; then
    TARGET=$1
else
    die "Argument needs to be 'cpu', 'cuda'; got '$1'" 1
fi
if [[ "$4" =~ ^(legacy|alpaka)$ ]]; then
    TARGET=$1
else
    die "Argument needs to be 'legacy', 'alpaka'; got '$4'" 1
fi

if [ "${TARGET}" == "cpu" ]; then
  echo "Running CPU-only test"
  cmsRun "${SCRIPT}" runNumber=$3 buBaseDir=${DATA} fuBaseDir=${DATA} buNumStreams=1 maxEvents=$2 unpackerType=$4 backend=serial_sync
elif [ "${TARGET}" == "cuda" ]; then
  echo "Running GPU-only test"
  cmsRun "${SCRIPT}" runNumber=$3 buBaseDir=${DATA} fuBaseDir=${DATA} buNumStreams=1 maxEvents=$2 unpackerType=$4 backend=cuda_async
fi