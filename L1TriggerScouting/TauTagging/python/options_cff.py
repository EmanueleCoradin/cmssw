import argparse
from enum import IntEnum

class Step(IntEnum):
    UNPACKING = 1
    CLUSTERING = 2
    SORTING = 3
    RESHAPING = 4
    TAGGING = 5

def parse_step(value: str) -> Step:
try:
    return Step[value.upper()]
except KeyError:
    valid = ", ".join(s.name.lower() for s in Step)
    raise argparse.ArgumentTypeError(
        f"Invalid step '{value}'. Valid choices: {valid}"
    )

class Dump(IntEnum):
    NONE = 0
    CANDIDATES = 1
    CLUSTERS = 2
    LOGITS = 3

def parse_dump(value: str) -> Step:
try:
    return Dump[value.upper()]
except KeyError:
    valid = ", ".join(s.name.lower() for s in Dump)
    raise argparse.ArgumentTypeError(
        f"Invalid dump '{value}'. Valid choices: {valid}"
    )

def parse_args():
    parser = argparse.ArgumentParser()

    # Basic CMSSW settings
    parser.add_argument(
        "-nt", "--numberOfThreads",
        type=int,
        default=1,
        help="Number of CMSSW threads"
    )
    parser.add_argument(
        "-ns", "--numberOfStreams",
        type=int,
        default=1,
        help="Number of CMSSW streams"
    )
    parser.add_argument(
        "-ne", "--numberOfEvents",
        type=int,
        default=1,
        help="Number of events to process"
    )
    
    # pipeline 
    parser.add_argument(
        "--step",
        type=parse_step,
        default=Step.TAGGING,
        help="Run only the specified pipeline stages: unpacking, clustering, sorting, reshaping, tagging"
    )

    # dump
    parser.add_argument(
        "-d", "--dump",
        type=parse_dump,
        default=Step.NONE,
        help="Select which artifacts to dump on NanoAOD file: none, candidates, clusters, logits"
    )

    # Backend and environment
    parser.add_argument(
        "-b", "--backend",
        type=str,
        default="serial_sync",
        choices=["serial_sync", "cuda_async", "rocm_async"],
        help="Hardware accelerator backend"
    )

    parser.add_argument(
        "-ws", "--wantSummary",
        action='store_true',
        help="Show timing report"
    )

    # Clustering parameters
    parser.add_argument(
        "--dc",
        type=float,
        default=0.2,
        help="Side of the box inside which the density of a point is calculated"
    )
    parser.add_argument(
        "--rhoc",
        type=float,
        default=5.0,
        help="Minimum rhoc required for a point to be considered a seed candidate"
    )
    parser.add_argument(
        "--dm",
        type=float,
        default=0.4,
        help="Side of the box inside which the followers of a point are searched"
    )
    parser.add_argument(
        "-wc", "--wrapCoords",
        action="store_true",
        help="Wrap phi coordinate in CLUEstering"
    )

    # Scouting configuration
    parser.add_argument(
        "-scout", "--runScouting",
        action="store_true",
        help="Run scouting-based tagging"
    )
    parser.add_argument(
        "-rn","--runNumber",
        type=int,
        default=38,
        help="Run number"
    )
    parser.add_argument(
        "-ln", "--lumiNumber",
        type=int,
        default=1,
        help="Lumisection number"
    )
    parser.add_argument(
        "-dsm", "--daqSourceMode",
        type=str,
        default="ScoutingPhase2",
        help="DAQ source data mode"
    )
    parser.add_argument(
        "-broker", "--broker",
        type=str,
        default="none",
        help="Broker: 'none' or 'hostname:port'"
    )

    # Tagger
    parser.add_argument(
        "-m","--model",
        type=str,
        default="L1TriggerScouting/TauTagging/data/softtauid_sigmoid.pt",
        help="Path to JIT compiled PyTorch model."
    )

    # Directories and I/O streams
    parser.add_argument(
        "-fbd", "--fuBaseDir",
        type=str,
        default="/dev/shm/ramdisk",
        help="FU base directory"
    )
    parser.add_argument(
        "-bbd", "--buBaseDir",
        nargs="+",
        default=["/dev/shm/ramdisk"],
        help="BU base directory (can specify multiple)"
    )
    parser.add_argument(
        "-bns", "--buNumStreams",
        nargs="+",
        type=int,
        default=[],
        help="Number of input streams (i.e. files) used simultaneously for each BU directory"
    )
    parser.add_argument(
        "-sf", "--splitFactor",
        type=int,
        default=1
    )
    parser.add_argument(
        "-s", "--streams",
        nargs="+",
        type=int,
        default=[],
        help="Input link IDs for the inputs"
    )

    # Fast Timer Service Json
    parser.add_argument(
        "--timer",
        action="store_true",
        help="Write json file with report of FastTimerService"
    )

    return parser.parse_args()
