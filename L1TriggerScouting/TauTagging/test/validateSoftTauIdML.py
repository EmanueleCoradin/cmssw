#!/usr/bin/env python3
"""Replay SoftTauIdML (STT_jit.pt) inference in pure python and compare with CMSSW.

This is NOT a cmsRun configuration. It is a standalone python script meant to run
inside the CMSSW runtime environment (after ``cmsenv``), using the python
interpreter and packages shipped with the release (uproot, torch, numpy).

Inputs
------
A NanoAOD (or OrbitFlatTable) ROOT file produced with the CMSSW producers:

* ``L1TriggerScouting/TauTagging/plugins/SoftTauInputTensorToNanoAODFlatTable.cc``
  (or ``SoftTauInputTensorToOrbitFlatTable.cc``) - per-constituent flat table of
  valid (unmasked) rows: 10 features + ``inputConstituentIndex``/
  ``constituentIndex`` + ``inputTensorIndex``. Named rows only:
  padded rows (pad_mask(row) != 1) are skipped by the dumper, so the padding
  mask is reconstructed here from the presence of the row.

* ``L1TriggerScouting/TauTagging/plugins/SoftTauOutputTensorToNanoAODFlatTable.cc``
  (or ``SoftTauOutputTensorToOrbitFlatTable.cc``) - per-cluster model outputs
  ``cls``, ``pt``, ``charge`` + ``outputTensorIndex`` (one row per cluster).

What it replicates
------------------
The C++ call convention of the SoftTauIdML producer
(``L1TriggerScouting/TauTagging/plugins/alpaka/SoftTauIdML.cc`` +
``PhysicsTools/PyTorchAlpaka/interface/SoAConversion.h``):

* the scripted module is invoked with two POSITIONAL tensors, in registration
  order (names are NOT passed): ``model(features, pad_mask)``;
* the model weights AND both inputs are casted to ``torch.half`` (the C++ side
  calls ``model_.forward(..., torch::kHalf)`` and separately casts the module
  via ``Model::to(device, kHalf)``; here ``model.half()`` replicates that);
* the model must return a tuple of three tensors, interpreted in registration
  order as ``(cls, pt, charge)``;
* inference is batched with contiguous chunks of ``--batch-size`` clusters
  (default 32, as ``SoftTauIdML::batchSize``).

Tensor shapes per batch of B clusters: features [B, 16, 10], pad_mask [B, 16].
Feature order is the dumper column order: pt, deta, dphi, z0, charge, oh0..oh4.
Assumption: clusters whose constituents are all masked have an all-zero tensor
row (as the SoA transform kernel zero-initializes padded rows).
"""

import argparse
import os
import sys

import numpy as np
import torch
import awkward as ak
import uproot

NUM_CONSTITUENTS = 16        # JetFeatures rows (Eigen::Matrix<float, 16, 10>)
NUM_FEATURES = 10            # JetFeatures columns
INPUT_COLUMNS = ["pt", "deta", "dphi", "z0", "charge", "oh0", "oh1", "oh2", "oh3", "oh4"]
OUTPUT_KEYS = ["cls", "pt", "charge"]  # registration order in SoftTauIdML (tuple position)


def file_in_path(relative):
    """Resolve a CMSSW-relative path like FileInPath does."""
    for base in (os.environ.get("CMSSW_BASE"), os.environ.get("CMSSW_RELEASE_BASE")):
        if base:
            full = os.path.join(base, "src", relative)
            if os.path.isfile(full):
                return full
    return None


def event_input(event, input_name, constituent_column):
    """Rebuild (features, pad_mask) for one event from the flat-input-table row."""
    n_rows = len(event[f"{input_name}_inputTensorIndex"])
    tensor_idx = np.asarray(ak.to_numpy(event[f"{input_name}_inputTensorIndex"]), dtype=np.int64)
    constituent_idx = np.asarray(ak.to_numpy(event[f"{input_name}_{constituent_column}"]), dtype=np.int64)
    rows = np.stack([np.asarray(ak.to_numpy(event[f"{input_name}_{c}"]), dtype=np.float32) for c in INPUT_COLUMNS], axis=1)

    n_clusters = int(tensor_idx.max()) + 1 if n_rows else 0
    features = np.zeros((n_clusters, NUM_CONSTITUENTS, NUM_FEATURES), dtype=np.float32)
    pad_mask = np.zeros((n_clusters, NUM_CONSTITUENTS), dtype=np.float32)
    if n_rows:
        features[tensor_idx, constituent_idx] = rows
        pad_mask[tensor_idx, constituent_idx] = 1.0
    return features, pad_mask


def event_output(event, output_name):
    """Expected CMSSW outputs for one event, indexed by cluster."""
    cluster_idx = np.asarray(ak.to_numpy(event[f"{output_name}_outputTensorIndex"]), dtype=np.int64)
    expected = np.stack([np.asarray(ak.to_numpy(event[f"{output_name}_{k}"]), dtype=np.float32) for k in OUTPUT_KEYS], axis=1)
    if cluster_idx.size and not np.array_equal(cluster_idx, np.sort(cluster_idx)):
        expected = expected[np.argsort(cluster_idx)]
    return expected


def infer(model, features, pad_mask, batch_size):
    """Batched inference replicating AlpakaModel::forward(..., kHalf)."""
    outputs = [[] for _ in OUTPUT_KEYS]
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            # the release torch is built without NumPy support: go through nested lists
            batch_features = torch.tensor(features[start:start + batch_size].tolist(), dtype=torch.half)
            batch_mask = torch.tensor(pad_mask[start:start + batch_size].tolist(), dtype=torch.half)
            result = model(batch_features, batch_mask)
            if not isinstance(result, (tuple, list)) or len(result) != len(OUTPUT_KEYS):
                raise RuntimeError(
                    f"Model returned {type(result).__name__}"
                    + (f" of length {len(result)}" if isinstance(result, (tuple, list)) else "")
                    + f", expected a tuple/list of {len(OUTPUT_KEYS)} tensors (cls, pt, charge)"
                )
            for i, tensor in enumerate(result):
                outputs[i].append(np.asarray(tensor.float().cpu().tolist(), dtype=np.float32).reshape(-1))
    return np.stack([np.concatenate(values) for values in outputs], axis=1)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Run with cmsenv active; reads the dumpers' flat tables (see module docstring for details).",
    )
    parser.add_argument("input", help="NanoAOD/OrbitFlatTable ROOT file")
    parser.add_argument("--model", default=file_in_path("L1TriggerScouting/TauTagging/data/STT_jit.pt"),
                        help="Path to the TorchScript model")
    parser.add_argument("--input-name", default="SoftTauInputs", help="Name of the input flat table")
    parser.add_argument("--output-name", default="SoftTauOutputs", help="Name of the output flat table")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Cluster batch size (must match the producer's batchSize)")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance (float16 inference replayed in python)")
    parser.add_argument("--atol", type=float, default=1e-3, help="Absolute tolerance")
    parser.add_argument("--max-events", type=int, default=0, help="Process at most this many events (0 = all)")
    args = parser.parse_args()

    if not args.model or not os.path.isfile(args.model):
        sys.exit(f"error: cannot find TorchScript model (set --model); tried: {args.model}")
    if args.batch_size <= 0:
        sys.exit("error: --batch-size must be positive")

    model = torch.jit.load(args.model, map_location="cpu")
    model.half()  # as Model::to(device, torch::kHalf) on the C++ side
    model.eval()

    tree = uproot.open(args.input)["Events"]
    branch_names = set(tree.keys())
    need = [f"{args.input_name}_inputTensorIndex",
            f"{args.input_name}_inputConstituentIndex",
            f"{args.output_name}_cls"]
    constituent_column = "inputConstituentIndex"
    if f"{args.input_name}_inputConstituentIndex" not in branch_names:
        # OrbitFlatTable dumper naming
        constituent_column = "constituentIndex"
        need[1] = f"{args.input_name}_constituentIndex"
    missing = [b for b in need if b not in branch_names]
    if missing:
        sys.exit(f"error: branches {missing} not in file; available tables: "
                 f"{sorted({b.split('_')[0] for b in branch_names})}")

    read_branches = ([f"{args.input_name}_{c}" for c in INPUT_COLUMNS]
                     + [f"{args.input_name}_inputTensorIndex", f"{args.input_name}_{constituent_column}"]
                     + [f"{args.output_name}_{k}" for k in OUTPUT_KEYS]
                     + [f"{args.output_name}_outputTensorIndex"])
    arrays = tree.arrays(read_branches, library="ak", entry_stop=args.max_events or None)

    n_events = 0
    n_clusters_total = 0
    n_mismatched = 0
    worst = {k: 0.0 for k in OUTPUT_KEYS}
    bad_events = []

    for event in arrays:
        n_events += 1
        features, pad_mask = event_input(event, args.input_name, constituent_column)
        expected = event_output(event, args.output_name)

        if len(expected) != len(features):
            # clusters with all constituents masked are absent from the input dump;
            # they were still inferred by CMSSW on all-zero padded rows
            if len(expected) < len(features):
                sys.exit(f"event {n_events}: more output rows ({len(expected)}) than input tensors ({len(features)}); cannot align")
            extra = len(expected) - len(features)
            features = np.concatenate([features, np.zeros((extra, NUM_CONSTITUENTS, NUM_FEATURES), np.float32)])
            pad_mask = np.concatenate([pad_mask, np.zeros((extra, NUM_CONSTITUENTS), np.float32)])

        if len(expected) == 0:
            continue

        predicted = infer(model, features, pad_mask, args.batch_size)
        n_clusters_total += len(expected)

        event_bad = []
        for i, key in enumerate(OUTPUT_KEYS):
            diff = np.abs(predicted[:, i] - expected[:, i])
            close = np.isclose(predicted[:, i], expected[:, i], rtol=args.rtol, atol=args.atol)
            worst[key] = max(worst[key], float(diff.max()))
            if not close.all():
                n_mismatched += int((~close).sum())
                event_bad.append((key, int((~close).sum()), float(diff.max())))
        if event_bad:
            bad_events.append((n_events, event_bad))

    print(f"processed {n_events} events, {n_clusters_total} clusters "
          f"(batch size {args.batch_size}, float16 inference)")
    for key in OUTPUT_KEYS:
        print(f"  max |Δ {key}| = {worst[key]:.6g}")
    if bad_events:
        print(f"FAIL: {n_mismatched} mismatched values (rtol={args.rtol}, atol={args.atol}) in {len(bad_events)} event(s)")
        for event_id, details in bad_events[:10]:
            for key, count, maxdiff in details:
                print(f"  event {event_id}: {key}: {count} mismatched, max |Δ| = {maxdiff:.6g}")
        if len(bad_events) > 10:
            print(f"  ... and {len(bad_events) - 10} more events")
        return 1
    print(f"PASS: all outputs within rtol={args.rtol}, atol={args.atol}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
