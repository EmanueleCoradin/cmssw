# Candidate and superclustering inference

This incremental implementation builds on `02223812f` (the four previous Work
commits), not just the GitHub branch. The reconciled remote development head was
`3ca7bfc14e16169b5df847653be120d62fa9d8cf`. The previous CNN/DNN/PFN feature kernels,
selection, tensor adapters and result policies are retained. CPU ONNX remains the
default. No production models are included, trained or exported.

## Candidate reconstruction

`TICLCandidateProducer` selects `inferenceAlgo="PyTorchAlpaka"` through its existing
`TracksterInferenceAlgoBase` hook. Each CPU EDM stream owns its backend plugin,
queue and independent optional PID/regression models. Inference still happens
**after both interpretation algorithms and PCA assignment, before the reconstructed
Tracksters are put into the event and before final candidates are constructed**.
There is no additional producer, published intermediate collection or changed token.

The bridge stages the reconstructed Tracksters' vertices, multiplicities, energies
and PCA-derived barycenter into the existing Trackster/CaloCluster SoAs. It runs the
same `TracksterFeatureExtractionRunner` and `TracksterInferenceRunner` used by the
previous device producers. PID and regression share feature extraction and update
the same private device collection. Only the two prediction blocks (eight PID
columns and regressed energy, including portable column padding) come back to the
CPU. No vertex/PCA payload is copied back. Only these predictions are written into
the existing CPU objects; timing, eigenvalues/eigenvectors, vertices, tracks, GSF
tracks and other associations are untouched. Input interpretation Tracksters remain
distinct from the reconstructed output Tracksters. Candidate construction, muon
momentum treatment and the original/reconstructed-object links are unchanged.

DNN/PFN reset both predictions for selected Tracksters, then run enabled tasks,
as in the CPU implementations. Unselected Tracksters retain their predictions.
PFN regression returns raw energy when raw energy is **not strictly greater than**
`minClusterEnergy`; PID still runs for those selected objects. CNN remains PID-only.
Disabling both tasks is a no-op and does not resolve any model paths. Setting the
producer's `regressionAndPid=False` bypasses plugin construction entirely.

Geometry comes from the first cluster hit, matching CPU preprocessing, whereas
barrel exclusion uses the cluster seed. Host layers are already endcap-local and
are not folded by the detector layer count. Empty-hit clusters are assigned layer
-1 and cannot occupy an image slot. Valid CPU inputs are expected to have hits.
CaloCluster coordinates/energy are staged as floats, as in the existing SoAs:
real-event checks must still assess rounding near selection thresholds and equal
energies. The PCA scalar ordering for PFN is unchanged; eigenvalues are not PFN
inputs in this implementation.

```python
# After loading the ordinary iterative TICL configuration:
from RecoTICL.Configuration.stageInference_cff import withCandidateInference

process.ticlCandidate = withCandidateInference(
    process.ticlCandidate,
    model="PFN", doPID=True, doRegression=True,
    pidModel="RecoTICL/Inference/data/candidate_pid_pfn.pt",
    regressionModel="RecoTICL/Inference/data/candidate_energy_pfn.pt",
    minClusterEnergy=2.5,
    backend="cuda_async", deviceIndex=0, batchSize=64,
    convertToFP16=False,
)
# DNN is selected with model="DNN" and the corresponding two model paths.
# CNN: model="CNN", doPID=True, doRegression=False.
```

Alternatively, `makeCandidateInference(...)` returns just the PSet:

```python
from RecoTICL.Configuration.stageInference_cff import makeCandidateInference
process.ticlCandidate.inferenceAlgo = "PyTorchAlpaka"
process.ticlCandidate.pluginInferenceAlgoPyTorchAlpaka = makeCandidateInference(
    model="DNN", doPID=False, doRegression=True,
    regressionModel="RecoTICL/Inference/data/candidate_energy_dnn.pt",
)
```

The CPU PFN PSet remains available; select `inferenceAlgo="TracksterInferenceByPFN"`
to restore it. The model factory helpers are also imported by `iterativeTICL_cff`.

## Superclustering

The existing `TracksterLinkingbySuperClusteringDNN` remains the reconstruction
algorithm. Its batch flush calls either ONNX or a per-stream `PairScoreInference`
backend. The backend consumes row-major features and returns one compatibility
score per row; it does not consume or return a TracksterDevice.

Pair selection and feature extraction intentionally remain on the CPU. They reuse
`SuperclusteringDNNInputs::computeInto()` directly for v1/v2/v3; that source is
unchanged. This preserves ROOT's rotation convention, projection degeneracies,
clamping, eigenvalue sums and timing handling without an approximate second GPU
implementation. In particular, the **feature** delta-phi is unwrapped, whereas the
**selection window** uses wrapped delta-phi. v3 retains delta-times in [-50,50],
replaces values outside that range with -5, and preserves NaNs as the CPU code does.

The existing stable pT sort, candidate loop, tile-bin traversal and within-bin seed
order are unchanged. Inference does not reorder rows. The original score application
loop, strict `score > bestScore` comparison, first-encountered tie winner, candidate
mask, best-seed state across batch flushes, final flush, merges and singleton output
order all remain in place. The two original linked-index products are unchanged.

```python
from RecoTICL.Configuration.stageInference_cff import withSuperclusteringInference

process.ticlTracksterLinksSuperclusteringDNN = withSuperclusteringInference(
    process.ticlTracksterLinksSuperclusteringDNN,
    engine="PyTorchAlpaka",
    modelPath="RecoTICL/Inference/data/superclustering_v3.pt",
    version="v3", batchSize=64,
    backend="cuda_async", deviceIndex=0, convertToFP16=False,
)
# Restore ONNX using its retained path (choose the matching version/model):
process.ticlTracksterLinksSuperclusteringDNN = withSuperclusteringInference(
    process.ticlTracksterLinksSuperclusteringDNN, engine="ONNX", version="v3",
)
```

The helper clones the producer/PSet and preserves every selection threshold and
working point. `batchSize` counts **pairs**; the older `inferenceBatchSize` counts
**floats**. The helper sets the latter to `batchSize * featureCount`. Direct C++
configuration can use independent outer batch capacity and model minibatch size;
row order is retained in both cases. The default CPU configuration is untouched.
When PyTorch is selected, neither the global cache nor the linking producer resolves
the unused ONNX linking-model path. Independent post-link CPU PID/regression remains
as configured. The helper is also imported by `superclustering_cff`.

## Production model contracts

All paths below are **required future data**, resolved with `edm::FileInPath` only
for enabled tasks. They are not supplied by these patches. Other CMSSW data-package
paths may be configured. Missing enabled models fail explicitly; there is no silent
ONNX fallback. Serial/CUDA/ROCm plugin selection is explicit via `serial_sync`,
`cuda_async`, `rocm_async`. An unavailable backend/device is an error.

| Task | Default path in `RecoTICL/Inference/data/` | Forward inputs | Forward output |
|---|---|---|---|
| Candidate CNN PID | `candidate_pid_cnn.pt` | `[B,3,50,10]` | `[B,8]` |
| Candidate DNN PID | `candidate_pid_dnn.pt` | `[B,50,10,3]` | `[B,8]` |
| Candidate DNN regression | `candidate_energy_dnn.pt` | `[B,50,10,3]` | `[B,1]` or `[B]` |
| Candidate PFN PID | `candidate_pid_pfn.pt` | `[B,50,10,7]`, then `[B,7]` | `[B,8]` |
| Candidate PFN regression | `candidate_energy_pfn.pt` | `[B,50,10,7]`, then `[B,7]` | `[B,1]` or `[B]` |
| Pair scoring v1/v2/v3 | `superclustering_v1.pt`, `superclustering_v2.pt`, `superclustering_v3.pt` | `[B,9]`, `[B,17]`, `[B,18]` respectively | `[B,1]` or `[B]` |

The TorchScript forward result must be one floating-point Tensor on the selected
Alpaka device, not a tuple/dictionary. Models must support dynamic positive B,
including the final partial minibatch, use no hard-coded device, and not mutate
inputs. No model is called for an empty collection. Missing image slots are zero;
minibatch tails contain only real rows, without extra padded samples. Portable SoA
alignment padding is not part of a tensor's logical shape. `convertToFP16=False`
uses FP32; True converts both weights and inputs to FP16 and copies predictions
into the FP32 output storage. FP16 suitability requires validation with real models.

CNN/DNN cluster channels are corrected cluster energy (energy/multiplicity),
absolute eta, phi. PFN appends x, y, absolute z, cell count. The PFN second tensor is
raw energy, raw EM energy, barycenter x, barycenter y, absolute barycenter z, absolute
barycenter eta, barycenter phi. Full descending cluster-energy ordering precedes
per-layer top-ten truncation. PID class order is photon, electron, muon, neutral
pion, charged hadron, neutral hadron, ambiguous, unknown. Predictions are used as
returned: no added activation/normalization. Regression must supply physical energy;
pair scoring must supply the compatibility score expected by the working point,
not uncalibrated logits. Candidate-stage models are distinct from first-stage models.

Pair input ordering is defined by the unchanged CPU source:

| Index | Feature |
|---|---|
| 0 | abs(candidate eta) - abs(seed eta) |
| 1 | candidate phi - seed phi (unwrapped) |
| 2–4 | candidate raw energy, signed eta, raw pT |
| 5–8 | seed signed eta, phi, raw energy, raw pT |
| 9 | angle between candidate and seed leading PCA vectors |
| 10–11 | xz and yz projected angles in the CPU-defined seed frame |
| 12–14 | xy, yz, xz projected angles in the CMS frame |
| 15–16 | candidate largest eigenvalue, its fraction of the float eigenvalue sum |
| 17 | v3 modified candidate-minus-seed time |

v1 uses indices 0–8; v2 uses 0–16; v3 uses all 18. Zero 2D denominators give
pi/2; a zero eigenvalue sum gives fraction zero. These rules are preserved by reuse,
not replaced with a different numerical regularization.

## Shared execution and remaining boundaries

`BatchedInferenceRunner` extracts the previous runner's model ownership and
minibatch loop. It uses the existing PyTorchAlpaka `AlpakaModel`, `QueueGuard`,
`BatchedTensorCollection`, `TensorCollection` and SoA conversion utilities. It
checks output shape/device/type and explicitly copies into registered storage.
`InputTransform` distinguishes identity, image channel permutation and flat rows.
`adaptInferenceInputs` can be tested without loading a model. The former
`TracksterInferenceRunner` delegates execution and retains its specialized reset,
PID/regression scatter and publication policies. No PhysicsTools changes are needed
in this incremental series.

Pair input/output adapters reuse the existing single-float portable layout as
scalar tensor storage; its historical `energy` column name has no energy semantics
in this backend. Flattened pair rows are reshaped as tensor views, so no pair SoA,
EDM dictionary or dummy Trackster result is introduced. The pair plugin reuses its
workspace for identical batch shapes, performs one feature upload and one score
download per flush, and avoids the const-input COW copy for its private scratch.
Different tail shapes can reallocate workspace. Warmup defaults to zero; optional
pair warmup runs once on the first nonempty input shape.

This is **not** a GPU-native reconstruction conversion. Candidate interpretation and
linking currently produce CPU objects. Their private queues block at the required
CPU boundaries: selection count then candidate predictions; scores at each linking
flush. No implicit device-to-device cross-queue ownership is introduced. The current
CPU interfaces cannot directly accept upstream device SoAs. Future fusion requires
GPU interpretation/PCA or a device-aware stage interface, device-count-aware
selection, and a decision about ordered superclustering grouping. Pair selection/
features should move together when upstream Tracksters are device-resident and
measurements justify it; porting them now would add geometry/PCA transfers and a
second numerical implementation for small workloads. Framework-managed accelerator
assignment, multistream memory/latency and sharing tensor materialization across
PID/regression are still follow-up work. These opt-in bridges require explicit
backend/device choice and do not implement automatic fallback or load balancing.

## Validation status

Executed in the authoring environment for this incremental series:

- Ten Python unittest methods (four existing, six new), using the repository's real
  CMSSW Python modules: passed. Covers model/task combinations, independent paths,
  disabled tasks, cloning, preserved inputs/thresholds, backend/size validation,
  v1/v2/v3 batch units and restoration of the saved ONNX path.
- Existing standalone C++20 cluster-order test, GCC 13.3: all 16,384 index-order
  comparisons with `std::sort` passed, including ties.
- Python compilation and `git diff --check`: passed.
- Source comparison against `02223812f`: CPU DNN/PFN and superclustering feature
  source unchanged; superclustering selection/order/grouping code byte-identical;
  candidate PCA/inference/publication/kinematics sequence unchanged except a null
  plugin guard for the already-supported empty-model configuration.
- All existing `#include <Eigen/Core>` directives preserved.

**Not executed:** the new CMSSW C++ tests, serial/CUDA/ROCm compilation, model runtime,
real-event preprocessing, multistream execution, performance, or inference-output
comparison. `scram b -j 4` cannot start (`scram: command not found`). A direct C++
syntax-check attempt stops at missing `Eigen/Core`; the CMSSW externals are absent.
`nvcc` and `hipcc` are absent. Thus compilation and physics equivalence are not
established. No assertion of inference-output equivalence is made without models.

C++ tests supplied for a complete CMSSW environment:

| Test | Model-free checks | Status here |
|---|---|---|
| `testTracksterPreprocessing*` | Existing all-family oracle comparisons, now also through the actual legacy bridge; selected indices, PCA barycenter feature order, images/padding, tensor tails, reset/scatter indices, preserved time/PCA/track links | Pending |
| `testPairPreprocessing*` | All v1/v2/v3 features against an analytic fixture; timing bounds/NaN, unwrapped phi, zero eigenvalue sum/projections; actual registered input/output tensor dimensions, row mapping, empty and partial minibatches | Pending |
| `testSuperclusteringOrder` | Actual production selection/grouping with an injected deterministic scorer: equal pT, score ties, candidate masking, 1/2/4/64-row flushes, split input collections, cuts, wrapped selection, links and merged energies | Pending |

Float feature tests use absolute-plus-relative tolerance `2e-5 * (1+abs(expected))`.
Integer mappings/order and grouping are exact. Synthetic scores only test result
application, not neural-network physics. The ordering test compiles the production
plugin source directly into its standalone binary and uses its event-independent
overload; it does not copy a second reconstruction algorithm.

In the appropriate initialized CMSSW release, build all affected packages and tests:

```bash
scram b -j 8
python3 RecoTICL/Inference/test/testTracksterInferenceConfig.py
python3 RecoTICL/Inference/test/testStageInferenceConfig.py
scram b runtests TEST=testSuperclusteringOrder
find "$CMSSW_BASE/test/$SCRAM_ARCH" -maxdepth 1 -type f \
  \( -name 'testTracksterPreprocessing*' -o -name 'testPairPreprocessing*' \) -print
```

Run each generated backend-specific executable on a matching serial, CUDA or ROCm
node supported by that release. Exit 77 means no device: it is not a pass. Then
compare real-event preprocessing against CPU objects/geometry, supply the specified
production models, test missing-model diagnostics and both disabled-task paths,
and separately validate inference outputs and candidate four-vectors. Building one
GPU backend does not establish support for the other.
