# PhysioML

<!-- badges: start -->
[![r-universe](https://x-biosignal.r-universe.dev/badges/PhysioML)](https://x-biosignal.r-universe.dev/PhysioML)
<!-- badges: end -->

PhysioML provides reproducible feature transforms and leakage-aware reference
deep-learning workflows for `PhysioExperiment` time series. Its toy fixtures
validate mechanics only; they do not select or clinically validate a model.

## Installation

```r
install.packages(
  "PhysioML",
  repos = c("https://x-biosignal.r-universe.dev", BiocManager::repositories())
)
```

`Rcatch22` is optional for `catch22()`. ROCKET and MiniRocket require the
optional R package `reticulate` and a caller-managed Python environment with
NumPy and aeon. PhysioML never installs or selects a Python environment at
runtime.

Deep-learning functions require the optional R packages `torch` and `luz` plus
a caller-installed LibTorch runtime. Install those dependencies explicitly
before use. PhysioML never calls `torch::install_torch()`, downloads weights,
or selects an accelerator from package code.

## Quick start

PhysioML reads assays as `time x channel x case`, preserving channel labels and
case IDs. Building that input contract needs no optional backend:

```r
library(PhysioML)

set.seed(1)
peCases <- function(n_case) {
  a <- array(rnorm(256 * 2 * n_case), dim = c(256, 2, n_case))
  PhysioExperiment(
    assays = list(raw = a), samplingRate = 128,
    colData = S4Vectors::DataFrame(label = c("C3", "C4"))
  )
}
train_pe <- peCases(6)
test_pe <- peCases(3)
train_labels <- factor(rep(c("rest", "task"), length.out = 6))
valid_labels <- factor(rep(c("rest", "task"), length.out = 3))
train_pe
```

The feature, dataset, model, and export workflows below call the optional
backends noted above (`reticulate` with aeon, or `torch`/`luz`). Each is guarded
so it runs only when that backend is installed, and reuses the objects built
here.

## Input and output contracts

PhysioML interprets assays as `time x channel` for one case or
`time x channel x case` for multiple cases. Channel labels and case IDs are
preserved exactly. Inputs must be numeric, finite, equal length, and at least
10 samples for catch22 or nine samples for convolution transforms. Missing
values, unequal-length series, implicit resampling, padding, and channel
averaging are not supported.

`catch22()` returns the 22 canonical Rcatch22 features for every case-channel
series. Set `catch24 = TRUE` to append the upstream `DN_Mean` and
`DN_Spread_Std` values. `peReducedFeatures()` widens these to a case-by-feature
matrix without adding an outcome.

ROCKET returns alternating maximum and proportion-positive-value features.
MiniRocket returns proportion-positive-value features. Both results record the
requested kernel count and actual backend feature width; MiniRocket may adjust
the requested width to its kernel family.

## Leakage boundary

Fit a convolution transform on training cases, then reuse its persistent model
for validation or test cases:

```r
if (requireNamespace("reticulate", quietly = TRUE)) {
  fit <- minirocket(train_pe, seed = 42)
  test_features <- minirocket(test_pe, model = fit$model)
}
```

Do not fit the transform on all cases before cross-validation.

## Torch Dataset contract

Split patients or cases before windowing or normalization. Fit a training
Dataset, then reuse its frozen channel-wise statistics:

```r
if (requireNamespace("torch", quietly = TRUE)) {
  train <- peDataset(
    train_pe,
    targets = train_labels,
    window_samples = 256,
    stride_samples = 128,
    normalization = "zscore"
  )
  valid <- peDataset(
    test_pe,
    targets = valid_labels,
    window_samples = 256,
    stride_samples = 128,
    normalization = "zscore",
    normalization_stats = train$contract$normalization_stats,
    class_levels = train$contract$class_levels
  )
}
```

Items are `channel x time`; DataLoader batches are
`batch x channel x time`, including a batch of one. Windows are complete,
case-major, and never cross a case boundary. Classification targets are
zero-based `torch_long` values with a reversible label table. Regression
targets and inputs are `torch_float32`. Targets are never appended to inputs.

Z-score scale is the population standard deviation. Robust scale is
`1.4826 * median(abs(x - median(x)))`. A constant training channel receives
scale one and a typed diagnostic. Passing all cases to `peDataset()` and
splitting its windows afterward leaks normalization information and can put
one patient's windows in both training and validation.

## Reference models and training

`peModel()` constructs four compact modules with a common
`batch x channel x time` input and `batch x output` result:

- EEGNet-8,2 uses explicit temporal padding, grouped spatial depthwise
  convolution, separable temporal convolution, average pooling, and mandatory
  post-step max norms of 1 for spatial filters and 0.25 for classifier rows.
- CNN1D is a small repository baseline, not a published clinical architecture.
- TCN blocks use left padding only. Their recorded receptive field is
  `1 + 2 * (kernel_size - 1) * sum(dilations)`.
- LSTM explicitly permutes to `batch x time x channel` and uses the final
  hidden representation. Bidirectionality must be requested explicitly.

Classification modules return logits; softmax is applied only for probability
prediction. `trainModel()` uses Adam through luz, requires caller-supplied
case-disjoint validation, never creates a row-level split, and rejects
independently fitted validation normalization. The initial implementation uses
CPU and `num_workers = 0`; it restores caller R and torch RNG states on success
and error. Accelerator runs are not claimed to be bitwise reproducible.

```r
if (requireNamespace("torch", quietly = TRUE) &&
    requireNamespace("luz", quietly = TRUE)) {
  fit <- trainModel(
    train,
    valid,
    model = "cnn1d",
    epochs = 20,
    seed = 42
  )
  probability <- predictModel(fit, valid, type = "probability")
}
```

Prediction runs with gradients disabled and evaluation mode enabled. Channel
order, time width, task, dtype, class mapping, and frozen normalization must
match training exactly.

## ONNX interoperability

`exportONNX()` crosses one explicit inference-only boundary:

```text
R torch module -> TorchScript -> Python PyTorch -> ONNX
```

The initial exporter uses PyTorch's legacy TorchScript ONNX path deliberately;
it does not rebuild an R architecture in Python. R LibTorch and Python torch
must have the same major/minor version. Export uses float32
`batch x channel x time` input named `signal` and float32
`batch x output` logits named `output`, opset 17 by default. Channel, time, and
output dimensions remain fixed. EEGNet, CNN1D, and TCN support a dynamic batch
axis. R torch's implicit LSTM initial state is traced at a fixed batch width,
so LSTM export rejects `dynamic_batch = TRUE` and supports the validated fixed
batch width of two.

Each graph is checked with ONNX, run through ONNX Runtime's
`CPUExecutionProvider` with provider fallback disabled, and compared to the R
torch logits before landing. The maximum absolute and relative errors must
both be at most `1e-4`. The graph and canonical JSON model card are landed as
a pair:

```r
if (requireNamespace("torch", quietly = TRUE) &&
    requireNamespace("luz", quietly = TRUE) &&
    requireNamespace("reticulate", quietly = TRUE)) {
  onnx_path <- file.path(tempdir(), "model.onnx")
  exportONNX(fit, onnx_path)
  portable <- importONNX(onnx_path)
  logits <- onnxPredict(portable, test_pe, type = "logit")
}
```

`importONNX()` retains graph bytes and the validated card, not a Python
`InferenceSession`, so the result survives `saveRDS()` and a fresh R process.
`onnxPredict()` recreates a CPU session for each call and requires the exact
training channel order, time width, task, class mapping, dtype, and frozen
normalization statistics. It never fits normalization or sends targets to
ONNX Runtime.

Python torch, ONNX, and ONNX Runtime are caller-managed optional dependencies.
PhysioML does not install them or select a Python environment. ONNX avoids
Python pickle loading during import, but an untrusted graph can still target
parser/runtime vulnerabilities or consume excessive resources. The model-card
SHA-256 proves byte integrity, not trust or clinical validity.

## Transfer learning and domain alignment

`freezeLayers()` clones a fitted R torch model and selects parameters by their
exact names. Prefix selection is component-aware. The default architecture
registry freezes the feature extractor and leaves only `classifier.*`
trainable. `fineTune()` passes exactly the resulting trainable tensors to Adam.
Frozen modules remain in evaluation mode, so batch-normalization buffers do
not drift during a linear probe.

```r
if (requireNamespace("torch", quietly = TRUE) &&
    requireNamespace("luz", quietly = TRUE)) {
  frozen <- freezeLayers(fit)
  adapted <- fineTune(
    frozen,
    train,
    valid,
    strategy = "partial",
    unfreeze = c("conv2", "bn2")
  )
}
```

Target normalization is fitted only on the target training split and then
reused unchanged by validation and test datasets. Case IDs must be disjoint.
The source model and caller RNG states are not mutated.

`domainAdapt()` implements CORAL for a training-source matrix `Xs` and an
explicitly designated unlabeled target adaptation matrix `Xt`:

```text
A   = (cov(Xs) + lambda I)^(-1/2) (cov(Xt) + lambda I)^(1/2)
Xs' = (Xs - mean(Xs)) A + mean(Xt)
```

Matrix powers use symmetric eigendecomposition with an explicit eigenvalue
floor. Target validation/test rows and labels do not belong in the fit.
Matching second-order statistics does not guarantee accuracy, fairness,
causal invariance, class-conditional alignment, or clinical validity.

## Foundation embeddings

`foundationEmbed()` resolves the same case-major, channel-ordered complete
windows as the feature and torch APIs. Every result records sampling rate,
units, axes, channel order, model ID, immutable 40-character revision, complete
manifest SHA-256, preprocessing, and backend versions. CPU execution is the
only public device.

`cacheFoundationModel()` requires explicit first-download consent, license
acceptance, an immutable revision, and a predeclared canonical manifest hash.
It verifies the entire declared file inventory, sizes, hashes, READY marker,
path containment, and symlink absence before use. A cache hit performs no
download. Safetensors is the supported checkpoint boundary;
`trust_remote_code`, unrestricted pickle loading, package installation, and
environment selection are not used. Checksums establish integrity, not trust.

MOMENT, Chronos-2, LaBraM, and BENDR have distinct adapter contracts. The
package-owned tiny safetensors fixture keeps routine acceptance reproducible
offline. An official adapter is enabled only after its exact revision,
preprocessing, output, repeatability, CPU behavior, and license pass a managed
preflight. Chronos-2 currently lacks a documented fixed-width embedding
method; LaBraM and BENDR official repository-code/pickle checkpoints do not
meet this safe generic loader boundary. Those adapters fail explicitly rather
than guessing an intermediate representation or montage.

## Persistence and trust

Fitted convolution models contain a Python pickle serialized into an R raw
vector, plus backend versions and a SHA-256 checksum. `saveRDS()` and
`readRDS()` preserve the model without a live Python pointer. A checksum
detects accidental corruption but does not establish trust: loading a pickle
from an untrusted source can execute code. Reuse only model objects obtained
from a trusted source and with the recorded aeon version.

Every result records exact package, Python, NumPy, and aeon provenance. These
generic transforms do not establish clinical validity for a downstream model.

A fitted `physio_dl_model` contains live torch/luz objects for the current
session. Upstream torch serialization is not promised here as durable across
versions. The plain model spec, input contract, class mapping, normalization
statistics, and backend versions are retained for governed export work. Never
deserialize an untrusted model file.

## References and upstream licenses

The feature definitions follow Lubba et al. (2019), ROCKET follows Dempster,
Petitjean and Webb (2020), MiniRocket follows Dempster, Schmidt and Webb
(2021), EEGNet follows Lawhern et al. (2018), and the causal TCN follows Bai,
Kolter and Koltun (2018). PhysioML is MIT-licensed and calls separately
installed maintained backends at runtime; it does not vendor their source.

## Part of the x-biosignal ecosystem

See the [x-biosignal](https://github.com/x-biosignal) organization and
[x-biosignal.r-universe.dev](https://x-biosignal.r-universe.dev) for the full
package suite.
