# Create a case-aware torch Dataset

Converts a PhysioExperiment assay to fixed channel-by-time items. Split
cases before calling this function, fit normalization on training cases
only, and pass `normalization_stats` to validation and prediction
datasets. Class tensors are zero-based `torch_long`; input and
regression tensors are `torch_float32`. A one-item batch always retains
its batch dimension.

## Usage

``` r
peDataset(
  x,
  targets = NULL,
  assay_name = NULL,
  cases = NULL,
  channels = NULL,
  window_samples = NULL,
  stride_samples = NULL,
  normalization = c("none", "zscore", "robust"),
  normalization_stats = NULL,
  task = c("classification", "regression"),
  class_levels = NULL,
  dtype = "float32"
)
```

## Arguments

- x:

  A `PhysioExperiment`; attach subject keys with
  [`withCaseData()`](https://x-biosignal.github.io/PhysioML/reference/withCaseData.md).

- targets:

  Optional one-per-selected-case target vector.

- assay_name, cases, channels:

  Exact assay and identity selections.

- window_samples, stride_samples:

  Optional complete-window dimensions.

- normalization:

  Exact normalization method.

- normalization_stats:

  Frozen statistics from a training dataset.

- task:

  Exact task name.

- class_levels:

  Optional ordered classification labels.

- dtype:

  Currently exactly `"float32"`.

## Value

A `physio_torch_dataset`.

## Examples

``` r
# Prepare the input offline: a time x channel x case assay with channel labels.
arr <- array(stats::rnorm(50 * 2 * 6), dim = c(50, 2, 6))
pe <- PhysioExperiment::PhysioExperiment(
  assays = list(raw = arr),
  colData = S4Vectors::DataFrame(label = c("C3", "C4")),
  samplingRate = 100
)
# Building the Dataset itself needs the optional torch backend.
if (requireNamespace("torch", quietly = TRUE)) {
  ds <- peDataset(
    pe,
    targets = rep(c("rest", "task"), 3),
    task = "classification",
    class_levels = c("rest", "task")
  )
}
#> Error: the R package `torch` is installed but LibTorch is unavailable; install it explicitly with `torch::install_torch()`
```
