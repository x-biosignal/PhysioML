# Freeze named parts of a fitted torch model

The fitted model is cloned before `requires_grad` is changed. Exact
matching is the default; prefix matching is component-aware.

## Usage

``` r
freezeLayers(
  model,
  trainable = NULL,
  frozen = NULL,
  match = c("exact", "prefix"),
  strict = TRUE
)
```

## Arguments

- model:

  A fitted `physio_dl_model`.

- trainable, frozen:

  Optional unique exact parameter names or prefixes.

- match:

  Exact selection mode.

- strict:

  Whether unmatched rules are errors.

## Value

A plain `physio_transfer_model`.

## Examples

``` r
arr <- array(stats::rnorm(50 * 2 * 6), dim = c(50, 2, 6))
pe <- PhysioExperiment::PhysioExperiment(
  assays = list(raw = arr),
  colData = S4Vectors::DataFrame(label = c("C3", "C4")),
  samplingRate = 100
)
# Needs the optional torch and luz backends.
if (requireNamespace("torch", quietly = TRUE) &&
    requireNamespace("luz", quietly = TRUE)) {
  train <- peDataset(
    pe, targets = rep(c("a", "b"), 3),
    task = "classification", class_levels = c("a", "b")
  )
  fit <- trainModel(train, model = "cnn1d", epochs = 1L)
  frozen <- freezeLayers(fit)
  frozen$freeze_manifest
}
#> Error: the R package `torch` is installed but LibTorch is unavailable; install it explicitly with `torch::install_torch()`
```
