# Fine-tune a fitted torch model

Training is CPU-only and passes only explicitly trainable parameters to
Adam. Frozen modules, including their batch-normalization buffers,
remain in evaluation mode. Validation and prediction data must reuse
frozen normalization statistics.

## Usage

``` r
fineTune(
  model,
  train,
  valid = NULL,
  strategy = c("linear_probe", "partial", "full"),
  unfreeze = NULL,
  epochs = 10L,
  batch_size = 32L,
  learning_rate = 1e-04,
  weight_decay = 0,
  callbacks = NULL,
  seed = 1L,
  device = "cpu",
  num_workers = 0L,
  verbose = FALSE
)
```

## Arguments

- model:

  A fitted `physio_dl_model` or `physio_transfer_model`.

- train, valid:

  Compatible targeted datasets.

- strategy:

  Exact tuning strategy.

- unfreeze:

  Component prefixes used only by partial tuning.

- epochs, batch_size:

  Positive whole-number settings.

- learning_rate:

  Positive Adam learning rate.

- weight_decay:

  Non-negative Adam weight decay.

- callbacks:

  Currently `NULL`; callback execution is owned by
  [`trainModel()`](https://x-biosignal.github.io/PhysioML/reference/trainModel.md)'s
  luz workflow.

- seed:

  Whole-number CPU seed.

- device:

  Currently exactly `"cpu"`.

- num_workers:

  Currently exactly zero.

- verbose:

  Whether to print one line per epoch.

## Value

A fitted `physio_dl_model` with a plain `transfer` record.

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
  tuned <- fineTune(fit, train, strategy = "linear_probe", epochs = 1L)
}
#> Error: the R package `torch` is installed but LibTorch is unavailable; install it explicitly with `torch::install_torch()`
```
