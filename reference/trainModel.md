# Train a reference torch model through luz

The caller supplies case-disjoint training and validation Datasets. When
subject metadata is attached with
[`withCaseData()`](https://x-biosignal.github.io/PhysioML/reference/withCaseData.md),
both datasets must carry it and subjects must also be disjoint. Legacy
unannotated inputs retain case-only checking; missing metadata does not
establish subject separation. Validation must reuse training
normalization statistics. CPU execution, no worker processes, and
deterministic caller-state restoration are enforced in this
implementation.

## Usage

``` r
trainModel(
  train,
  valid = NULL,
  model = c("eegnet", "cnn1d", "tcn", "lstm"),
  epochs = 20L,
  batch_size = 32L,
  learning_rate = 0.001,
  weight_decay = 0,
  callbacks = NULL,
  seed = 1L,
  device = "cpu",
  num_workers = 0L,
  verbose = FALSE,
  ...
)
```

## Arguments

- train, valid:

  Targeted `physio_torch_dataset` objects.

- model:

  Exact architecture name or a `physio_torch_module`.

- epochs, batch_size:

  Positive whole-number training settings.

- learning_rate:

  Positive Adam learning rate.

- weight_decay:

  Non-negative Adam weight decay.

- callbacks:

  NULL or a list of luz callback instances.

- seed:

  Whole-number CPU seed.

- device:

  Currently exactly `"cpu"`.

- num_workers:

  Currently exactly zero.

- verbose:

  Whether luz prints progress.

- ...:

  Architecture settings when `model` is a name.

## Value

A `physio_dl_model`.

## Examples

``` r
arr <- array(stats::rnorm(50 * 2 * 6), dim = c(50, 2, 6))
pe <- PhysioExperiment::PhysioExperiment(
  assays = list(raw = arr),
  colData = S4Vectors::DataFrame(label = c("C3", "C4")),
  samplingRate = 100
)
# Training needs the optional torch and luz backends.
if (requireNamespace("torch", quietly = TRUE) &&
    requireNamespace("luz", quietly = TRUE)) {
  train <- peDataset(
    pe, targets = rep(c("a", "b"), 3),
    task = "classification", class_levels = c("a", "b")
  )
  fit <- trainModel(train, model = "cnn1d", epochs = 1L)
}
#> Error: the R package `torch` is installed but LibTorch is unavailable; install it explicitly with `torch::install_torch()`
```
