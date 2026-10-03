# Predict with a trained deep-learning model

Inference uses evaluation mode and disabled gradients. The Dataset must
exactly match the training channel, time, task, class, dtype, and frozen
normalization contracts.

## Usage

``` r
predictModel(
  model,
  newdata,
  batch_size = 64L,
  type = c("class", "probability", "response", "logit"),
  num_workers = 0L
)
```

## Arguments

- model:

  A fitted `physio_dl_model`.

- newdata:

  A compatible target-free or targeted Dataset.

- batch_size:

  Positive whole-number inference batch size.

- type:

  Exact output type.

- num_workers:

  Currently exactly zero.

## Value

A numeric matrix or identity-preserving data frame.

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
  predictModel(fit, train, type = "class")
}
#> Error: the R package `torch` is installed but LibTorch is unavailable; install it explicitly with `torch::install_torch()`
```
