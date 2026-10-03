# Export a fitted PhysioML torch model to ONNX

Uses the governed R torch to TorchScript to Python PyTorch bridge. Only
the batch axis may be dynamic. The graph and canonical model card are
validated before they are atomically landed.

## Usage

``` r
exportONNX(
  model,
  path,
  opset = 17L,
  dynamic_batch = TRUE,
  validate = TRUE,
  metadata_path = paste0(path, ".json"),
  overwrite = FALSE
)
```

## Arguments

- model:

  A fitted `physio_dl_model`.

- path:

  Destination `.onnx` path.

- opset:

  Positive ONNX opset; the initial default is 17.

- dynamic_batch:

  Whether the batch axis is dynamic.

- validate:

  Whether to enforce the full ONNX checker, graph/card consistency, and
  logit parity tolerance. Basic graph parsing, I/O contract checks, and
  CPU inference are always required.

- metadata_path:

  Destination canonical JSON model-card path.

- overwrite:

  Whether an existing graph/card pair may be replaced.

## Value

Invisibly, a plain `physio_onnx_export` list.

## Examples

``` r
arr <- array(stats::rnorm(50 * 2 * 6), dim = c(50, 2, 6))
pe <- PhysioExperiment::PhysioExperiment(
  assays = list(raw = arr),
  colData = S4Vectors::DataFrame(label = c("C3", "C4")),
  samplingRate = 100
)
path <- tempfile(fileext = ".onnx")
# \donttest{
# Export needs torch plus the governed torch->ONNX Python bridge.
train <- peDataset(
  pe, targets = rep(c("a", "b"), 3),
  task = "classification", class_levels = c("a", "b")
)
#> Error: the R package `torch` is installed but LibTorch is unavailable; install it explicitly with `torch::install_torch()`
fit <- trainModel(train, model = "cnn1d", epochs = 1L)
#> Error: the R package `torch` is installed but LibTorch is unavailable; install it explicitly with `torch::install_torch()`
exportONNX(fit, path)
#> Error: the R package `torch` is installed but LibTorch is unavailable; install it explicitly with `torch::install_torch()`
# }
```
