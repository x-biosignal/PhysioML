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
