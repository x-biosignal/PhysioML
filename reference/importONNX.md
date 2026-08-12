# Import a governed PhysioML ONNX graph

Validates a graph against its canonical, hash-bound PhysioML model card
and returns raw persistent bytes rather than a live Python runtime
session.

## Usage

``` r
importONNX(
  path,
  metadata_path = paste0(path, ".json"),
  providers = "CPUExecutionProvider",
  verify = TRUE
)
```

## Arguments

- path:

  Path to an ONNX graph.

- metadata_path:

  Path to its canonical JSON model card.

- providers:

  Execution provider; currently CPU only.

- verify:

  Whether to run the ONNX checker and graph/card introspection.

## Value

A persistent `physio_onnx_model`.
