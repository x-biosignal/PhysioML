# Predict through ONNX Runtime on CPU

Materializes a validated raw ONNX payload only for the current call,
disables provider fallback, and reuses the shared prediction formatter.

## Usage

``` r
onnxPredict(
  model,
  newdata,
  batch_size = 64L,
  type = c("class", "probability", "response", "logit"),
  providers = "CPUExecutionProvider"
)
```

## Arguments

- model:

  A validated `physio_onnx_model`.

- newdata:

  A compatible `physio_torch_dataset`.

- batch_size:

  Positive whole-number inference batch size.

- type:

  Exact output type.

- providers:

  Execution provider; currently CPU only.

## Value

The same identity-preserving output shapes as
[`predictModel()`](https://x-biosignal.github.io/PhysioML/reference/predictModel.md).
