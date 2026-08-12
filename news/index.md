# Changelog

## PhysioML 0.4.1

- Made the foundation-cache symlink-rejection test portable: the test
  now skips when the build filesystem cannot create a symlink (e.g. the
  Windows r-universe binary builder), where
  [`file.symlink()`](https://rdrr.io/r/base/files.html) fails silently
  so no unsafe condition exists to assert on. The security assertion
  itself is unchanged and still runs wherever symlinks work.

## PhysioML 0.3.0

- Adds governed R torch to TorchScript to Python PyTorch ONNX export
  with canonical hash-bound model cards and pre-landing checker/logit
  parity gates.
- Adds persistent raw-payload ONNX import and CPU-only ONNX Runtime
  prediction with provider fallback disabled and exact Dataset contract
  validation.

## PhysioML 0.2.0

- Adds
  [`peDataset()`](https://x-biosignal.github.io/PhysioML/reference/peDataset.md)
  for exact case-aware windowing, zero-based class targets, and
  train-only z-score or robust normalization.
- Adds compact EEGNet, CNN1D, causal TCN, and LSTM modules through the
  optional R torch backend.
- Adds CPU
  [`trainModel()`](https://x-biosignal.github.io/PhysioML/reference/trainModel.md)
  workflows through luz with case-disjoint validation, caller RNG
  restoration, and mandatory canonical EEGNet max-norm projection.
- Adds contract-checked
  [`predictModel()`](https://x-biosignal.github.io/PhysioML/reference/predictModel.md)
  logits, probabilities, classes, and regression responses without
  changing batch-normalization buffers.

## PhysioML 0.1.0

- Adds exact PhysioExperiment input-axis and identity contracts.
- Adds canonical catch22 features through the optional Rcatch22 backend.
- Adds fitted, reusable ROCKET and MiniRocket transforms through the
  optional aeon backend.
- Adds
  [`peReducedFeatures()`](https://x-biosignal.github.io/PhysioML/reference/peReducedFeatures.md)
  for case-by-feature design matrices with transform provenance.
