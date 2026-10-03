# PhysioML 0.4.2

## Documentation

* A vignette carries one task end to end on synthetic or bundled data, offline,
  and is built and run by `R CMD check`.
* Runnable `@examples` added or corrected across 20 help pages. Each runs
  offline in seconds, writes nothing outside `tempdir()`, and is executed by
  `R CMD check`; anything needing a device, a download or an optional backend is
  fenced with the reason stated.
* The README's quick start runs as written: it attaches the package, builds its
  own inputs, and uses only hard dependencies.

# Development

- `withCaseData()` attaches exact case-to-subject/session/cycle identity to experiments or feature rows. Feature transforms, torch windows and serializable input contracts retain these identifiers.
- `validateSubjectSplit()` provides a backend-free check for shared cases/subjects. Training and fine-tuning contract checks reject shared subjects or one-sided identity metadata when subject mappings are supplied; legacy unannotated inputs retain case-only checking.

# PhysioML 0.4.1

- Made the foundation-cache symlink-rejection test portable: the test now
  skips when the build filesystem cannot create a symlink (e.g. the Windows
  r-universe binary builder), where `file.symlink()` fails silently so no
  unsafe condition exists to assert on. The security assertion itself is
  unchanged and still runs wherever symlinks work.

# PhysioML 0.3.0

- Adds governed R torch to TorchScript to Python PyTorch ONNX export with
  canonical hash-bound model cards and pre-landing checker/logit parity gates.
- Adds persistent raw-payload ONNX import and CPU-only ONNX Runtime prediction
  with provider fallback disabled and exact Dataset contract validation.

# PhysioML 0.2.0

- Adds `peDataset()` for exact case-aware windowing, zero-based class targets,
  and train-only z-score or robust normalization.
- Adds compact EEGNet, CNN1D, causal TCN, and LSTM modules through the optional
  R torch backend.
- Adds CPU `trainModel()` workflows through luz with case-disjoint validation,
  caller RNG restoration, and mandatory canonical EEGNet max-norm projection.
- Adds contract-checked `predictModel()` logits, probabilities, classes, and
  regression responses without changing batch-normalization buffers.

# PhysioML 0.1.0

- Adds exact PhysioExperiment input-axis and identity contracts.
- Adds canonical catch22 features through the optional Rcatch22 backend.
- Adds fitted, reusable ROCKET and MiniRocket transforms through the optional
  aeon backend.
- Adds `peReducedFeatures()` for case-by-feature design matrices with transform
  provenance.
