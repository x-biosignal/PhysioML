# PhysioML: leakage-aware machine learning for physiological time series

Leakage-aware catch22, ROCKET, and MiniRocket transforms for
[`PhysioCore::PhysioExperiment()`](https://x-biosignal.github.io/PhysioCore//reference/PhysioExperiment.html)
objects. Fitted convolution transforms are persisted as validated raw
payloads rather than live Python pointers. Optional torch and luz
workflows provide case-aware window Datasets, compact reference neural
modules, CPU training, and contract-checked prediction. Governed ONNX
export/import retains hash-bound plain contracts and uses CPU ONNX
Runtime without provider fallback. Named transfer-learning policies keep
frozen parameters and batch-normalization state fixed, CORAL aligns
explicitly designated unlabeled adaptation domains, and governed
foundation adapters require immutable revisions and complete cache
manifests. These mechanics do not establish clinical validity.

## See also

Useful links:

- <https://github.com/x-biosignal/PhysioML>

- <https://x-biosignal.r-universe.dev/PhysioML>

- <https://x-biosignal.github.io/PhysioML/>

- Report bugs at <https://github.com/x-biosignal/PhysioML/issues>

## Author

**Maintainer**: Yusuke Matsui <mail.to.matsui@gmail.com>

Authors:

- Yusuke Matsui <mail.to.matsui@gmail.com>
