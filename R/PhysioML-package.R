#' PhysioML: leakage-aware machine learning for physiological time series
#'
#' Leakage-aware catch22, ROCKET, and MiniRocket transforms for
#' [PhysioExperiment::PhysioExperiment()] objects. Fitted convolution transforms are
#' persisted as validated raw payloads rather than live Python pointers.
#' Optional torch and luz workflows provide case-aware window Datasets,
#' compact reference neural modules, CPU training, and contract-checked
#' prediction. Governed ONNX export/import retains hash-bound plain contracts
#' and uses CPU ONNX Runtime without provider fallback. Named transfer-learning
#' policies keep frozen parameters and batch-normalization state fixed, CORAL
#' aligns explicitly designated unlabeled adaptation domains, and governed
#' foundation adapters require immutable revisions and complete cache
#' manifests. These mechanics do not establish clinical validity.
#'
#' @keywords internal
#' @importFrom PhysioExperiment defaultAssay samplingRate
"_PACKAGE"

utils::globalVariables(c("ctx", "self"))
