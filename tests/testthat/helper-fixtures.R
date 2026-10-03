test_venv <- normalizePath(
  test_path("..", "..", "..", "..", ".venv"),
  mustWork = FALSE
)
test_python <- file.path(test_venv, "bin", "python")
test_numba_cache <- file.path(tempdir(), "physioml-numba-cache")
dir.create(test_numba_cache, recursive = TRUE, showWarnings = FALSE)
Sys.setenv(NUMBA_CACHE_DIR = test_numba_cache)
if (!nzchar(Sys.getenv("RETICULATE_PYTHON")) && file.exists(test_python)) {
  Sys.setenv(RETICULATE_PYTHON = test_python)
}

tiny_revision <- "eba1a9334849d25ba420c25c4157d0743409726b"
tiny_manifest <- "702d25944e23a5cf609ba83e2872ed377161949f94f02ec8321f987f9f57f10e"

ml_fixture <- function(data, labels = NULL, case_ids = NULL, sr = 100) {
  if (is.null(labels)) {
    labels <- paste0("ch", seq_len(dim(data)[[2L]]))
  }
  if (length(dim(data)) == 3L && !is.null(case_ids)) {
    dimnames(data)[[3L]] <- case_ids
  }
  PhysioCore::PhysioExperiment(
    assays = list(raw = data),
    colData = S4Vectors::DataFrame(label = labels),
    samplingRate = sr
  )
}

ml_backend_available <- function() {
  requireNamespace("reticulate", quietly = TRUE) &&
    reticulate::py_available(initialize = TRUE) &&
    reticulate::py_module_available("numpy") &&
    reticulate::py_module_available("aeon")
}

ml_torch_available <- function(require_luz = FALSE) {
  available <- requireNamespace("torch", quietly = TRUE) &&
    isTRUE(tryCatch(torch::torch_is_installed(), error = function(e) FALSE))
  if (require_luz) {
    available <- available && requireNamespace("luz", quietly = TRUE)
  }
  available
}

ml_skip_torch <- function(require_luz = FALSE) {
  testthat::skip_if_not_installed("torch")
  testthat::skip_if_not(torch::torch_is_installed(), "LibTorch is unavailable")
  if (require_luz) {
    testthat::skip_if_not_installed("luz")
  }
}

ml_onnx_available <- function() {
  ml_torch_available() &&
    requireNamespace("reticulate", quietly = TRUE) &&
    reticulate::py_available(initialize = TRUE) &&
    all(vapply(
      c("torch", "onnx", "onnxruntime"),
      reticulate::py_module_available,
      logical(1)
    ))
}

ml_skip_onnx <- function() {
  ml_skip_torch()
  testthat::skip_if_not_installed("reticulate")
  testthat::skip_if_not(reticulate::py_available(initialize = TRUE))
  for (module in c("torch", "onnx", "onnxruntime")) {
    testthat::skip_if_not(
      reticulate::py_module_available(module),
      paste("Python module unavailable:", module)
    )
  }
}

ml_untrained_dl_model <- function(
    dataset,
    architecture = "cnn1d",
    task = dataset$contract$task,
    hidden = 4L,
    seed = 29L) {
  outputs <- if (task == "classification") {
    length(dataset$contract$class_levels)
  } else {
    1L
  }
  module <- peModel(
    architecture,
    n_channels = dim(dataset$data)[[2L]],
    n_time = dim(dataset$data)[[3L]],
    n_outputs = outputs,
    task = task,
    sampling_rate = dataset$contract$sampling_rate,
    dropout = 0,
    hidden = hidden,
    seed = seed
  )
  output <- list(
    fit = list(model = module),
    model_spec = attr(module, "model_spec"),
    input_contract = dataset$contract,
    task = task,
    class_levels = dataset$contract$class_levels,
    normalization_stats = dataset$contract$normalization_stats,
    versions = .ml_torch_versions()
  )
  class(output) <- "physio_dl_model"
  output
}

ml_toy_classification <- function(n_cases = 16L, n_time = 32L) {
  labels <- rep(c("low", "high"), each = n_cases / 2L)
  data <- array(
    0,
    c(n_time, 1L, n_cases),
    dimnames = list(NULL, NULL, paste0("patient_", seq_len(n_cases)))
  )
  wave <- sin(seq(0, 2 * pi, length.out = n_time))
  for (case in seq_len(n_cases)) {
    data[, 1L, case] <- wave + if (labels[[case]] == "high") 2 else -2
  }
  list(
    x = ml_fixture(data, labels = "Cz", sr = 100),
    targets = labels
  )
}

ml_toy_regression <- function(n_cases = 16L, n_time = 32L) {
  targets <- seq(-1, 1, length.out = n_cases)
  data <- array(
    0,
    c(n_time, 1L, n_cases),
    dimnames = list(NULL, NULL, paste0("subject_", seq_len(n_cases)))
  )
  trend <- seq(-0.1, 0.1, length.out = n_time)
  for (case in seq_len(n_cases)) {
    data[, 1L, case] <- targets[[case]] + trend
  }
  list(
    x = ml_fixture(data, labels = "Cz", sr = 100),
    targets = targets
  )
}

ml_transfer_fixture <- function(seed, n_cases, prefix) {
  set.seed(seed)
  targets <- rep(c("left", "right"), length.out = n_cases)
  data <- array(
    stats::rnorm(128L * 4L * n_cases, sd = 0.08),
    c(128L, 4L, n_cases)
  )
  motif <- pmax(0, sin(seq(0, 4 * pi, length.out = 128L)))
  amplitude <- stats::runif(n_cases, 0.8, 1.2)
  for (case in seq_len(n_cases)) {
    channel <- if (targets[[case]] == "left") 3L else 4L
    data[, channel, case] <- data[, channel, case] +
      amplitude[[case]] * motif
  }
  dimnames(data)[[3L]] <- paste0(prefix, "_", seq_len(n_cases))
  list(
    x = PhysioCore::PhysioExperiment(
      assays = list(raw = data),
      colData = S4Vectors::DataFrame(
        label = paste0("ch", 1:4),
        unit = rep("uV", 4)
      ),
      samplingRate = 128
    ),
    targets = targets
  )
}

ml_transfer_reference_model <- function(dataset) {
  module <- peModel(
    "cnn1d",
    n_channels = 4L,
    n_time = 128L,
    n_outputs = 2L,
    task = "classification",
    sampling_rate = 128,
    dropout = 0,
    hidden = 4L,
    seed = 91L
  )
  torch::with_no_grad({
    module$conv1$weight$zero_()
    for (index in 1:4) {
      module$conv1$weight[index, index, 4] <- 1
    }
    module$conv1$bias$zero_()
    module$bn1$weight$fill_(1)
    module$bn1$bias$zero_()
    module$bn1$running_mean$zero_()
    module$bn1$running_var$fill_(1)
    module$conv2$weight$zero_()
    module$conv2$weight[1, 1, 3] <- 1
    module$conv2$weight[2, 2, 3] <- 1
    module$conv2$weight[3, 1, 3] <- 0.5
    module$conv2$weight[4, 2, 3] <- 0.5
    module$conv2$bias$zero_()
    module$bn2$weight$fill_(1)
    module$bn2$bias$zero_()
    module$bn2$running_mean$zero_()
    module$bn2$running_var$fill_(1)
    module$classifier$weight$zero_()
    module$classifier$weight[1, 1] <- 1
    module$classifier$weight[2, 2] <- 1
    module$classifier$weight[1, 7] <- 1
    module$classifier$weight[2, 8] <- 1
    module$classifier$bias$zero_()
  })
  output <- list(
    fit = list(model = module),
    model_spec = attr(module, "model_spec"),
    input_contract = dataset$contract,
    task = "classification",
    class_levels = dataset$contract$class_levels,
    normalization_stats = dataset$contract$normalization_stats,
    versions = .ml_torch_versions()
  )
  class(output) <- "physio_dl_model"
  output
}
