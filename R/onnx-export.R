.ml_assert_dl_model <- function(model) {
  if (!inherits(model, "physio_dl_model") ||
      !is.list(model$model_spec) ||
      !is.list(model$input_contract) ||
      is.null(model$fit$model) ||
      !inherits(model$fit$model, "nn_module")) {
    .ml_stop("`model` must be a complete fitted `physio_dl_model`")
  }
  spec <- model$model_spec
  contract <- model$input_contract
  if (!identical(spec$task, model$task) ||
      !identical(contract$task, model$task) ||
      spec$n_channels != length(contract$channel_ids) ||
      spec$n_time != contract$window_samples ||
      !identical(contract$dtype, "float32") ||
      spec$n_outputs < 1L) {
    .ml_stop("fitted model input, output, and task contracts are inconsistent")
  }
  if (model$task == "classification" &&
      (!identical(model$class_levels, contract$class_levels) ||
       length(model$class_levels) != spec$n_outputs)) {
    .ml_stop("fitted model class mapping is inconsistent")
  }
  if (!identical(
    model$normalization_stats,
    contract$normalization_stats
  )) {
    .ml_stop("fitted model normalization statistics are inconsistent")
  }
  .ml_character_ids(contract$channel_ids, "model channel_ids")
  invisible(TRUE)
}

.ml_onnx_output_path <- function(path, name, extension = NULL) {
  if (!is.character(path) || length(path) != 1L || is.na(path) ||
      !nzchar(path) || dir.exists(path)) {
    .ml_stop(sprintf("`%s` must be one non-empty file path", name))
  }
  if (!is.null(extension) &&
      !endsWith(tolower(path), tolower(extension))) {
    .ml_stop(sprintf("`%s` must end in `%s`", name, extension))
  }
  parent <- dirname(path)
  if (!dir.exists(parent)) {
    .ml_stop(sprintf("parent directory for `%s` does not exist", name))
  }
  file.path(normalizePath(parent, mustWork = TRUE), basename(path))
}

.ml_clone_module <- function(module, spec) {
  state <- .ml_clone_state_dict(module)
  clone <- .ml_model_generator(spec)()
  clone$load_state_dict(state)
  clone
}

.ml_state_arrays <- function(module) {
  lapply(module$state_dict(), function(value) {
    torch::as_array(value$detach()$to(device = "cpu")$clone())
  })
}

.ml_landing_pair <- function(onnx_temp, card_temp, path, metadata_path, overwrite) {
  backup_onnx <- backup_card <- NULL
  if (overwrite && file.exists(path)) {
    backup_onnx <- tempfile(".onnx-backup-", dirname(path))
    if (!file.rename(path, backup_onnx)) {
      .ml_stop("could not prepare existing ONNX file for atomic overwrite")
    }
  }
  if (overwrite && file.exists(metadata_path)) {
    backup_card <- tempfile(".card-backup-", dirname(metadata_path))
    if (!file.rename(metadata_path, backup_card)) {
      if (!is.null(backup_onnx)) file.rename(backup_onnx, path)
      .ml_stop("could not prepare existing model card for atomic overwrite")
    }
  }
  landed_onnx <- FALSE
  on.exit({
    if (!landed_onnx && !is.null(backup_onnx) && file.exists(backup_onnx)) {
      file.rename(backup_onnx, path)
    }
    if (!landed_onnx && !is.null(backup_card) && file.exists(backup_card)) {
      file.rename(backup_card, metadata_path)
    }
  }, add = TRUE)
  if (!file.rename(onnx_temp, path)) {
    .ml_stop("could not atomically land the ONNX graph")
  }
  if (!file.rename(card_temp, metadata_path)) {
    unlink(path)
    .ml_stop("could not atomically land the ONNX model card")
  }
  landed_onnx <- TRUE
  if (!is.null(backup_onnx)) unlink(backup_onnx)
  if (!is.null(backup_card)) unlink(backup_card)
  invisible(TRUE)
}

#' Export a fitted PhysioML torch model to ONNX
#'
#' Uses the governed R torch to TorchScript to Python PyTorch bridge. Only the
#' batch axis may be dynamic. The graph and canonical model card are validated
#' before they are atomically landed.
#'
#' @param model A fitted `physio_dl_model`.
#' @param path Destination `.onnx` path.
#' @param opset Positive ONNX opset; the initial default is 17.
#' @param dynamic_batch Whether the batch axis is dynamic.
#' @param validate Whether to enforce the full ONNX checker, graph/card
#'   consistency, and logit parity tolerance. Basic graph parsing, I/O contract
#'   checks, and CPU inference are always required.
#' @param metadata_path Destination canonical JSON model-card path.
#' @param overwrite Whether an existing graph/card pair may be replaced.
#'
#' @return Invisibly, a plain `physio_onnx_export` list.
#' @export
#' @examples
#' arr <- array(stats::rnorm(50 * 2 * 6), dim = c(50, 2, 6))
#' pe <- PhysioExperiment::PhysioExperiment(
#'   assays = list(raw = arr),
#'   colData = S4Vectors::DataFrame(label = c("C3", "C4")),
#'   samplingRate = 100
#' )
#' path <- tempfile(fileext = ".onnx")
#' \donttest{
#' # Export needs torch plus the governed torch->ONNX Python bridge.
#' train <- peDataset(
#'   pe, targets = rep(c("a", "b"), 3),
#'   task = "classification", class_levels = c("a", "b")
#' )
#' fit <- trainModel(train, model = "cnn1d", epochs = 1L)
#' exportONNX(fit, path)
#' }
exportONNX <- function(
    model,
    path,
    opset = 17L,
    dynamic_batch = TRUE,
    validate = TRUE,
    metadata_path = paste0(path, ".json"),
    overwrite = FALSE) {
  .ml_require_torch()
  .ml_assert_dl_model(model)
  path <- .ml_onnx_output_path(path, "path", ".onnx")
  metadata_path <- .ml_onnx_output_path(metadata_path, "metadata_path", ".json")
  if (identical(path, metadata_path)) {
    .ml_stop("ONNX and model-card paths must be distinct")
  }
  opset <- .ml_whole_number(opset, "opset", minimum = 1L)
  dynamic_batch <- .ml_flag(dynamic_batch, "dynamic_batch")
  validate <- .ml_flag(validate, "validate")
  overwrite <- .ml_flag(overwrite, "overwrite")
  if (!overwrite && (file.exists(path) || file.exists(metadata_path))) {
    .ml_stop("output exists; set `overwrite = TRUE` to replace the pair")
  }
  if (model$model_spec$architecture == "lstm" && dynamic_batch) {
    .ml_stop(paste0(
      "dynamic-batch ONNX export is unsupported for the R torch LSTM ",
      "because its implicit initial state is traced at a fixed batch size; ",
      "use `dynamic_batch = FALSE`"
    ))
  }
  bridge <- .ml_onnx_python()
  python_versions <- bridge$versions()
  versions <- .ml_check_torch_bridge_versions(python_versions)
  if (!"CPUExecutionProvider" %in% unlist(python_versions$providers)) {
    .ml_stop("ONNX Runtime CPUExecutionProvider is unavailable")
  }

  spec <- model$model_spec
  n_channels <- as.integer(spec$n_channels)
  n_time <- as.integer(spec$n_time)
  example_array <- array(
    seq(-1, 1, length.out = 2L * n_channels * n_time),
    c(2L, n_channels, n_time)
  )
  onnx_temp <- tempfile(".onnx-", dirname(path), fileext = ".onnx")
  card_temp <- tempfile(".card-", dirname(metadata_path), fileext = ".json")
  torchscript_temp <- tempfile("physioml-", fileext = ".pt")
  on.exit(unlink(c(onnx_temp, card_temp, torchscript_temp)), add = TRUE)

  source_state <- .ml_state_arrays(model$fit$model)
  source_training <- isTRUE(model$fit$model$training)
  result <- .ml_preserve_r_rng(.ml_preserve_torch_rng({
    module <- .ml_clone_module(model$fit$model, spec)
    module$eval()
    example <- torch::torch_tensor(
      example_array,
      dtype = torch::torch_float32(),
      device = "cpu"
    )
    expected <- torch::with_no_grad({
      torch::as_array(module(example)$to(device = "cpu"))
    })
    if (!identical(dim(expected), c(2L, as.integer(spec$n_outputs))) ||
        any(!is.finite(expected))) {
      .ml_stop("R torch export logits have an invalid shape or value")
    }
    traced <- torch::jit_trace(
      module,
      example,
      respect_mode = FALSE
    )
    torch::jit_save(traced, torchscript_temp)
    exported <- bridge$export_model(
      torchscript_temp,
      onnx_temp,
      example_array,
      opset,
      dynamic_batch,
      validate
    )
    parity <- bridge$predict_model(onnx_temp, example_array)
    actual <- as.matrix(parity$output)
    if (!identical(dim(actual), dim(expected)) || any(!is.finite(actual))) {
      .ml_stop("ONNX Runtime export logits have an invalid shape or value")
    }
    absolute <- max(abs(actual - expected))
    relative <- max(
      abs(actual - expected) /
        pmax(abs(actual), abs(expected), 1e-7)
    )
    if (validate && (absolute > 1e-4 || relative > 1e-4)) {
      .ml_stop(sprintf(
        "R torch and ONNX logits differ (absolute %.8g, relative %.8g)",
        absolute,
        relative
      ))
    }
    list(
      summary = exported$summary,
      providers = unlist(parity$providers),
      absolute_error = absolute,
      relative_error = relative
    )
  }))
  if (!identical(source_state, .ml_state_arrays(model$fit$model)) ||
      !identical(source_training, isTRUE(model$fit$model$training))) {
    .ml_stop("source model state changed during ONNX export")
  }
  if (!identical(result$providers, "CPUExecutionProvider")) {
    .ml_stop("ONNX Runtime provider fallback was not disabled")
  }

  summary <- result$summary
  batch_dim <- if (dynamic_batch) "batch" else 2L
  expected_summary <- list(
    input = list(
      name = "signal",
      dtype = "float32",
      shape = list(batch_dim, n_channels, n_time)
    ),
    output = list(
      name = "output",
      dtype = "float32",
      shape = list(batch_dim, as.integer(spec$n_outputs))
    )
  )
  if (!identical(unname(unlist(summary$input)), unname(unlist(expected_summary$input))) ||
      !identical(unname(unlist(summary$output)), unname(unlist(expected_summary$output))) ||
      as.integer(summary$opset) != opset) {
    .ml_stop("exported ONNX graph does not match the requested I/O contract")
  }

  size <- file.info(onnx_temp)$size
  hash <- .ml_sha256_file(onnx_temp)
  contract <- model$input_contract
  normalization <- model$normalization_stats
  if (!is.null(normalization)) {
    normalization <- normalization[c(
      "method", "channel_ids", "center", "scale", "zero_scale"
    )]
  }
  card <- list(
    schema_version = "1.0.0",
    format = "onnx",
    onnx_sha256 = hash,
    onnx_size_bytes = as.numeric(size),
    opset = as.integer(summary$opset),
    ir_version = as.integer(summary$ir_version),
    producer_name = as.character(summary$producer_name),
    producer_version = as.character(summary$producer_version),
    exporter_mode = "torchscript-legacy-onnx",
    input = c(
      expected_summary$input,
      list(dynamic_axes = if (dynamic_batch) list("0" = "batch") else NULL)
    ),
    output = c(
      expected_summary$output,
      list(dynamic_axes = if (dynamic_batch) list("0" = "batch") else NULL)
    ),
    task = model$task,
    architecture = spec$architecture,
    n_outputs = as.integer(spec$n_outputs),
    model_spec = spec,
    input_contract = list(
      channel_ids = contract$channel_ids,
      n_channels = n_channels,
      n_time = n_time,
      window_samples = as.integer(contract$window_samples),
      stride_samples = as.integer(contract$stride_samples),
      dtype = "float32"
    ),
    normalization_method = contract$normalization_method,
    normalization_stats = normalization,
    class_levels = if (model$task == "classification") model$class_levels else NULL,
    package_versions = list(
      PhysioML = as.character(utils::packageVersion("PhysioML")),
      torch = versions$r$torch,
      luz = as.character(utils::packageVersion("luz")),
      reticulate = as.character(utils::packageVersion("reticulate"))
    ),
    backend_versions = list(
      r_torch = versions$r$torch,
      libtorch = versions$r$libtorch,
      python = python_versions$python,
      python_torch = python_versions$python_torch,
      onnx = python_versions$onnx,
      onnxruntime = python_versions$onnxruntime
    ),
    citations = c(
      "ONNX specification: https://onnx.ai/",
      "ONNX Runtime: https://onnxruntime.ai/",
      "PyTorch ONNX: https://pytorch.org/docs/stable/onnx.html"
    )
  )
  .ml_validate_card(card)
  writeChar(
    .ml_canonical_json(card),
    card_temp,
    eos = NULL,
    useBytes = TRUE
  )
  if (validate) {
    .ml_validate_graph_card(summary, card)
  }
  .ml_landing_pair(
    onnx_temp,
    card_temp,
    path,
    metadata_path,
    overwrite
  )
  output <- list(
    path = path,
    metadata_path = metadata_path,
    onnx_sha256 = hash,
    onnx_size_bytes = as.numeric(size),
    graph = summary,
    max_absolute_error = result$absolute_error,
    max_relative_error = result$relative_error,
    versions = card$backend_versions
  )
  class(output) <- "physio_onnx_export"
  invisible(output)
}
