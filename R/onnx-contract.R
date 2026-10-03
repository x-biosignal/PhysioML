.ml_onnx_max_bytes <- 512 * 1024^2
.ml_onnx_card_max_bytes <- 8 * 1024^2

.ml_onnx_python <- function(modules = c("torch", "onnx", "onnxruntime")) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    .ml_stop("the optional R package `reticulate` is required for ONNX")
  }
  if (!reticulate::py_available(initialize = TRUE)) {
    .ml_stop("Python is unavailable in the caller-managed reticulate environment")
  }
  for (module in modules) {
    if (!reticulate::py_module_available(module)) {
      .ml_stop(sprintf(
        "the Python module `%s` is required in the active environment",
        module
      ))
    }
  }
  path <- system.file("python", package = "PhysioML")
  if (!nzchar(path)) {
    candidate <- file.path("inst", "python")
    if (file.exists(file.path(candidate, "physio_onnx.py"))) {
      path <- normalizePath(candidate, mustWork = TRUE)
    }
  }
  if (!nzchar(path) || !file.exists(file.path(path, "physio_onnx.py"))) {
    .ml_stop("the installed PhysioML ONNX bridge is unavailable")
  }
  reticulate::import_from_path("physio_onnx", path = path, convert = TRUE)
}

.ml_major_minor <- function(version) {
  version <- sub("[+!-].*$", "", as.character(version))
  fields <- strsplit(version, ".", fixed = TRUE)[[1L]]
  if (length(fields) < 2L ||
      any(!grepl("^[0-9]+$", fields[1:2]))) {
    .ml_stop(sprintf("cannot parse backend version `%s`", version))
  }
  paste(fields[1:2], collapse = ".")
}

.ml_check_torch_bridge_versions <- function(python_versions) {
  r_versions <- .ml_torch_versions()
  if (!identical(
    .ml_major_minor(r_versions$libtorch),
    .ml_major_minor(python_versions$python_torch)
  )) {
    .ml_stop(sprintf(
      paste0(
        "R LibTorch %s and Python torch %s must have the same major/minor ",
        "version before loading TorchScript"
      ),
      r_versions$libtorch,
      python_versions$python_torch
    ))
  }
  list(r = r_versions, python = python_versions)
}

.ml_sha256_file <- function(path) {
  unname(digest::digest(file = path, algo = "sha256", serialize = FALSE))
}

.ml_sha256_raw <- function(value) {
  unname(digest::digest(value, algo = "sha256", serialize = FALSE))
}

.ml_sort_json <- function(value) {
  if (is.list(value) && !is.data.frame(value)) {
    value <- lapply(value, .ml_sort_json)
    if (!is.null(names(value))) {
      value <- value[order(names(value), method = "radix")]
    }
  }
  value
}

.ml_json_arrays <- function(value) {
  if (is.null(value)) {
    return(NULL)
  }
  if (is.atomic(value) && length(value) != 1L) {
    return(lapply(as.list(value), .ml_json_arrays))
  }
  if (is.list(value) && !is.data.frame(value)) {
    return(lapply(value, .ml_json_arrays))
  }
  value
}

.ml_simplify_json_arrays <- function(value) {
  if (!is.list(value) || is.data.frame(value)) {
    return(value)
  }
  value <- lapply(value, .ml_simplify_json_arrays)
  if (is.null(names(value)) && length(value)) {
    scalar <- vapply(value, function(item) {
      is.atomic(item) && length(item) == 1L
    }, logical(1))
    types <- unique(vapply(value, typeof, character(1)))
    if (all(scalar) && length(types) == 1L) {
      return(unlist(value, use.names = FALSE))
    }
  }
  value
}

.ml_assert_json_finite <- function(value, path = "card") {
  if (is.numeric(value) && any(!is.finite(value))) {
    .ml_stop(sprintf("`%s` contains a non-finite number", path))
  }
  if (is.list(value)) {
    labels <- names(value)
    if (is.null(labels)) labels <- as.character(seq_along(value))
    for (index in seq_along(value)) {
      .ml_assert_json_finite(
        value[[index]],
        paste0(path, "$", labels[[index]])
      )
    }
  }
  invisible(TRUE)
}

.ml_canonical_json <- function(value) {
  .ml_assert_json_finite(value)
  value <- .ml_json_arrays(value)
  value <- .ml_sort_json(value)
  paste0(
    jsonlite::toJSON(
      value,
      auto_unbox = TRUE,
      null = "null",
      na = "null",
      digits = 17,
      pretty = TRUE,
      dataframe = "rows",
      matrix = "rowmajor"
    ),
    "\n"
  )
}

.ml_read_card <- function(path) {
  if (!file.exists(path) || dir.exists(path)) {
    .ml_stop("ONNX model card must be a readable regular file")
  }
  size <- file.info(path)$size
  if (!is.finite(size) || size < 1 || size > .ml_onnx_card_max_bytes) {
    .ml_stop("ONNX model card is empty or exceeds the 8 MiB size ceiling")
  }
  bytes <- rawToChar(readBin(path, "raw", n = size))
  if (!jsonlite::validate(bytes)) {
    .ml_stop("ONNX model card is not valid JSON")
  }
  card <- jsonlite::fromJSON(bytes, simplifyVector = FALSE)
  if (!identical(bytes, .ml_canonical_json(card))) {
    .ml_stop(paste0(
      "ONNX model card must use canonical JSON with unique sorted keys"
    ))
  }
  card <- .ml_simplify_json_arrays(card)
  .ml_validate_card(card)
  card
}

.ml_validate_card <- function(card) {
  required <- c(
    "architecture", "backend_versions", "citations", "class_levels",
    "exporter_mode", "format", "input", "input_contract", "ir_version",
    "model_spec", "n_outputs", "normalization_method",
    "normalization_stats", "onnx_sha256", "onnx_size_bytes", "opset",
    "output", "package_versions", "producer_name", "producer_version",
    "schema_version", "task"
  )
  if (!is.list(card) || !identical(sort(names(card)), sort(required))) {
    .ml_stop("ONNX model card schema is incomplete or contains unknown fields")
  }
  scalar_text <- function(value, name, allow_empty = FALSE) {
    if (!is.character(value) || length(value) != 1L || is.na(value) ||
        (!allow_empty && !nzchar(value))) {
      .ml_stop(sprintf("model card `%s` must be one string", name))
    }
  }
  scalar_text(card$schema_version, "schema_version")
  scalar_text(card$format, "format")
  scalar_text(card$onnx_sha256, "onnx_sha256")
  scalar_text(card$task, "task")
  scalar_text(card$architecture, "architecture")
  scalar_text(card$exporter_mode, "exporter_mode")
  if (!identical(card$schema_version, "1.0.0") ||
      !identical(card$format, "onnx") ||
      !identical(card$exporter_mode, "torchscript-legacy-onnx") ||
      !grepl("^[0-9a-f]{64}$", card$onnx_sha256) ||
      !card$task %in% c("classification", "regression") ||
      !card$architecture %in% c("eegnet", "cnn1d", "tcn", "lstm")) {
    .ml_stop("ONNX model card contains an unsupported contract value")
  }
  for (name in c("onnx_size_bytes", "opset", "ir_version", "n_outputs")) {
    value <- card[[name]]
    if (!is.numeric(value) || length(value) != 1L || !is.finite(value) ||
        value < 1 || value != floor(value)) {
      .ml_stop(sprintf("model card `%s` must be a positive whole number", name))
    }
  }
  if (card$onnx_size_bytes > .ml_onnx_max_bytes) {
    .ml_stop("ONNX model card exceeds the 512 MiB graph size ceiling")
  }
  for (name in c("input", "output")) {
    item <- card[[name]]
    if (!is.list(item) ||
        !identical(
          sort(names(item)),
          c("dtype", "dynamic_axes", "name", "shape")
        ) ||
        !identical(item$dtype, "float32") ||
        !is.character(item$name) || length(item$name) != 1L ||
        !is.list(item$shape) && !is.atomic(item$shape)) {
      .ml_stop(sprintf("model card `%s` graph contract is invalid", name))
    }
  }
  if (!identical(card$input$name, "signal") ||
      !identical(card$output$name, "output")) {
    .ml_stop("ONNX graph input/output names must be `signal` and `output`")
  }
  contract <- card$input_contract
  required_contract <- c(
    "channel_ids", "dtype", "n_channels", "n_time", "stride_samples",
    "window_samples"
  )
  if (!is.list(contract) ||
      !identical(sort(names(contract)), sort(required_contract)) ||
      !identical(contract$dtype, "float32")) {
    .ml_stop("ONNX input contract is incomplete")
  }
  contract$channel_ids <- .ml_character_ids(
    contract$channel_ids,
    "card input_contract channel_ids"
  )
  if (length(contract$channel_ids) != contract$n_channels) {
    .ml_stop("ONNX card channel count does not match channel identities")
  }
  for (name in c(
    "n_channels", "n_time", "window_samples", "stride_samples"
  )) {
    value <- contract[[name]]
    if (!is.numeric(value) || length(value) != 1L || !is.finite(value) ||
        value < 1 || value != floor(value)) {
      .ml_stop(sprintf(
        "ONNX input contract `%s` must be a positive whole number",
        name
      ))
    }
  }
  if (contract$n_time != contract$window_samples) {
    .ml_stop("ONNX input time and window dimensions must match")
  }
  input_shape <- unname(vapply(
    card$input$shape,
    as.character,
    character(1)
  ))
  output_shape <- unname(vapply(
    card$output$shape,
    as.character,
    character(1)
  ))
  valid_batch <- function(value) {
    identical(value, "batch") ||
      (grepl("^[0-9]+$", value) && as.numeric(value) >= 1)
  }
  expected_dynamic_axes <- if (identical(input_shape[[1L]], "batch")) {
    list("0" = "batch")
  } else {
    NULL
  }
  if (length(input_shape) != 3L || !valid_batch(input_shape[[1L]]) ||
      !identical(input_shape[[2L]], as.character(contract$n_channels)) ||
      !identical(input_shape[[3L]], as.character(contract$n_time)) ||
      length(output_shape) != 2L ||
      !identical(output_shape[[1L]], input_shape[[1L]]) ||
      !identical(output_shape[[2L]], as.character(card$n_outputs)) ||
      !identical(card$input$dynamic_axes, expected_dynamic_axes) ||
      !identical(card$output$dynamic_axes, expected_dynamic_axes)) {
    .ml_stop("ONNX card graph shapes do not match its input/output contract")
  }
  if (!is.list(card$model_spec) ||
      !identical(card$model_spec$architecture, card$architecture) ||
      !identical(card$model_spec$task, card$task) ||
      card$model_spec$n_channels != contract$n_channels ||
      card$model_spec$n_time != contract$n_time ||
      card$model_spec$n_outputs != card$n_outputs) {
    .ml_stop("ONNX model specification does not match the model card")
  }
  if (!is.character(card$normalization_method) ||
      length(card$normalization_method) != 1L ||
      !card$normalization_method %in% c("none", "zscore", "robust")) {
    .ml_stop("ONNX card normalization method is unsupported")
  }
  if (card$normalization_method == "none") {
    if (!is.null(card$normalization_stats)) {
      .ml_stop("unnormalized ONNX cards must have null normalization statistics")
    }
  } else {
    .ml_validate_normalization(
      card$normalization_stats,
      card$normalization_method,
      contract$channel_ids
    )
  }
  if (card$task == "classification") {
    levels <- .ml_character_ids(card$class_levels, "card class_levels")
    if (length(levels) != card$n_outputs) {
      .ml_stop("ONNX card class levels do not match output width")
    }
  } else if (!is.null(card$class_levels)) {
    .ml_stop("regression ONNX cards must have null class levels")
  }
  .ml_assert_json_finite(card)
  invisible(TRUE)
}

.ml_validate_graph_card <- function(summary, card) {
  expected_input <- list(
    name = "signal",
    dtype = "float32",
    shape = c(
      if (identical(card$input$shape[[1L]], "batch")) "batch" else {
        as.integer(card$input$shape[[1L]])
      },
      as.integer(card$input_contract$n_channels),
      as.integer(card$input_contract$n_time)
    )
  )
  expected_output <- list(
    name = "output",
    dtype = "float32",
    shape = c(
      if (identical(card$output$shape[[1L]], "batch")) "batch" else {
        as.integer(card$output$shape[[1L]])
      },
      as.integer(card$n_outputs)
    )
  )
  normalize_shape <- function(value) {
    unname(vapply(value, function(item) {
      if (is.null(item)) "?" else as.character(item)
    }, character(1)))
  }
  same_io <- identical(summary$input$name, expected_input$name) &&
    identical(summary$input$dtype, expected_input$dtype) &&
    identical(
      normalize_shape(summary$input$shape),
      normalize_shape(expected_input$shape)
    ) &&
    identical(summary$output$name, expected_output$name) &&
    identical(summary$output$dtype, expected_output$dtype) &&
    identical(
      normalize_shape(summary$output$shape),
      normalize_shape(expected_output$shape)
    )
  if (!same_io || !identical(as.integer(summary$opset), as.integer(card$opset)) ||
      !identical(
        as.integer(summary$ir_version),
        as.integer(card$ir_version)
      )) {
    .ml_stop("ONNX graph and model card contracts do not match")
  }
  invisible(TRUE)
}
