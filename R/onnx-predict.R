.ml_validate_onnx_model <- function(model) {
  if (!inherits(model, "physio_onnx_model") ||
      !is.raw(model$onnx_payload) ||
      !is.character(model$onnx_sha256) ||
      length(model$onnx_sha256) != 1L ||
      !is.list(model$metadata) ||
      !identical(model$providers, "CPUExecutionProvider")) {
    .ml_stop("`model` must be a validated `physio_onnx_model`")
  }
  .ml_validate_card(model$metadata)
  if (!identical(.ml_sha256_raw(model$onnx_payload), model$onnx_sha256) ||
      !identical(model$onnx_sha256, model$metadata$onnx_sha256) ||
      length(model$onnx_payload) != model$metadata$onnx_size_bytes) {
    .ml_stop("ONNX model payload integrity validation failed")
  }
  invisible(TRUE)
}

.ml_validate_onnx_dataset <- function(card, newdata) {
  .ml_assert_dataset(newdata, "newdata")
  contract <- newdata$contract
  expected <- card$input_contract
  checks <- list(
    task = identical(contract$task, card$task),
    channel_ids = identical(contract$channel_ids, expected$channel_ids),
    window_samples = identical(
      as.integer(contract$window_samples),
      as.integer(expected$window_samples)
    ),
    dtype = identical(contract$dtype, expected$dtype),
    normalization_method = identical(
      contract$normalization_method,
      card$normalization_method
    ),
    class_levels = identical(contract$class_levels, card$class_levels)
  )
  failed <- names(checks)[!unlist(checks)]
  if (length(failed)) {
    .ml_stop(sprintf(
      "prediction `%s` contract does not match ONNX training",
      failed[[1L]]
    ))
  }
  if (dim(newdata$data)[[2L]] != expected$n_channels ||
      dim(newdata$data)[[3L]] != expected$n_time) {
    .ml_stop("prediction channel/time dimensions do not match ONNX training")
  }
  if (contract$normalization_method != "none" &&
      isTRUE(contract$normalization_fitted)) {
    .ml_stop(paste0(
      "prediction normalization was independently fitted; pass frozen ",
      "training `normalization_stats`"
    ))
  }
  if (!.ml_same_normalization(
    contract$normalization_stats,
    card$normalization_stats
  )) {
    if (!(is.null(contract$normalization_stats) &&
          is.null(card$normalization_stats))) {
      .ml_stop("prediction normalization statistics do not match ONNX training")
    }
  }
  invisible(TRUE)
}

#' Predict through ONNX Runtime on CPU
#'
#' Materializes a validated raw ONNX payload only for the current call, disables
#' provider fallback, and reuses the shared prediction formatter.
#'
#' @param model A validated `physio_onnx_model`.
#' @param newdata A compatible `physio_torch_dataset`.
#' @param batch_size Positive whole-number inference batch size.
#' @param type Exact output type.
#' @param providers Execution provider; currently CPU only.
#'
#' @return The same identity-preserving output shapes as [predictModel()].
#' @export
#' @examples
#' \donttest{
#' # `model` is from importONNX(); `newdata` is a matching peDataset() (CPU ORT).
#' onnxPredict(model, newdata, type = "class")
#' }
onnxPredict <- function(
    model,
    newdata,
    batch_size = 64L,
    type = c("class", "probability", "response", "logit"),
    providers = "CPUExecutionProvider") {
  .ml_validate_onnx_model(model)
  providers <- .ml_exact_enum(
    providers,
    "CPUExecutionProvider",
    "providers"
  )
  batch_size <- .ml_whole_number(batch_size, "batch_size", minimum = 1L)
  card <- model$metadata
  if (missing(type)) {
    type <- if (card$task == "classification") "class" else "response"
  }
  type <- .ml_exact_enum(
    type,
    c("class", "probability", "response", "logit"),
    "type"
  )
  if (card$task == "classification" && type == "response") {
    .ml_stop("`type = \"response\"` is available only for regression")
  }
  if (card$task == "regression" &&
      type %in% c("class", "probability", "logit")) {
    .ml_stop(sprintf("`type = \"%s\"` is available only for classification", type))
  }
  .ml_validate_onnx_dataset(card, newdata)
  graph_batch <- card$input$shape[[1L]]
  if (!identical(graph_batch, "batch")) {
    graph_batch <- as.integer(graph_batch)
    n_items <- nrow(newdata$contract$window_table)
    if (batch_size != graph_batch || n_items %% graph_batch != 0L) {
      .ml_stop(sprintf(
        paste0(
          "fixed-batch ONNX graph requires `batch_size = %d` and an item ",
          "count divisible by %d"
        ),
        graph_batch,
        graph_batch
      ))
    }
  }
  bridge <- .ml_onnx_python(c("onnxruntime"))
  if (!providers %in% unlist(bridge$versions()$providers)) {
    .ml_stop("ONNX Runtime CPUExecutionProvider is unavailable")
  }
  path <- .ml_write_raw_temp(model$onnx_payload)
  on.exit(unlink(path), add = TRUE)
  n <- nrow(newdata$contract$window_table)
  groups <- split(seq_len(n), ceiling(seq_len(n) / batch_size))
  output <- vector("list", length(groups))
  for (index in seq_along(groups)) {
    batch <- newdata$data[groups[[index]], , , drop = FALSE]
    result <- bridge$predict_model(path, batch)
    if (!identical(unlist(result$providers), "CPUExecutionProvider")) {
      .ml_stop("ONNX Runtime provider fallback was not disabled")
    }
    value <- as.matrix(result$output)
    if (nrow(value) != length(groups[[index]]) ||
        ncol(value) != card$n_outputs ||
        any(!is.finite(value))) {
      .ml_stop("ONNX Runtime returned an invalid batch x output result")
    }
    output[[index]] <- value
  }
  values <- do.call(rbind, output)
  format_model <- list(
    task = card$task,
    class_levels = card$class_levels,
    model_spec = list(n_outputs = as.integer(card$n_outputs))
  )
  .ml_format_predictions(values, format_model, newdata, type)
}
