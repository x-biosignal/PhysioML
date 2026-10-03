.ml_validate_prediction_contract <- function(model, newdata) {
  contract <- model$input_contract
  incoming <- newdata$contract
  checks <- c(
    "task", "channel_ids", "window_samples", "dtype",
    "normalization_method", "class_levels"
  )
  for (field in checks) {
    if (!identical(contract[[field]], incoming[[field]])) {
      .ml_stop(sprintf(
        "prediction `%s` contract does not match training",
        field
      ))
    }
  }
  if (dim(newdata$data)[[3L]] != model$model_spec$n_time) {
    .ml_stop("prediction time length does not match the trained model")
  }
  if (incoming$normalization_method != "none" &&
      isTRUE(incoming$normalization_fitted)) {
    .ml_stop(paste0(
      "prediction normalization was independently fitted; pass frozen ",
      "training `normalization_stats`"
    ))
  }
  if (!.ml_same_normalization(
    contract$normalization_stats,
    incoming$normalization_stats
  )) {
    if (!(is.null(contract$normalization_stats) &&
          is.null(incoming$normalization_stats))) {
      .ml_stop("prediction normalization statistics do not match training")
    }
  }
  invisible(TRUE)
}

.ml_predict_batches <- function(module, dataset, batch_size) {
  n <- nrow(dataset$contract$window_table)
  groups <- split(seq_len(n), ceiling(seq_len(n) / batch_size))
  predictions <- vector("list", length(groups))
  was_training <- isTRUE(module$training)
  module$eval()
  on.exit(module$train(mode = was_training), add = TRUE)
  torch::with_no_grad({
    for (index in seq_along(groups)) {
      batch <- dataset$.getbatch(groups[[index]])
      if (is.list(batch)) {
        batch <- batch[[1L]]
      }
      predictions[[index]] <- module(batch)
    }
  })
  output <- torch::torch_cat(predictions, dim = 1L)
  output <- output$to(device = "cpu")
  torch::as_array(output)
}

.ml_format_predictions <- function(values, model, newdata, type) {
  if (is.null(dim(values))) {
    values <- matrix(values, ncol = model$model_spec$n_outputs)
  }
  identity <- newdata$contract$window_table[c(
    "item", "case_id", "input_case_index", "window_in_case",
    "first_sample", "last_sample"
  )]
  subject_columns <- intersect(c("subject_id", "participant_id", "session_id",
    "trial_id", "side", "cycle_id"), names(newdata$contract$window_table))
  for (name in subject_columns) identity[[name]] <- newdata$contract$window_table[[name]]
  if (model$task == "classification") {
    colnames(values) <- model$class_levels
    if (type == "logit") {
      return(values)
    }
    shifted <- values - apply(values, 1L, max)
    probabilities <- exp(shifted)
    probabilities <- probabilities / rowSums(probabilities)
    colnames(probabilities) <- model$class_levels
    if (type == "probability") {
      return(probabilities)
    }
    predicted <- max.col(probabilities, ties.method = "first") - 1L
    identity$predicted_index <- as.integer(predicted)
    identity$predicted_label <- model$class_levels[predicted + 1L]
    return(identity)
  }
  colnames(values) <- paste0("response_", seq_len(ncol(values)))
  cbind(identity, as.data.frame(values, stringsAsFactors = FALSE))
}

#' Predict with a trained deep-learning model
#'
#' Inference uses evaluation mode and disabled gradients. The Dataset must
#' exactly match the training channel, time, task, class, dtype, and frozen
#' normalization contracts.
#'
#' @param model A fitted `physio_dl_model`.
#' @param newdata A compatible target-free or targeted Dataset.
#' @param batch_size Positive whole-number inference batch size.
#' @param type Exact output type.
#' @param num_workers Currently exactly zero.
#'
#' @return A numeric matrix or identity-preserving data frame.
#' @export
#' @examples
#' arr <- array(stats::rnorm(50 * 2 * 6), dim = c(50, 2, 6))
#' pe <- PhysioExperiment::PhysioExperiment(
#'   assays = list(raw = arr),
#'   colData = S4Vectors::DataFrame(label = c("C3", "C4")),
#'   samplingRate = 100
#' )
#' # Needs the optional torch and luz backends.
#' if (requireNamespace("torch", quietly = TRUE) &&
#'     requireNamespace("luz", quietly = TRUE)) {
#'   train <- peDataset(
#'     pe, targets = rep(c("a", "b"), 3),
#'     task = "classification", class_levels = c("a", "b")
#'   )
#'   fit <- trainModel(train, model = "cnn1d", epochs = 1L)
#'   predictModel(fit, train, type = "class")
#' }
predictModel <- function(
    model,
    newdata,
    batch_size = 64L,
    type = c("class", "probability", "response", "logit"),
    num_workers = 0L) {
  .ml_require_torch()
  if (!inherits(model, "physio_dl_model")) {
    .ml_stop("`model` must be a fitted `physio_dl_model`")
  }
  .ml_assert_dataset(newdata, "newdata")
  if (missing(type)) {
    type <- if (model$task == "classification") "class" else "response"
  }
  type <- .ml_exact_enum(
    type,
    c("class", "probability", "response", "logit"),
    "type"
  )
  batch_size <- .ml_whole_number(batch_size, "batch_size", minimum = 1L)
  num_workers <- .ml_whole_number(
    num_workers,
    "num_workers",
    minimum = 0L,
    maximum = 0L
  )
  .ml_validate_prediction_contract(model, newdata)
  if (model$task == "classification" && type == "response") {
    .ml_stop("`type = \"response\"` is available only for regression")
  }
  if (model$task == "regression" &&
      type %in% c("class", "probability", "logit")) {
    .ml_stop(sprintf("`type = \"%s\"` is available only for classification", type))
  }

  values <- .ml_predict_batches(model$fit$model, newdata, batch_size)
  .ml_format_predictions(values, model, newdata, type)
}
