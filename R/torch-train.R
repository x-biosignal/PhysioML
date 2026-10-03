.ml_assert_dataset <- function(x, name, require_targets = FALSE) {
  if (!inherits(x, "physio_torch_dataset")) {
    .ml_stop(sprintf("`%s` must be a `physio_torch_dataset`", name))
  }
  if (require_targets && is.null(x$targets)) {
    .ml_stop(sprintf("`%s` must contain targets", name))
  }
  invisible(TRUE)
}

.ml_compare_dataset_contracts <- function(train, valid) {
  left <- train$contract
  right <- valid$contract
  exact <- c(
    "task", "channel_ids", "window_samples", "dtype",
    "normalization_method", "class_levels"
  )
  for (field in exact) {
    if (!identical(left[[field]], right[[field]])) {
      .ml_stop(sprintf(
        "train/valid `%s` contracts must match exactly",
        field
      ))
    }
  }
  shared <- intersect(left$case_ids, right$case_ids)
  if (length(shared)) {
    .ml_stop(sprintf(
      "train/valid case IDs must be disjoint; shared: %s",
      paste(shared, collapse = ", ")
    ))
  }
  has_subject <- function(contract) !is.null(contract$case_data) &&
    "subject_id" %in% names(contract$case_data)
  annotated <- c(has_subject(left), has_subject(right))
  if (any(annotated)) {
    if (!all(annotated))
      .ml_stop("Both train and valid must carry subject case_data; identity is missing on one side")
    identity <- function(contract) {
      d <- contract$case_data
      d$input_case_index <- NULL
      .ml_align_case_data(d, contract$case_ids)
    }
    validateSubjectSplit(identity(left), identity(right))
  }
  if (left$normalization_method != "none" &&
      isTRUE(right$normalization_fitted)) {
    .ml_stop(paste0(
      "validation normalization was independently fitted; pass frozen ",
      "training `normalization_stats`"
    ))
  }
  if (!.ml_same_normalization(
    left$normalization_stats,
    right$normalization_stats
  )) {
    if (!(is.null(left$normalization_stats) &&
          is.null(right$normalization_stats))) {
      .ml_stop("train/valid normalization statistics must match exactly")
    }
  }
  invisible(TRUE)
}

.ml_clone_state_dict <- function(module) {
  lapply(module$state_dict(), function(value) {
    value <- value$detach()
    value$clone()
  })
}

.ml_initial_state_callback <- function(state) {
  callback <- luz::luz_callback(
    "physio_initial_state",
    initialize = function(state) {
      self$state <- state
    },
    on_fit_begin = function() {
      ctx$model$load_state_dict(self$state)
    }
  )
  callback(state)
}

.ml_project_rows <- function(weight, max_norm) {
  flat <- weight$view(c(weight$size(1L), -1L))
  norms <- flat$norm(p = 2, dim = 2L, keepdim = TRUE)
  scale <- torch::torch_clamp(max_norm / norms, max = 1)
  weight$copy_((flat * scale)$view_as(weight))
  invisible(NULL)
}

.ml_eegnet_max_norm_callback <- function() {
  callback <- luz::luz_callback(
    "physio_eegnet_max_norm",
    on_train_batch_end = function() {
      torch::with_no_grad({
        .ml_project_rows(ctx$model$depthwise$weight, 1)
        .ml_project_rows(ctx$model$classifier$weight, 0.25)
      })
    }
  )
  callback()
}

.ml_accuracy_metric <- function() {
  metric <- luz::luz_metric(
    "physio_zero_based_accuracy",
    abbrev = "accuracy",
    initialize = function() {
      self$correct <- 0
      self$total <- 0
    },
    update = function(predictions, targets) {
      predicted <- torch::torch_argmax(predictions, dim = 2L) - 1L
      self$correct <- self$correct +
        as.numeric((predicted == targets)$sum()$item())
      self$total <- self$total + as.numeric(targets$numel())
    },
    compute = function() {
      self$correct / self$total
    }
  )
  metric()
}

.ml_rmse_metric <- function() {
  metric <- luz::luz_metric(
    "physio_rmse",
    abbrev = "rmse",
    initialize = function() {
      self$squared_error <- 0
      self$total <- 0
    },
    update = function(predictions, targets) {
      error <- predictions - targets
      self$squared_error <- self$squared_error +
        as.numeric((error^2)$sum()$item())
      self$total <- self$total + as.numeric(targets$numel())
    },
    compute = function() {
      sqrt(self$squared_error / self$total)
    }
  )
  metric()
}

.ml_history_frame <- function(fit) {
  records <- fit$records$metrics
  rows <- list()
  index <- 0L
  for (set in c("train", "valid")) {
    values <- records[[set]]
    if (!length(values)) {
      next
    }
    for (epoch in seq_along(values)) {
      index <- index + 1L
      row <- data.frame(
        epoch = as.integer(epoch),
        set = set,
        stringsAsFactors = FALSE
      )
      metric_names <- names(values[[epoch]])
      metric_names[is.na(metric_names) | !nzchar(metric_names)] <-
        paste0("metric_", which(is.na(metric_names) | !nzchar(metric_names)))
      for (position in seq_along(values[[epoch]])) {
        value <- as.numeric(values[[epoch]][[position]])
        if (length(value) == 1L) {
          row[[metric_names[[position]]]] <- value
        }
      }
      rows[[index]] <- row
    }
  }
  if (!length(rows)) {
    return(data.frame(
      epoch = integer(),
      set = character(),
      stringsAsFactors = FALSE
    ))
  }
  names_all <- unique(unlist(lapply(rows, names), use.names = FALSE))
  rows <- lapply(rows, function(row) {
    missing <- setdiff(names_all, names(row))
    for (name in missing) {
      row[[name]] <- NA_real_
    }
    row[names_all]
  })
  do.call(rbind, rows)
}

.ml_validate_callbacks <- function(callbacks) {
  if (is.null(callbacks)) {
    return(list())
  }
  if (!is.list(callbacks) ||
      any(!vapply(callbacks, inherits, logical(1), what = "LuzCallback"))) {
    .ml_stop("`callbacks` must be NULL or a list of luz callback instances")
  }
  callbacks
}

.ml_prepare_training_model <- function(model, train, seed, dots) {
  if (inherits(model, "physio_torch_module")) {
    if (length(dots)) {
      .ml_stop("`...` must be empty when `model` is a module")
    }
    module <- model
    spec <- attr(module, "model_spec")
  } else {
    architecture <- .ml_exact_enum(
      model,
      c("eegnet", "cnn1d", "tcn", "lstm"),
      "model"
    )
    n_outputs <- if (train$contract$task == "classification") {
      length(train$contract$class_levels)
    } else {
      1L
    }
    module <- do.call(
      peModel,
      c(
        list(
          architecture = architecture,
          n_channels = dim(train$data)[[2L]],
          n_time = dim(train$data)[[3L]],
          n_outputs = n_outputs,
          task = train$contract$task,
          sampling_rate = train$contract$sampling_rate,
          seed = seed
        ),
        dots
      )
    )
    spec <- attr(module, "model_spec")
  }
  if (spec$n_channels != dim(train$data)[[2L]] ||
      spec$n_time != dim(train$data)[[3L]] ||
      spec$task != train$contract$task) {
    .ml_stop("model input shape and task must match the training Dataset")
  }
  if (spec$task == "classification" &&
      spec$n_outputs != length(train$contract$class_levels)) {
    .ml_stop("classification model outputs must match declared class levels")
  }
  if (spec$task == "regression" && spec$n_outputs != 1L) {
    .ml_stop("training currently supports one regression target per case")
  }
  list(module = module, spec = spec)
}

#' Train a reference torch model through luz
#'
#' The caller supplies case-disjoint training and validation Datasets.
#' When subject metadata is attached with [withCaseData()], both datasets must
#' carry it and subjects must also be disjoint. Legacy unannotated inputs retain
#' case-only checking; missing metadata does not establish subject separation.
#' Validation must reuse training normalization statistics. CPU execution,
#' no worker processes, and deterministic caller-state restoration are enforced
#' in this implementation.
#'
#' @param train,valid Targeted `physio_torch_dataset` objects.
#' @param model Exact architecture name or a `physio_torch_module`.
#' @param epochs,batch_size Positive whole-number training settings.
#' @param learning_rate Positive Adam learning rate.
#' @param weight_decay Non-negative Adam weight decay.
#' @param callbacks NULL or a list of luz callback instances.
#' @param seed Whole-number CPU seed.
#' @param device Currently exactly `"cpu"`.
#' @param num_workers Currently exactly zero.
#' @param verbose Whether luz prints progress.
#' @param ... Architecture settings when `model` is a name.
#'
#' @return A `physio_dl_model`.
#' @export
#' @examples
#' arr <- array(stats::rnorm(50 * 2 * 6), dim = c(50, 2, 6))
#' pe <- PhysioExperiment::PhysioExperiment(
#'   assays = list(raw = arr),
#'   colData = S4Vectors::DataFrame(label = c("C3", "C4")),
#'   samplingRate = 100
#' )
#' # Training needs the optional torch and luz backends.
#' if (requireNamespace("torch", quietly = TRUE) &&
#'     requireNamespace("luz", quietly = TRUE)) {
#'   train <- peDataset(
#'     pe, targets = rep(c("a", "b"), 3),
#'     task = "classification", class_levels = c("a", "b")
#'   )
#'   fit <- trainModel(train, model = "cnn1d", epochs = 1L)
#' }
trainModel <- function(
    train,
    valid = NULL,
    model = c("eegnet", "cnn1d", "tcn", "lstm"),
    epochs = 20L,
    batch_size = 32L,
    learning_rate = 1e-3,
    weight_decay = 0,
    callbacks = NULL,
    seed = 1L,
    device = "cpu",
    num_workers = 0L,
    verbose = FALSE,
    ...) {
  .ml_require_torch(need_luz = TRUE)
  .ml_assert_dataset(train, "train", require_targets = TRUE)
  if (missing(model)) {
    model <- "eegnet"
  }
  if (!is.null(valid)) {
    .ml_assert_dataset(valid, "valid", require_targets = TRUE)
    .ml_compare_dataset_contracts(train, valid)
  }
  epochs <- .ml_whole_number(epochs, "epochs", minimum = 1L)
  batch_size <- .ml_whole_number(batch_size, "batch_size", minimum = 1L)
  learning_rate <- .ml_positive_number(learning_rate, "learning_rate")
  weight_decay <- .ml_positive_number(
    weight_decay,
    "weight_decay",
    allow_zero = TRUE
  )
  seed <- .ml_seed(seed)
  device <- .ml_exact_enum(device, "cpu", "device")
  num_workers <- .ml_whole_number(
    num_workers,
    "num_workers",
    minimum = 0L,
    maximum = 0L
  )
  verbose <- .ml_flag(verbose, "verbose")
  callbacks <- .ml_validate_callbacks(callbacks)

  if (train$contract$task == "classification") {
    if (length(train$contract$class_levels) < 2L) {
      .ml_stop("classification training requires at least two classes")
    }
    expected <- seq_along(train$contract$class_levels) - 1L
    if (!identical(sort(unique(train$targets)), as.integer(expected))) {
      .ml_stop("training data must contain every declared classification class")
    }
  }

  prepared <- .ml_prepare_training_model(model, train, seed, list(...))
  module <- prepared$module
  spec <- prepared$spec
  initial_state <- .ml_clone_state_dict(module)
  generator <- attr(module, "physio_generator")
  if (is.null(generator)) {
    generator <- .ml_model_generator(spec)
  }

  result <- .ml_preserve_r_rng(.ml_preserve_torch_rng({
    set.seed(seed)
    torch::torch_manual_seed(seed)
    train_loader <- torch::dataloader(
      train,
      batch_size = batch_size,
      shuffle = TRUE,
      num_workers = num_workers,
      drop_last = FALSE
    )
    valid_loader <- if (is.null(valid)) {
      NULL
    } else {
      torch::dataloader(
        valid,
        batch_size = batch_size,
        shuffle = FALSE,
        num_workers = num_workers,
        drop_last = FALSE
      )
    }
    if (spec$task == "classification") {
      loss <- function(input, target) {
        torch::nn_cross_entropy_loss()(input, target + 1L)
      }
      metrics <- list(.ml_accuracy_metric())
    } else {
      loss <- torch::nn_mse_loss()
      metrics <- list(.ml_rmse_metric())
    }
    configured <- luz::setup(
      generator,
      loss = loss,
      optimizer = torch::optim_adam,
      metrics = metrics
    )
    configured <- luz::set_opt_hparams(
      configured,
      lr = learning_rate,
      weight_decay = weight_decay
    )
    fit_callbacks <- c(
      list(.ml_initial_state_callback(initial_state)),
      callbacks
    )
    if (spec$architecture == "eegnet") {
      fit_callbacks <- c(
        fit_callbacks,
        list(.ml_eegnet_max_norm_callback())
      )
    }
    fit <- luz::fit(
      configured,
      data = train_loader,
      epochs = epochs,
      callbacks = fit_callbacks,
      valid_data = valid_loader,
      accelerator = luz::accelerator(cpu = TRUE),
      verbose = verbose
    )
    list(
      fit = fit,
      history = .ml_history_frame(fit)
    )
  }))

  versions <- .ml_torch_versions()
  output <- list(
    fit = result$fit,
    model_spec = spec,
    input_contract = train$contract,
    task = train$contract$task,
    class_levels = train$contract$class_levels,
    normalization_stats = train$contract$normalization_stats,
    training_history = result$history,
    settings = list(
      epochs = epochs,
      batch_size = batch_size,
      learning_rate = learning_rate,
      weight_decay = weight_decay,
      seed = seed,
      device = device,
      num_workers = num_workers,
      validation_shuffle = FALSE,
      drop_last = FALSE
    ),
    versions = c(
      versions,
      list(
        luz = as.character(utils::packageVersion("luz")),
        torch_threads = torch::torch_get_num_threads(),
        torch_interop_threads = torch::torch_get_num_interop_threads()
      )
    ),
    diagnostics = data.frame(
      type = character(),
      message = character(),
      stringsAsFactors = FALSE
    )
  )
  class(output) <- "physio_dl_model"
  output
}
