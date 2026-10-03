.ml_window_plan <- function(resolved, window_samples, stride_samples) {
  n_cases <- dim(resolved$data)[[1L]]
  n_time <- dim(resolved$data)[[3L]]
  if (is.null(window_samples)) {
    if (!is.null(stride_samples)) {
      .ml_stop("`stride_samples` must be NULL when `window_samples` is NULL")
    }
    window_samples <- n_time
    stride_samples <- n_time
  } else {
    window_samples <- .ml_whole_number(
      window_samples,
      "window_samples",
      minimum = 1L,
      maximum = n_time
    )
    if (is.null(stride_samples)) {
      stride_samples <- window_samples
    } else {
      stride_samples <- .ml_whole_number(
        stride_samples,
        "stride_samples",
        minimum = 1L
      )
    }
  }
  starts <- seq.int(1L, n_time - window_samples + 1L, by = stride_samples)
  n_windows <- length(starts)
  table <- data.frame(
    item = seq_len(n_cases * n_windows),
    case_id = rep(resolved$case_data$case_id, each = n_windows),
    input_case_index = rep(
      resolved$case_data$input_case_index,
      each = n_windows
    ),
    window_in_case = rep(seq_len(n_windows), times = n_cases),
    first_sample = rep(starts, times = n_cases),
    last_sample = rep(starts + window_samples - 1L, times = n_cases),
    target_index = rep(seq_len(n_cases), each = n_windows),
    stringsAsFactors = FALSE
  )
  identity_names <- setdiff(names(resolved$case_data), c("case_id", "input_case_index"))
  for (name in identity_names)
    table[[name]] <- resolved$case_data[[name]][table$target_index]
  trailing <- n_time - max(starts + window_samples - 1L)
  list(
    table = table,
    window_samples = as.integer(window_samples),
    stride_samples = as.integer(stride_samples),
    trailing_samples = as.integer(trailing)
  )
}

.ml_window_array <- function(data, plan) {
  n_items <- nrow(plan$table)
  n_channels <- dim(data)[[2L]]
  out <- array(
    0,
    dim = c(n_items, n_channels, plan$window_samples)
  )
  for (item in seq_len(n_items)) {
    case <- plan$table$target_index[[item]]
    samples <- plan$table$first_sample[[item]]:plan$table$last_sample[[item]]
    out[item, , ] <- data[case, , samples, drop = FALSE]
  }
  out
}

.ml_class_targets <- function(targets, class_levels, n_cases) {
  if (length(targets) != n_cases || anyNA(targets)) {
    .ml_stop("`targets` must contain exactly one non-missing value per selected case")
  }
  if (is.factor(targets)) {
    labels <- enc2utf8(as.character(targets))
    declared <- enc2utf8(levels(targets))
    if (length(setdiff(declared, unique(labels)))) {
      .ml_stop("factor `targets` contain unused declared levels")
    }
  } else if (is.character(targets)) {
    labels <- enc2utf8(targets)
    declared <- NULL
  } else if (is.numeric(targets) && !is.complex(targets) &&
             all(is.finite(targets)) && all(targets == floor(targets))) {
    labels <- format(targets, scientific = FALSE, trim = TRUE)
    declared <- NULL
  } else {
    .ml_stop(paste0(
      "classification `targets` must be character, factor, or finite ",
      "integer-like values"
    ))
  }
  if (any(!nzchar(labels))) {
    .ml_stop("classification `targets` must contain non-empty labels")
  }
  if (!is.null(class_levels)) {
    if (is.factor(class_levels)) {
      class_levels <- as.character(class_levels)
    } else if (is.numeric(class_levels) && !is.complex(class_levels) &&
               all(is.finite(class_levels)) &&
               all(class_levels == floor(class_levels))) {
      class_levels <- format(class_levels, scientific = FALSE, trim = TRUE)
    }
    class_levels <- .ml_character_ids(
      enc2utf8(class_levels),
      "class_levels"
    )
    if (length(setdiff(unique(labels), class_levels))) {
      .ml_stop("`class_levels` must cover every selected classification target")
    }
  } else if (!is.null(declared)) {
    class_levels <- declared
  } else {
    class_levels <- sort(unique(labels), method = "radix")
  }
  index <- match(labels, class_levels) - 1L
  list(
    values = as.integer(index),
    levels = class_levels,
    mapping = data.frame(
      class_label = class_levels,
      class_index = seq_along(class_levels) - 1L,
      stringsAsFactors = FALSE
    )
  )
}

.ml_regression_targets <- function(targets, n_cases) {
  if (!is.numeric(targets) || is.complex(targets) ||
      length(targets) != n_cases || anyNA(targets) ||
      any(!is.finite(targets))) {
    .ml_stop(paste0(
      "regression `targets` must contain exactly one finite numeric value ",
      "per selected case"
    ))
  }
  as.numeric(targets)
}

.ml_fit_normalization <- function(windows, method, channel_ids) {
  n_channels <- dim(windows)[[2L]]
  center <- scale <- numeric(n_channels)
  zero <- logical(n_channels)
  for (channel in seq_len(n_channels)) {
    values <- as.numeric(windows[, channel, , drop = FALSE])
    if (method == "zscore") {
      center[[channel]] <- mean(values)
      scale[[channel]] <- sqrt(mean((values - center[[channel]])^2))
    } else {
      center[[channel]] <- stats::median(values)
      scale[[channel]] <- 1.4826 *
        stats::median(abs(values - center[[channel]]))
    }
    if (scale[[channel]] == 0) {
      scale[[channel]] <- 1
      zero[[channel]] <- TRUE
    }
  }
  list(
    method = method,
    channel_ids = channel_ids,
    center = center,
    scale = scale,
    zero_scale = zero
  )
}

.ml_validate_normalization <- function(stats, method, channel_ids) {
  if (!is.list(stats) ||
      !identical(stats$method, method) ||
      !identical(stats$channel_ids, channel_ids) ||
      !is.numeric(stats$center) || !is.numeric(stats$scale) ||
      length(stats$center) != length(channel_ids) ||
      length(stats$scale) != length(channel_ids) ||
      anyNA(stats$center) || anyNA(stats$scale) ||
      any(!is.finite(stats$center)) ||
      any(!is.finite(stats$scale)) ||
      any(stats$scale <= 0)) {
    .ml_stop(paste0(
      "`normalization_stats` must exactly match method and channel order ",
      "with finite centers and positive finite scales"
    ))
  }
  stats
}

.ml_apply_normalization <- function(windows, stats) {
  for (channel in seq_len(dim(windows)[[2L]])) {
    windows[, channel, ] <- (
      windows[, channel, , drop = FALSE] - stats$center[[channel]]
    ) / stats$scale[[channel]]
  }
  windows
}

.ml_torch_dataset_generator <- function() {
  torch::dataset(
    "physio_torch_dataset",
    initialize = function(data, targets, target_type, contract, diagnostics) {
      self$data <- data
      self$targets <- targets
      self$target_type <- target_type
      self$contract <- contract
      self$diagnostics <- diagnostics
    },
    .length = function() {
      dim(self$data)[[1L]]
    },
    .getitem = function(index) {
      index <- as.integer(index)
      input <- array(
        self$data[index, , ],
        dim = dim(self$data)[2:3]
      )
      input <- torch::torch_tensor(input, dtype = torch::torch_float32())
      if (is.null(self$targets)) {
        return(input)
      }
      target <- self$targets[[index]]
      if (self$target_type == "classification") {
        target <- torch::torch_tensor(
          as.integer(target),
          dtype = torch::torch_long()
        )
      } else {
        target <- torch::torch_tensor(
          c(as.numeric(target)),
          dtype = torch::torch_float32()
        )
      }
      list(input, target)
    },
    .getbatch = function(index) {
      if (inherits(index, "torch_tensor")) {
        index <- index$to(device = "cpu")
        index <- as.integer(torch::as_array(index))
      } else {
        index <- as.integer(index)
      }
      input <- self$data[index, , , drop = FALSE]
      input <- torch::torch_tensor(input, dtype = torch::torch_float32())
      if (is.null(self$targets)) {
        return(input)
      }
      target <- self$targets[index]
      if (self$target_type == "classification") {
        target <- torch::torch_tensor(
          as.integer(target),
          dtype = torch::torch_long()
        )
      } else {
        target <- torch::torch_tensor(
          matrix(as.numeric(target), ncol = 1L),
          dtype = torch::torch_float32()
        )
      }
      list(input, target)
    }
  )
}

#' Create a case-aware torch Dataset
#'
#' Converts a PhysioExperiment assay to fixed channel-by-time items. Split
#' cases before calling this function, fit normalization on training cases
#' only, and pass `normalization_stats` to validation and prediction datasets.
#' Class tensors are zero-based `torch_long`; input and regression tensors are
#' `torch_float32`. A one-item batch always retains its batch dimension.
#'
#' @param x A `PhysioExperiment`; attach subject keys with [withCaseData()].
#' @param targets Optional one-per-selected-case target vector.
#' @param assay_name,cases,channels Exact assay and identity selections.
#' @param window_samples,stride_samples Optional complete-window dimensions.
#' @param normalization Exact normalization method.
#' @param normalization_stats Frozen statistics from a training dataset.
#' @param task Exact task name.
#' @param class_levels Optional ordered classification labels.
#' @param dtype Currently exactly `"float32"`.
#'
#' @return A `physio_torch_dataset`.
#' @export
#' @examples
#' # Prepare the input offline: a time x channel x case assay with channel labels.
#' arr <- array(stats::rnorm(50 * 2 * 6), dim = c(50, 2, 6))
#' pe <- PhysioExperiment::PhysioExperiment(
#'   assays = list(raw = arr),
#'   colData = S4Vectors::DataFrame(label = c("C3", "C4")),
#'   samplingRate = 100
#' )
#' # Building the Dataset itself needs the optional torch backend.
#' if (requireNamespace("torch", quietly = TRUE)) {
#'   ds <- peDataset(
#'     pe,
#'     targets = rep(c("rest", "task"), 3),
#'     task = "classification",
#'     class_levels = c("rest", "task")
#'   )
#' }
peDataset <- function(
    x,
    targets = NULL,
    assay_name = NULL,
    cases = NULL,
    channels = NULL,
    window_samples = NULL,
    stride_samples = NULL,
    normalization = c("none", "zscore", "robust"),
    normalization_stats = NULL,
    task = c("classification", "regression"),
    class_levels = NULL,
    dtype = "float32") {
  if (missing(normalization)) {
    normalization <- "none"
  }
  if (missing(task)) {
    task <- "classification"
  }
  normalization <- .ml_exact_enum(
    normalization,
    c("none", "zscore", "robust"),
    "normalization"
  )
  task <- .ml_exact_enum(task, c("classification", "regression"), "task")
  .ml_require_torch()
  .ml_dtype(dtype)

  resolved <- .ml_resolve_input(
    x,
    assay_name = assay_name,
    channels = channels,
    cases = cases
  )
  plan <- .ml_window_plan(resolved, window_samples, stride_samples)
  windows <- .ml_window_array(resolved$data, plan)
  n_cases <- dim(resolved$data)[[1L]]

  target_values <- NULL
  target_type <- task
  mapping <- NULL
  levels <- NULL
  if (!is.null(targets)) {
    if (task == "classification") {
      parsed <- .ml_class_targets(targets, class_levels, n_cases)
      target_values <- parsed$values[plan$table$target_index]
      levels <- parsed$levels
      mapping <- parsed$mapping
    } else {
      if (!is.null(class_levels)) {
        .ml_stop("`class_levels` must be NULL for regression")
      }
      parsed <- .ml_regression_targets(targets, n_cases)
      target_values <- parsed[plan$table$target_index]
    }
  } else if (!is.null(class_levels)) {
    levels <- .ml_character_ids(enc2utf8(class_levels), "class_levels")
    if (task == "classification") {
      mapping <- data.frame(
        class_label = levels,
        class_index = seq_along(levels) - 1L,
        stringsAsFactors = FALSE
      )
    } else {
      .ml_stop("`class_levels` must be NULL for regression")
    }
  }

  fitted <- FALSE
  diagnostics <- data.frame(
    type = character(),
    channel_id = character(),
    message = character(),
    stringsAsFactors = FALSE
  )
  stats <- NULL
  if (normalization == "none") {
    if (!is.null(normalization_stats)) {
      .ml_stop("`normalization_stats` must be NULL when normalization is `none`")
    }
  } else if (is.null(normalization_stats)) {
    stats <- .ml_fit_normalization(
      windows,
      normalization,
      resolved$channel_data$channel_id
    )
    fitted <- TRUE
  } else {
    stats <- .ml_validate_normalization(
      normalization_stats,
      normalization,
      resolved$channel_data$channel_id
    )
  }
  if (!is.null(stats)) {
    zero_scale <- stats$zero_scale
    if (is.null(zero_scale)) {
      zero_scale <- rep(FALSE, length(stats$scale))
    }
    zero <- which(zero_scale)
    if (length(zero)) {
      diagnostics <- data.frame(
        type = rep("zero_scale_channel", length(zero)),
        channel_id = stats$channel_ids[zero],
        message = rep("zero scale replaced by one", length(zero)),
        stringsAsFactors = FALSE
      )
    }
    windows <- .ml_apply_normalization(windows, stats)
  }

  versions <- .ml_torch_versions()
  contract <- list(
    assay_name = resolved$assay_name,
    input_dimensions = resolved$input_dimensions,
    case_ids = resolved$case_data$case_id,
    case_data = resolved$case_data,
    input_case_indices = resolved$case_data$input_case_index,
    channel_ids = resolved$channel_data$channel_id,
    input_channel_indices = resolved$channel_data$input_channel_index,
    sampling_rate = resolved$sampling_rate,
    window_samples = plan$window_samples,
    stride_samples = plan$stride_samples,
    window_table = plan$table,
    trailing_samples = plan$trailing_samples,
    task = task,
    class_levels = levels,
    class_mapping = mapping,
    normalization_method = normalization,
    normalization_stats = stats,
    normalization_fitted = fitted,
    dtype = dtype,
    torch_version = versions$torch,
    libtorch_version = versions$libtorch
  )
  generator <- .ml_torch_dataset_generator()
  dataset <- generator(
    data = windows,
    targets = target_values,
    target_type = target_type,
    contract = .ml_plain_torch_contract(contract),
    diagnostics = diagnostics
  )
  class(dataset) <- unique(c("physio_torch_dataset", class(dataset)))
  dataset
}
