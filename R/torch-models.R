.ml_model_spec <- function(
    architecture,
    n_channels,
    n_time,
    n_outputs,
    task,
    sampling_rate,
    dropout,
    hidden,
    seed,
    dots) {
  n_channels <- .ml_whole_number(n_channels, "n_channels", minimum = 1L)
  n_time <- .ml_whole_number(n_time, "n_time", minimum = 1L)
  n_outputs <- .ml_whole_number(n_outputs, "n_outputs", minimum = 1L)
  dropout <- .ml_positive_number(dropout, "dropout", allow_zero = TRUE)
  if (dropout >= 1) {
    .ml_stop("`dropout` must be less than one")
  }
  hidden <- .ml_whole_number(hidden, "hidden", minimum = 1L)
  seed <- .ml_seed(seed)
  if (!is.null(sampling_rate)) {
    sampling_rate <- .ml_positive_number(sampling_rate, "sampling_rate")
  }
  allowed <- switch(
    architecture,
    eegnet = c("F1", "D", "kernel_length", "pool1", "pool2"),
    cnn1d = character(),
    tcn = c("levels", "kernel_size"),
    lstm = c("num_layers", "bidirectional")
  )
  unknown <- setdiff(names(dots), allowed)
  if (length(unknown) || any(!nzchar(names(dots)))) {
    .ml_stop(sprintf(
      "unsupported `%s` constructor argument(s): %s",
      architecture,
      paste(if (length(unknown)) unknown else "<unnamed>", collapse = ", ")
    ))
  }
  value <- function(name, default) {
    if (is.null(dots[[name]])) default else dots[[name]]
  }
  hyperparameters <- list(dropout = dropout, hidden = hidden)
  citations <- character()
  if (architecture == "eegnet") {
    f1 <- .ml_whole_number(value("F1", 8L), "F1", minimum = 1L)
    depth <- .ml_whole_number(value("D", 2L), "D", minimum = 1L)
    kernel <- value(
      "kernel_length",
      if (is.null(sampling_rate)) 64L else round(sampling_rate / 2)
    )
    kernel <- .ml_whole_number(kernel, "kernel_length", minimum = 2L)
    pool1 <- .ml_whole_number(value("pool1", 4L), "pool1", minimum = 1L)
    pool2 <- .ml_whole_number(value("pool2", 8L), "pool2", minimum = 1L)
    if (n_time < pool1 * pool2) {
      .ml_stop("EEGNet requires `n_time >= pool1 * pool2`")
    }
    hyperparameters <- c(
      hyperparameters,
      list(
        F1 = f1,
        D = depth,
        F2 = f1 * depth,
        kernel_length = kernel,
        pool1 = pool1,
        pool2 = pool2,
        separable_kernel = 16L,
        max_norm_spatial = 1,
        max_norm_classifier = 0.25
      )
    )
    citations <- "Lawhern et al. (2018), doi:10.1088/1741-2552/aace8c"
  } else if (architecture == "cnn1d") {
    if (n_time < 2L) {
      .ml_stop("CNN1D requires at least two time samples")
    }
    hyperparameters <- c(
      hyperparameters,
      list(kernel1 = 7L, kernel2 = 5L, pool = 2L)
    )
  } else if (architecture == "tcn") {
    levels <- .ml_whole_number(value("levels", 3L), "levels", minimum = 1L)
    kernel <- .ml_whole_number(
      value("kernel_size", 3L),
      "kernel_size",
      minimum = 2L
    )
    dilations <- 2^(seq_len(levels) - 1L)
    receptive_field <- 1L + 2L * (kernel - 1L) * sum(dilations)
    if (receptive_field > n_time) {
      .ml_stop(sprintf(
        "TCN receptive field %d exceeds `n_time` %d",
        receptive_field,
        n_time
      ))
    }
    hyperparameters <- c(
      hyperparameters,
      list(
        levels = levels,
        kernel_size = kernel,
        dilations = as.integer(dilations),
        receptive_field = as.integer(receptive_field)
      )
    )
    citations <- "Bai, Kolter & Koltun (2018), arXiv:1803.01271"
  } else {
    layers <- .ml_whole_number(
      value("num_layers", 1L),
      "num_layers",
      minimum = 1L
    )
    bidirectional <- value("bidirectional", FALSE)
    bidirectional <- .ml_flag(bidirectional, "bidirectional")
    hyperparameters <- c(
      hyperparameters,
      list(num_layers = layers, bidirectional = bidirectional)
    )
  }
  list(
    architecture = architecture,
    n_channels = n_channels,
    n_time = n_time,
    n_outputs = n_outputs,
    task = task,
    sampling_rate = sampling_rate,
    hyperparameters = hyperparameters,
    seed = seed,
    citations = citations,
    eegnet_variant = if (architecture == "eegnet") {
      "eegnet_canonical_max_norm"
    } else {
      NULL
    }
  )
}

.ml_eegnet_generator <- function(spec) {
  hp <- spec$hyperparameters
  torch::nn_module(
    "physio_eegnet",
    initialize = function() {
      self$temporal <- torch::nn_conv2d(
        1L,
        hp$F1,
        kernel_size = c(1L, hp$kernel_length),
        bias = FALSE
      )
      self$bn1 <- torch::nn_batch_norm2d(hp$F1)
      self$depthwise <- torch::nn_conv2d(
        hp$F1,
        hp$F2,
        kernel_size = c(spec$n_channels, 1L),
        groups = hp$F1,
        bias = FALSE
      )
      self$bn2 <- torch::nn_batch_norm2d(hp$F2)
      self$pool1 <- torch::nn_avg_pool2d(c(1L, hp$pool1))
      self$drop1 <- torch::nn_dropout(hp$dropout)
      self$separable_depthwise <- torch::nn_conv2d(
        hp$F2,
        hp$F2,
        kernel_size = c(1L, hp$separable_kernel),
        groups = hp$F2,
        bias = FALSE
      )
      self$separable_pointwise <- torch::nn_conv2d(
        hp$F2,
        hp$F2,
        kernel_size = c(1L, 1L),
        bias = FALSE
      )
      self$bn3 <- torch::nn_batch_norm2d(hp$F2)
      self$pool2 <- torch::nn_avg_pool2d(c(1L, hp$pool2))
      self$drop2 <- torch::nn_dropout(hp$dropout)
      flattened <- hp$F2 * floor(floor(spec$n_time / hp$pool1) / hp$pool2)
      self$classifier <- torch::nn_linear(flattened, spec$n_outputs)
    },
    forward = function(x) {
      x <- x$unsqueeze(2)
      left <- floor((hp$kernel_length - 1L) / 2L)
      right <- hp$kernel_length - 1L - left
      x <- torch::nnf_pad(x, c(left, right, 0L, 0L))
      x <- self$temporal(x)
      x <- self$bn1(x)
      x <- self$depthwise(x)
      x <- self$bn2(x)
      x <- torch::nnf_elu(x)
      x <- self$pool1(x)
      x <- self$drop1(x)
      left <- floor((hp$separable_kernel - 1L) / 2L)
      right <- hp$separable_kernel - 1L - left
      x <- torch::nnf_pad(x, c(left, right, 0L, 0L))
      x <- self$separable_depthwise(x)
      x <- self$separable_pointwise(x)
      x <- self$bn3(x)
      x <- torch::nnf_elu(x)
      x <- self$pool2(x)
      x <- self$drop2(x)
      x <- torch::torch_flatten(x, start_dim = 2L)
      self$classifier(x)
    }
  )
}

.ml_cnn1d_generator <- function(spec) {
  hp <- spec$hyperparameters
  torch::nn_module(
    "physio_cnn1d",
    initialize = function() {
      self$conv1 <- torch::nn_conv1d(
        spec$n_channels,
        hp$hidden,
        kernel_size = 7L,
        padding = 3L
      )
      self$bn1 <- torch::nn_batch_norm1d(hp$hidden)
      self$pool <- torch::nn_max_pool1d(2L)
      self$conv2 <- torch::nn_conv1d(
        hp$hidden,
        2L * hp$hidden,
        kernel_size = 5L,
        padding = 2L
      )
      self$bn2 <- torch::nn_batch_norm1d(2L * hp$hidden)
      self$average <- torch::nn_adaptive_avg_pool1d(1L)
      self$classifier <- torch::nn_linear(2L * hp$hidden, spec$n_outputs)
    },
    forward = function(x) {
      x <- self$pool(torch::nnf_relu(self$bn1(self$conv1(x))))
      x <- torch::nnf_relu(self$bn2(self$conv2(x)))
      x <- self$average(x)$squeeze(3)
      self$classifier(x)
    }
  )
}

.ml_tcn_block_generator <- function() {
  torch::nn_module(
    "physio_tcn_block",
    initialize = function(in_channels, out_channels, kernel_size, dilation,
                          dropout) {
      self$padding <- as.integer(dilation * (kernel_size - 1L))
      self$conv1 <- torch::nn_conv1d(
        in_channels,
        out_channels,
        kernel_size,
        dilation = dilation
      )
      self$conv2 <- torch::nn_conv1d(
        out_channels,
        out_channels,
        kernel_size,
        dilation = dilation
      )
      self$drop1 <- torch::nn_dropout(dropout)
      self$drop2 <- torch::nn_dropout(dropout)
      self$residual <- if (in_channels == out_channels) {
        torch::nn_identity()
      } else {
        torch::nn_conv1d(in_channels, out_channels, 1L)
      }
    },
    forward = function(x) {
      residual <- self$residual(x)
      out <- torch::nnf_pad(x, c(self$padding, 0L))
      out <- self$drop1(torch::nnf_relu(self$conv1(out)))
      out <- torch::nnf_pad(out, c(self$padding, 0L))
      out <- self$drop2(torch::nnf_relu(self$conv2(out)))
      torch::nnf_relu(out + residual)
    }
  )
}

.ml_tcn_generator <- function(spec) {
  hp <- spec$hyperparameters
  block <- .ml_tcn_block_generator()
  torch::nn_module(
    "physio_tcn",
    initialize = function() {
      modules <- vector("list", hp$levels)
      in_channels <- spec$n_channels
      for (level in seq_len(hp$levels)) {
        modules[[level]] <- block(
          in_channels = in_channels,
          out_channels = hp$hidden,
          kernel_size = hp$kernel_size,
          dilation = hp$dilations[[level]],
          dropout = hp$dropout
        )
        in_channels <- hp$hidden
      }
      self$blocks <- torch::nn_module_list(modules)
      self$classifier <- torch::nn_linear(hp$hidden, spec$n_outputs)
    },
    representations = function(x) {
      for (index in seq_len(length(self$blocks))) {
        x <- self$blocks[[index]](x)
      }
      x
    },
    forward = function(x) {
      x <- self$representations(x)
      x <- x$select(3L, x$size(3L))
      self$classifier(x)
    }
  )
}

.ml_lstm_generator <- function(spec) {
  hp <- spec$hyperparameters
  torch::nn_module(
    "physio_lstm",
    initialize = function() {
      self$lstm <- torch::nn_lstm(
        input_size = spec$n_channels,
        hidden_size = hp$hidden,
        num_layers = hp$num_layers,
        batch_first = TRUE,
        dropout = if (hp$num_layers > 1L) hp$dropout else 0,
        bidirectional = hp$bidirectional
      )
      directions <- if (hp$bidirectional) 2L else 1L
      self$classifier <- torch::nn_linear(
        hp$hidden * directions,
        spec$n_outputs
      )
    },
    representations = function(x) {
      output <- self$lstm(x$permute(c(1L, 3L, 2L)))[[1L]]
      output$select(2L, output$size(2L))
    },
    forward = function(x) {
      self$classifier(self$representations(x))
    }
  )
}

.ml_model_generator <- function(spec) {
  switch(
    spec$architecture,
    eegnet = .ml_eegnet_generator(spec),
    cnn1d = .ml_cnn1d_generator(spec),
    tcn = .ml_tcn_generator(spec),
    lstm = .ml_lstm_generator(spec)
  )
}

.ml_parameter_count <- function(module) {
  sum(vapply(
    module$parameters,
    function(parameter) as.numeric(parameter$numel()),
    numeric(1)
  ))
}

#' Construct a reference torch model
#'
#' Builds compact EEGNet, CNN1D, causal TCN, or unidirectional/bidirectional
#' LSTM modules. Classification models return logits; regression models return
#' raw values. Input is always batch by channel by time.
#'
#' @param architecture Exact architecture name.
#' @param n_channels,n_time,n_outputs Positive whole dimensions.
#' @param task Exact task name.
#' @param sampling_rate Optional positive sampling rate.
#' @param dropout Dropout probability in `[0, 1)`.
#' @param hidden Positive hidden width.
#' @param seed CPU initialization seed.
#' @param ... Architecture-specific named settings.
#'
#' @return A `physio_torch_module`.
#' @export
#' @examples
#' # Requires the optional torch backend.
#' if (requireNamespace("torch", quietly = TRUE)) {
#'   model <- peModel(
#'     architecture = "cnn1d",
#'     n_channels = 2, n_time = 50, n_outputs = 2,
#'     task = "classification"
#'   )
#' }
peModel <- function(
    architecture = c("eegnet", "cnn1d", "tcn", "lstm"),
    n_channels,
    n_time,
    n_outputs,
    task = c("classification", "regression"),
    sampling_rate = NULL,
    dropout = 0.5,
    hidden = 32L,
    seed = 1L,
    ...) {
  if (missing(architecture)) {
    architecture <- "eegnet"
  }
  if (missing(task)) {
    task <- "classification"
  }
  architecture <- .ml_exact_enum(
    architecture,
    c("eegnet", "cnn1d", "tcn", "lstm"),
    "architecture"
  )
  task <- .ml_exact_enum(task, c("classification", "regression"), "task")
  .ml_require_torch()
  spec <- .ml_model_spec(
    architecture,
    n_channels,
    n_time,
    n_outputs,
    task,
    sampling_rate,
    dropout,
    hidden,
    seed,
    list(...)
  )
  module <- .ml_preserve_r_rng(.ml_preserve_torch_rng({
    set.seed(spec$seed)
    torch::torch_manual_seed(spec$seed)
    .ml_model_generator(spec)()
  }))
  versions <- .ml_torch_versions()
  spec$torch_version <- versions$torch
  spec$libtorch_version <- versions$libtorch
  spec$parameter_count <- .ml_parameter_count(module)
  attr(module, "model_spec") <- spec
  attr(module, "physio_generator") <- .ml_model_generator(spec)
  class(module) <- unique(c("physio_torch_module", class(module)))
  module
}
