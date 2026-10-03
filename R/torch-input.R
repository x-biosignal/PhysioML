.ml_with_envvar <- function(name, value, code) {
  old <- Sys.getenv(name, unset = NA_character_)
  on.exit({
    if (is.na(old)) {
      Sys.unsetenv(name)
    } else {
      do.call(Sys.setenv, stats::setNames(list(old), name))
    }
  }, add = TRUE)
  do.call(Sys.setenv, stats::setNames(list(value), name))
  force(code)
}

.ml_require_torch <- function(need_luz = FALSE) {
  available <- .ml_with_envvar(
    "TORCH_INSTALL",
    "0",
    requireNamespace("torch", quietly = TRUE)
  )
  if (!available) {
    .ml_stop(paste0(
      "the optional R package `torch` is required; install it and then ",
      "install LibTorch explicitly with `torch::install_torch()`"
    ))
  }
  installed <- tryCatch(
    isTRUE(torch::torch_is_installed()),
    error = function(e) FALSE
  )
  if (!installed) {
    .ml_stop(paste0(
      "the R package `torch` is installed but LibTorch is unavailable; ",
      "install it explicitly with `torch::install_torch()`"
    ))
  }
  if (need_luz && !requireNamespace("luz", quietly = TRUE)) {
    .ml_stop("the optional R package `luz` is required for training")
  }
  invisible(TRUE)
}

.ml_torch_versions <- function() {
  namespace <- asNamespace("torch")
  libtorch <- NA_character_
  if (exists("torch_version", envir = namespace, inherits = FALSE)) {
    candidate <- get("torch_version", envir = namespace)
    libtorch <- tryCatch({
      if (is.function(candidate)) {
        candidate <- candidate()
      }
      as.character(candidate)
    }, error = function(e) NA_character_)
  }
  if (is.na(libtorch) &&
      exists("torch_config", envir = namespace, inherits = FALSE)) {
    config <- tryCatch(
      get("torch_config", envir = namespace)(),
      error = function(e) NULL
    )
    for (name in c("libtorch_version", "version")) {
      if (!is.null(config[[name]])) {
        libtorch <- as.character(config[[name]])
        break
      }
    }
  }
  list(
    torch = as.character(utils::packageVersion("torch")),
    libtorch = libtorch
  )
}

.ml_positive_number <- function(value, name, allow_zero = FALSE) {
  lower_ok <- if (allow_zero) value >= 0 else value > 0
  if (!is.numeric(value) || length(value) != 1L || is.na(value) ||
      !is.finite(value) || !lower_ok) {
    qualifier <- if (allow_zero) "non-negative" else "positive"
    .ml_stop(sprintf("`%s` must be one finite %s number", name, qualifier))
  }
  as.numeric(value)
}

.ml_seed <- function(seed) {
  .ml_whole_number(seed, "seed", minimum = 0, maximum = 2^31 - 1)
}

.ml_dtype <- function(dtype) {
  dtype <- .ml_exact_enum(dtype, "float32", "dtype")
  torch::torch_float32()
}

.ml_plain_torch_contract <- function(contract) {
  fields <- c(
    "assay_name", "input_dimensions", "case_ids", "case_data", "input_case_indices",
    "channel_ids", "input_channel_indices", "sampling_rate",
    "window_samples", "stride_samples", "window_table", "trailing_samples",
    "task", "class_levels", "class_mapping", "normalization_method",
    "normalization_stats", "normalization_fitted", "dtype", "torch_version",
    "libtorch_version"
  )
  contract[intersect(fields, names(contract))]
}

.ml_same_numeric <- function(x, y, tolerance = 0) {
  isTRUE(all.equal(
    unname(x),
    unname(y),
    tolerance = tolerance,
    check.attributes = FALSE
  ))
}

.ml_same_normalization <- function(x, y) {
  identical(x$method, y$method) &&
    identical(x$channel_ids, y$channel_ids) &&
    .ml_same_numeric(x$center, y$center) &&
    .ml_same_numeric(x$scale, y$scale)
}

.ml_preserve_torch_rng <- function(code) {
  state <- torch::torch_get_rng_state()
  on.exit(torch::torch_set_rng_state(state), add = TRUE)
  force(code)
}
