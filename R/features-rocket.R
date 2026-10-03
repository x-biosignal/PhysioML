.ml_require_aeon <- function() {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    .ml_stop(paste0(
      "ROCKET transforms require the optional R package `reticulate` and a ",
      "caller-managed Python environment with NumPy and aeon"
    ))
  }
  if (!reticulate::py_available(initialize = TRUE)) {
    .ml_stop(paste0(
      "reticulate cannot initialize Python; configure a Python environment ",
      "with NumPy and aeon before calling this transform"
    ))
  }
  missing_modules <- c("numpy", "aeon")[
    !vapply(
      c("numpy", "aeon"),
      reticulate::py_module_available,
      logical(1)
    )
  ]
  if (length(missing_modules)) {
    .ml_stop(sprintf(
      "the active reticulate Python lacks required module(s): %s",
      paste(missing_modules, collapse = ", ")
    ))
  }
  convolution <- reticulate::import(
    "aeon.transformations.collection.convolution_based",
    delay_load = TRUE,
    convert = FALSE
  )
  for (class_name in c("Rocket", "MiniRocket")) {
    if (!reticulate::py_has_attr(convolution, class_name)) {
      .ml_stop(sprintf(
        "the installed aeon backend does not provide `%s`",
        class_name
      ))
    }
  }
  aeon <- reticulate::import("aeon", delay_load = TRUE, convert = TRUE)
  numpy <- reticulate::import("numpy", delay_load = TRUE, convert = TRUE)
  config <- reticulate::py_config()
  list(
    convolution = convolution,
    aeon_version = as.character(aeon$`__version__`),
    numpy_version = as.character(numpy$`__version__`),
    python_version = as.character(config$version)
  )
}

.ml_pickle_payload <- function(object) {
  path <- tempfile(fileext = ".pickle")
  on.exit(unlink(path), add = TRUE)
  reticulate::py_save_object(object, path, pickle = "pickle")
  size <- file.info(path)$size
  if (!is.finite(size) || size < 1 || size > .Machine$integer.max) {
    .ml_stop("aeon produced an invalid serialized model payload")
  }
  readBin(path, what = "raw", n = as.integer(size))
}

.ml_unpickle_payload <- function(payload) {
  path <- tempfile(fileext = ".pickle")
  on.exit(unlink(path), add = TRUE)
  writeBin(payload, path)
  reticulate::py_load_object(path, pickle = "pickle", convert = FALSE)
}

.ml_payload_hash <- function(payload) {
  digest::digest(payload, algo = "sha256", serialize = FALSE)
}

.ml_validate_rocket_model <- function(model, method) {
  required <- c(
    "method", "backend", "backend_version", "python_version",
    "numpy_version", "payload", "payload_sha256", "n_kernels_requested",
    "n_features_realized", "seed", "normalize", "deterministic",
    "channel_ids", "n_time", "feature_names", "fit_case_ids",
    "input_dtype", "citations"
  )
  if (!inherits(model, "physio_rocket_model") || !is.list(model) ||
      !all(required %in% names(model))) {
    .ml_stop("`model` is not a complete `physio_rocket_model`")
  }
  scalar_text <- function(value) {
    is.character(value) && length(value) == 1L &&
      !is.na(value) && nzchar(value)
  }
  if (!scalar_text(model$method) || !scalar_text(model$backend) ||
      !scalar_text(model$backend_version) ||
      !scalar_text(model$python_version) ||
      !scalar_text(model$numpy_version) ||
      !scalar_text(model$input_dtype) ||
      !identical(model$method, method) || !identical(model$backend, "aeon") ||
      !identical(model$input_dtype, "float64")) {
    .ml_stop(sprintf("`model` is not an aeon %s model", method))
  }
  n_kernels <- .ml_whole_number(
    model$n_kernels_requested,
    "model$n_kernels_requested",
    minimum = 1
  )
  n_features <- .ml_whole_number(
    model$n_features_realized,
    "model$n_features_realized",
    minimum = 1
  )
  seed <- .ml_whole_number(model$seed, "model$seed")
  n_time <- .ml_whole_number(model$n_time, "model$n_time", minimum = 9)
  if (method == "rocket" && n_features != 2 * as.double(n_kernels)) {
    .ml_stop("`model` has an invalid ROCKET requested/realized width")
  }
  if (method == "rocket") {
    .ml_flag(model$normalize, "model$normalize")
    if (!is.logical(model$deterministic) || length(model$deterministic) != 1L ||
        !is.na(model$deterministic)) {
      .ml_stop("`model$deterministic` must be NA for a ROCKET model")
    }
  } else {
    .ml_flag(model$deterministic, "model$deterministic")
    if (!is.logical(model$normalize) || length(model$normalize) != 1L ||
        !is.na(model$normalize)) {
      .ml_stop("`model$normalize` must be NA for a MiniRocket model")
    }
  }
  if (!is.raw(model$payload) || !length(model$payload) ||
      !is.character(model$payload_sha256) ||
      length(model$payload_sha256) != 1L ||
      is.na(model$payload_sha256) ||
      !grepl("^[0-9a-f]{64}$", model$payload_sha256) ||
      !identical(.ml_payload_hash(model$payload), model$payload_sha256)) {
    .ml_stop("`model` payload is corrupt or its SHA-256 does not match")
  }
  if (!is.character(model$feature_names) ||
      length(model$feature_names) != n_features ||
      anyNA(model$feature_names) || anyDuplicated(model$feature_names)) {
    .ml_stop("`model` has an invalid feature-name contract")
  }
  expected_names <- .ml_rocket_feature_contract(
    matrix(0, nrow = 1L, ncol = n_features),
    method
  )$feature_names
  if (!identical(model$feature_names, expected_names)) {
    .ml_stop("`model` feature names do not match the canonical contract")
  }
  .ml_character_ids(model$channel_ids, "model$channel_ids")
  .ml_character_ids(model$fit_case_ids, "model$fit_case_ids")
  if (!is.character(model$citations) || !length(model$citations) ||
      anyNA(model$citations) || any(!nzchar(trimws(model$citations)))) {
    .ml_stop("`model` has invalid fit-case or citation provenance")
  }
  invisible(n_kernels)
  invisible(seed)
  invisible(n_time)
  invisible(model)
}

.ml_python_matrix <- function(value, n_cases) {
  value <- reticulate::py_to_r(value)
  if (is.null(dim(value))) {
    if (length(value) %% n_cases != 0L) {
      .ml_stop("aeon returned a transform with an invalid case dimension")
    }
    value <- matrix(value, nrow = n_cases, byrow = TRUE)
  } else {
    value <- as.matrix(value)
  }
  storage.mode(value) <- "double"
  if (nrow(value) != n_cases || !ncol(value) || any(!is.finite(value))) {
    .ml_stop("aeon returned a non-finite or dimensionally invalid transform")
  }
  value
}

.ml_rocket_feature_contract <- function(features, method) {
  if (method == "rocket") {
    if (ncol(features) %% 2L != 0L) {
      .ml_stop("aeon Rocket returned an odd transformed width")
    }
    kernels <- ncol(features) %/% 2L
    # aeon emits PPV then maximum; the public PhysioML contract is max then PPV.
    order <- as.vector(rbind(
      seq.int(2L, ncol(features), by = 2L),
      seq.int(1L, ncol(features), by = 2L)
    ))
    features <- features[, order, drop = FALSE]
    feature_names <- sprintf(
      "rocket_%05d_%s",
      rep(seq_len(kernels), each = 2L),
      rep(c("max", "ppv"), kernels)
    )
  } else {
    feature_names <- sprintf(
      "minirocket_%05d_ppv",
      seq_len(ncol(features))
    )
  }
  colnames(features) <- feature_names
  list(features = features, feature_names = feature_names)
}

.ml_check_model_argument <- function(value, model_value, was_missing, name) {
  if (was_missing) {
    return(model_value)
  }
  if (!identical(value, model_value)) {
    .ml_stop(sprintf(
      "`%s` conflicts with the supplied fitted model",
      name
    ))
  }
  value
}

.ml_transform_settings <- function(input, method, backend_info, model,
                                   reused) {
  list(
    method = method,
    backend = "aeon",
    backend_version = backend_info$aeon_version,
    numpy_version = backend_info$numpy_version,
    python_version = backend_info$python_version,
    assay_name = input$assay_name,
    sampling_rate = input$sampling_rate,
    input_dimensions = input$input_dimensions,
    case_ids = input$case_data$case_id,
    channel_ids = input$channel_data$channel_id,
    n_time = dim(input$data)[[3L]],
    n_kernels_requested = model$n_kernels_requested,
    n_features_realized = model$n_features_realized,
    seed = model$seed,
    normalize = model$normalize,
    deterministic = model$deterministic,
    model_reused = reused,
    payload_sha256 = model$payload_sha256,
    citations = model$citations
  )
}

.ml_rocket_impl <- function(x, method, model, assay_name, channels, cases,
                            n_kernels, seed, normalize, deterministic, backend,
                            missing_arguments) {
  backend <- .ml_exact_enum(backend, "aeon", "backend")
  if (is.null(model)) {
    n_kernels <- .ml_whole_number(n_kernels, "n_kernels", minimum = 1)
    seed <- .ml_whole_number(seed, "seed")
    if (method == "rocket") {
      normalize <- .ml_flag(normalize, "normalize")
    } else {
      deterministic <- .ml_flag(deterministic, "deterministic")
    }
  } else {
    .ml_validate_rocket_model(model, method)
    n_kernels <- .ml_check_model_argument(
      n_kernels,
      model$n_kernels_requested,
      missing_arguments$n_kernels,
      "n_kernels"
    )
    seed <- .ml_check_model_argument(
      seed,
      model$seed,
      missing_arguments$seed,
      "seed"
    )
    if (method == "rocket") {
      normalize <- .ml_check_model_argument(
        normalize,
        model$normalize,
        missing_arguments$normalize,
        "normalize"
      )
    } else {
      deterministic <- .ml_check_model_argument(
        deterministic,
        model$deterministic,
        missing_arguments$deterministic,
        "deterministic"
      )
    }
    n_kernels <- .ml_whole_number(n_kernels, "n_kernels", minimum = 1)
    seed <- .ml_whole_number(seed, "seed")
    if (method == "rocket") {
      normalize <- .ml_flag(normalize, "normalize")
    } else {
      deterministic <- .ml_flag(deterministic, "deterministic")
    }
  }

  input <- .ml_resolve_input(
    x,
    assay_name = assay_name,
    channels = channels,
    cases = cases,
    min_time = 9L
  )
  if (!is.null(model)) {
    if (!identical(input$channel_data$channel_id, model$channel_ids)) {
      .ml_stop("selected channel labels/order do not match the fitted model")
    }
    if (!identical(dim(input$data)[[3L]], model$n_time)) {
      .ml_stop("selected time length does not match the fitted model")
    }
  }

  .ml_preserve_r_rng({
    backend_info <- .ml_require_aeon()
    if (!is.null(model) &&
        !identical(model$backend_version, backend_info$aeon_version)) {
      .ml_stop(sprintf(
        "fitted model requires aeon %s, but the active backend is %s",
        model$backend_version,
        backend_info$aeon_version
      ))
    }
    python_data <- reticulate::np_array(
      input$data,
      dtype = "float64",
      order = "C"
    )

    reused <- !is.null(model)
    if (!reused) {
      constructor <- reticulate::py_get_attr(
        backend_info$convolution,
        if (method == "rocket") "Rocket" else "MiniRocket"
      )
      if (method == "rocket") {
        python_model <- constructor(
          n_kernels = n_kernels,
          normalise = normalize,
          n_jobs = 1L,
          random_state = seed
        )
      } else {
        python_model <- constructor(
          n_kernels = n_kernels,
          n_jobs = 1L,
          random_state = seed
        )
      }
      raw_features <- python_model$fit_transform(python_data)
      features <- .ml_python_matrix(raw_features, dim(input$data)[[1L]])
      contract <- .ml_rocket_feature_contract(features, method)
      features <- contract$features
      payload <- .ml_pickle_payload(python_model)
      citations <- if (method == "rocket") {
        c(
          "Dempster, Petitjean & Webb (2020) doi:10.1007/s10618-020-00701-z",
          "aeon Rocket"
        )
      } else {
        c(
          "Dempster, Schmidt & Webb (2021) doi:10.1145/3447548.3467231",
          "aeon MiniRocket"
        )
      }
      model <- structure(
        list(
          method = method,
          backend = "aeon",
          backend_version = backend_info$aeon_version,
          python_version = backend_info$python_version,
          numpy_version = backend_info$numpy_version,
          payload = payload,
          payload_sha256 = .ml_payload_hash(payload),
          n_kernels_requested = n_kernels,
          n_features_realized = ncol(features),
          seed = seed,
          normalize = if (method == "rocket") normalize else NA,
          deterministic = if (method == "minirocket") deterministic else NA,
          channel_ids = input$channel_data$channel_id,
          n_time = dim(input$data)[[3L]],
          feature_names = contract$feature_names,
          fit_case_ids = input$case_data$case_id,
          input_dtype = "float64",
          citations = citations
        ),
        class = "physio_rocket_model"
      )
    } else {
      # Class, metadata, version, and hash were validated before deserialization.
      python_model <- .ml_unpickle_payload(model$payload)
      raw_features <- python_model$transform(python_data)
      features <- .ml_python_matrix(raw_features, dim(input$data)[[1L]])
      contract <- .ml_rocket_feature_contract(features, method)
      features <- contract$features
      if (!identical(contract$feature_names, model$feature_names) ||
          ncol(features) != model$n_features_realized) {
        .ml_stop("aeon transform output violates the fitted feature contract")
      }
    }

    rownames(features) <- input$case_data$case_id
    result <- structure(
      list(
        features = features,
        case_data = input$case_data,
        model = model,
        settings = .ml_transform_settings(
          input,
          method,
          backend_info,
          model,
          reused
        ),
        diagnostics = .ml_empty_diagnostics()
      ),
      class = "physio_rocket_transform"
    )
    result
  })
}

#' Fit or apply the ROCKET transform
#'
#' Uses `aeon.transformations.collection.convolution_based.Rocket` through an
#' optional, caller-configured [reticulate][reticulate::reticulate] backend.
#' With `model = NULL`, fits only on the selected cases and returns a reusable
#' persistent model. With a model, validates the complete input contract and
#' transforms without refitting.
#'
#' @param x A `PhysioExperiment`.
#' @param model `NULL` or a fitted `physio_rocket_model`.
#' @param assay_name,channels,cases Input selection; see [catch22()].
#' @param n_kernels One finite whole number of kernels.
#' @param seed One whole-number backend seed in `[0, 2^31 - 1]`.
#' @param normalize One non-missing logical passed to aeon's `normalise`.
#' @param backend Exactly `"aeon"`.
#'
#' @details
#' Input assays are time x channel or time x channel x case and must contain
#' finite, equal-length series of at least nine samples. The fitted model binds
#' exact channel labels/order and time length. Requested kernel count and
#' realized feature width are recorded separately because backend transforms
#' may adjust the width.
#'
#' Models persist the backend with Python pickle in a hashed raw vector. A
#' matching SHA-256 detects corruption but does not make deserialization safe:
#' only reuse model objects from trusted sources. PhysioML validates class,
#' metadata, hash, input identity, and backend version before unpickling.
#'
#' @return A `physio_rocket_transform` containing a finite case-by-feature
#'   matrix, exact case identities, a persistent fitted model, settings, and
#'   typed diagnostics.
#'
#' @references
#' Dempster A, Petitjean F, Webb GI (2020). ROCKET: Exceptionally fast and
#' accurate time series classification using random convolutional kernels.
#' *Data Mining and Knowledge Discovery*, 34, 1454-1495.
#' \doi{10.1007/s10618-020-00701-z}
#'
#' @export
#' @examples
#' arr <- array(stats::rnorm(60 * 2 * 4), dim = c(60, 2, 4))
#' pe <- PhysioExperiment::PhysioExperiment(
#'   assays = list(raw = arr),
#'   colData = S4Vectors::DataFrame(label = c("C3", "C4")),
#'   samplingRate = 100
#' )
#' \donttest{
#' # The transform needs a caller-managed Python environment with NumPy and aeon.
#' fit <- rocket(pe, n_kernels = 50)
#' dim(fit$features)
#' }
rocket <- function(x, model = NULL, assay_name = NULL, channels = NULL,
                   cases = NULL, n_kernels = 10000L, seed = 1L,
                   normalize = TRUE, backend = "aeon") {
  missing_arguments <- list(
    n_kernels = missing(n_kernels),
    seed = missing(seed),
    normalize = missing(normalize),
    deterministic = TRUE
  )
  .ml_rocket_impl(
    x, "rocket", model, assay_name, channels, cases, n_kernels, seed,
    normalize, NA, backend, missing_arguments
  )
}

#' Fit or apply the MiniRocket transform
#'
#' Uses `aeon.transformations.collection.convolution_based.MiniRocket`.
#' A fitted model records the realized feature width because aeon rounds the
#' requested kernel count according to the MiniRocket kernel family.
#'
#' @inheritParams rocket
#' @param deterministic One non-missing logical. `TRUE` requires the
#'   fixed-seed, single-worker aeon path and records that contract. Determinism
#'   is scoped to the recorded backend versions and platform.
#'
#' @return A `physio_rocket_transform`; see [rocket()].
#'
#' @references
#' Dempster A, Schmidt DF, Webb GI (2021). MiniRocket: A Very Fast (Almost)
#' Deterministic Transform for Time Series Classification.
#' \doi{10.1145/3447548.3467231}
#'
#' @export
#' @examples
#' arr <- array(stats::rnorm(60 * 2 * 4), dim = c(60, 2, 4))
#' pe <- PhysioExperiment::PhysioExperiment(
#'   assays = list(raw = arr),
#'   colData = S4Vectors::DataFrame(label = c("C3", "C4")),
#'   samplingRate = 100
#' )
#' \donttest{
#' # The transform needs a caller-managed Python environment with NumPy and aeon.
#' fit <- minirocket(pe, n_kernels = 84)
#' dim(fit$features)
#' }
minirocket <- function(x, model = NULL, assay_name = NULL, channels = NULL,
                       cases = NULL, n_kernels = 10000L, seed = 1L,
                       deterministic = TRUE, backend = "aeon") {
  missing_arguments <- list(
    n_kernels = missing(n_kernels),
    seed = missing(seed),
    normalize = TRUE,
    deterministic = missing(deterministic)
  )
  .ml_rocket_impl(
    x, "minirocket", model, assay_name, channels, cases, n_kernels, seed,
    NA, deterministic, backend, missing_arguments
  )
}

#' @export
print.physio_rocket_model <- function(x, ...) {
  cat(
    "<physio_rocket_model> ", x$method, " via ", x$backend, " ",
    x$backend_version, "\n",
    "  channels: ", length(x$channel_ids),
    "; time: ", x$n_time,
    "; features: ", x$n_features_realized, "\n",
    "  payload SHA-256: ", x$payload_sha256, "\n",
    sep = ""
  )
  invisible(x)
}

#' @export
print.physio_rocket_transform <- function(x, ...) {
  cat(
    "<physio_rocket_transform> ", x$settings$method, "\n",
    "  cases: ", nrow(x$features),
    "; features: ", ncol(x$features),
    "; model reused: ", x$settings$model_reused, "\n",
    sep = ""
  )
  invisible(x)
}
