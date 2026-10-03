.ml_reduced_catch22 <- function(result) {
  result_frame <- as.data.frame(result)
  identity_names <- c(
    "case_id", "input_case_index", "channel_id", "input_channel_index"
  )
  feature_names <- setdiff(names(result_frame), identity_names)
  provenance <- attr(result, "feature_provenance", exact = TRUE)
  channel_ids <- provenance$channel_ids
  case_ids <- provenance$case_ids
  output_names <- as.vector(vapply(
    channel_ids,
    function(channel_id) {
      paste0(enc2utf8(channel_id), "__", enc2utf8(feature_names))
    },
    character(length(feature_names))
  ))
  if (anyNA(output_names) || anyDuplicated(output_names)) {
    .ml_stop(paste0(
      "catch22 channel and feature identities collide after UTF-8 ",
      "concatenation"
    ))
  }
  output <- matrix(
    NA_real_,
    nrow = length(case_ids),
    ncol = length(output_names),
    dimnames = list(case_ids, output_names)
  )
  for (case in seq_along(case_ids)) {
    rows <- which(result_frame$case_id == case_ids[[case]])
    if (!identical(result_frame$channel_id[rows], channel_ids)) {
      .ml_stop("catch22 row identities violate the case-channel contract")
    }
    output[case, ] <- as.numeric(t(as.matrix(
      result_frame[rows, feature_names, drop = FALSE]
    )))
  }
  output
}

#' Create a case-by-feature design matrix
#'
#' Adapts [catch22()], [rocket()], or [minirocket()] output without adding an
#' outcome or changing the feature transform. For convolution methods, pass a
#' model fitted only on training cases when transforming validation/test cases.
#'
#' @param x A `PhysioExperiment`.
#' @param method Exactly one of `"catch22"`, `"rocket"`, or `"minirocket"`.
#' @param model `NULL` or a fitted convolution-transform model. Must be `NULL`
#'   for catch22.
#' @param assay_name,channels,cases Input selection; see [catch22()].
#' @param ... Method-specific arguments.
#'
#' @return A numeric matrix with class `physio_feature_matrix`. Attributes
#'   preserve exact case identity, feature provenance, typed diagnostics, and
#'   the fitted model when applicable.
#'
#' @export
#' @examples
#' arr <- array(stats::rnorm(40 * 2 * 3), dim = c(40, 2, 3))
#' pe <- PhysioExperiment::PhysioExperiment(
#'   assays = list(raw = arr),
#'   colData = S4Vectors::DataFrame(label = c("C3", "C4")),
#'   samplingRate = 100
#' )
#' # catch22 features need the optional Rcatch22 backend.
#' if (requireNamespace("Rcatch22", quietly = TRUE)) {
#'   features <- peReducedFeatures(pe, method = "catch22")
#'   dim(features)
#' }
peReducedFeatures <- function(
    x,
    method = c("catch22", "rocket", "minirocket"),
    model = NULL,
    assay_name = NULL,
    channels = NULL,
    cases = NULL,
    ...) {
  method <- .ml_exact_enum(method, c("catch22", "rocket", "minirocket"), "method")
  arguments <- list(
    x = x,
    assay_name = assay_name,
    channels = channels,
    cases = cases
  )
  if (method == "catch22") {
    if (!is.null(model)) {
      .ml_stop("`model` must be NULL when `method = \"catch22\"`")
    }
    result <- do.call(catch22, c(arguments, list(...)))
    features <- .ml_reduced_catch22(result)
    case_data <- attr(result, "case_data", exact = TRUE)
    provenance <- attr(result, "feature_provenance", exact = TRUE)
    diagnostics <- attr(result, "diagnostics", exact = TRUE)
    fitted_model <- NULL
  } else {
    arguments$model <- model
    transform <- do.call(
      if (method == "rocket") rocket else minirocket,
      c(arguments, list(...))
    )
    features <- transform$features
    case_data <- transform$case_data
    provenance <- transform$settings
    diagnostics <- transform$diagnostics
    fitted_model <- transform$model
  }
  class(features) <- unique(c(
    "physio_feature_matrix",
    class(features)
  ))
  attr(features, "case_data") <- case_data
  attr(features, "feature_provenance") <- provenance
  attr(features, "diagnostics") <- diagnostics
  attr(features, "model") <- fitted_model
  features
}
