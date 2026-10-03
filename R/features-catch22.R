.ml_require_catch22 <- function() {
  if (!requireNamespace("Rcatch22", quietly = TRUE)) {
    .ml_stop(paste0(
      "`catch22()` requires the optional R package `Rcatch22`; install it ",
      "from CRAN before calling this feature backend"
    ))
  }
}

.ml_catch22_names <- function(catch24) {
  feature_list <- getExportedValue("Rcatch22", "feature_list")
  if (is.character(feature_list)) {
    feature_names <- feature_list
  } else {
    feature_list <- as.data.frame(feature_list, stringsAsFactors = FALSE)
    if (!"feature" %in% names(feature_list)) {
      .ml_stop("the installed Rcatch22 feature list has an unsupported schema")
    }
    feature_names <- as.character(feature_list$feature)
  }
  if (length(feature_names) != 22L ||
      anyNA(feature_names) ||
      anyDuplicated(feature_names)) {
    .ml_stop("the installed Rcatch22 feature list has an unsupported schema")
  }
  feature_names <- enc2utf8(feature_names)
  if (catch24) {
    feature_names <- c(feature_names, "DN_Mean", "DN_Spread_Std")
  }
  feature_names
}

.ml_normalize_catch22 <- function(output, feature_names) {
  if (is.numeric(output) && !is.null(names(output))) {
    values <- as.numeric(output)
    names(values) <- enc2utf8(names(output))
  } else {
    output <- as.data.frame(output, stringsAsFactors = FALSE)
    feature_column <- intersect(c("feature", "name", "names"), names(output))
    value_column <- intersect(c("value", "values", "result"), names(output))
    if (length(feature_column) == 1L && length(value_column) == 1L) {
      values <- as.numeric(output[[value_column]])
      names(values) <- enc2utf8(as.character(output[[feature_column]]))
    } else if (nrow(output) == 1L &&
               all(vapply(output, is.numeric, logical(1)))) {
      values <- as.numeric(output[1L, , drop = TRUE])
      names(values) <- enc2utf8(names(output))
    } else {
      .ml_stop("Rcatch22 returned an unsupported structured result")
    }
  }
  if (!identical(names(values), feature_names)) {
    .ml_stop(paste0(
      "Rcatch22 feature names/order differ from the canonical installed ",
      "feature list"
    ))
  }
  values
}

.ml_catch22_one <- function(series, catch24, feature_names,
                            case_id, channel_id) {
  backend_error <- NULL
  output <- tryCatch(
    suppressWarnings(
      Rcatch22::catch22_all(series, catch24 = catch24)
    ),
    error = function(e) {
      backend_error <<- conditionMessage(e)
      NULL
    }
  )

  diagnostics <- .ml_empty_diagnostics()
  if (is.null(output)) {
    values <- stats::setNames(
      rep(NA_real_, length(feature_names)),
      feature_names
    )
    diagnostics <- data.frame(
      case_id = case_id,
      channel_id = channel_id,
      feature = NA_character_,
      type = "backend_error",
      message = backend_error,
      stringsAsFactors = FALSE
    )
  } else {
    values <- .ml_normalize_catch22(output, feature_names)
    bad <- which(!is.finite(values))
    if (length(bad)) {
      diagnostics <- data.frame(
        case_id = rep(case_id, length(bad)),
        channel_id = rep(channel_id, length(bad)),
        feature = names(values)[bad],
        type = rep("non_finite_backend_feature", length(bad)),
        message = rep(
          "Rcatch22 returned a non-finite value; retained as NA_real_",
          length(bad)
        ),
        stringsAsFactors = FALSE
      )
      values[bad] <- NA_real_
    }
  }
  list(values = values, diagnostics = diagnostics)
}

#' Calculate canonical catch22 features
#'
#' Calls the official C-backed [Rcatch22::catch22_all()] implementation for
#' every selected case-channel series. Input samples are passed unchanged.
#'
#' @param x A `PhysioExperiment`.
#' @param assay_name `NULL` for [PhysioExperiment::defaultAssay()] or an exact assay
#'   name.
#' @param channels `NULL` or exact unique channel labels in requested order.
#' @param cases `NULL`, exact unique case labels, or exact unique positive case
#'   indices in requested order.
#' @param catch24 One non-missing logical. When `TRUE`, append upstream mean and
#'   standard-deviation features after the canonical 22.
#'
#' @details
#' A 2-D assay is interpreted as time x channel for one case. A 3-D assay is
#' interpreted as time x channel x case. Series must be finite, equal length
#' within an assay, and contain at least 10 samples. PhysioML does not impute,
#' resample, normalize, detrend, or collapse channels before calling Rcatch22.
#'
#' @return An [S4Vectors::DataFrame()] with one row per case-channel pair,
#'   identity columns followed by 22 or 24 numeric feature columns. Stable
#'   transform provenance and typed diagnostics are stored as attributes.
#'
#' @references
#' Lubba CH, Sethi SS, Knaute P, Schultz SR, Fulcher BD, Jones NS (2019).
#' catch22: CAnonical Time-series CHaracteristics.
#' *Data Mining and Knowledge Discovery*, 33, 1821-1852.
#' \doi{10.1007/s10618-019-00647-x}
#'
#' Henderson T (2026). Rcatch22: Calculation of 22 Canonical Time-Series
#' Characteristics. R package.
#'
#' @export
#' @examples
#' if (requireNamespace("Rcatch22", quietly = TRUE)) {
#'   values <- matrix(sin(seq(0, 8 * pi, length.out = 100)), 100, 1)
#'   pe <- PhysioExperiment::PhysioExperiment(
#'     assays = list(raw = values),
#'     colData = S4Vectors::DataFrame(label = "signal"),
#'     samplingRate = 100
#'   )
#'   catch22(pe)
#' }
catch22 <- function(x, assay_name = NULL, channels = NULL, cases = NULL,
                    catch24 = FALSE) {
  catch24 <- .ml_flag(catch24, "catch24")
  .ml_require_catch22()
  input <- .ml_resolve_input(
    x,
    assay_name = assay_name,
    channels = channels,
    cases = cases,
    min_time = 10L
  )
  feature_names <- .ml_catch22_names(catch24)
  n_rows <- dim(input$data)[[1L]] * dim(input$data)[[2L]]
  feature_values <- matrix(
    NA_real_,
    nrow = n_rows,
    ncol = length(feature_names),
    dimnames = list(NULL, feature_names)
  )
  identity <- data.frame(
    case_id = character(n_rows),
    input_case_index = integer(n_rows),
    channel_id = character(n_rows),
    input_channel_index = integer(n_rows),
    stringsAsFactors = FALSE
  )
  diagnostics <- .ml_empty_diagnostics()

  row <- 0L
  for (case in seq_len(dim(input$data)[[1L]])) {
    for (channel in seq_len(dim(input$data)[[2L]])) {
      row <- row + 1L
      case_id <- input$case_data$case_id[[case]]
      channel_id <- input$channel_data$channel_id[[channel]]
      computed <- .ml_catch22_one(
        input$data[case, channel, ],
        catch24,
        feature_names,
        case_id,
        channel_id
      )
      feature_values[row, ] <- computed$values
      identity[row, ] <- list(
        case_id,
        input$case_data$input_case_index[[case]],
        channel_id,
        input$channel_data$input_channel_index[[channel]]
      )
      diagnostics <- rbind(diagnostics, computed$diagnostics)
    }
  }

  answer <- S4Vectors::DataFrame(
    identity,
    as.data.frame(feature_values, check.names = FALSE),
    check.names = FALSE
  )
  provenance <- list(
    method = "catch22",
    backend = "Rcatch22",
    backend_version = as.character(utils::packageVersion("Rcatch22")),
    catch24 = catch24,
    assay_name = input$assay_name,
    sampling_rate = input$sampling_rate,
    input_dimensions = input$input_dimensions,
    case_ids = input$case_data$case_id,
    channel_ids = input$channel_data$channel_id,
    feature_names = feature_names,
    citations = c(
      "Lubba et al. (2019) doi:10.1007/s10618-019-00647-x",
      "Rcatch22 CRAN package"
    )
  )
  attr(answer, "case_data") <- input$case_data
  attr(answer, "feature_provenance") <- provenance
  attr(answer, "diagnostics") <- diagnostics
  answer
}
