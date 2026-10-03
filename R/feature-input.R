.ml_stop <- function(message) {
  stop(message, call. = FALSE)
}

.ml_exact_enum <- function(value, choices, name) {
  if (!is.character(value) || length(value) != 1L || is.na(value) ||
      !value %in% choices) {
    .ml_stop(sprintf(
      "`%s` must be exactly one of: %s",
      name,
      paste(sprintf('"%s"', choices), collapse = ", ")
    ))
  }
  value
}

.ml_flag <- function(value, name) {
  if (!is.logical(value) || length(value) != 1L || is.na(value)) {
    .ml_stop(sprintf("`%s` must be one non-missing logical value", name))
  }
  value
}

.ml_whole_number <- function(value, name, minimum = 0, maximum = .Machine$integer.max) {
  if (!is.numeric(value) || length(value) != 1L || is.na(value) ||
      !is.finite(value) || value != floor(value) ||
      value < minimum || value > maximum) {
    .ml_stop(sprintf(
      "`%s` must be one finite whole number in [%s, %s]",
      name,
      format(minimum, scientific = FALSE),
      format(maximum, scientific = FALSE)
    ))
  }
  as.integer(value)
}

.ml_character_ids <- function(value, name) {
  if (!is.character(value) || !length(value) || anyNA(value) ||
      any(!nzchar(trimws(value))) || anyDuplicated(value)) {
    .ml_stop(sprintf(
      "`%s` must contain unique, non-empty, exact character labels",
      name
    ))
  }
  enc2utf8(value)
}

.ml_resolve_assay <- function(x, assay_name) {
  if (!methods::is(x, "PhysioExperiment")) {
    .ml_stop("`x` must be a PhysioExperiment")
  }
  assay_names <- SummarizedExperiment::assayNames(x)
  if (is.null(assay_name)) {
    assay_name <- PhysioExperiment::defaultAssay(x)
  } else if (!is.character(assay_name) || length(assay_name) != 1L ||
             is.na(assay_name) || !nzchar(assay_name)) {
    .ml_stop("`assay_name` must be NULL or one non-empty character value")
  }
  if (length(assay_name) != 1L || is.na(assay_name) ||
      !assay_name %in% assay_names) {
    .ml_stop(sprintf(
      "assay `%s` is unavailable; choices are: %s",
      assay_name,
      paste(assay_names, collapse = ", ")
    ))
  }
  assay_name
}

.ml_case_indices <- function(cases, case_ids) {
  if (is.null(cases)) {
    return(seq_along(case_ids))
  }
  if (is.character(cases)) {
    cases <- .ml_character_ids(cases, "cases")
    unknown <- setdiff(cases, case_ids)
    if (length(unknown)) {
      .ml_stop(sprintf(
        "unknown case label(s): %s",
        paste(unknown, collapse = ", ")
      ))
    }
    return(match(cases, case_ids))
  }
  if (is.factor(cases) || is.logical(cases) || !is.numeric(cases) ||
      !length(cases) || anyNA(cases) || any(!is.finite(cases)) ||
      any(cases != floor(cases)) || any(cases < 1) ||
      any(cases > length(case_ids)) || anyDuplicated(cases)) {
    .ml_stop(paste0(
      "`cases` must contain unique exact labels or unique positive ",
      "whole-number indices"
    ))
  }
  as.integer(cases)
}

.ml_resolve_input <- function(x, assay_name = NULL, channels = NULL,
                              cases = NULL, min_time = 1L) {
  assay_name <- .ml_resolve_assay(x, assay_name)
  signal <- SummarizedExperiment::assay(x, assay_name)
  dims <- dim(signal)
  if (!is.numeric(signal) || is.complex(signal) ||
      !length(dims) %in% c(2L, 3L)) {
    .ml_stop(paste0(
      "the selected assay must be a finite numeric time x channel matrix ",
      "or time x channel x case array"
    ))
  }
  if (dims[[1L]] < min_time) {
    .ml_stop(sprintf(
      "the selected assay has %d time samples; at least %d are required",
      dims[[1L]],
      min_time
    ))
  }
  if (!length(signal) || any(!is.finite(signal))) {
    .ml_stop("the selected assay must contain only finite numeric values")
  }

  col_data <- SummarizedExperiment::colData(x)
  if (!"label" %in% names(col_data)) {
    .ml_stop("`colData(x)$label` is required for exact channel identity")
  }
  channel_ids <- as.character(col_data[["label"]])
  if (length(channel_ids) != dims[[2L]] || anyNA(channel_ids) ||
      any(!nzchar(trimws(channel_ids))) || anyDuplicated(channel_ids)) {
    .ml_stop(paste0(
      "`colData(x)$label` must exactly match the assay channels and contain ",
      "unique non-empty labels"
    ))
  }
  channel_ids <- enc2utf8(channel_ids)
  if (is.null(channels)) {
    channel_index <- seq_along(channel_ids)
  } else {
    channels <- .ml_character_ids(channels, "channels")
    unknown <- setdiff(channels, channel_ids)
    if (length(unknown)) {
      .ml_stop(sprintf(
        "unknown channel label(s): %s",
        paste(unknown, collapse = ", ")
      ))
    }
    channel_index <- match(channels, channel_ids)
  }

  n_cases <- if (length(dims) == 3L) dims[[3L]] else 1L
  case_ids <- NULL
  if (length(dims) == 3L) {
    assay_dimnames <- dimnames(signal)
    if (length(assay_dimnames) >= 3L) {
      candidate <- assay_dimnames[[3L]]
      if (length(candidate) == n_cases && !anyNA(candidate) &&
          all(nzchar(trimws(candidate))) && !anyDuplicated(candidate)) {
        case_ids <- enc2utf8(candidate)
      }
    }
  }
  if (is.null(case_ids)) {
    case_ids <- paste0("case_", seq_len(n_cases))
  }
  identity <- S4Vectors::metadata(x)$ml_case_data
  if (!is.null(identity)) identity <- .ml_align_case_data(identity, case_ids)
  case_index <- .ml_case_indices(cases, case_ids)

  if (length(dims) == 2L) {
    selected <- array(
      signal[, channel_index, drop = FALSE],
      dim = c(dims[[1L]], length(channel_index), 1L)
    )
  } else {
    selected <- signal[, channel_index, case_index, drop = FALSE]
  }
  selected <- aperm(selected, c(3L, 2L, 1L))
  storage.mode(selected) <- "double"
  dimnames(selected) <- list(
    case = case_ids[case_index],
    channel = channel_ids[channel_index],
    time = NULL
  )

  case_data <- data.frame(case_id = case_ids[case_index],
    input_case_index = as.integer(case_index), stringsAsFactors = FALSE)
  if (!is.null(identity)) {
    for (name in setdiff(names(identity), c("case_id", "input_case_index")))
      case_data[[name]] <- identity[[name]][case_index]
  }
  list(
    data = selected,
    case_data = case_data,
    channel_data = data.frame(
      channel_id = channel_ids[channel_index],
      input_channel_index = as.integer(channel_index),
      stringsAsFactors = FALSE
    ),
    assay_name = assay_name,
    sampling_rate = as.numeric(PhysioExperiment::samplingRate(x)),
    input_dimensions = as.integer(dims)
  )
}

.ml_empty_diagnostics <- function() {
  data.frame(
    case_id = character(),
    channel_id = character(),
    feature = character(),
    type = character(),
    message = character(),
    stringsAsFactors = FALSE
  )
}

.ml_preserve_r_rng <- function(code) {
  had_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  if (had_seed) {
    seed <- get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  }
  on.exit({
    if (had_seed) {
      assign(".Random.seed", seed, envir = .GlobalEnv)
    } else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
      rm(".Random.seed", envir = .GlobalEnv)
    }
  }, add = TRUE)
  force(code)
}

.ml_align_case_data <- function(case_data, case_ids) {
  allowed <- c("case_id", "subject_id", "participant_id", "session_id",
               "trial_id", "side", "cycle_id", "input_case_index")
  if (!is.data.frame(case_data) || !nrow(case_data) ||
      anyDuplicated(names(case_data)) ||
      !all(c("case_id", "subject_id") %in% names(case_data)) ||
      any(!names(case_data) %in% allowed)) {
    .ml_stop("`case_data` requires case_id and subject_id, with optional participant_id/session_id/trial_id/side/cycle_id only")
  }
  case_data <- as.data.frame(case_data)
  if ("input_case_index" %in% names(case_data)) {
    index <- case_data$input_case_index
    if (!is.numeric(index) || is.complex(index) || !is.null(dim(index)) ||
        any(!is.finite(index)) ||
        any(index < 1 | index != floor(index)) || anyDuplicated(index))
      .ml_stop("input_case_index must contain unique positive whole numbers")
  }
  labels <- case_data[setdiff(names(case_data), "input_case_index")]
  if (!all(vapply(labels, function(v) is.character(v) && is.null(dim(v)) && !anyNA(v) &&
      all(nzchar(trimws(v))), logical(1)))) {
    .ml_stop("case identity columns must be complete non-empty character labels")
  }
  .ml_character_ids(case_data$case_id, "case_data$case_id")
  if (!setequal(case_data$case_id, case_ids))
    .ml_stop("case_data must match all input case IDs exactly")
  if ("participant_id" %in% names(case_data) &&
      !identical(case_data$participant_id, case_data$subject_id))
    .ml_stop("participant_id must equal subject_id")
  case_data <- case_data[match(case_ids, case_data$case_id), , drop = FALSE]
  rownames(case_data) <- NULL
  case_data
}

#' Attach explicit subject identity to signal cases or feature rows
#'
#' Establishes a case-to-subject mapping without inferring subjects from case
#' names. Row order is reconciled by exact case ID. A cycle is a case, not an
#' independent subject. Use cohort-derived subject/session keys when available.
#'
#' @param x A `PhysioExperiment` or finite numeric case-by-feature matrix with
#'   unique nonempty row names used as case IDs.
#' @param case_data Data frame with unique `case_id` and repeated `subject_id`.
#'   Optional character identity columns are `participant_id`, `session_id`,
#'   `trial_id`, `side`, `cycle_id`. All supplied columns must be complete.
#'   An optional `input_case_index` from a PhysioML transform is accepted as
#'   unique positive whole numbers; signal selection computes its own indices.
#'   The table must cover all input cases exactly, before any case selection.
#' @return The input with validated identity metadata. Experiments store it in
#'   `metadata(x)$ml_case_data`; matrices store it in attribute `case_data`.
#'   PhysioML input transforms and datasets preserve this information. Ordinary
#'   external matrix manipulation is not guaranteed to preserve attributes;
#'   reattach the corresponding table after subsetting feature rows.
#' @seealso [validateSubjectSplit()], [peDataset()], [peReducedFeatures()]
#' @export
#' @examples
#' # Attach subject identity to a case-by-feature matrix.
#' features <- matrix(
#'   stats::rnorm(8), nrow = 2,
#'   dimnames = list(c("c1", "c2"), c("a", "b", "c", "d"))
#' )
#' case_data <- data.frame(case_id = c("c1", "c2"), subject_id = c("s1", "s2"))
#' labelled <- withCaseData(features, case_data)
#' attr(labelled, "case_data")
withCaseData <- function(x, case_data) {
  if (methods::is(x, "PhysioExperiment")) {
    # Resolve signal identity independently of any previously attached table.
    md <- S4Vectors::metadata(x)
    md$ml_case_data <- NULL
    S4Vectors::metadata(x) <- md
    ids <- .ml_resolve_input(x)$case_data$case_id
    md$ml_case_data <- .ml_align_case_data(case_data, ids)
    S4Vectors::metadata(x) <- md
  } else {
    if (!is.matrix(x) || !is.numeric(x) || is.complex(x) ||
        !nrow(x) || !ncol(x) || any(!is.finite(x)))
      .ml_stop("x must be a PhysioExperiment or finite numeric feature matrix")
    ids <- .ml_character_ids(rownames(x), "feature row names")
    attr(x, "case_data") <- .ml_align_case_data(case_data, ids)
  }
  x
}

#' Validate a split intended to generalize to unseen subjects
#'
#' Checks explicit identifiers only; this does not audit upstream preprocessing
#' fit scopes, verify biological identity, or prevent leakage in external model
#' software unless called before fitting. No backend is needed.
#'
#' @param train,valid Case identity data frames as in [withCaseData()].
#' @return Invisibly `TRUE`, or an error for shared cases or subjects, invalid
#'   labels, or inconsistent participant aliases.
#' @export
#' @examples
#' train <- data.frame(case_id = c("c1", "c2"), subject_id = c("s1", "s2"))
#' valid <- data.frame(case_id = c("c3", "c4"), subject_id = c("s3", "s4"))
#' validateSubjectSplit(train, valid)
validateSubjectSplit <- function(train, valid) {
  left <- .ml_align_case_data(train, train$case_id)
  right <- .ml_align_case_data(valid, valid$case_id)
  shared <- intersect(left$case_id, right$case_id)
  if (length(shared)) .ml_stop(paste0("train/valid case IDs must be disjoint; shared: ",
                                    paste(shared, collapse = ", ")))
  shared <- intersect(left$subject_id, right$subject_id)
  if (length(shared)) .ml_stop(paste0("train/valid subject IDs must be disjoint; shared: ",
                                    paste(shared, collapse = ", ")))
  invisible(TRUE)
}
