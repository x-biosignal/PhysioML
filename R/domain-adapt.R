.ml_embedding_matrix <- function(x, name) {
  if (inherits(x, "physio_foundation_embedding")) {
    x <- x$embeddings
  }
  if (!is.matrix(x) || !is.numeric(x) || is.complex(x) ||
      nrow(x) < 2L || ncol(x) < 1L || anyNA(x) ||
      any(!is.finite(x))) {
    .ml_stop(sprintf(
      "`%s` must be a finite numeric observation-by-feature matrix with at least two rows",
      name
    ))
  }
  if (is.null(colnames(x))) {
    .ml_stop(sprintf("`%s` must have exact feature names", name))
  }
  .ml_character_ids(colnames(x), sprintf("%s feature names", name))
  if (!is.null(rownames(x))) {
    .ml_character_ids(rownames(x), sprintf("%s observation names", name))
  }
  x
}

.ml_symmetric_power <- function(matrix, power, regularization) {
  matrix <- (matrix + t(matrix)) / 2
  decomposition <- eigen(matrix, symmetric = TRUE)
  values <- decomposition$values
  if (any(!is.finite(values)) || any(!is.finite(decomposition$vectors))) {
    .ml_stop("CORAL covariance eigendecomposition is non-finite")
  }
  floor_value <- max(
    regularization,
    max(values) * .Machine$double.eps * nrow(matrix)
  )
  values <- pmax(values, floor_value)
  result <- decomposition$vectors %*%
    diag(values^power, nrow = length(values)) %*%
    t(decomposition$vectors)
  result <- (result + t(result)) / 2
  residual <- max(abs(result - t(result)))
  if (!all(is.finite(result)) || residual > 1e-10) {
    .ml_stop("CORAL covariance matrix power failed its symmetry check")
  }
  list(value = result, floor = floor_value)
}

.ml_apply_coral <- function(x, model) {
  x <- .ml_embedding_matrix(x, "x")
  if (!identical(colnames(x), model$feature_names)) {
    .ml_stop("new source feature names and order must match the CORAL model")
  }
  identity <- dimnames(x)
  aligned <- sweep(x, 2L, model$source_center, "-") %*%
    model$transform
  aligned <- sweep(aligned, 2L, model$target_center, "+")
  dimnames(aligned) <- identity
  aligned
}

#' Align source embeddings to an unlabeled target domain with CORAL
#'
#' Fit CORAL only with training-source observations and an explicitly
#' designated unlabeled target adaptation set. Held-out target validation and
#' test observations must not be supplied. CORAL matches first- and
#' second-order statistics; it does not establish causal, class-conditional,
#' fairness, or clinical invariance.
#'
#' @param source,target Finite observation-by-feature matrices, or foundation
#'   embedding results, with identical exact feature names and order.
#' @param method Currently exactly `"coral"`.
#' @param regularization Positive covariance regularization.
#' @param return_model Whether to return the fitted transform and diagnostics.
#'
#' @return A matrix, or a plain list containing aligned source data and model.
#' @export
#' @examples
#' set.seed(1)
#' feat <- c("f1", "f2", "f3")
#' source <- matrix(stats::rnorm(30), ncol = 3, dimnames = list(NULL, feat))
#' target <- matrix(stats::rnorm(30, mean = 1), ncol = 3,
#'                  dimnames = list(NULL, feat))
#' fit <- domainAdapt(source, target)
#' fit$diagnostics$source_mean_residual
domainAdapt <- function(
    source,
    target,
    method = c("coral"),
    regularization = 1e-6,
    return_model = TRUE) {
  if (missing(method)) method <- "coral"
  method <- .ml_exact_enum(method, "coral", "method")
  regularization <- .ml_positive_number(
    regularization,
    "regularization"
  )
  return_model <- .ml_flag(return_model, "return_model")
  source <- .ml_embedding_matrix(source, "source")
  target <- .ml_embedding_matrix(target, "target")
  if (!identical(colnames(source), colnames(target))) {
    .ml_stop("source and target feature names and order must match exactly")
  }
  source_center <- colMeans(source)
  target_center <- colMeans(target)
  p <- ncol(source)
  source_covariance <- stats::cov(source) +
    diag(regularization, nrow = p)
  target_covariance <- stats::cov(target) +
    diag(regularization, nrow = p)
  source_power <- .ml_symmetric_power(
    source_covariance,
    -0.5,
    regularization
  )
  target_power <- .ml_symmetric_power(
    target_covariance,
    0.5,
    regularization
  )
  transform <- source_power$value %*% target_power$value
  model <- list(
    method = method,
    feature_names = colnames(source),
    source_center = source_center,
    target_center = target_center,
    source_covariance = source_covariance,
    target_covariance = target_covariance,
    transform = transform,
    regularization = regularization,
    eigenvalue_floor = c(
      source = source_power$floor,
      target = target_power$floor
    ),
    fit_counts = c(source = nrow(source), target = nrow(target))
  )
  aligned <- .ml_apply_coral(source, model)
  if (!return_model) {
    attr(aligned, "coral_model") <- model
    return(aligned)
  }
  list(
    source_aligned = aligned,
    target = target,
    model = model,
    diagnostics = list(
      source_covariance_residual = max(abs(
        stats::cov(aligned) - stats::cov(target)
      )),
      source_mean_residual = max(abs(
        colMeans(aligned) - colMeans(target)
      ))
    ),
    citations = c(
      "Sun, Feng & Saenko (2016), doi:10.48550/arXiv.1511.05547",
      "Sun & Saenko (2016), doi:10.48550/arXiv.1607.01719"
    )
  )
}
