coral_oracle <- function(source, target, regularization) {
  power <- function(value, exponent) {
    value <- (value + t(value)) / 2
    decomposition <- eigen(value, symmetric = TRUE)
    floor <- max(
      regularization,
      max(decomposition$values) * .Machine$double.eps * nrow(value)
    )
    decomposition$vectors %*%
      diag(
        pmax(decomposition$values, floor)^exponent,
        nrow = nrow(value)
      ) %*%
      t(decomposition$vectors)
  }
  source_covariance <- cov(source) + diag(regularization, ncol(source))
  target_covariance <- cov(target) + diag(regularization, ncol(target))
  transform <- power(source_covariance, -0.5) %*%
    power(target_covariance, 0.5)
  output <- sweep(
    sweep(source, 2L, colMeans(source), "-") %*% transform,
    2L,
    colMeans(target),
    "+"
  )
  dimnames(output) <- dimnames(source)
  output
}

test_that("domainAdapt matches an independent CORAL oracle", {
  set.seed(4)
  source <- matrix(rnorm(120), 30L, 4L)
  target <- matrix(rnorm(160), 40L, 4L) %*%
    matrix(c(
      1.0, 0.2, 0.0, 0.0,
      0.0, 1.3, 0.1, 0.0,
      0.0, 0.0, 0.8, 0.3,
      0.1, 0.0, 0.0, 1.1
    ), 4L, 4L)
  colnames(source) <- colnames(target) <- paste0("feature_", 1:4)
  rownames(source) <- paste0("source_", seq_len(nrow(source)))
  rownames(target) <- paste0("target_", seq_len(nrow(target)))

  result <- domainAdapt(source, target, regularization = 1e-6)
  expected <- coral_oracle(source, target, 1e-6)
  expect_equal(result$source_aligned, expected, tolerance = 1e-10)
  expect_identical(rownames(result$source_aligned), rownames(source))
  expect_identical(colnames(result$source_aligned), colnames(source))
  expect_identical(result$target, target)
  expect_lt(result$diagnostics$source_mean_residual, 1e-12)
  expect_equal(result$model$fit_counts, c(source = 30L, target = 40L))
})

test_that("domainAdapt handles rank-deficient covariances", {
  source <- cbind(a = 1:6, b = 2 * (1:6), c = rep(1, 6))
  target <- cbind(a = 2:8, b = 3 * (2:8), c = rep(4, 7))
  result <- domainAdapt(source, target, regularization = 1e-5)
  expect_true(all(is.finite(result$source_aligned)))
  expect_true(all(result$model$eigenvalue_floor >= 1e-5))
  expect_equal(
    result$source_aligned,
    coral_oracle(source, target, 1e-5),
    tolerance = 1e-9
  )
})

test_that("fitted CORAL applies without refitting", {
  source <- cbind(a = 1:5, b = c(2, 5, 4, 8, 7))
  target <- cbind(a = 11:16, b = c(8, 7, 10, 12, 14, 16))
  fitted <- domainAdapt(source, target)
  new_source <- cbind(a = c(3, 9), b = c(1, 12))
  rownames(new_source) <- c("new_1", "new_2")
  applied <- .ml_apply_coral(new_source, fitted$model)
  expected <- sweep(
    sweep(new_source, 2L, fitted$model$source_center, "-") %*%
      fitted$model$transform,
    2L,
    fitted$model$target_center,
    "+"
  )
  dimnames(expected) <- dimnames(new_source)
  expect_equal(applied, expected, tolerance = 0)
  expect_identical(rownames(applied), rownames(new_source))

  held_out <- target + 1000
  expect_false(any(fitted$model$target_center == colMeans(held_out)))
  expect_false("labels" %in% names(formals(domainAdapt)))
})

test_that("domainAdapt rejects malformed identity and feature contracts", {
  source <- cbind(a = 1:4, b = 5:8)
  target <- cbind(a = 2:5, b = 6:9)
  expect_error(domainAdapt(source, target[, 2:1]), "order")
  expect_error(domainAdapt(unname(source), target), "feature names")
  colnames(source) <- c("a", "a")
  expect_error(domainAdapt(source, target), "unique")
  source <- cbind(a = 1:4, b = 5:8)
  source[1L, 1L] <- Inf
  expect_error(domainAdapt(source, target), "finite")
  expect_error(domainAdapt(matrix(1, 1L, 1L), target), "at least two")
  expect_error(domainAdapt(c(1, 2), target), "matrix")
  expect_error(domainAdapt(target, target, regularization = 0), "positive")
  expect_error(domainAdapt(target, target, method = "riemannian"), "exactly")
})

test_that("return_model false preserves the fitted identity transform", {
  source <- cbind(a = 1:4, b = c(2, 1, 4, 3))
  target <- cbind(a = 8:12, b = c(1, 5, 7, 8, 11))
  rownames(source) <- paste0("s", 1:4)
  aligned <- domainAdapt(source, target, return_model = FALSE)
  expect_true(is.matrix(aligned))
  expect_identical(rownames(aligned), rownames(source))
  expect_true(is.list(attr(aligned, "coral_model")))
})
