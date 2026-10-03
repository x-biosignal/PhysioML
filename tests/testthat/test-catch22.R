test_that("catch22 matches direct Rcatch22 output and preserves row order", {
  skip_if_not_installed("Rcatch22")
  t <- seq(0, 4 * pi, length.out = 100)
  data <- array(0, c(100, 2, 3))
  data[, 1, 1] <- sin(t)
  data[, 2, 1] <- cos(t)
  data[, 1, 2] <- seq(-1, 1, length.out = 100)
  data[, 2, 2] <- rep(c(-1, 1), 50)
  data[, 1, 3] <- sin(t) + seq_along(t) / 100
  data[, 2, 3] <- cos(2 * t)
  pe <- ml_fixture(
    data,
    labels = c("A", "B"),
    case_ids = c("first", "second", "third")
  )

  result <- catch22(
    pe,
    channels = c("B", "A"),
    cases = c("third", "first")
  )
  frame <- as.data.frame(result)
  expect_s4_class(result, "DFrame")
  expect_identical(
    frame$case_id,
    rep(c("third", "first"), each = 2L)
  )
  expect_identical(frame$channel_id, rep(c("B", "A"), 2L))
  expect_identical(ncol(frame), 26L)

  direct <- Rcatch22::catch22_all(data[, 2, 3])
  direct <- as.data.frame(direct)
  expect_identical(names(frame)[5:26], as.character(direct$names))
  expect_equal(
    as.numeric(frame[1, 5:26]),
    as.numeric(direct$values),
    tolerance = 1e-12
  )
  provenance <- attr(result, "feature_provenance", exact = TRUE)
  expect_identical(provenance$backend, "Rcatch22")
  expect_identical(provenance$channel_ids, c("B", "A"))
  expect_identical(provenance$case_ids, c("third", "first"))
})

test_that("catch24 appends upstream mean and standard deviation", {
  skip_if_not_installed("Rcatch22")
  values <- seq(-3, 7, length.out = 100)
  pe <- ml_fixture(matrix(values, 100, 1), labels = "signal")
  result <- as.data.frame(catch22(pe, catch24 = TRUE))

  expect_identical(
    tail(names(result), 2L),
    c("DN_Mean", "DN_Spread_Std")
  )
  expect_equal(result$DN_Mean, mean(values), tolerance = 1e-12)
  expect_equal(result$DN_Spread_Std, stats::sd(values), tolerance = 1e-12)
})

test_that("degenerate catch22 series retain rows with typed diagnostics", {
  skip_if_not_installed("Rcatch22")
  data <- cbind(constant = rep(4, 50), tied = rep(c(0, 1), 25))
  result <- catch22(ml_fixture(data, labels = c("constant", "tied")))
  frame <- as.data.frame(result)
  diagnostics <- attr(result, "diagnostics", exact = TRUE)

  expect_identical(nrow(frame), 2L)
  expect_identical(
    names(diagnostics),
    c("case_id", "channel_id", "feature", "type", "message")
  )
  expect_false(any(vapply(frame[5:26], function(x) any(is.infinite(x)), logical(1))))
  if (anyNA(frame[1, 5:26])) {
    expect_true(any(diagnostics$channel_id == "constant"))
  }
})

test_that("catch22 repeat serialization is byte-identical", {
  skip_if_not_installed("Rcatch22")
  pe <- ml_fixture(
    cbind(x = sin(seq(0, 5, length.out = 80))),
    labels = "x"
  )
  first <- serialize(catch22(pe), NULL, version = 3)
  second <- serialize(catch22(pe), NULL, version = 3)
  expect_identical(first, second)
})

test_that("catch22 reduced features are channel-major and carry no outcome", {
  skip_if_not_installed("Rcatch22")
  data <- array(seq_len(60 * 2 * 2), c(60, 2, 2))
  pe <- ml_fixture(
    data,
    labels = c("C2", "C1"),
    case_ids = c("train", "test")
  )
  reduced <- peReducedFeatures(pe, method = "catch22")

  expect_s3_class(reduced, "physio_feature_matrix")
  expect_identical(dim(reduced), c(2L, 44L))
  expect_true(all(startsWith(colnames(reduced)[1:22], "C2__")))
  expect_true(all(startsWith(colnames(reduced)[23:44], "C1__")))
  expect_identical(rownames(reduced), c("train", "test"))
  expect_false(any(grepl("outcome|target", colnames(reduced), ignore.case = TRUE)))
  expect_null(attr(reduced, "model", exact = TRUE))
})
