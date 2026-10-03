test_that("peDataset preserves exact axes, identity, and complete windows", {
  ml_skip_torch()
  input <- array(
    seq_len(11 * 3 * 2),
    c(11, 3, 2),
    dimnames = list(NULL, NULL, c("p1", "p2"))
  )
  data <- peDataset(
    ml_fixture(input, labels = c("A", "B", "C")),
    targets = factor(c("control", "case"), levels = c("case", "control")),
    cases = c("p2", "p1"),
    channels = c("C", "A"),
    window_samples = 4,
    stride_samples = 3
  )

  expect_s3_class(data, "physio_torch_dataset")
  expect_identical(dim(data$data), c(6L, 2L, 4L))
  expect_identical(
    data$contract$window_table$first_sample,
    rep(c(1L, 4L, 7L), 2L)
  )
  expect_identical(data$contract$window_table$case_id, rep(c("p2", "p1"), each = 3))
  expect_identical(data$contract$trailing_samples, 1L)
  expect_equal(data$data[1, 1, ], input[1:4, 3, 2])
  expect_identical(data$contract$class_levels, c("case", "control"))
  expect_identical(data$targets, c(rep(1L, 3), rep(0L, 3)))
})

test_that("Dataset item and batch tensors retain exact dimensions and dtypes", {
  ml_skip_torch()
  x <- ml_fixture(matrix(seq_len(18), 9, 2), labels = c("x", "y"))
  classification <- peDataset(x, targets = "case", class_levels = c("case", "other"))
  item <- classification$.getitem(1)
  batch <- classification$.getbatch(1)

  expect_identical(item[[1]]$size(), c(2L, 9L))
  expect_identical(batch[[1]]$size(), c(1L, 2L, 9L))
  expect_true(item[[1]]$dtype == torch::torch_float32())
  expect_true(item[[2]]$dtype == torch::torch_long())
  expect_identical(as.integer(torch::as_array(batch[[2]])), 0L)

  prediction <- peDataset(x, task = "regression")
  expect_s3_class(prediction$.getitem(1), "torch_tensor")
  expect_identical(prediction$.getbatch(1)$size(), c(1L, 2L, 9L))
})

test_that("targets never enter model input and map reversibly", {
  ml_skip_torch()
  input <- array(
    seq_len(8 * 2 * 3),
    c(8, 2, 3),
    dimnames = list(NULL, NULL, c("a", "b", "c"))
  )
  x <- ml_fixture(input, labels = c("u", "v"))
  first <- peDataset(x, targets = c("z", "a", "z"))
  second <- peDataset(x, targets = c("a", "z", "a"))

  expect_identical(first$data, second$data)
  expect_false(identical(first$targets, second$targets))
  expect_identical(
    first$contract$class_mapping$class_index,
    c(0L, 1L)
  )
  expect_identical(
    first$contract$class_levels[first$targets + 1L],
    rep(c("z", "a", "z"), each = 1L)
  )
  expect_identical(dim(first$data)[[2L]], 2L)
})

test_that("normalization uses population training statistics and freezes them", {
  ml_skip_torch()
  train_array <- array(
    c(1, 2, 3, 4, 7, 7, 7, 7),
    c(4, 2, 1),
    dimnames = list(NULL, NULL, "train")
  )
  train <- peDataset(
    ml_fixture(train_array, labels = c("vary", "constant")),
    normalization = "zscore"
  )
  stats <- train$contract$normalization_stats
  expect_equal(stats$center, c(2.5, 7))
  expect_equal(stats$scale, c(sqrt(1.25), 1))
  expect_identical(stats$zero_scale, c(FALSE, TRUE))
  expect_identical(train$diagnostics$type, "zero_scale_channel")

  held_out <- train_array
  held_out[, 1, 1] <- c(1000, 1001, 1002, 1003)
  dimnames(held_out)[[3L]] <- "held_out"
  valid <- peDataset(
    ml_fixture(held_out, labels = c("vary", "constant")),
    normalization = "zscore",
    normalization_stats = stats
  )
  expect_false(valid$contract$normalization_fitted)
  expect_identical(valid$contract$normalization_stats, stats)
  expect_gt(min(valid$data[, 1, ]), 800)
})

test_that("robust normalization matches median absolute deviation arithmetic", {
  ml_skip_torch()
  x <- ml_fixture(matrix(c(1, 2, 10, 20, 5, 5, 5, 5), 4, 2), c("a", "b"))
  data <- peDataset(x, normalization = "robust")
  stats <- data$contract$normalization_stats

  expect_equal(stats$center, c(6, 5))
  expect_equal(stats$scale, c(1.4826 * 4.5, 1))
  expect_identical(stats$zero_scale, c(FALSE, TRUE))
})

test_that("dataset validation rejects ambiguous leakage-prone contracts", {
  ml_skip_torch()
  x <- ml_fixture(matrix(seq_len(20), 10, 2), c("a", "b"))
  expect_error(peDataset(x, window_samples = 6, stride_samples = 0), "whole number")
  expect_error(peDataset(x, stride_samples = 2), "must be NULL")
  expect_error(peDataset(x, normalization_stats = list()), "must be NULL")
  expect_error(
    peDataset(x, targets = factor("a", levels = c("a", "unused"))),
    "unused"
  )
  expect_error(peDataset(x, targets = 1.5), "integer-like")
  expect_error(peDataset(x, targets = Inf, task = "regression"), "finite")
  expect_error(peDataset(x, dtype = "float"), "exactly one")

  fitted <- peDataset(x, normalization = "zscore")
  bad <- fitted$contract$normalization_stats
  bad$channel_ids <- rev(bad$channel_ids)
  expect_error(
    peDataset(x, normalization = "zscore", normalization_stats = bad),
    "channel order"
  )
})
