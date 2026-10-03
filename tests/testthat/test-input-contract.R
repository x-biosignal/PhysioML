test_that("two-dimensional input remains one case with exact channel order", {
  data <- cbind(a = 1:20, b = 101:120, c = 201:220)
  pe <- ml_fixture(data, labels = c("left", "mid", "right"))
  resolved <- PhysioML:::.ml_resolve_input(
    pe,
    channels = c("right", "left"),
    cases = "case_1",
    min_time = 10L
  )

  expect_identical(dim(resolved$data), c(1L, 2L, 20L))
  expect_identical(resolved$case_data$case_id, "case_1")
  expect_identical(resolved$case_data$input_case_index, 1L)
  expect_identical(
    resolved$channel_data$channel_id,
    c("right", "left")
  )
  expect_equal(resolved$data[1, 1, ], data[, 3])
  expect_equal(resolved$data[1, 2, ], data[, 1])
})

test_that("three-dimensional input is explicitly case by channel by time", {
  data <- array(seq_len(12 * 3 * 4), c(12, 3, 4))
  pe <- ml_fixture(
    data,
    labels = c("A", "B", "C"),
    case_ids = c("p1", "p2", "p3", "p4")
  )
  resolved <- PhysioML:::.ml_resolve_input(
    pe,
    channels = c("C", "A"),
    cases = c("p4", "p2"),
    min_time = 9L
  )

  expect_identical(dim(resolved$data), c(2L, 2L, 12L))
  expect_equal(resolved$data[1, 1, ], data[, 3, 4])
  expect_equal(resolved$data[1, 2, ], data[, 1, 4])
  expect_equal(resolved$data[2, 1, ], data[, 3, 2])
  expect_identical(
    resolved$case_data,
    data.frame(
      case_id = c("p4", "p2"),
      input_case_index = c(4L, 2L)
    )
  )
})

test_that("incomplete case dimnames fall back to stable case ids", {
  data <- array(seq_len(10 * 2 * 3), c(10, 2, 3))
  dimnames(data)[[3L]] <- c("one", "", "three")
  resolved <- PhysioML:::.ml_resolve_input(
    ml_fixture(data, labels = c("x", "y")),
    cases = c(3, 1),
    min_time = 10L
  )
  expect_identical(resolved$case_data$case_id, c("case_3", "case_1"))
})

test_that("input contract rejects ambiguous or invalid data", {
  good <- ml_fixture(matrix(seq_len(30), 15, 2), labels = c("x", "y"))
  expect_error(
    PhysioML:::.ml_resolve_input(good, channels = "z"),
    "unknown channel"
  )
  expect_error(
    PhysioML:::.ml_resolve_input(good, channels = c("x", "x")),
    "unique"
  )
  expect_error(
    PhysioML:::.ml_resolve_input(good, cases = 0),
    "positive"
  )
  expect_error(
    PhysioML:::.ml_resolve_input(good, cases = factor("case_1")),
    "labels"
  )
  expect_error(
    PhysioML:::.ml_resolve_input(good, assay_name = "ra"),
    "unavailable"
  )

  short <- ml_fixture(matrix(1:18, 9, 2), labels = c("x", "y"))
  expect_error(
    PhysioML:::.ml_resolve_input(short, min_time = 10L),
    "at least 10"
  )
  nonfinite <- good
  SummarizedExperiment::assay(nonfinite, "raw")[1, 1] <- Inf
  expect_error(
    PhysioML:::.ml_resolve_input(nonfinite),
    "finite"
  )
  wrong_dim <- ml_fixture(
    array(seq_len(10 * 2 * 2 * 2), c(10, 2, 2, 2)),
    labels = c("x", "y")
  )
  expect_error(
    PhysioML:::.ml_resolve_input(wrong_dim),
    "matrix or"
  )
})

test_that("exact enums and scalar settings do not partially match", {
  pe <- ml_fixture(matrix(seq_len(20), 20, 1), labels = "x")
  expect_error(peReducedFeatures(pe, method = "cat"), "exactly one")
  expect_error(rocket(pe, backend = "a"), "exactly one")
  expect_error(rocket(pe, n_kernels = 1.5), "whole number")
  expect_error(rocket(pe, seed = -1), "whole number")
  expect_error(minirocket(pe, deterministic = NA), "non-missing")
})
