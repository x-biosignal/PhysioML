test_that("ROCKET matches direct aeon values with explicit max-PPV order", {
  skip_if_not(ml_backend_available())
  data <- array(sin(seq_len(80 * 2 * 3) / 7), c(80, 2, 3))
  pe <- ml_fixture(
    data,
    labels = c("left", "right"),
    case_ids = c("p1", "p2", "p3")
  )
  result <- rocket(pe, n_kernels = 7L, seed = 17L, normalize = FALSE)

  conv <- reticulate::import(
    "aeon.transformations.collection.convolution_based",
    convert = FALSE
  )
  direct_model <- conv$Rocket(
    n_kernels = 7L,
    normalise = FALSE,
    n_jobs = 1L,
    random_state = 17L
  )
  direct_input <- reticulate::np_array(
    aperm(data, c(3L, 2L, 1L)),
    dtype = "float64",
    order = "C"
  )
  direct <- as.matrix(reticulate::py_to_r(
    direct_model$fit_transform(direct_input)
  ))
  direct <- direct[, as.vector(rbind(
    seq.int(2L, ncol(direct), 2L),
    seq.int(1L, ncol(direct), 2L)
  )), drop = FALSE]

  expect_equal(unname(result$features), unname(direct), tolerance = 0)
  expect_identical(
    colnames(result$features)[1:4],
    c(
      "rocket_00001_max", "rocket_00001_ppv",
      "rocket_00002_max", "rocket_00002_ppv"
    )
  )
  expect_identical(ncol(result$features), 14L)
  expect_s3_class(result$model, "physio_rocket_model")
  expect_true(is.raw(result$model$payload))
  expect_false(any(vapply(result$model, inherits, logical(1), "python.builtin.object")))
})

test_that("MiniRocket fit and transform reuse payload without refitting", {
  skip_if_not(ml_backend_available())
  train_data <- array(cos(seq_len(60 * 2 * 4) / 11), c(60, 2, 4))
  test_data <- array(sin(seq_len(60 * 2 * 2) / 9), c(60, 2, 2))
  train <- ml_fixture(
    train_data,
    labels = c("A", "B"),
    case_ids = paste0("train_", 1:4)
  )
  test <- ml_fixture(
    test_data,
    labels = c("A", "B"),
    case_ids = paste0("test_", 1:2)
  )

  fitted <- minirocket(train, n_kernels = 100L, seed = 42L)
  payload <- fitted$model$payload
  transformed <- minirocket(test, model = fitted$model)
  repeated <- minirocket(test, model = fitted$model)

  expect_identical(fitted$model$fit_case_ids, paste0("train_", 1:4))
  expect_identical(transformed$model$payload, payload)
  expect_identical(repeated$model$payload, payload)
  expect_identical(transformed$features, repeated$features)
  expect_identical(transformed$settings$model_reused, TRUE)
  expect_true(all(startsWith(
    colnames(transformed$features),
    "minirocket_"
  )))
  expect_identical(ncol(transformed$features) %% 84L, 0L)
})

test_that("convolution models reject contract conflicts before transform", {
  skip_if_not(ml_backend_available())
  data <- array(seq_len(50 * 2 * 3), c(50, 2, 3))
  pe <- ml_fixture(
    data,
    labels = c("A", "B"),
    case_ids = paste0("c", 1:3)
  )
  fitted <- rocket(pe, n_kernels = 3L, seed = 5L)

  expect_error(
    rocket(pe, model = fitted$model, seed = 6L),
    "conflicts"
  )
  expect_error(
    rocket(pe, model = fitted$model, n_kernels = 4L),
    "conflicts"
  )
  expect_error(
    rocket(pe, model = fitted$model, normalize = FALSE),
    "conflicts"
  )
  expect_error(
    rocket(pe, model = fitted$model, channels = c("B", "A")),
    "channel labels/order"
  )
  longer <- ml_fixture(
    array(seq_len(51 * 2), c(51, 2, 1)),
    labels = c("A", "B"),
    case_ids = "x"
  )
  expect_error(
    rocket(longer, model = fitted$model),
    "time length"
  )
})

test_that("payload hash and model metadata validate before unpickling", {
  skip_if_not(ml_backend_available())
  pe <- ml_fixture(
    array(seq_len(40 * 2 * 2), c(40, 2, 2)),
    labels = c("A", "B"),
    case_ids = c("x", "y")
  )
  fitted <- rocket(pe, n_kernels = 2L, seed = 3L)

  corrupt <- fitted$model
  corrupt$payload[[1L]] <- as.raw(bitwXor(as.integer(corrupt$payload[[1L]]), 1L))
  expect_error(rocket(pe, model = corrupt), "corrupt|SHA-256")

  wrong_method <- fitted$model
  wrong_method$method <- "minirocket"
  expect_error(rocket(pe, model = wrong_method), "not an aeon rocket")

  malformed <- unclass(fitted$model)
  expect_error(rocket(pe, model = malformed), "complete")

  invalid_contract <- fitted$model
  invalid_contract$feature_names[[1L]] <- "tampered_name"
  invalid_contract$payload <- charToRaw("not a Python pickle")
  invalid_contract$payload_sha256 <- digest::digest(
    invalid_contract$payload,
    algo = "sha256",
    serialize = FALSE
  )
  expect_error(
    rocket(pe, model = invalid_contract),
    "canonical contract"
  )
})

test_that("R RNG state is restored on convolution success and error", {
  skip_if_not(ml_backend_available())
  pe <- ml_fixture(
    array(seq_len(40 * 1 * 2), c(40, 1, 2)),
    labels = "A",
    case_ids = c("x", "y")
  )
  set.seed(1001)
  before <- .Random.seed
  fitted <- rocket(pe, n_kernels = 2L, seed = 9L)
  expect_identical(.Random.seed, before)

  bad <- fitted$model
  bad$backend_version <- "0.0.invalid"
  expect_error(rocket(pe, model = bad), "requires aeon")
  expect_identical(.Random.seed, before)
})

test_that("ROCKET does not advance NumPy global RNG state", {
  skip_if_not(ml_backend_available())
  numpy <- reticulate::import("numpy", convert = FALSE)
  previous_state <- numpy$random$get_state()
  on.exit(numpy$random$set_state(previous_state), add = TRUE)
  numpy$random$seed(2468L)
  expected <- reticulate::py_to_r(numpy$random$random(5L))
  numpy$random$seed(2468L)

  pe <- ml_fixture(
    matrix(sin(seq_len(50L)), ncol = 1L),
    labels = "signal"
  )
  invisible(rocket(pe, n_kernels = 2L, seed = 19L))
  observed <- reticulate::py_to_r(numpy$random$random(5L))

  expect_identical(observed, expected)
})

test_that("same seeds reproduce and ROCKET different seeds change features", {
  skip_if_not(ml_backend_available())
  pe <- ml_fixture(
    array(sin(seq_len(70 * 2 * 3)), c(70, 2, 3)),
    labels = c("A", "B"),
    case_ids = paste0("c", 1:3)
  )
  first <- rocket(pe, n_kernels = 4L, seed = 11L)
  second <- rocket(pe, n_kernels = 4L, seed = 11L)
  other <- rocket(pe, n_kernels = 4L, seed = 12L)

  expect_identical(first$features, second$features)
  expect_false(isTRUE(all.equal(first$features, other$features)))
})

test_that("model survives RDS and reduced adapter preserves transform", {
  skip_if_not(ml_backend_available())
  pe <- ml_fixture(
    array(cos(seq_len(50 * 2 * 2)), c(50, 2, 2)),
    labels = c("A", "B"),
    case_ids = c("one", "two")
  )
  fitted <- minirocket(pe, n_kernels = 84L, seed = 8L)
  path <- tempfile(fileext = ".rds")
  on.exit(unlink(path), add = TRUE)
  saveRDS(fitted$model, path)
  restored <- readRDS(path)
  transformed <- minirocket(pe, model = restored)
  reduced <- peReducedFeatures(pe, method = "minirocket", model = restored)

  expect_identical(transformed$features, fitted$features)
  expect_equal(
    as.vector(reduced),
    as.vector(transformed$features),
    tolerance = 0,
    ignore_attr = TRUE
  )
  expect_identical(attr(reduced, "model")$payload, restored$payload)
  expect_identical(
    attr(reduced, "case_data")$case_id,
    c("one", "two")
  )
})
