foundation_fixture <- function(n_cases = 3L, n_time = 16L) {
  data <- array(
    seq(-1, 1, length.out = n_time * 2L * n_cases),
    c(n_time, 2L, n_cases),
    dimnames = list(NULL, NULL, paste0("foundation_", seq_len(n_cases)))
  )
  PhysioCore::PhysioExperiment(
    assays = list(raw = data),
    colData = S4Vectors::DataFrame(
      label = c("C3", "C4"),
      unit = c("uV", "uV")
    ),
    samplingRate = 100
  )
}

foundation_backend_available <- function() {
  requireNamespace("reticulate", quietly = TRUE) &&
    reticulate::py_available(initialize = TRUE) &&
    all(vapply(
      c("numpy", "torch", "safetensors"),
      reticulate::py_module_available,
      logical(1)
    ))
}

test_that("foundationEmbed returns stable fixed-width tiny embeddings", {
  skip_if_not(foundation_backend_available())
  cache <- tempfile("foundation-cache-")
  dir.create(cache)
  x <- foundation_fixture(n_cases = 5L)
  one <- foundationEmbed(
    x,
    model = "moment",
    model_id = "physioml/foundation-tiny",
    revision = tiny_revision,
    manifest_sha256 = tiny_manifest,
    batch_size = 2L,
    cache_dir = cache,
    offline = TRUE
  )
  two <- foundationEmbed(
    x,
    model = "moment",
    model_id = "physioml/foundation-tiny",
    revision = tiny_revision,
    manifest_sha256 = tiny_manifest,
    batch_size = 4L,
    cache_dir = cache,
    offline = TRUE
  )
  expect_s3_class(one, "physio_foundation_embedding")
  expect_equal(dim(one$embeddings), c(5L, 6L))
  expect_identical(one$embeddings, two$embeddings)
  expect_identical(one$identity$case_id, paste0("foundation_", 1:5))
  expect_identical(one$diagnostics$final_batch_size, 1L)
  expect_false(one$diagnostics$target_values_passed)
  expect_false(any(grepl(cache, unlist(one), fixed = TRUE)))
  expect_identical(unserialize(serialize(one, NULL)), one)
})

test_that("foundationEmbed preserves the documented channel token axis", {
  skip_if_not(foundation_backend_available())
  cache <- tempfile("foundation-cache-")
  dir.create(cache)
  x <- foundation_fixture(n_cases = 3L)
  output <- foundationEmbed(
    x,
    model = "moment",
    model_id = "physioml/foundation-tiny",
    revision = tiny_revision,
    manifest_sha256 = tiny_manifest,
    reduction = "none",
    batch_size = 2L,
    cache_dir = cache,
    offline = TRUE
  )
  expect_equal(dim(output$embeddings), c(3L, 2L, 6L))
  expect_identical(output$contract$reduction, "none")
  expect_identical(output$contract$channel_ids, c("C3", "C4"))
})

test_that("tiny Python adapter agrees with its independent CSV reference", {
  skip_if_not(foundation_backend_available())
  directory <- .ml_foundation_fixture_dir()
  input <- utils::read.csv(file.path(directory, "inputs.csv"))
  expected <- utils::read.csv(file.path(directory, "embeddings.csv"))
  values <- array(0, c(101L, 2L, 16L))
  for (row in seq_len(nrow(input))) {
    values[
      input$item[[row]],
      input$channel[[row]],
      input$sample[[row]]
    ] <- input$value[[row]]
  }
  bridge <- .ml_foundation_python(c("numpy", "torch", "safetensors"))
  for (batch_size in c(1L, 2L, 5L, 17L)) {
    actual <- bridge$embed(
      directory,
      "moment",
      "physioml/foundation-tiny",
      values,
      "mean",
      batch_size
    )$embeddings
    expect_equal(
      unname(as.matrix(actual)),
      unname(as.matrix(expected[, -1L])),
      tolerance = 2e-6
    )
  }
  expect_error(
    bridge$embed(
      directory,
      "moment",
      "physioml/foundation-tiny",
      aperm(values[1:2, , , drop = FALSE], c(1L, 3L, 2L)),
      "mean",
      2L
    ),
    "last axis"
  )
})

test_that("foundationEmbed distinguishes input and provider failures", {
  cache <- tempfile("foundation-cache-")
  dir.create(cache)
  x <- foundation_fixture()
  no_units <- x
  SummarizedExperiment::colData(no_units)$unit <- NULL
  expect_error(
    foundationEmbed(
      no_units,
      model = "moment",
      model_id = "physioml/foundation-tiny",
      revision = tiny_revision,
      manifest_sha256 = tiny_manifest,
      cache_dir = cache,
      offline = TRUE
    ),
    "explicit.*unit"
  )
  mixed <- x
  SummarizedExperiment::colData(mixed)$unit <- c("uV", "mV")
  expect_error(
    foundationEmbed(
      mixed,
      model = "moment",
      model_id = "physioml/foundation-tiny",
      revision = tiny_revision,
      manifest_sha256 = tiny_manifest,
      cache_dir = cache,
      offline = TRUE
    ),
    "common unit"
  )
  expect_error(
    foundationEmbed(
      x,
      model = "chronos2",
      revision = .ml_foundation_registry()$default_revision[[2L]],
      manifest_sha256 = tiny_manifest,
      cache_dir = cache,
      offline = TRUE
    ),
    "no version-gated"
  )
  expect_error(
    foundationEmbed(
      x,
      model = "labram",
      revision = .ml_foundation_registry()$default_revision[[3L]],
      manifest_sha256 = tiny_manifest,
      cache_dir = cache,
      offline = TRUE
    ),
    "200 Hz"
  )
  expect_error(
    foundationEmbed(
      x,
      model = "mom",
      revision = tiny_revision,
      manifest_sha256 = tiny_manifest,
      cache_dir = cache,
      offline = TRUE
    ),
    "exactly"
  )
})

test_that("foundationEmbed retains one-item final batches and window identity", {
  skip_if_not(foundation_backend_available())
  cache <- tempfile("foundation-cache-")
  dir.create(cache)
  x <- foundation_fixture(n_cases = 1L, n_time = 33L)
  output <- foundationEmbed(
    x,
    model = "moment",
    model_id = "physioml/foundation-tiny",
    revision = tiny_revision,
    manifest_sha256 = tiny_manifest,
    window_samples = 16L,
    stride_samples = 16L,
    batch_size = 3L,
    cache_dir = cache,
    offline = TRUE
  )
  expect_equal(dim(output$embeddings), c(2L, 6L))
  expect_identical(output$identity$first_sample, c(1L, 17L))
  expect_identical(output$identity$last_sample, c(16L, 32L))
  expect_identical(output$diagnostics$trailing_samples, 1L)
})
