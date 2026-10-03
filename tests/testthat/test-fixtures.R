test_that("shipped feature fixtures have valid provenance and hashes", {
  fixture_dir <- system.file("extdata", package = "PhysioML")
  if (!nzchar(fixture_dir)) {
    fixture_dir <- test_path("..", "..", "inst", "extdata")
  }
  manifest <- readLines(
    file.path(fixture_dir, "feature_reference.sha256"),
    warn = FALSE
  )
  expect_length(manifest, 4L)
  valid <- vapply(manifest, function(line) {
    fields <- strsplit(line, "[[:space:]]+")[[1L]]
    path <- file.path(fixture_dir, fields[[2L]])
    file.exists(path) && identical(
      digest::digest(path, algo = "sha256", file = TRUE, serialize = FALSE),
      fields[[1L]]
    )
  }, logical(1))
  expect_true(all(valid))

  dcf <- read.dcf(file.path(fixture_dir, "feature_reference.dcf"))
  expect_identical(unname(dcf[1L, "Schema"]), "1")
  expect_identical(unname(dcf[1L, "UCR-Data-Included"]), "false")
})

test_that("shipped catch22 references match the active official backend", {
  skip_if_not_installed("Rcatch22")
  fixture_dir <- system.file("extdata", package = "PhysioML")
  if (!nzchar(fixture_dir)) {
    fixture_dir <- test_path("..", "..", "inst", "extdata")
  }
  vectors <- read.csv(
    file.path(fixture_dir, "feature_vectors.csv"),
    stringsAsFactors = FALSE
  )
  reference <- read.csv(
    file.path(fixture_dir, "catch22_reference.csv"),
    stringsAsFactors = FALSE
  )
  for (id in unique(vectors$series_id)) {
    observed <- suppressWarnings(Rcatch22::catch22_all(
      vectors$value[vectors$series_id == id]
    ))
    expected <- reference[reference$series_id == id, ]
    expect_identical(as.character(observed$names), expected$feature)
    expect_equal(as.numeric(observed$values), expected$value, tolerance = 1e-12)
  }
})

test_that("synthetic fixture has a fixed non-overlapping TRAIN TEST split", {
  fixture_dir <- system.file("extdata", package = "PhysioML")
  if (!nzchar(fixture_dir)) {
    fixture_dir <- test_path("..", "..", "inst", "extdata")
  }
  split <- read.csv(
    file.path(fixture_dir, "synthetic_train_test.csv"),
    stringsAsFactors = FALSE
  )
  train <- unique(split$case_id[split$split == "TRAIN"])
  test <- unique(split$case_id[split$split == "TEST"])
  expect_length(train, 30L)
  expect_length(test, 20L)
  expect_length(intersect(train, test), 0L)
  expect_true(all(table(split$case_id) == 150L))
})

test_that("torch reference assets are complete and internally hashed", {
  fixture_dir <- system.file("extdata", package = "PhysioML")
  if (!nzchar(fixture_dir)) {
    fixture_dir <- test_path("..", "..", "inst", "extdata")
  }
  paths <- file.path(
    fixture_dir,
    c(
      "torch_reference.rds",
      "torch_reference.csv",
      "torch_reference.dcf",
      "torch_reference.sha256"
    )
  )
  expect_true(all(file.exists(paths)))
  hashes <- read.table(
    paths[[4L]],
    col.names = c("sha256", "file"),
    stringsAsFactors = FALSE
  )
  observed <- vapply(file.path(fixture_dir, hashes$file), function(path) {
    digest::digest(path, algo = "sha256", file = TRUE, serialize = FALSE)
  }, character(1))
  expect_identical(unname(observed), hashes$sha256)

  reference <- readRDS(paths[[1L]])
  expect_identical(reference$schema_version, 1L)
  expect_identical(dim(reference$window$values), c(4L, 2L, 4L))
  expect_equal(reference$forwards$convolution, c(5, 7, 9, 11, 13))
  expect_equal(rowSums(reference$forwards$probabilities), c(1, 1))
  expect_identical(reference$toy$classification_epochs, 10L)
  expect_identical(reference$toy$regression_epochs, 20L)
})
