test_that("ONNX exports and metadata are registered", {
  expect_true(all(
    c("exportONNX", "importONNX", "onnxPredict") %in%
      getNamespaceExports("PhysioML")
  ))
  description_path <- test_path("..", "..", "DESCRIPTION")
  if (!file.exists(description_path)) {
    description_path <- system.file("DESCRIPTION", package = "PhysioML")
  }
  description <- read.dcf(description_path)
  expect_identical(unname(description[1L, "Version"]), "0.4.1")
  expect_match(unname(description[1L, "Imports"]), "jsonlite")
  collate <- unname(description[1L, "Collate"])
  expect_match(collate, "onnx-contract.R")
  expect_match(collate, "onnx-predict.R")
})

test_that("ONNX path, provider, and model validation is exact", {
  expect_error(importONNX("missing.onnx"), "regular file")
  expect_error(
    importONNX("missing.onnx", providers = "CPU"),
    "exactly one of"
  )
  expect_error(onnxPredict(list(), list()), "physio_onnx_model")
  expect_error(
    .ml_onnx_output_path(tempdir(), "path", ".onnx"),
    "file path"
  )
  expect_error(
    .ml_onnx_output_path(file.path(tempdir(), "x.bin"), "path", ".onnx"),
    "end in"
  )
})

test_that("canonical card reader rejects duplicate and noncanonical JSON", {
  path <- tempfile(fileext = ".json")
  writeLines('{"a":1,"a":2}', path, useBytes = TRUE)
  expect_error(.ml_read_card(path), "canonical JSON")
  writeLines('{ "a": 1 }', path, useBytes = TRUE)
  expect_error(.ml_read_card(path), "canonical JSON")
})

test_that("canonical JSON rejects non-finite values in unnamed arrays", {
  expect_error(
    .ml_canonical_json(list(values = list(1, Inf))),
    "non-finite"
  )
})

test_that("ONNX runtime code never installs or selects dependencies", {
  namespace <- asNamespace("PhysioML")
  code <- paste(vapply(
    c(
      ".ml_onnx_python", "exportONNX", "importONNX", "onnxPredict",
      ".ml_validate_onnx_model"
    ),
    function(name) paste(deparse(get(name, namespace)), collapse = "\n"),
    character(1)
  ), collapse = "\n")
  python_path <- system.file(
    "python",
    "physio_onnx.py",
    package = "PhysioML"
  )
  expect_true(file.exists(python_path))
  code <- paste(code, paste(readLines(python_path, warn = FALSE), collapse = "\n"))
  prohibited <- paste0(
    "(py_install|use_(python|virtualenv|condaenv)|install_torch",
    "|pip[.]main|subprocess|os[.]system)"
  )
  expect_false(grepl(prohibited, code))
  expect_false(grepl("trust_remote_code", code, fixed = TRUE))
})

test_that("stable prediction formatting handles extreme finite logits", {
  ml_skip_torch()
  fixture <- ml_toy_classification(4)
  data <- peDataset(
    fixture$x,
    class_levels = c("high", "low")
  )
  model <- list(
    task = "classification",
    class_levels = c("high", "low"),
    model_spec = list(n_outputs = 2L)
  )
  logits <- matrix(c(1e30, -1e30, -1e30, 1e30, 0, 0, 5, 5), ncol = 2)
  probability <- .ml_format_predictions(logits, model, data, "probability")
  expect_true(all(is.finite(probability)))
  expect_equal(rowSums(probability), rep(1, 4), tolerance = 1e-15)
  classes <- .ml_format_predictions(logits, model, data, "class")
  expect_identical(classes$predicted_index, c(0L, 1L, 1L, 0L))
})
