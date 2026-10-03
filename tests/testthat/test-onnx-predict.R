test_that("import and prediction reject graph, card, and payload mutation", {
  ml_skip_onnx()
  fixture <- ml_toy_classification(4)
  train <- peDataset(fixture$x, targets = fixture$targets)
  model <- ml_untrained_dl_model(train)
  path <- tempfile(fileext = ".onnx")
  exportONNX(model, path)
  card_path <- paste0(path, ".json")

  graph_copy <- tempfile(fileext = ".onnx")
  card_copy <- paste0(graph_copy, ".json")
  file.copy(path, graph_copy)
  file.copy(card_path, card_copy)
  stream <- file(graph_copy, "r+b")
  seek(stream, where = 10L)
  byte <- readBin(stream, "raw", n = 1L)
  seek(stream, where = 10L)
  writeBin(as.raw(bitwXor(as.integer(byte), 1L)), stream)
  close(stream)
  expect_error(importONNX(graph_copy), "hash and size")

  card <- .ml_read_card(card_path)
  card$opset <- 16L
  writeChar(.ml_canonical_json(card), card_copy, eos = NULL, useBytes = TRUE)
  file.copy(path, graph_copy, overwrite = TRUE)
  expect_error(importONNX(graph_copy), "graph and model card")

  imported <- importONNX(path)
  imported$onnx_payload[[1L]] <- as.raw(bitwXor(
    as.integer(imported$onnx_payload[[1L]]),
    1L
  ))
  prediction_data <- peDataset(
    fixture$x,
    class_levels = train$contract$class_levels
  )
  expect_error(
    onnxPredict(imported, prediction_data),
    "integrity"
  )
})

test_that("ONNX prediction preserves order, final batch, and output formatting", {
  ml_skip_onnx()
  fixture <- ml_toy_classification(6)
  train <- peDataset(fixture$x, targets = fixture$targets)
  model <- ml_untrained_dl_model(train)
  prediction_data <- peDataset(
    fixture$x,
    class_levels = train$contract$class_levels
  )
  path <- tempfile(fileext = ".onnx")
  exportONNX(model, path)
  imported <- importONNX(path)
  logits <- onnxPredict(imported, prediction_data, batch_size = 5L, type = "logit")
  probability <- onnxPredict(
    imported,
    prediction_data,
    batch_size = 5L,
    type = "probability"
  )
  classes <- onnxPredict(
    imported,
    prediction_data,
    batch_size = 5L,
    type = "class"
  )

  expect_identical(dim(logits), c(6L, 2L))
  expect_equal(rowSums(probability), rep(1, 6), tolerance = 1e-7)
  expect_identical(classes$case_id, paste0("patient_", 1:6))
  expect_identical(classes$item, 1:6)
  expect_equal(
    logits,
    predictModel(model, prediction_data, batch_size = 5L, type = "logit"),
    tolerance = 1e-4
  )
})

test_that("ONNX prediction enforces frozen normalization and class order", {
  ml_skip_onnx()
  fixture <- ml_toy_classification(6)
  train <- peDataset(
    fixture$x,
    targets = fixture$targets,
    normalization = "zscore"
  )
  model <- ml_untrained_dl_model(train)
  path <- tempfile(fileext = ".onnx")
  exportONNX(model, path)
  imported <- importONNX(path)
  good <- peDataset(
    fixture$x,
    normalization = "zscore",
    normalization_stats = train$contract$normalization_stats,
    class_levels = train$contract$class_levels
  )
  expect_silent(onnxPredict(imported, good))
  independent <- peDataset(
    fixture$x,
    normalization = "zscore",
    class_levels = train$contract$class_levels
  )
  expect_error(onnxPredict(imported, independent), "independently fitted")
  wrong_levels <- peDataset(
    fixture$x,
    normalization = "zscore",
    normalization_stats = train$contract$normalization_stats,
    class_levels = rev(train$contract$class_levels)
  )
  expect_error(onnxPredict(imported, wrong_levels), "class_levels")
})

test_that("imported model is persistent plain data without Python pointers", {
  ml_skip_onnx()
  fixture <- ml_toy_classification(4)
  train <- peDataset(fixture$x, targets = fixture$targets)
  model <- ml_untrained_dl_model(train)
  path <- tempfile(fileext = ".onnx")
  exportONNX(model, path)
  imported <- importONNX(path)
  saved <- tempfile(fileext = ".rds")
  saveRDS(imported, saved)
  restored <- readRDS(saved)

  expect_identical(restored, imported)
  expect_true(is.raw(imported$onnx_payload))
  expect_false(any(vapply(
    imported,
    inherits,
    logical(1),
    what = "python.builtin.object"
  )))
})

test_that("regression ONNX response matches R torch", {
  ml_skip_onnx()
  fixture <- ml_toy_regression(4)
  train <- peDataset(
    fixture$x,
    targets = fixture$targets,
    task = "regression"
  )
  model <- ml_untrained_dl_model(train, task = "regression")
  prediction_data <- peDataset(fixture$x, task = "regression")
  path <- tempfile(fileext = ".onnx")
  exportONNX(model, path)
  imported <- importONNX(path)
  actual <- onnxPredict(imported, prediction_data, batch_size = 3L)
  expected <- predictModel(model, prediction_data, batch_size = 3L)

  expect_equal(actual$response_1, expected$response_1, tolerance = 1e-4)
  expect_identical(actual$case_id, expected$case_id)
  expect_error(
    onnxPredict(imported, prediction_data, type = "logit"),
    "only for classification"
  )
})

test_that("prediction preserves R RNG and rejects wrong fixed batches early", {
  ml_skip_onnx()
  fixture <- ml_toy_classification(4, 64)
  train <- peDataset(fixture$x, targets = fixture$targets)
  prediction_data <- peDataset(
    fixture$x,
    class_levels = train$contract$class_levels
  )
  cnn <- ml_untrained_dl_model(train, "cnn1d")
  cnn_path <- tempfile(fileext = ".onnx")
  exportONNX(cnn, cnn_path)
  imported_cnn <- importONNX(cnn_path)
  set.seed(2901)
  r_state <- .Random.seed
  expect_silent(onnxPredict(imported_cnn, prediction_data, batch_size = 3L))
  expect_identical(.Random.seed, r_state)

  lstm <- ml_untrained_dl_model(train, "lstm")
  lstm_path <- tempfile(fileext = ".onnx")
  exportONNX(lstm, lstm_path, dynamic_batch = FALSE)
  imported_lstm <- importONNX(lstm_path)
  expect_error(
    onnxPredict(imported_lstm, prediction_data, batch_size = 3L),
    "fixed-batch ONNX graph"
  )
})
