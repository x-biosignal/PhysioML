test_that("all reference architectures export with exact dynamic I/O parity", {
  ml_skip_onnx()
  fixture <- ml_toy_classification(4, 64)
  train <- peDataset(fixture$x, targets = fixture$targets)
  prediction_data <- peDataset(
    fixture$x,
    class_levels = train$contract$class_levels
  )
  for (architecture in c("eegnet", "cnn1d", "tcn")) {
    model <- ml_untrained_dl_model(train, architecture)
    before <- .ml_state_arrays(model$fit$model)
    was_training <- model$fit$model$training
    r_state <- torch::torch_get_rng_state()$clone()
    set.seed(1029)
    random_state <- .Random.seed
    path <- tempfile(fileext = ".onnx")
    export <- exportONNX(model, path)

    expect_lte(export$max_absolute_error, 1e-4)
    expect_lte(export$max_relative_error, 1e-4)
    expect_identical(export$graph$input$name, "signal")
    expect_identical(export$graph$output$name, "output")
    expect_identical(export$graph$input$shape[[1L]], "batch")
    expect_identical(export$graph$output$shape[[1L]], "batch")
    expect_identical(export$graph$opset, 17L)
    expect_identical(before, .ml_state_arrays(model$fit$model))
    expect_identical(model$fit$model$training, was_training)
    expect_true(torch::torch_equal(torch::torch_get_rng_state(), r_state))
    expect_identical(.Random.seed, random_state)

    imported <- importONNX(path)
    expect_identical(imported$metadata$input$dynamic_axes, list("0" = "batch"))
    expect_identical(imported$metadata$output$dynamic_axes, list("0" = "batch"))
    onnx_logits <- onnxPredict(
      imported,
      prediction_data,
      batch_size = 3L,
      type = "logit"
    )
    r_logits <- predictModel(model, prediction_data, type = "logit")
    expect_equal(onnx_logits, r_logits, tolerance = 1e-4, info = architecture)
    expect_identical(dim(onnx_logits), c(4L, 2L))
  }
  lstm <- ml_untrained_dl_model(train, "lstm")
  lstm_path <- tempfile(fileext = ".onnx")
  expect_error(
    exportONNX(lstm, lstm_path),
    "dynamic-batch ONNX export is unsupported"
  )
  lstm_export <- exportONNX(lstm, lstm_path, dynamic_batch = FALSE)
  expect_identical(unlist(lstm_export$graph$input$shape), c(2L, 1L, 64L))
  lstm_import <- importONNX(lstm_path)
  expect_null(lstm_import$metadata$input$dynamic_axes)
  expect_null(lstm_import$metadata$output$dynamic_axes)
  lstm_onnx <- onnxPredict(
    lstm_import,
    prediction_data,
    batch_size = 2L,
    type = "logit"
  )
  expect_equal(
    lstm_onnx,
    predictModel(lstm, prediction_data, batch_size = 2L, type = "logit"),
    tolerance = 1e-4
  )
})

test_that("export writes a canonical identity-free hash-bound card", {
  ml_skip_onnx()
  fixture <- ml_toy_classification(4)
  train <- peDataset(
    fixture$x,
    targets = fixture$targets,
    normalization = "zscore"
  )
  model <- ml_untrained_dl_model(train)
  path <- tempfile(fileext = ".onnx")
  exportONNX(model, path)
  card_path <- paste0(path, ".json")
  card_text <- readChar(
    card_path,
    nchars = file.info(card_path)$size,
    useBytes = TRUE
  )
  card <- .ml_read_card(card_path)

  expect_identical(card_text, .ml_canonical_json(card))
  expect_identical(card$onnx_sha256, .ml_sha256_file(path))
  expect_identical(
    as.numeric(card$onnx_size_bytes),
    unname(file.info(path)$size)
  )
  expect_identical(card$input$shape, list("batch", 1L, 32L))
  expect_identical(card$output$shape, list("batch", 2L))
  expect_identical(card$backend_versions$libtorch, "2.8.0")
  expect_identical(card$exporter_mode, "torchscript-legacy-onnx")
  expect_false(grepl("patient_", card_text, fixed = TRUE))
  expect_false(grepl(tempdir(), card_text, fixed = TRUE))
  expect_false(any(c(
    "case_ids", "window_table", "input_case_indices"
  ) %in% names(card$input_contract)))
})

test_that("export refuses collisions and preserves an existing pair", {
  ml_skip_onnx()
  fixture <- ml_toy_classification(4)
  train <- peDataset(fixture$x, targets = fixture$targets)
  model <- ml_untrained_dl_model(train)
  path <- tempfile(fileext = ".onnx")
  first <- exportONNX(model, path)
  old_graph <- readBin(path, "raw", n = file.info(path)$size)
  old_card <- readBin(paste0(path, ".json"), "raw", n = file.info(paste0(path, ".json"))$size)

  expect_error(exportONNX(model, path), "output exists")
  second <- exportONNX(model, path, overwrite = TRUE)
  expect_identical(first$onnx_sha256, second$onnx_sha256)
  expect_identical(
    readBin(path, "raw", n = file.info(path)$size),
    old_graph
  )
  expect_identical(
    readBin(paste0(path, ".json"), "raw", n = file.info(paste0(path, ".json"))$size),
    old_card
  )
})

test_that("version mismatch is rejected before bridge loading", {
  ml_skip_torch()
  expect_error(
    .ml_check_torch_bridge_versions(list(python_torch = "9.1.0")),
    "before loading TorchScript"
  )
})
