test_that("all reference models preserve batch and output dimensions", {
  ml_skip_torch()
  settings <- list(
    eegnet = list(),
    cnn1d = list(),
    tcn = list(levels = 2L),
    lstm = list()
  )
  for (architecture in names(settings)) {
    model <- do.call(
      peModel,
      c(
        list(
          architecture = architecture,
          n_channels = 2L,
          n_time = 64L,
          n_outputs = 3L,
          dropout = 0,
          hidden = 4L,
          seed = 7L
        ),
        settings[[architecture]]
      )
    )
    expect_s3_class(model, "physio_torch_module")
    expect_identical(model(torch::torch_randn(c(1, 2, 64)))$size(), c(1L, 3L))
    expect_identical(model(torch::torch_randn(c(5, 2, 64)))$size(), c(5L, 3L))
    expect_gt(attr(model, "model_spec")$parameter_count, 0)
  }
})

test_that("EEGNet records and implements compact depthwise contracts", {
  ml_skip_torch()
  model <- peModel(
    "eegnet",
    n_channels = 3,
    n_time = 128,
    n_outputs = 2,
    sampling_rate = 128,
    dropout = 0
  )
  spec <- attr(model, "model_spec")
  depthwise <- attr(model$depthwise, "module")
  separable <- attr(model$separable_depthwise, "module")

  expect_identical(spec$hyperparameters$F1, 8L)
  expect_identical(spec$hyperparameters$D, 2L)
  expect_identical(spec$hyperparameters$kernel_length, 64L)
  expect_identical(depthwise$groups, 8L)
  expect_identical(depthwise$kernel_size, c(3L, 1L))
  expect_identical(separable$groups, 16L)
  expect_identical(spec$eegnet_variant, "eegnet_canonical_max_norm")
})

test_that("TCN representations are causal", {
  ml_skip_torch()
  model <- peModel(
    "tcn",
    n_channels = 1,
    n_time = 64,
    n_outputs = 2,
    hidden = 4,
    levels = 2,
    dropout = 0,
    seed = 4
  )
  model$eval()
  original <- torch::torch_randn(c(1, 1, 64))
  changed <- original$clone()
  changed[, , 40:64] <- changed[, , 40:64] + 100

  left <- torch::as_array(model$representations(original)[, , 1:39])
  right <- torch::as_array(model$representations(changed)[, , 1:39])
  expect_equal(left, right, tolerance = 0)
  expect_identical(
    attr(model, "model_spec")$hyperparameters$receptive_field,
    13L
  )
})

test_that("LSTM explicitly uses time-major sequence input and final output", {
  ml_skip_torch()
  model <- peModel(
    "lstm",
    n_channels = 2,
    n_time = 12,
    n_outputs = 1,
    task = "regression",
    hidden = 3,
    dropout = 0,
    seed = 9
  )
  model$eval()
  input <- torch::torch_randn(c(2, 2, 12))
  sequence <- model$lstm(input$permute(c(1, 3, 2)))[[1L]]
  expected <- sequence$select(2L, sequence$size(2L))

  expect_equal(
    torch::as_array(model$representations(input)),
    torch::as_array(expected),
    tolerance = 0
  )
  expect_identical(model(input)$size(), c(2L, 1L))
})

test_that("model constructors reject partial and invalid configurations", {
  ml_skip_torch()
  expect_error(peModel("cnn", 1, 32, 2), "exactly one")
  expect_error(peModel("cnn1d", 1, 1, 2), "at least two")
  expect_error(peModel("eegnet", 1, 16, 2), "pool1")
  expect_error(peModel("tcn", 1, 8, 2, levels = 3), "receptive field")
  expect_error(peModel("lstm", 1, 8, 1, bidirectional = NA), "non-missing")
  expect_error(peModel("cnn1d", 1, 8, 1, unknown = 2), "unsupported")
  expect_error(peModel("lstm", 1, 8, 1, dropout = 1), "less than one")
})
