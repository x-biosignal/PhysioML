test_that("training rejects shared cases and independently fitted validation", {
  ml_skip_torch(require_luz = TRUE)
  fixture <- ml_toy_classification(8)
  train <- peDataset(
    fixture$x,
    targets = fixture$targets[1:6],
    cases = paste0("patient_", 1:6),
    normalization = "zscore"
  )
  shared <- peDataset(
    fixture$x,
    targets = fixture$targets[6:8],
    cases = paste0("patient_", 6:8),
    normalization = "zscore",
    normalization_stats = train$contract$normalization_stats,
    class_levels = train$contract$class_levels
  )
  expect_error(
    trainModel(train, shared, model = "cnn1d", epochs = 1),
    "disjoint"
  )

  independent <- peDataset(
    fixture$x,
    targets = fixture$targets[7:8],
    cases = paste0("patient_", 7:8),
    normalization = "zscore",
    class_levels = train$contract$class_levels
  )
  expect_error(
    trainModel(train, independent, model = "cnn1d", epochs = 1),
    "independently fitted"
  )
})

test_that("fixed CPU classification fixture exceeds predeclared accuracy", {
  ml_skip_torch(require_luz = TRUE)
  fixture <- ml_toy_classification()
  train <- peDataset(
    fixture$x,
    targets = fixture$targets,
    normalization = "zscore"
  )
  fit <- trainModel(
    train,
    model = "cnn1d",
    epochs = 10,
    batch_size = 16,
    learning_rate = 0.03,
    dropout = 0,
    hidden = 4,
    seed = 11
  )
  prediction_data <- peDataset(
    fixture$x,
    normalization = "zscore",
    normalization_stats = train$contract$normalization_stats,
    class_levels = train$contract$class_levels
  )
  prediction <- predictModel(fit, prediction_data, type = "class")

  expect_gt(mean(prediction$predicted_label == fixture$targets), 0.95)
  expect_true(all(c("loss", "accuracy") %in% names(fit$training_history)))
})

test_that("fixed CPU regression fixture meets predeclared RMSE", {
  ml_skip_torch(require_luz = TRUE)
  fixture <- ml_toy_regression()
  train <- peDataset(
    fixture$x,
    targets = fixture$targets,
    task = "regression",
    normalization = "zscore"
  )
  fit <- trainModel(
    train,
    model = "cnn1d",
    epochs = 20,
    batch_size = 16,
    learning_rate = 0.03,
    dropout = 0,
    hidden = 4,
    seed = 12
  )
  prediction_data <- peDataset(
    fixture$x,
    task = "regression",
    normalization = "zscore",
    normalization_stats = train$contract$normalization_stats
  )
  prediction <- predictModel(fit, prediction_data)
  rmse <- sqrt(mean((prediction$response_1 - fixture$targets)^2))

  expect_lt(rmse, 0.3)
  expect_identical(dim(prediction), c(16L, 7L))
  expect_true("rmse" %in% names(fit$training_history))
})

test_that("CPU fits are repeatable and restore R and torch RNG states", {
  ml_skip_torch(require_luz = TRUE)
  fixture <- ml_toy_classification(8)
  data <- peDataset(fixture$x, targets = fixture$targets)
  prediction_data <- peDataset(
    fixture$x,
    class_levels = data$contract$class_levels
  )
  set.seed(314)
  r_state <- .Random.seed
  torch_state <- torch::torch_get_rng_state()$clone()
  first <- trainModel(
    data,
    model = "cnn1d",
    epochs = 2,
    batch_size = 4,
    dropout = 0,
    hidden = 3,
    seed = 22
  )
  expect_identical(.Random.seed, r_state)
  expect_true(torch::torch_equal(torch::torch_get_rng_state(), torch_state))
  second <- trainModel(
    data,
    model = "cnn1d",
    epochs = 2,
    batch_size = 4,
    dropout = 0,
    hidden = 3,
    seed = 22
  )

  expect_identical(first$training_history, second$training_history)
  expect_equal(
    predictModel(first, prediction_data, type = "logit"),
    predictModel(second, prediction_data, type = "logit"),
    tolerance = 0
  )
})

test_that("R and torch RNG states are restored when a callback errors", {
  ml_skip_torch(require_luz = TRUE)
  fixture <- ml_toy_classification(8)
  data <- peDataset(fixture$x, targets = fixture$targets)
  callback <- luz::luz_callback(
    "injected_training_error",
    on_train_batch_begin = function() stop("injected callback failure")
  )()
  set.seed(2718)
  r_state <- .Random.seed
  torch_state <- torch::torch_get_rng_state()$clone()

  expect_error(
    trainModel(
      data,
      model = "cnn1d",
      epochs = 1,
      callbacks = list(callback),
      seed = 8
    ),
    "injected callback failure"
  )
  expect_identical(.Random.seed, r_state)
  expect_true(torch::torch_equal(torch::torch_get_rng_state(), torch_state))
})

test_that("prediction is stable, identity-preserving, and contract guarded", {
  ml_skip_torch(require_luz = TRUE)
  fixture <- ml_toy_classification(8)
  train <- peDataset(
    fixture$x,
    targets = fixture$targets,
    normalization = "zscore"
  )
  fit <- trainModel(
    train,
    model = "cnn1d",
    epochs = 1,
    batch_size = 4,
    dropout = 0,
    hidden = 3
  )
  prediction_data <- peDataset(
    fixture$x,
    normalization = "zscore",
    normalization_stats = train$contract$normalization_stats,
    class_levels = train$contract$class_levels
  )
  before <- lapply(
    fit$fit$model$state_dict(),
    function(x) torch::as_array(x$clone())
  )
  probabilities <- predictModel(fit, prediction_data, type = "probability")
  after <- lapply(
    fit$fit$model$state_dict(),
    function(x) torch::as_array(x$clone())
  )

  expect_equal(rowSums(probabilities), rep(1, 8), tolerance = 1e-7)
  expect_identical(before, after)
  expect_true(fit$fit$model$training)
  classes <- predictModel(fit, prediction_data, type = "class")
  expect_identical(classes$case_id, paste0("patient_", 1:8))
  expect_true(all(classes$predicted_index %in% 0:1))

  independent <- peDataset(
    fixture$x,
    normalization = "zscore",
    class_levels = train$contract$class_levels
  )
  expect_error(predictModel(fit, independent), "independently fitted")
  reordered <- peDataset(
    fixture$x,
    channels = "Cz",
    normalization = "zscore",
    normalization_stats = train$contract$normalization_stats,
    class_levels = rev(train$contract$class_levels)
  )
  expect_error(predictModel(fit, reordered), "class_levels")
  expect_error(predictModel(fit, prediction_data, type = "response"), "only")
})

test_that("EEGNet training always applies canonical max-norm projections", {
  ml_skip_torch(require_luz = TRUE)
  fixture <- ml_toy_classification(4, 64)
  data <- peDataset(fixture$x, targets = fixture$targets)
  fit <- trainModel(
    data,
    model = "eegnet",
    epochs = 1,
    batch_size = 2,
    learning_rate = 0.01,
    dropout = 0,
    seed = 3
  )
  spatial <- fit$fit$model$depthwise$weight
  spatial <- spatial$view(c(spatial$size(1), -1))
  classifier <- fit$fit$model$classifier$weight

  expect_lte(max(torch::as_array(spatial$norm(p = 2, dim = 2))), 1 + 1e-6)
  expect_lte(max(torch::as_array(classifier$norm(p = 2, dim = 2))), 0.25 + 1e-6)
  expect_identical(fit$model_spec$eegnet_variant, "eegnet_canonical_max_norm")
})
