test_that("freezeLayers uses exact named parameters for every architecture", {
  ml_skip_torch()
  fixture <- ml_toy_classification(n_cases = 8L, n_time = 128L)
  dataset <- peDataset(fixture$x, fixture$targets, task = "classification")

  for (architecture in c("eegnet", "cnn1d", "tcn", "lstm")) {
    model <- ml_untrained_dl_model(
      dataset,
      architecture = architecture,
      hidden = 4L,
      seed = 12L
    )
    source <- .ml_state_arrays(model$fit$model)
    source_modes <- .ml_module_modes(model$fit$model)
    frozen <- freezeLayers(model)
    manifest <- frozen$freeze_manifest

    expect_s3_class(frozen, "physio_transfer_model")
    expect_identical(
      manifest$parameter_name,
      sort(manifest$parameter_name, method = "radix")
    )
    expect_true(all(manifest$requires_grad_after[
      startsWith(manifest$parameter_name, "classifier.")
    ]))
    expect_false(any(manifest$requires_grad_after[
      !startsWith(manifest$parameter_name, "classifier.")
    ]))
    expect_identical(
      frozen$source_state_sha256,
      frozen$result_state_sha256
    )
    expect_identical(source, .ml_state_arrays(model$fit$model))
    expect_identical(source_modes, .ml_module_modes(model$fit$model))
  }
})

test_that("freezeLayers rejects ambiguous or empty rules", {
  ml_skip_torch()
  fixture <- ml_toy_classification(n_cases = 8L, n_time = 32L)
  dataset <- peDataset(fixture$x, fixture$targets, task = "classification")
  model <- ml_untrained_dl_model(dataset)

  expect_error(freezeLayers(model, trainable = ""), "unique, non-empty")
  expect_error(
    freezeLayers(model, trainable = c("classifier", "classifier")),
    "unique"
  )
  expect_error(freezeLayers(model, trainable = "class"), "unknown")
  expect_error(
    freezeLayers(
      model,
      trainable = "classifier",
      frozen = "classifier.weight",
      match = "prefix"
    ),
    "overlap"
  )
  expect_error(
    freezeLayers(
      model,
      frozen = names(model$fit$model$named_parameters())
    ),
    "zero trainable"
  )
  expect_error(
    freezeLayers(model, trainable = "class", match = "prefix"),
    "unknown"
  )
  prefix <- freezeLayers(
    model,
    trainable = "classifier",
    match = "prefix"
  )
  expect_identical(
    prefix$freeze_manifest$parameter_name[
      prefix$freeze_manifest$requires_grad_after
    ],
    c("classifier.bias", "classifier.weight")
  )
})

test_that("freezeLayers clones tensor storage and restores RNG", {
  ml_skip_torch()
  fixture <- ml_toy_classification(n_cases = 8L, n_time = 32L)
  dataset <- peDataset(fixture$x, fixture$targets, task = "classification")
  model <- ml_untrained_dl_model(dataset)
  set.seed(33)
  r_state <- .Random.seed
  torch::torch_manual_seed(44)
  torch_state <- torch::torch_get_rng_state()$clone()

  frozen <- freezeLayers(model)
  expect_identical(.Random.seed, r_state)
  expect_identical(
    torch::as_array(torch::torch_get_rng_state()),
    torch::as_array(torch_state)
  )
  source_before <- .ml_state_arrays(model$fit$model)
  torch::with_no_grad({
    frozen$model$fit$model$classifier$weight$add_(1)
  })
  expect_identical(source_before, .ml_state_arrays(model$fit$model))

  expect_error(freezeLayers(model, trainable = "missing"), "unknown")
  expect_identical(.Random.seed, r_state)
  expect_identical(
    torch::as_array(torch::torch_get_rng_state()),
    torch::as_array(torch_state)
  )
})

test_that("fineTune optimizer and frozen-state manifests agree", {
  ml_skip_torch()
  fixture <- ml_toy_classification(n_cases = 16L, n_time = 32L)
  dataset <- peDataset(fixture$x, fixture$targets, task = "classification")
  model <- ml_untrained_dl_model(dataset, hidden = 4L, seed = 8L)
  source_state <- .ml_state_arrays(model$fit$model)

  tuned <- fineTune(
    model,
    dataset,
    strategy = "linear_probe",
    epochs = 2L,
    batch_size = 5L,
    learning_rate = 0.01,
    seed = 19L
  )
  manifest <- tuned$transfer$freeze_manifest
  expect_identical(
    tuned$transfer$optimizer_parameter_names,
    manifest$parameter_name[manifest$requires_grad_after]
  )
  final_state <- .ml_state_arrays(tuned$fit$model)
  frozen_names <- manifest$parameter_name[!manifest$requires_grad_after]
  expect_identical(final_state[frozen_names], source_state[frozen_names])
  expect_false(identical(
    final_state[c("classifier.weight", "classifier.bias")],
    source_state[c("classifier.weight", "classifier.bias")]
  ))
  expect_identical(
    tuned$transfer$module_mode_policy$mode[
      tuned$transfer$module_mode_policy$module_name %in% c("bn1", "bn2")
    ],
    c("eval", "eval")
  )
  expect_identical(source_state, .ml_state_arrays(model$fit$model))
})

test_that("fineTune strategies update only selected components", {
  ml_skip_torch()
  fixture <- ml_toy_classification(n_cases = 16L, n_time = 32L)
  dataset <- peDataset(fixture$x, fixture$targets, task = "classification")
  model <- ml_untrained_dl_model(dataset, hidden = 4L, seed = 14L)

  partial <- fineTune(
    model,
    dataset,
    strategy = "partial",
    unfreeze = c("conv2", "bn2"),
    epochs = 1L,
    batch_size = 4L,
    learning_rate = 0.005,
    seed = 5L
  )
  selected <- partial$transfer$optimizer_parameter_names
  expect_true(all(c(
    "classifier.bias", "classifier.weight",
    "conv2.bias", "conv2.weight", "bn2.bias", "bn2.weight"
  ) %in% selected))
  expect_false(any(startsWith(selected, "conv1.")))

  full <- fineTune(
    model,
    dataset,
    strategy = "full",
    epochs = 1L,
    batch_size = 4L,
    learning_rate = 0.001,
    seed = 5L
  )
  expect_identical(
    full$transfer$optimizer_parameter_names,
    sort(names(model$fit$model$named_parameters()), method = "radix")
  )
  expect_error(
    fineTune(model, dataset, strategy = "partial"),
    "must name"
  )
  expect_error(
    fineTune(model, dataset, strategy = "full", unfreeze = "conv2"),
    "only for partial"
  )
  expect_error(
    fineTune(
      model,
      dataset,
      strategy = "partial",
      unfreeze = "conv"
    ),
    "unknown"
  )
})

test_that("fineTune enforces validation and callback boundaries", {
  ml_skip_torch()
  fixture <- ml_toy_classification(n_cases = 16L, n_time = 32L)
  train_index <- c(1:4, 9:12)
  valid_index <- c(5:8, 13:16)
  train <- peDataset(
    fixture$x,
    fixture$targets[train_index],
    cases = train_index,
    normalization = "zscore",
    task = "classification"
  )
  valid_bad <- peDataset(
    fixture$x,
    fixture$targets[valid_index],
    cases = valid_index,
    normalization = "zscore",
    task = "classification",
    class_levels = train$contract$class_levels
  )
  valid <- peDataset(
    fixture$x,
    fixture$targets[valid_index],
    cases = valid_index,
    normalization = "zscore",
    normalization_stats = train$contract$normalization_stats,
    task = "classification",
    class_levels = train$contract$class_levels
  )
  model <- ml_untrained_dl_model(train, hidden = 4L)

  expect_error(
    fineTune(model, train, valid_bad, epochs = 1L),
    "independently fitted"
  )
  expect_s3_class(
    fineTune(model, train, valid, epochs = 1L, batch_size = 4L),
    "physio_dl_model"
  )
  expect_error(
    fineTune(model, train, epochs = 1L, callbacks = list(identity)),
    "callbacks.*NULL"
  )
  shared <- valid
  shared$contract$case_ids[[1L]] <- train$contract$case_ids[[1L]]
  expect_error(
    fineTune(model, train, shared, epochs = 1L),
    "case IDs must be disjoint"
  )
})

test_that("fineTune is deterministic and keeps caller state", {
  ml_skip_torch()
  fixture <- ml_toy_classification(n_cases = 16L, n_time = 32L)
  dataset <- peDataset(fixture$x, fixture$targets, task = "classification")
  model <- ml_untrained_dl_model(dataset, hidden = 4L, seed = 21L)
  set.seed(55)
  r_state <- .Random.seed
  torch::torch_manual_seed(66)
  torch_state <- torch::torch_get_rng_state()$clone()
  source <- .ml_state_arrays(model$fit$model)

  one <- fineTune(
    model,
    dataset,
    epochs = 2L,
    batch_size = 7L,
    learning_rate = 0.005,
    seed = 23L
  )
  two <- fineTune(
    model,
    dataset,
    epochs = 2L,
    batch_size = 7L,
    learning_rate = 0.005,
    seed = 23L
  )
  expect_identical(
    one$transfer$final_state_sha256,
    two$transfer$final_state_sha256
  )
  expect_identical(.Random.seed, r_state)
  expect_identical(
    torch::as_array(torch::torch_get_rng_state()),
    torch::as_array(torch_state)
  )
  expect_identical(source, .ml_state_arrays(model$fit$model))
  expect_identical(
    predictModel(one, dataset, type = "class"),
    predictModel(two, dataset, type = "class")
  )
})

test_that("partial tuning beats the linear probe on fixed transfer seeds", {
  ml_skip_torch()
  scores <- data.frame(linear = numeric(), partial = numeric())
  for (seed in c(1L, 7L, 19L)) {
    train_fixture <- ml_transfer_fixture(
      1000L + seed,
      32L,
      paste0("transfer_train_", seed)
    )
    test_fixture <- ml_transfer_fixture(
      2000L + seed,
      128L,
      paste0("transfer_test_", seed)
    )
    train <- peDataset(
      train_fixture$x,
      train_fixture$targets,
      task = "classification"
    )
    test <- peDataset(
      test_fixture$x,
      class_levels = train$contract$class_levels,
      task = "classification"
    )
    model <- ml_transfer_reference_model(train)
    linear <- fineTune(
      model,
      train,
      strategy = "linear_probe",
      epochs = 8L,
      batch_size = 16L,
      learning_rate = 0.01,
      seed = seed
    )
    partial <- fineTune(
      model,
      train,
      strategy = "partial",
      unfreeze = c("conv2", "bn2"),
      epochs = 8L,
      batch_size = 16L,
      learning_rate = 0.01,
      seed = seed
    )
    scores <- rbind(
      scores,
      data.frame(
        linear = mean(
          predictModel(linear, test, type = "class")$predicted_label ==
            test_fixture$targets
        ),
        partial = mean(
          predictModel(partial, test, type = "class")$predicted_label ==
            test_fixture$targets
        )
      )
    )
  }
  expect_true(all(scores$partial > scores$linear))
  expect_gte(stats::median(scores$partial - scores$linear), 0.25)
})
