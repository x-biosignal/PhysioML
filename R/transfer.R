.ml_transfer_citations <- c(
  "Yosinski et al. (2014), doi:10.48550/arXiv.1411.1792"
)

.ml_transfer_heads <- c(
  eegnet = "classifier",
  cnn1d = "classifier",
  tcn = "classifier",
  lstm = "classifier"
)

.ml_sorted_names <- function(x) {
  x[order(enc2utf8(x), method = "radix")]
}

.ml_parameter_shape <- function(parameter) {
  paste(as.integer(parameter$size()), collapse = "x")
}

.ml_state_sha256 <- function(module) {
  state <- module$state_dict()
  state <- state[.ml_sorted_names(names(state))]
  payload <- lapply(names(state), function(name) {
    value <- state[[name]]$detach()$to(device = "cpu")$contiguous()
    list(
      name = enc2utf8(name),
      shape = as.integer(value$size()),
      dtype = as.character(value$dtype),
      values = torch::as_array(value)
    )
  })
  .ml_sha256_raw(serialize(payload, NULL, version = 3L))
}

.ml_module_modes <- function(module) {
  out <- logical()
  visit <- function(current, prefix = "") {
    label <- if (nzchar(prefix)) prefix else "<root>"
    out[[label]] <<- isTRUE(current$training)
    children <- current$children
    if (!length(children)) {
      return(invisible(NULL))
    }
    for (name in names(children)) {
      path <- if (nzchar(prefix)) paste(prefix, name, sep = ".") else name
      visit(children[[name]], path)
    }
    invisible(NULL)
  }
  visit(module)
  out
}

.ml_restore_module_modes <- function(module, modes) {
  visit <- function(current, prefix = "") {
    label <- if (nzchar(prefix)) prefix else "<root>"
    if (is.null(modes[[label]])) {
      .ml_stop("module-mode manifest does not match the cloned architecture")
    }
    current$train(mode = isTRUE(modes[[label]]))
    children <- current$children
    if (length(children)) {
      for (name in names(children)) {
        path <- if (nzchar(prefix)) paste(prefix, name, sep = ".") else name
        visit(children[[name]], path)
      }
    }
    invisible(NULL)
  }
  visit(module)
  invisible(module)
}

.ml_clone_fitted_model <- function(model) {
  .ml_assert_dl_model(model)
  source <- model$fit$model
  source_parameters <- source$named_parameters()
  first <- source_parameters[[1L]]
  clone <- .ml_model_generator(model$model_spec)()
  clone$to(device = first$device, dtype = first$dtype)
  clone$load_state_dict(.ml_clone_state_dict(source))
  .ml_restore_module_modes(clone, .ml_module_modes(source))
  attr(clone, "model_spec") <- model$model_spec
  attr(clone, "physio_generator") <- .ml_model_generator(model$model_spec)
  class(clone) <- unique(c("physio_torch_module", class(clone)))
  output <- model
  output$fit <- list(model = clone)
  if (!is.null(model$fit$records)) {
    output$fit$records <- model$fit$records
  }
  output
}

.ml_validate_rules <- function(value, name) {
  if (is.null(value)) {
    return(character())
  }
  if (!is.character(value) || !length(value) || anyNA(value) ||
      any(!nzchar(value)) || anyDuplicated(value)) {
    .ml_stop(sprintf(
      "`%s` must be NULL or unique, non-empty parameter rules",
      name
    ))
  }
  enc2utf8(value)
}

.ml_rule_hits <- function(rules, parameter_names, match) {
  if (!length(rules)) {
    return(stats::setNames(vector("list", 0L), character()))
  }
  stats::setNames(lapply(rules, function(rule) {
    if (match == "exact") {
      which(parameter_names == rule)
    } else {
      which(
        parameter_names == rule |
          startsWith(parameter_names, paste0(rule, "."))
      )
    }
  }), rules)
}

.ml_apply_freeze_rules <- function(module, architecture, trainable, frozen,
                                   match, strict) {
  parameters <- module$named_parameters()
  parameters <- parameters[.ml_sorted_names(names(parameters))]
  parameter_names <- names(parameters)
  before <- vapply(parameters, function(x) isTRUE(x$requires_grad), logical(1))
  trainable <- .ml_validate_rules(trainable, "trainable")
  frozen <- .ml_validate_rules(frozen, "frozen")
  train_hits <- .ml_rule_hits(trainable, parameter_names, match)
  frozen_hits <- .ml_rule_hits(frozen, parameter_names, match)
  unknown <- c(
    names(train_hits)[!lengths(train_hits)],
    names(frozen_hits)[!lengths(frozen_hits)]
  )
  if (strict && length(unknown)) {
    .ml_stop(sprintf(
      "unknown parameter rule(s): %s",
      paste(unique(unknown), collapse = ", ")
    ))
  }
  train_index <- unique(unlist(train_hits, use.names = FALSE))
  frozen_index <- unique(unlist(frozen_hits, use.names = FALSE))
  overlap <- intersect(train_index, frozen_index)
  if (length(overlap)) {
    .ml_stop(sprintf(
      "trainable and frozen rules overlap at: %s",
      paste(parameter_names[overlap], collapse = ", ")
    ))
  }

  if (!length(trainable) && !length(frozen)) {
    head <- unname(.ml_transfer_heads[[architecture]])
    train_index <- .ml_rule_hits(head, parameter_names, "prefix")[[1L]]
    frozen_index <- setdiff(seq_along(parameters), train_index)
    rule <- rep("default:frozen_feature", length(parameters))
    rule[train_index] <- "default:trainable_head"
  } else {
    after <- before
    if (length(trainable)) {
      after[] <- FALSE
      after[train_index] <- TRUE
    }
    if (length(frozen_index)) {
      after[frozen_index] <- FALSE
    }
    rule <- rep("unchanged", length(parameters))
    rule[train_index] <- "trainable"
    rule[frozen_index] <- "frozen"
    train_index <- which(after)
  }
  if (!length(train_index)) {
    .ml_stop("freeze rules leave zero trainable parameters")
  }
  after <- seq_along(parameters) %in% train_index
  for (index in seq_along(parameters)) {
    parameters[[index]]$requires_grad_(after[[index]])
  }
  data.frame(
    parameter_name = parameter_names,
    shape = vapply(parameters, .ml_parameter_shape, character(1)),
    numel = as.numeric(vapply(
      parameters,
      function(x) as.numeric(x$numel()),
      numeric(1)
    )),
    requires_grad_before = before,
    requires_grad_after = after,
    rule = rule,
    stringsAsFactors = FALSE
  )
}

#' Freeze named parts of a fitted torch model
#'
#' The fitted model is cloned before `requires_grad` is changed. Exact matching
#' is the default; prefix matching is component-aware.
#'
#' @param model A fitted `physio_dl_model`.
#' @param trainable,frozen Optional unique exact parameter names or prefixes.
#' @param match Exact selection mode.
#' @param strict Whether unmatched rules are errors.
#'
#' @return A plain `physio_transfer_model`.
#' @export
#' @examples
#' arr <- array(stats::rnorm(50 * 2 * 6), dim = c(50, 2, 6))
#' pe <- PhysioExperiment::PhysioExperiment(
#'   assays = list(raw = arr),
#'   colData = S4Vectors::DataFrame(label = c("C3", "C4")),
#'   samplingRate = 100
#' )
#' # Needs the optional torch and luz backends.
#' if (requireNamespace("torch", quietly = TRUE) &&
#'     requireNamespace("luz", quietly = TRUE)) {
#'   train <- peDataset(
#'     pe, targets = rep(c("a", "b"), 3),
#'     task = "classification", class_levels = c("a", "b")
#'   )
#'   fit <- trainModel(train, model = "cnn1d", epochs = 1L)
#'   frozen <- freezeLayers(fit)
#'   frozen$freeze_manifest
#' }
freezeLayers <- function(
    model,
    trainable = NULL,
    frozen = NULL,
    match = c("exact", "prefix"),
    strict = TRUE) {
  .ml_require_torch()
  .ml_assert_dl_model(model)
  if (missing(match)) match <- "exact"
  match <- .ml_exact_enum(match, c("exact", "prefix"), "match")
  strict <- .ml_flag(strict, "strict")
  source_hash <- .ml_state_sha256(model$fit$model)
  source_mode <- .ml_module_modes(model$fit$model)
  result <- .ml_preserve_r_rng(.ml_preserve_torch_rng({
    clone <- .ml_clone_fitted_model(model)
    manifest <- .ml_apply_freeze_rules(
      clone$fit$model,
      model$model_spec$architecture,
      trainable,
      frozen,
      match,
      strict
    )
    list(
      model = clone,
      freeze_manifest = manifest,
      source_model_spec = model$model_spec,
      source_state_sha256 = source_hash,
      result_state_sha256 = .ml_state_sha256(clone$fit$model),
      citations = .ml_transfer_citations
    )
  }))
  if (!identical(source_hash, .ml_state_sha256(model$fit$model)) ||
      !identical(source_mode, .ml_module_modes(model$fit$model))) {
    .ml_stop("source model state or mode changed while freezing layers")
  }
  if (!identical(result$source_state_sha256, result$result_state_sha256)) {
    .ml_stop("freezing changed parameter or buffer values")
  }
  class(result) <- c("physio_transfer_model", "list")
  result
}

.ml_validate_fine_tune_contract <- function(model, train, valid) {
  .ml_assert_dataset(train, "train", require_targets = TRUE)
  if (!is.null(valid)) {
    .ml_assert_dataset(valid, "valid", require_targets = TRUE)
    .ml_compare_dataset_contracts(train, valid)
  }
  reference <- model$input_contract
  incoming <- train$contract
  for (field in c(
    "task", "channel_ids", "window_samples", "dtype", "normalization_method",
    "class_levels"
  )) {
    if (!identical(reference[[field]], incoming[[field]])) {
      .ml_stop(sprintf(
        "fine-tuning `%s` contract does not match the fitted model",
        field
      ))
    }
  }
  if (dim(train$data)[[3L]] != model$model_spec$n_time ||
      model$model_spec$n_outputs != if (model$task == "classification") {
        length(train$contract$class_levels)
      } else {
        1L
      }) {
    .ml_stop("fine-tuning input or output dimensions do not match the model")
  }
  invisible(TRUE)
}

.ml_set_module_mode_policy <- function(module) {
  parameters <- module$named_parameters()
  trainable <- names(parameters)[vapply(
    parameters,
    function(x) isTRUE(x$requires_grad),
    logical(1)
  )]
  policy <- data.frame(
    module_name = character(),
    mode = character(),
    has_parameters = logical(),
    has_trainable_parameters = logical(),
    stringsAsFactors = FALSE
  )
  visit <- function(current, prefix = "", parent_mode = FALSE) {
    label <- if (nzchar(prefix)) prefix else "<root>"
    direct <- current$named_parameters(recursive = FALSE)
    subtree <- current$named_parameters(recursive = TRUE)
    subtree_names <- if (nzchar(prefix)) {
      paste(prefix, names(subtree), sep = ".")
    } else {
      names(subtree)
    }
    has_parameters <- length(subtree_names) > 0L
    has_trainable <- any(subtree_names %in% trainable)
    mode <- if (has_parameters) has_trainable else parent_mode
    current$train(mode = mode)
    policy <<- rbind(
      policy,
      data.frame(
        module_name = label,
        mode = if (mode) "train" else "eval",
        has_parameters = length(direct) > 0L,
        has_trainable_parameters = any(
          if (nzchar(prefix)) {
            paste(prefix, names(direct), sep = ".") %in% trainable
          } else {
            names(direct) %in% trainable
          }
        ),
        stringsAsFactors = FALSE
      )
    )
    children <- current$children
    if (length(children)) {
      for (name in names(children)) {
        path <- if (nzchar(prefix)) paste(prefix, name, sep = ".") else name
        visit(children[[name]], path, mode)
      }
    }
    invisible(NULL)
  }
  visit(module)
  policy
}

.ml_batch_metric <- function(output, target, task) {
  if (task == "classification") {
    predicted <- torch::torch_argmax(output, dim = 2L) - 1L
    return(c(
      numerator = as.numeric((predicted == target)$sum()$item()),
      denominator = as.numeric(target$numel())
    ))
  }
  error <- output - target
  c(
    numerator = as.numeric((error^2)$sum()$item()),
    denominator = as.numeric(target$numel())
  )
}

.ml_transfer_epoch <- function(module, dataset, batch_size, task,
                               optimizer = NULL, shuffle = FALSE,
                               eegnet = FALSE) {
  n <- nrow(dataset$contract$window_table)
  index <- if (shuffle) {
    as.integer(torch::as_array(torch::torch_randperm(n))) + 1L
  } else {
    seq_len(n)
  }
  groups <- split(index, ceiling(seq_along(index) / batch_size))
  total_loss <- total_n <- numerator <- denominator <- 0
  for (group in groups) {
    batch <- dataset$.getbatch(group)
    input <- batch[[1L]]
    target <- batch[[2L]]
    if (!is.null(optimizer)) optimizer$zero_grad()
    output <- module(input)
    loss <- if (task == "classification") {
      torch::nn_cross_entropy_loss()(output, target + 1L)
    } else {
      torch::nn_mse_loss()(output, target)
    }
    if (!is.null(optimizer)) {
      loss$backward()
      optimizer$step()
      if (eegnet) {
        torch::with_no_grad({
          if (isTRUE(module$depthwise$weight$requires_grad)) {
            .ml_project_rows(module$depthwise$weight, 1)
          }
          if (isTRUE(module$classifier$weight$requires_grad)) {
            .ml_project_rows(module$classifier$weight, 0.25)
          }
        })
      }
    }
    count <- as.numeric(target$size(1L))
    total_loss <- total_loss + as.numeric(loss$item()) * count
    total_n <- total_n + count
    metric <- .ml_batch_metric(output$detach(), target, task)
    numerator <- numerator + metric[["numerator"]]
    denominator <- denominator + metric[["denominator"]]
  }
  metric <- if (task == "classification") {
    numerator / denominator
  } else {
    sqrt(numerator / denominator)
  }
  c(loss = total_loss / total_n, metric = metric)
}

.ml_dataset_digest <- function(x) {
  digest::digest(
    list(data = x$data, targets = x$targets, contract = x$contract),
    algo = "sha256",
    serialize = TRUE
  )
}

#' Fine-tune a fitted torch model
#'
#' Training is CPU-only and passes only explicitly trainable parameters to
#' Adam. Frozen modules, including their batch-normalization buffers, remain in
#' evaluation mode. Validation and prediction data must reuse frozen
#' normalization statistics.
#'
#' @param model A fitted `physio_dl_model` or `physio_transfer_model`.
#' @param train,valid Compatible targeted datasets.
#' @param strategy Exact tuning strategy.
#' @param unfreeze Component prefixes used only by partial tuning.
#' @param epochs,batch_size Positive whole-number settings.
#' @param learning_rate Positive Adam learning rate.
#' @param weight_decay Non-negative Adam weight decay.
#' @param callbacks Currently `NULL`; callback execution is owned by
#'   [trainModel()]'s luz workflow.
#' @param seed Whole-number CPU seed.
#' @param device Currently exactly `"cpu"`.
#' @param num_workers Currently exactly zero.
#' @param verbose Whether to print one line per epoch.
#'
#' @return A fitted `physio_dl_model` with a plain `transfer` record.
#' @export
#' @examples
#' arr <- array(stats::rnorm(50 * 2 * 6), dim = c(50, 2, 6))
#' pe <- PhysioExperiment::PhysioExperiment(
#'   assays = list(raw = arr),
#'   colData = S4Vectors::DataFrame(label = c("C3", "C4")),
#'   samplingRate = 100
#' )
#' # Needs the optional torch and luz backends.
#' if (requireNamespace("torch", quietly = TRUE) &&
#'     requireNamespace("luz", quietly = TRUE)) {
#'   train <- peDataset(
#'     pe, targets = rep(c("a", "b"), 3),
#'     task = "classification", class_levels = c("a", "b")
#'   )
#'   fit <- trainModel(train, model = "cnn1d", epochs = 1L)
#'   tuned <- fineTune(fit, train, strategy = "linear_probe", epochs = 1L)
#' }
fineTune <- function(
    model,
    train,
    valid = NULL,
    strategy = c("linear_probe", "partial", "full"),
    unfreeze = NULL,
    epochs = 10L,
    batch_size = 32L,
    learning_rate = 1e-4,
    weight_decay = 0,
    callbacks = NULL,
    seed = 1L,
    device = "cpu",
    num_workers = 0L,
    verbose = FALSE) {
  .ml_require_torch()
  if (inherits(model, "physio_transfer_model")) {
    if (!is.list(model$model)) {
      .ml_stop("`model` is an incomplete `physio_transfer_model`")
    }
    source <- model$model
  } else {
    source <- model
  }
  .ml_assert_dl_model(source)
  .ml_validate_fine_tune_contract(source, train, valid)
  if (missing(strategy)) strategy <- "linear_probe"
  strategy <- .ml_exact_enum(
    strategy,
    c("linear_probe", "partial", "full"),
    "strategy"
  )
  unfreeze <- .ml_validate_rules(unfreeze, "unfreeze")
  if (strategy == "partial" && !length(unfreeze)) {
    .ml_stop("`unfreeze` must name at least one component for partial tuning")
  }
  if (strategy != "partial" && length(unfreeze)) {
    .ml_stop("`unfreeze` is available only for partial tuning")
  }
  epochs <- .ml_whole_number(epochs, "epochs", minimum = 1L)
  batch_size <- .ml_whole_number(batch_size, "batch_size", minimum = 1L)
  learning_rate <- .ml_positive_number(learning_rate, "learning_rate")
  weight_decay <- .ml_positive_number(
    weight_decay,
    "weight_decay",
    allow_zero = TRUE
  )
  seed <- .ml_seed(seed)
  device <- .ml_exact_enum(device, "cpu", "device")
  num_workers <- .ml_whole_number(
    num_workers,
    "num_workers",
    minimum = 0L,
    maximum = 0L
  )
  verbose <- .ml_flag(verbose, "verbose")
  if (!is.null(callbacks)) {
    .ml_stop(paste0(
      "`callbacks` must be NULL for fineTune(); use trainModel() for the ",
      "luz callback lifecycle"
    ))
  }
  if (train$contract$task == "classification") {
    expected <- seq_along(train$contract$class_levels) - 1L
    if (!identical(sort(unique(train$targets)), as.integer(expected))) {
      .ml_stop("fine-tuning data must contain every declared class")
    }
  }

  source_hash <- .ml_state_sha256(source$fit$model)
  source_modes <- .ml_module_modes(source$fit$model)
  train_digest <- .ml_dataset_digest(train)
  valid_digest <- if (is.null(valid)) NULL else .ml_dataset_digest(valid)
  result <- .ml_preserve_r_rng(.ml_preserve_torch_rng({
    output <- .ml_clone_fitted_model(source)
    module <- output$fit$model
    architecture <- source$model_spec$architecture
    if (strategy == "full") {
      manifest <- .ml_apply_freeze_rules(
        module,
        architecture,
        trainable = names(module$named_parameters()),
        frozen = NULL,
        match = "exact",
        strict = TRUE
      )
    } else {
      head <- unname(.ml_transfer_heads[[architecture]])
      rules <- if (strategy == "partial") c(head, unfreeze) else head
      manifest <- .ml_apply_freeze_rules(
        module,
        architecture,
        trainable = rules,
        frozen = NULL,
        match = "prefix",
        strict = TRUE
      )
    }
    initial_hash <- .ml_state_sha256(module)
    mode_policy <- .ml_set_module_mode_policy(module)
    parameters <- module$named_parameters()
    optimizer_names <- names(parameters)[vapply(
      parameters,
      function(x) isTRUE(x$requires_grad),
      logical(1)
    )]
    optimizer_names <- .ml_sorted_names(optimizer_names)
    expected_names <- manifest$parameter_name[manifest$requires_grad_after]
    if (!identical(optimizer_names, expected_names)) {
      .ml_stop("optimizer parameter manifest does not match freeze manifest")
    }
    optimizer <- torch::optim_adam(
      unname(parameters[optimizer_names]),
      lr = learning_rate,
      weight_decay = weight_decay
    )
    set.seed(seed)
    torch::torch_manual_seed(seed)
    history <- list()
    for (epoch in seq_len(epochs)) {
      .ml_set_module_mode_policy(module)
      train_metric <- .ml_transfer_epoch(
        module,
        train,
        batch_size,
        source$task,
        optimizer = optimizer,
        shuffle = TRUE,
        eegnet = architecture == "eegnet"
      )
      history[[length(history) + 1L]] <- data.frame(
        epoch = epoch,
        set = "train",
        loss = unname(train_metric[["loss"]]),
        metric = unname(train_metric[["metric"]]),
        stringsAsFactors = FALSE
      )
      if (!is.null(valid)) {
        module$eval()
        valid_metric <- torch::with_no_grad({
          .ml_transfer_epoch(
            module,
            valid,
            batch_size,
            source$task,
            optimizer = NULL,
            shuffle = FALSE
          )
        })
        history[[length(history) + 1L]] <- data.frame(
          epoch = epoch,
          set = "valid",
          loss = unname(valid_metric[["loss"]]),
          metric = unname(valid_metric[["metric"]]),
          stringsAsFactors = FALSE
        )
      }
      if (verbose) {
        message(sprintf(
          "epoch %d: loss=%.6f, metric=%.6f",
          epoch,
          train_metric[["loss"]],
          train_metric[["metric"]]
        ))
      }
    }
    .ml_set_module_mode_policy(module)
    output$fit <- list(model = module)
    output$input_contract <- train$contract
    output$normalization_stats <- train$contract$normalization_stats
    output$training_history <- do.call(rbind, history)
    output$settings <- list(
      epochs = epochs,
      batch_size = batch_size,
      learning_rate = learning_rate,
      weight_decay = weight_decay,
      seed = seed,
      device = device,
      num_workers = num_workers,
      validation_shuffle = FALSE,
      drop_last = FALSE
    )
    output$transfer <- list(
      strategy = strategy,
      source_state_sha256 = source_hash,
      initial_state_sha256 = initial_hash,
      final_state_sha256 = .ml_state_sha256(module),
      freeze_manifest = manifest,
      optimizer_parameter_names = optimizer_names,
      module_mode_policy = mode_policy,
      source_versions = source$versions,
      citations = .ml_transfer_citations
    )
    class(output) <- "physio_dl_model"
    output
  }))
  if (!identical(source_hash, .ml_state_sha256(source$fit$model)) ||
      !identical(source_modes, .ml_module_modes(source$fit$model))) {
    .ml_stop("source model state or mode changed during fine-tuning")
  }
  if (!identical(train_digest, .ml_dataset_digest(train)) ||
      !identical(valid_digest, if (is.null(valid)) NULL else {
        .ml_dataset_digest(valid)
      })) {
    .ml_stop("a Dataset changed during fine-tuning")
  }
  result
}
