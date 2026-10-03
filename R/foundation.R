.ml_foundation_units <- function(x, channel_indices) {
  metadata <- SummarizedExperiment::colData(x)
  if (!"unit" %in% names(metadata)) {
    .ml_stop("foundation input requires explicit `colData(x)$unit` values")
  }
  units <- as.character(metadata[["unit"]])[channel_indices]
  if (!length(units) || anyNA(units) || any(!nzchar(trimws(units)))) {
    .ml_stop("foundation input channel units must be explicit and non-empty")
  }
  if (length(unique(units)) != 1L) {
    .ml_stop("foundation input channels must use one explicit common unit")
  }
  enc2utf8(units[[1L]])
}

.ml_foundation_preflight <- function(row, model_id, resolved, units) {
  tiny <- identical(model_id, "physioml/foundation-tiny")
  if (tiny) {
    return(list(
      adapter = "physioml_tiny",
      operations = "none",
      input_axes = "batch,channel,time"
    ))
  }
  if (row$adapter[[1L]] == "labram") {
    if (!identical(as.numeric(resolved$sampling_rate), 200) ||
        !units %in% c("uV", "\u00b5V")) {
      .ml_stop(
        "LaBraM requires declared 200 Hz input in microvolts; no unit conversion is inferred"
      )
    }
    .ml_stop(paste0(
      "LaBraM official checkpoint execution is not CPU-verified: its ",
      "ordered montage, repository code, and pickle checkpoint do not meet ",
      "the safe adapter boundary"
    ))
  }
  if (row$adapter[[1L]] == "bendr") {
    .ml_stop(paste0(
      "BENDR official checkpoint execution is not CPU-verified: its DN3 ",
      "repository-code and pickle checkpoint contract is unsupported"
    ))
  }
  if (row$adapter[[1L]] == "chronos2") {
    .ml_stop(paste0(
      "Chronos-2 has no version-gated documented fixed-width embedding ",
      "method at registered revision ", row$default_revision[[1L]]
    ))
  }
  if (!isTRUE(row$cpu_verified[[1L]])) {
    .ml_stop(sprintf(
      paste0(
        "%s official adapter is not CPU-verified at revision %s; ",
        "run the recorded provider preflight before enabling it"
      ),
      row$model_family[[1L]],
      row$default_revision[[1L]]
    ))
  }
  list(
    adapter = row$adapter[[1L]],
    operations = "provider-defined",
    input_axes = row$input_axes[[1L]]
  )
}

.ml_foundation_result_plain <- function(x) {
  if (inherits(x, "python.builtin.object") ||
      typeof(x) == "externalptr") {
    return(FALSE)
  }
  if (is.list(x)) {
    return(all(vapply(x, .ml_foundation_result_plain, logical(1))))
  }
  TRUE
}

#' Obtain governed foundation-model embeddings
#'
#' The exact model ID, immutable revision, complete manifest hash, ordered
#' channels, sampling rate, units, and window contract are recorded. Python and
#' all provider packages remain caller-managed. A first upstream download is
#' explicit; package code never installs software or selects an environment.
#'
#' Stable dimensions are promised only for one exact model, revision, and
#' preprocessing contract. The output is not a clinical prediction and does
#' not establish fairness, causal invariance, or domain invariance.
#'
#' @param x A `PhysioExperiment`.
#' @param model Exact adapter name.
#' @param model_id Optional exact upstream model identifier.
#' @param revision Immutable 40-character commit hash.
#' @param manifest_sha256 Expected canonical cache-manifest SHA-256.
#' @param assay_name,cases,channels Exact input selections.
#' @param window_samples,stride_samples Complete-window settings.
#' @param reduction Exact documented embedding reduction.
#' @param batch_size Positive whole-number batch size.
#' @param device Currently exactly `"cpu"`.
#' @param cache_dir Caller-owned cache root.
#' @param allow_download Explicit first-download consent.
#' @param offline Whether all network access is forbidden.
#'
#' @return A path-free, serializable `physio_foundation_embedding`.
#' @export
#' @examples
#' arr <- array(stats::rnorm(400 * 2 * 2), dim = c(400, 2, 2))
#' pe <- PhysioExperiment::PhysioExperiment(
#'   assays = list(raw = arr),
#'   colData = S4Vectors::DataFrame(label = c("C3", "C4"), unit = "uV"),
#'   samplingRate = 200
#' )
#' \donttest{
#' # Needs a pinned revision, the expected manifest hash, a caller-managed
#' # Python env, and (on first use) explicit download consent.
#' emb <- foundationEmbed(
#'   pe,
#'   model = "moment",
#'   revision = "411e288267f82cce86296dbe4d6c8bc533cc162f",
#'   manifest_sha256 = strrep("0", 64), # replace with the real manifest hash
#'   allow_download = TRUE
#' )
#' }
foundationEmbed <- function(
    x,
    model = c("moment", "chronos2", "labram", "bendr"),
    model_id = NULL,
    revision,
    manifest_sha256,
    assay_name = NULL,
    cases = NULL,
    channels = NULL,
    window_samples = NULL,
    stride_samples = NULL,
    reduction = c("mean", "none"),
    batch_size = 16L,
    device = "cpu",
    cache_dir = tools::R_user_dir("PhysioML", "cache"),
    allow_download = FALSE,
    offline = FALSE) {
  if (missing(model)) model <- "moment"
  row <- .ml_foundation_row(model)
  model <- row$adapter[[1L]]
  model_id <- .ml_foundation_model_id(model_id, row)
  revision <- .ml_foundation_revision(revision)
  manifest_sha256 <- .ml_manifest_sha256(manifest_sha256)
  if (missing(reduction)) reduction <- "mean"
  reduction <- .ml_exact_enum(reduction, c("mean", "none"), "reduction")
  batch_size <- .ml_whole_number(batch_size, "batch_size", minimum = 1L)
  device <- .ml_exact_enum(device, "cpu", "device")
  allow_download <- .ml_flag(allow_download, "allow_download")
  offline <- .ml_flag(offline, "offline")

  resolved <- .ml_resolve_input(
    x,
    assay_name = assay_name,
    channels = channels,
    cases = cases
  )
  units <- .ml_foundation_units(
    x,
    resolved$channel_data$input_channel_index
  )
  preprocessing <- .ml_foundation_preflight(
    row,
    model_id,
    resolved,
    units
  )
  plan <- .ml_window_plan(resolved, window_samples, stride_samples)
  windows <- .ml_window_array(resolved$data, plan)
  storage.mode(windows) <- "double"
  cached <- cacheFoundationModel(
    model = model,
    model_id = model_id,
    revision = revision,
    manifest_sha256 = manifest_sha256,
    cache_dir = cache_dir,
    allow_download = allow_download,
    offline = offline,
    license_accepted = allow_download,
    overwrite = FALSE
  )
  root <- .ml_cache_root(cache_dir)
  entry <- .ml_cache_entry(root, model, model_id, revision)
  verified <- .ml_verify_foundation_entry(
    entry,
    model,
    model_id,
    revision,
    manifest_sha256
  )
  required_modules <- c("numpy", "torch", "safetensors")
  if (!identical(model_id, "physioml/foundation-tiny")) {
    required_modules <- c(required_modules, row$python_module[[1L]])
  }
  bridge <- .ml_foundation_python(required_modules)
  result <- bridge$embed(
    verified$asset_dir,
    model,
    model_id,
    windows,
    reduction,
    batch_size
  )
  values <- result$embeddings
  if (!is.numeric(values) || anyNA(values) || any(!is.finite(values))) {
    .ml_stop("foundation adapter returned non-finite or non-numeric embeddings")
  }
  n_items <- nrow(plan$table)
  if (reduction == "mean") {
    values <- as.matrix(values)
    if (nrow(values) != n_items || ncol(values) < 1L) {
      .ml_stop("foundation mean embedding has an invalid item or feature dimension")
    }
    feature_names <- sprintf(
      "%s_%04d",
      model,
      seq_len(ncol(values))
    )
    colnames(values) <- feature_names
    embedding_width <- ncol(values)
  } else {
    if (length(dim(values)) != 3L || dim(values)[[1L]] != n_items ||
        dim(values)[[2L]] != nrow(resolved$channel_data) ||
        dim(values)[[3L]] < 1L) {
      .ml_stop("foundation unreduced embedding has an invalid token-axis contract")
    }
    feature_names <- sprintf(
      "%s_%04d",
      model,
      seq_len(dim(values)[[3L]])
    )
    embedding_width <- dim(values)[[3L]]
  }
  identity <- plan$table[c(
    "item", "case_id", "input_case_index", "window_in_case",
    "first_sample", "last_sample"
  )]
  output <- list(
    embeddings = values,
    identity = identity,
    feature_names = feature_names,
    contract = list(
      adapter = model,
      model_family = if (identical(
        model_id,
        "physioml/foundation-tiny"
      )) {
        "PhysioML tiny validation adapter"
      } else {
        row$model_family[[1L]]
      },
      model_id = model_id,
      revision = revision,
      manifest_sha256 = manifest_sha256,
      input_axes = "batch,channel,time",
      channel_ids = resolved$channel_data$channel_id,
      sampling_rate = resolved$sampling_rate,
      units = units,
      window_samples = plan$window_samples,
      stride_samples = plan$stride_samples,
      reduction = reduction,
      embedding_width = as.integer(embedding_width)
    ),
    preprocessing = preprocessing,
    cache_provenance = list(
      schema_version = cached$manifest$schema_version,
      checkpoint_format = cached$manifest$checkpoint_format,
      manifest_sha256 = cached$manifest_sha256,
      asset_count = as.integer(cached$asset_count),
      asset_bytes = as.numeric(cached$asset_bytes)
    ),
    backend_versions = unlist(result$versions, use.names = TRUE),
    citations = c(row$paper_url[[1L]], row$source_url[[1L]]),
    diagnostics = list(
      n_items = n_items,
      n_batches = ceiling(n_items / batch_size),
      final_batch_size = {
        remainder <- n_items %% batch_size
        if (remainder) remainder else min(batch_size, n_items)
      },
      trailing_samples = plan$trailing_samples,
      target_values_passed = FALSE
    )
  )
  if (!.ml_foundation_result_plain(output)) {
    .ml_stop("foundation result contains a live Python or external pointer")
  }
  class(output) <- c("physio_foundation_embedding", "list")
  output
}
