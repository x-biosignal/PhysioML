.ml_foundation_schema <- "1.0.0"
.ml_foundation_max_files <- 10000L
.ml_foundation_max_file_bytes <- 2 * 1024^3
.ml_foundation_max_total_bytes <- 4 * 1024^3

.ml_foundation_registry <- function() {
  data.frame(
    adapter = c("moment", "chronos2", "labram", "bendr"),
    model_family = c("MOMENT", "Chronos-2", "LaBraM", "BENDR"),
    default_model_id = c(
      "AutonLab/MOMENT-1-small",
      "amazon/chronos-2",
      "935963004/LaBraM",
      "SPOClab-ca/BENDR"
    ),
    default_revision = c(
      "411e288267f82cce86296dbe4d6c8bc533cc162f",
      "29ec3766d36d6f73f0696f85560a422f50e8498c",
      "c431221e6cfd23dbfa9950e0180682fb322b0548",
      "ac918abaec111d15fcaa2a8fcd2bd3d8b0d81a10"
    ),
    python_module = c("momentfm", "chronos", "labram", "bendr"),
    loader = c(
      "MOMENTPipeline embedding task",
      "unsupported:no documented Chronos2 embedding method",
      "unsupported:repository code plus pickle checkpoint",
      "unsupported:DN3 repository code plus pickle checkpoint"
    ),
    input_axes = rep("batch,channel,time", 4L),
    sampling_rate = c("model-specific", "model-specific", "200", "256"),
    units = c("explicit", "explicit", "uV", "uV"),
    channel_contract = c(
      "ordered channels",
      "ordered series",
      "official ordered EEG montage",
      "official ordered EEG montage"
    ),
    embedding_reduction = c(
      "documented mean/none",
      "unavailable",
      "unavailable",
      "unavailable"
    ),
    cpu_verified = rep(FALSE, 4L),
    license = c("MIT", "Apache-2.0", "MIT", "research code; verify weights"),
    source_url = c(
      "https://github.com/moment-timeseries-foundation-model/moment",
      "https://github.com/amazon-science/chronos-forecasting",
      "https://github.com/935963004/LaBraM",
      "https://github.com/SPOClab-ca/BENDR"
    ),
    paper_url = c(
      "https://arxiv.org/abs/2402.03885",
      "https://arxiv.org/abs/2510.15821",
      "https://openreview.net/forum?id=QzTpTRVtrP",
      "https://arxiv.org/abs/2101.12037"
    ),
    stringsAsFactors = FALSE
  )
}

.ml_foundation_row <- function(model) {
  model <- .ml_exact_enum(
    model,
    c("moment", "chronos2", "labram", "bendr"),
    "model"
  )
  registry <- .ml_foundation_registry()
  registry[registry$adapter == model, , drop = FALSE]
}

.ml_foundation_revision <- function(revision) {
  if (missing(revision) || !is.character(revision) ||
      length(revision) != 1L || is.na(revision) ||
      !grepl("^[0-9a-f]{40}$", revision)) {
    .ml_stop(
      "`revision` must be one immutable 40-character lowercase commit hash"
    )
  }
  revision
}

.ml_manifest_sha256 <- function(manifest_sha256) {
  if (missing(manifest_sha256) || !is.character(manifest_sha256) ||
      length(manifest_sha256) != 1L || is.na(manifest_sha256) ||
      !grepl("^[0-9a-f]{64}$", manifest_sha256)) {
    .ml_stop("`manifest_sha256` must be one lowercase SHA-256 value")
  }
  manifest_sha256
}

.ml_foundation_model_id <- function(model_id, row) {
  if (is.null(model_id)) {
    return(row$default_model_id[[1L]])
  }
  if (!is.character(model_id) || length(model_id) != 1L ||
      is.na(model_id) || !nzchar(model_id) ||
      grepl("[[:cntrl:]]", model_id)) {
    .ml_stop("`model_id` must be NULL or one non-empty identifier")
  }
  enc2utf8(model_id)
}

.ml_foundation_slug <- function(model_id) {
  text <- tolower(gsub("[^A-Za-z0-9._-]+", "-", model_id))
  text <- sub("^-+", "", sub("-+$", "", text))
  if (!nzchar(text)) text <- "model"
  paste0(
    substr(text, 1L, 80L),
    "-",
    substr(digest::digest(model_id, algo = "sha256"), 1L, 12L)
  )
}

.ml_cache_root <- function(cache_dir) {
  if (!is.character(cache_dir) || length(cache_dir) != 1L ||
      is.na(cache_dir) || !nzchar(cache_dir)) {
    .ml_stop("`cache_dir` must be one non-empty directory path")
  }
  if (!dir.exists(cache_dir) && !dir.create(cache_dir, recursive = TRUE)) {
    .ml_stop("could not create the foundation cache directory")
  }
  normalizePath(cache_dir, mustWork = TRUE)
}

.ml_cache_entry <- function(root, adapter, model_id, revision) {
  file.path(
    root,
    "schema-v1",
    adapter,
    .ml_foundation_slug(model_id),
    revision
  )
}

.ml_safe_relative <- function(path) {
  if (!is.character(path) || length(path) != 1L || is.na(path) ||
      !nzchar(path) || startsWith(path, "/") ||
      grepl("^[A-Za-z]:", path) || grepl("\\\\", path) ||
      any(strsplit(path, "/", fixed = TRUE)[[1L]] %in% c("", ".", ".."))) {
    .ml_stop("foundation asset path is absolute, empty, or traverses a directory")
  }
  enc2utf8(path)
}

.ml_foundation_fixture_dir <- function() {
  path <- system.file("extdata", "foundation_tiny", package = "PhysioML")
  if (!nzchar(path)) {
    candidate <- file.path("inst", "extdata", "foundation_tiny")
    if (dir.exists(candidate)) {
      path <- normalizePath(candidate, mustWork = TRUE)
    }
  }
  path
}

.ml_read_foundation_manifest <- function(path) {
  if (!file.exists(path) || dir.exists(path) || nzchar(Sys.readlink(path))) {
    .ml_stop("foundation manifest must be a regular non-symlink file")
  }
  size <- file.info(path)$size
  if (!is.finite(size) || size < 1L || size > 8 * 1024^2) {
    .ml_stop("foundation manifest is empty or exceeds 8 MiB")
  }
  text <- rawToChar(readBin(path, "raw", n = size))
  if (!jsonlite::validate(text)) {
    .ml_stop("foundation manifest is not valid JSON")
  }
  manifest <- jsonlite::fromJSON(text, simplifyVector = FALSE)
  if (!identical(text, .ml_canonical_json(manifest))) {
    .ml_stop("foundation manifest must use canonical JSON")
  }
  .ml_simplify_json_arrays(manifest)
}

.ml_validate_foundation_manifest <- function(
    manifest,
    adapter = NULL,
    model_id = NULL,
    revision = NULL) {
  required <- c(
    "adapter", "adapter_version", "checkpoint_format", "download_consent",
    "files", "license", "model_id", "revision", "schema_version", "sources"
  )
  if (!is.list(manifest) ||
      !identical(sort(names(manifest)), sort(required))) {
    .ml_stop("foundation manifest schema is incomplete or has unknown fields")
  }
  for (name in c(
    "adapter", "adapter_version", "checkpoint_format", "license",
    "model_id", "revision", "schema_version"
  )) {
    value <- manifest[[name]]
    if (!is.character(value) || length(value) != 1L ||
        is.na(value) || !nzchar(value)) {
      .ml_stop(sprintf("foundation manifest `%s` must be one string", name))
    }
  }
  if (!identical(manifest$schema_version, .ml_foundation_schema)) {
    .ml_stop("unsupported foundation manifest schema version")
  }
  if (!identical(manifest$checkpoint_format, "safetensors")) {
    .ml_stop("foundation checkpoints must use the safe safetensors format")
  }
  .ml_foundation_revision(manifest$revision)
  if (!is.logical(manifest$download_consent) ||
      length(manifest$download_consent) != 1L ||
      is.na(manifest$download_consent)) {
    .ml_stop("foundation manifest `download_consent` must be one logical value")
  }
  if (!is.null(adapter) && !identical(manifest$adapter, adapter)) {
    .ml_stop("foundation manifest adapter does not match the cache key")
  }
  if (!is.null(model_id) && !identical(manifest$model_id, model_id)) {
    .ml_stop("foundation manifest model ID does not match the cache key")
  }
  if (!is.null(revision) && !identical(manifest$revision, revision)) {
    .ml_stop("foundation manifest revision does not match the cache key")
  }
  if (!is.character(manifest$sources) || !length(manifest$sources) ||
      anyNA(manifest$sources) || any(!nzchar(manifest$sources))) {
    .ml_stop("foundation manifest sources must be non-empty strings")
  }
  files <- manifest$files
  if (!is.data.frame(files)) {
    files <- tryCatch(
      do.call(rbind, lapply(files, as.data.frame, stringsAsFactors = FALSE)),
      error = function(e) NULL
    )
  }
  if (is.null(files) ||
      !identical(sort(names(files)), c("path", "sha256", "size")) ||
      !nrow(files) || nrow(files) > .ml_foundation_max_files) {
    .ml_stop("foundation manifest file inventory is invalid")
  }
  files$path <- vapply(files$path, .ml_safe_relative, character(1))
  if (anyDuplicated(files$path) ||
      anyDuplicated(tolower(enc2utf8(files$path)))) {
    .ml_stop("foundation manifest has duplicate or case-colliding paths")
  }
  if (!is.numeric(files$size) || anyNA(files$size) ||
      any(!is.finite(files$size)) || any(files$size < 1) ||
      any(files$size != floor(files$size)) ||
      any(files$size > .ml_foundation_max_file_bytes) ||
      sum(files$size) > .ml_foundation_max_total_bytes) {
    .ml_stop("foundation manifest asset sizes are invalid or exceed ceilings")
  }
  if (!is.character(files$sha256) || anyNA(files$sha256) ||
      any(!grepl("^[0-9a-f]{64}$", files$sha256))) {
    .ml_stop("foundation manifest asset hashes are invalid")
  }
  files <- files[order(enc2utf8(files$path), method = "radix"), , drop = FALSE]
  manifest$files <- files
  manifest
}

.ml_foundation_inventory <- function(asset_dir) {
  paths <- list.files(
    asset_dir,
    recursive = TRUE,
    all.files = TRUE,
    no.. = TRUE,
    include.dirs = FALSE
  )
  if (!length(paths)) {
    .ml_stop("foundation asset directory is empty")
  }
  paths <- gsub("\\\\", "/", paths)
  paths <- vapply(paths, .ml_safe_relative, character(1))
  paths <- paths[order(enc2utf8(paths), method = "radix")]
  if (anyDuplicated(tolower(paths))) {
    .ml_stop("foundation assets have case-colliding paths")
  }
  full <- file.path(asset_dir, paths)
  has_symlink <- vapply(paths, function(path) {
    components <- strsplit(path, "/", fixed = TRUE)[[1L]]
    candidates <- file.path(
      asset_dir,
      vapply(seq_along(components), function(index) {
        paste(components[seq_len(index)], collapse = "/")
      }, character(1))
    )
    any(nzchar(Sys.readlink(candidates)))
  }, logical(1))
  if (any(has_symlink)) {
    .ml_stop("foundation assets must not contain symlinks")
  }
  info <- file.info(full)
  if (any(!is.finite(info$size)) || any(info$size < 1) ||
      length(paths) > .ml_foundation_max_files ||
      any(info$size > .ml_foundation_max_file_bytes) ||
      sum(info$size) > .ml_foundation_max_total_bytes) {
    .ml_stop("foundation asset inventory exceeds safety ceilings")
  }
  text_candidates <- grepl("\\.(json|txt)$", tolower(paths))
  for (path in full[text_candidates]) {
    bytes <- readBin(path, "raw", n = min(512L, file.info(path)$size))
    prefix <- tolower(trimws(rawToChar(bytes)))
    if (startsWith(prefix, "<!doctype html") ||
        startsWith(prefix, "<html")) {
      .ml_stop("foundation asset contains an HTML error body")
    }
  }
  data.frame(
    path = paths,
    sha256 = vapply(full, .ml_sha256_file, character(1)),
    size = as.numeric(info$size),
    stringsAsFactors = FALSE
  )
}

.ml_verify_foundation_entry <- function(
    entry,
    adapter = NULL,
    model_id = NULL,
    revision = NULL,
    manifest_sha256 = NULL) {
  manifest_path <- file.path(entry, "manifest.json")
  ready_path <- file.path(entry, "READY")
  asset_dir <- file.path(entry, "assets")
  if (!dir.exists(entry) || !dir.exists(asset_dir) ||
      !file.exists(ready_path) || nzchar(Sys.readlink(ready_path))) {
    .ml_stop("foundation cache entry is incomplete; READY is absent")
  }
  manifest <- .ml_validate_foundation_manifest(
    .ml_read_foundation_manifest(manifest_path),
    adapter,
    model_id,
    revision
  )
  actual_manifest_sha256 <- .ml_sha256_file(manifest_path)
  ready <- readLines(ready_path, warn = FALSE, encoding = "UTF-8")
  if (!identical(ready, actual_manifest_sha256)) {
    .ml_stop("foundation cache READY marker does not match its manifest")
  }
  if (!is.null(manifest_sha256) &&
      !identical(actual_manifest_sha256, manifest_sha256)) {
    .ml_stop("foundation manifest SHA-256 does not match the requested value")
  }
  actual <- .ml_foundation_inventory(asset_dir)
  expected <- manifest$files
  if (!identical(actual$path, expected$path) ||
      !identical(actual$sha256, expected$sha256) ||
      !isTRUE(all.equal(actual$size, expected$size, tolerance = 0))) {
    .ml_stop("foundation asset hash, size, or inventory mismatch")
  }
  list(
    manifest = manifest,
    manifest_sha256 = actual_manifest_sha256,
    asset_dir = asset_dir,
    asset_count = nrow(actual),
    asset_bytes = sum(actual$size)
  )
}

.ml_acquire_foundation_lock <- function(path, attempts = 100L) {
  parent <- dirname(path)
  if (!dir.exists(parent) && !dir.create(parent, recursive = TRUE)) {
    .ml_stop("could not create foundation cache-key directory")
  }
  for (attempt in seq_len(attempts)) {
    if (dir.create(path, showWarnings = FALSE)) {
      return(invisible(TRUE))
    }
    Sys.sleep(0.05)
  }
  .ml_stop("foundation cache entry is locked by another process")
}

.ml_foundation_python <- function(modules = character()) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    .ml_stop("the optional R package `reticulate` is required for foundation models")
  }
  if (!reticulate::py_available(initialize = TRUE)) {
    .ml_stop("Python is unavailable in the caller-managed reticulate environment")
  }
  for (module in modules) {
    if (!reticulate::py_module_available(module)) {
      .ml_stop(sprintf(
        "the Python module `%s` is required in the active environment",
        module
      ))
    }
  }
  path <- system.file("python", package = "PhysioML")
  if (!nzchar(path)) {
    candidate <- file.path("inst", "python")
    if (file.exists(file.path(candidate, "physio_foundation.py"))) {
      path <- normalizePath(candidate, mustWork = TRUE)
    }
  }
  if (!nzchar(path) ||
      !file.exists(file.path(path, "physio_foundation.py"))) {
    .ml_stop("the installed PhysioML foundation bridge is unavailable")
  }
  reticulate::import_from_path(
    "physio_foundation",
    path = path,
    convert = TRUE
  )
}

.ml_prepare_download_manifest <- function(
    asset_dir,
    row,
    model_id,
    revision) {
  inventory <- .ml_foundation_inventory(asset_dir)
  unsafe <- grepl(
    "\\.(bin|pkl|pickle|pt|pth|ckpt|py|pyc|so|dll|dylib|zip|tar|gz)$",
    tolower(inventory$path)
  )
  if (any(unsafe)) {
    .ml_stop(paste0(
      "download contains an unsafe executable, archive, or pickle asset: ",
      paste(inventory$path[unsafe], collapse = ", ")
    ))
  }
  if (!any(endsWith(tolower(inventory$path), ".safetensors"))) {
    .ml_stop("download contains no safetensors checkpoint")
  }
  list(
    adapter = row$adapter[[1L]],
    adapter_version = .ml_foundation_schema,
    checkpoint_format = "safetensors",
    download_consent = TRUE,
    files = inventory,
    license = row$license[[1L]],
    model_id = model_id,
    revision = revision,
    schema_version = .ml_foundation_schema,
    sources = c(
      row$source_url[[1L]],
      sprintf("https://huggingface.co/%s", model_id)
    )
  )
}

.ml_land_foundation_entry <- function(temp, entry, manifest, overwrite = FALSE) {
  manifest_text <- .ml_canonical_json(manifest)
  writeBin(
    charToRaw(manifest_text),
    file.path(temp, "manifest.json")
  )
  hash <- .ml_sha256_file(file.path(temp, "manifest.json"))
  writeLines(hash, file.path(temp, "READY"), useBytes = TRUE)
  .ml_verify_foundation_entry(
    temp,
    manifest$adapter,
    manifest$model_id,
    manifest$revision,
    hash
  )
  backup <- NULL
  if (dir.exists(entry)) {
    if (!overwrite) {
      .ml_stop("foundation cache entry already exists")
    }
    backup <- tempfile(
      paste0(".", basename(entry), "-backup-"),
      tmpdir = dirname(entry)
    )
    if (!file.rename(entry, backup)) {
      .ml_stop("could not prepare foundation cache entry for atomic overwrite")
    }
  }
  landed <- FALSE
  on.exit({
    if (!landed && !is.null(backup) && dir.exists(backup)) {
      file.rename(backup, entry)
    }
  }, add = TRUE)
  if (!file.rename(temp, entry)) {
    .ml_stop("could not atomically land the foundation cache entry")
  }
  landed <- TRUE
  if (!is.null(backup)) {
    unlink(backup, recursive = TRUE, force = TRUE)
  }
  hash
}

#' Cache an immutable physiological foundation model
#'
#' A first network download requires explicit consent, a fixed 40-character
#' revision, a predeclared canonical manifest hash, and license acceptance.
#' Cache hits revalidate the complete inventory and perform no download.
#'
#' @param model Exact adapter name.
#' @param model_id Optional exact upstream model identifier.
#' @param revision Immutable 40-character commit hash.
#' @param manifest_sha256 Expected canonical manifest SHA-256.
#' @param cache_dir Caller-owned cache root.
#' @param allow_download Explicit first-download consent.
#' @param offline Whether all network access is forbidden.
#' @param license_accepted Whether the upstream license was accepted.
#' @param overwrite Whether an existing valid entry may be replaced.
#'
#' @return Invisibly, path-free cache provenance.
#' @export
#' @examples
#' \donttest{
#' # A first download needs explicit consent, the exact pinned revision, and the
#' # expected canonical manifest hash; it also needs a caller-managed Python env.
#' cacheFoundationModel(
#'   model = "moment",
#'   revision = "411e288267f82cce86296dbe4d6c8bc533cc162f",
#'   manifest_sha256 = strrep("0", 64), # replace with the real manifest hash
#'   cache_dir = tempfile("physioml-cache"),
#'   allow_download = TRUE,
#'   license_accepted = TRUE
#' )
#' }
cacheFoundationModel <- function(
    model = c("moment", "chronos2", "labram", "bendr"),
    model_id = NULL,
    revision,
    manifest_sha256,
    cache_dir = tools::R_user_dir("PhysioML", "cache"),
    allow_download = FALSE,
    offline = FALSE,
    license_accepted = FALSE,
    overwrite = FALSE) {
  if (missing(model)) model <- "moment"
  row <- .ml_foundation_row(model)
  model <- row$adapter[[1L]]
  model_id <- .ml_foundation_model_id(model_id, row)
  revision <- .ml_foundation_revision(revision)
  manifest_sha256 <- .ml_manifest_sha256(manifest_sha256)
  allow_download <- .ml_flag(allow_download, "allow_download")
  offline <- .ml_flag(offline, "offline")
  license_accepted <- .ml_flag(license_accepted, "license_accepted")
  overwrite <- .ml_flag(overwrite, "overwrite")
  if (offline && allow_download) {
    .ml_stop("`offline = TRUE` is incompatible with `allow_download = TRUE`")
  }
  root <- .ml_cache_root(cache_dir)
  entry <- .ml_cache_entry(root, model, model_id, revision)
  if (dir.exists(entry) && !overwrite) {
    verified <- .ml_verify_foundation_entry(
      entry,
      model,
      model_id,
      revision,
      manifest_sha256
    )
    return(invisible(verified[c(
      "manifest", "manifest_sha256", "asset_count", "asset_bytes"
    )]))
  }
  if (offline && !identical(model_id, "physioml/foundation-tiny")) {
    .ml_stop("foundation cache miss in offline mode")
  }
  if (!allow_download &&
      !identical(model_id, "physioml/foundation-tiny")) {
    .ml_stop("foundation cache miss; set `allow_download = TRUE` explicitly")
  }
  if (allow_download && !license_accepted) {
    .ml_stop("upstream license acceptance is required before download")
  }

  lock <- paste0(entry, ".lock")
  .ml_acquire_foundation_lock(lock)
  on.exit(unlink(lock, recursive = TRUE, force = TRUE), add = TRUE)
  if (dir.exists(entry) && !overwrite) {
    verified <- .ml_verify_foundation_entry(
      entry,
      model,
      model_id,
      revision,
      manifest_sha256
    )
    return(invisible(verified[c(
      "manifest", "manifest_sha256", "asset_count", "asset_bytes"
    )]))
  }
  temp <- tempfile(
    paste0(".", basename(entry), "-"),
    tmpdir = dirname(entry)
  )
  if (!dir.create(file.path(temp, "assets"), recursive = TRUE)) {
    .ml_stop("could not create a temporary foundation cache entry")
  }
  landed <- FALSE
  on.exit(if (!landed) {
    unlink(temp, recursive = TRUE, force = TRUE)
  }, add = TRUE)

  if (identical(model_id, "physioml/foundation-tiny")) {
    fixture <- .ml_foundation_fixture_dir()
    if (!nzchar(fixture) || !dir.exists(fixture)) {
      .ml_stop("the package-owned tiny foundation fixture is unavailable")
    }
    fixture_manifest <- .ml_validate_foundation_manifest(
      .ml_read_foundation_manifest(file.path(fixture, "manifest.json")),
      model,
      model_id,
      revision
    )
    files <- fixture_manifest$files$path
    for (path in files) {
      destination <- file.path(temp, "assets", path)
      dir.create(dirname(destination), recursive = TRUE, showWarnings = FALSE)
      if (!file.copy(file.path(fixture, path), destination, copy.mode = FALSE)) {
        .ml_stop("could not copy the package-owned tiny foundation fixture")
      }
    }
    manifest <- fixture_manifest
  } else {
    bridge <- .ml_foundation_python("huggingface_hub")
    bridge$download_snapshot(
      model_id,
      revision,
      file.path(temp, "assets")
    )
    manifest <- .ml_prepare_download_manifest(
      file.path(temp, "assets"),
      row,
      model_id,
      revision
    )
  }
  manifest <- .ml_validate_foundation_manifest(
    manifest,
    model,
    model_id,
    revision
  )
  actual_text <- .ml_canonical_json(manifest)
  actual_hash <- .ml_sha256_raw(charToRaw(actual_text))
  if (!identical(actual_hash, manifest_sha256)) {
    .ml_stop("downloaded foundation manifest does not match `manifest_sha256`")
  }
  landed_hash <- .ml_land_foundation_entry(
    temp,
    entry,
    manifest,
    overwrite = overwrite
  )
  landed <- TRUE
  verified <- .ml_verify_foundation_entry(
    entry,
    model,
    model_id,
    revision,
    landed_hash
  )
  invisible(verified[c(
    "manifest", "manifest_sha256", "asset_count", "asset_bytes"
  )])
}

#' Inspect immutable foundation-model cache entries
#'
#' @param model Optional exact adapter filter.
#' @param cache_dir Caller-owned cache root.
#' @param verify Whether to revalidate every manifest and asset.
#'
#' @return A path-free data frame, one row per READY entry.
#' @export
#' @examples
#' # An empty cache root returns a zero-row inventory, offline.
#' foundationCacheInfo(cache_dir = tempfile("physioml-cache"))
foundationCacheInfo <- function(
    model = NULL,
    cache_dir = tools::R_user_dir("PhysioML", "cache"),
    verify = TRUE) {
  if (!is.null(model)) {
    model <- .ml_foundation_row(model)$adapter[[1L]]
  }
  verify <- .ml_flag(verify, "verify")
  root <- .ml_cache_root(cache_dir)
  schema <- file.path(root, "schema-v1")
  if (!dir.exists(schema)) {
    return(data.frame(
      adapter = character(),
      model_id = character(),
      revision = character(),
      manifest_sha256 = character(),
      asset_count = integer(),
      asset_bytes = numeric(),
      verified = logical(),
      stringsAsFactors = FALSE
    ))
  }
  ready <- list.files(
    schema,
    pattern = "^READY$",
    recursive = TRUE,
    full.names = TRUE
  )
  rows <- list()
  for (path in ready) {
    entry <- dirname(path)
    manifest <- .ml_validate_foundation_manifest(
      .ml_read_foundation_manifest(file.path(entry, "manifest.json"))
    )
    if (!is.null(model) && manifest$adapter != model) next
    result <- if (verify) {
      .ml_verify_foundation_entry(entry)
    } else {
      list(
        manifest = manifest,
        manifest_sha256 = .ml_sha256_file(file.path(entry, "manifest.json")),
        asset_count = nrow(manifest$files),
        asset_bytes = sum(manifest$files$size)
      )
    }
    rows[[length(rows) + 1L]] <- data.frame(
      adapter = manifest$adapter,
      model_id = manifest$model_id,
      revision = manifest$revision,
      manifest_sha256 = result$manifest_sha256,
      asset_count = as.integer(result$asset_count),
      asset_bytes = as.numeric(result$asset_bytes),
      verified = verify,
      stringsAsFactors = FALSE
    )
  }
  if (!length(rows)) {
    return(data.frame(
      adapter = character(),
      model_id = character(),
      revision = character(),
      manifest_sha256 = character(),
      asset_count = integer(),
      asset_bytes = numeric(),
      verified = logical(),
      stringsAsFactors = FALSE
    ))
  }
  output <- do.call(rbind, rows)
  output[order(
    output$adapter,
    output$model_id,
    output$revision,
    method = "radix"
  ), , drop = FALSE]
}
