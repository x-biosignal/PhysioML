test_that("foundation registry is complete and conservative", {
  registry <- .ml_foundation_registry()
  expect_identical(
    registry$adapter,
    c("moment", "chronos2", "labram", "bendr")
  )
  expect_true(all(c(
    "adapter", "model_family", "default_model_id", "default_revision",
    "python_module", "loader", "input_axes", "sampling_rate", "units",
    "channel_contract", "embedding_reduction", "cpu_verified", "license",
    "source_url", "paper_url"
  ) %in% names(registry)))
  expect_true(all(grepl("^[0-9a-f]{40}$", registry$default_revision)))
  expect_false(any(registry$cpu_verified))
  expect_error(.ml_foundation_row("mom"), "exactly")
})

test_that("package-owned tiny model caches and verifies offline", {
  cache <- tempfile("foundation-cache-")
  dir.create(cache)
  result <- cacheFoundationModel(
    "moment",
    model_id = "physioml/foundation-tiny",
    revision = tiny_revision,
    manifest_sha256 = tiny_manifest,
    cache_dir = cache,
    offline = TRUE
  )
  expect_identical(result$manifest_sha256, tiny_manifest)
  expect_equal(result$asset_count, 5L)
  info <- foundationCacheInfo(cache_dir = cache)
  expect_equal(nrow(info), 1L)
  expect_true(info$verified)
  expect_false(any(grepl(normalizePath(cache), unlist(result), fixed = TRUE)))

  testthat::local_mocked_bindings(
    .ml_foundation_python = function(...) {
      stop("cache hit attempted a downloader", call. = FALSE)
    },
    .package = "PhysioML"
  )
  hit <- cacheFoundationModel(
    "moment",
    model_id = "physioml/foundation-tiny",
    revision = tiny_revision,
    manifest_sha256 = tiny_manifest,
    cache_dir = cache,
    offline = TRUE
  )
  expect_identical(hit$manifest_sha256, tiny_manifest)
})

test_that("cache separates miss, offline, license, and hash errors", {
  cache <- tempfile("foundation-cache-")
  dir.create(cache)
  revision <- .ml_foundation_registry()$default_revision[[1L]]
  expect_error(
    cacheFoundationModel(
      "moment",
      revision = revision,
      manifest_sha256 = tiny_manifest,
      cache_dir = cache
    ),
    "cache miss"
  )
  expect_error(
    cacheFoundationModel(
      "moment",
      revision = revision,
      manifest_sha256 = tiny_manifest,
      cache_dir = cache,
      offline = TRUE
    ),
    "offline"
  )
  expect_error(
    cacheFoundationModel(
      "moment",
      revision = revision,
      manifest_sha256 = tiny_manifest,
      cache_dir = cache,
      allow_download = TRUE
    ),
    "license acceptance"
  )
  expect_error(
    cacheFoundationModel(
      "moment",
      model_id = "physioml/foundation-tiny",
      revision = tiny_revision,
      manifest_sha256 = paste0("0", substring(tiny_manifest, 2L)),
      cache_dir = cache,
      offline = TRUE
    ),
    "manifest"
  )
  expect_error(
    cacheFoundationModel(
      "moment",
      revision = "main",
      manifest_sha256 = tiny_manifest,
      cache_dir = cache
    ),
    "immutable"
  )
  expect_error(
    cacheFoundationModel(
      "moment",
      revision = revision,
      manifest_sha256 = "bad",
      cache_dir = cache
    ),
    "SHA-256"
  )
})

test_that("cache detects asset and READY corruption", {
  cache <- tempfile("foundation-cache-")
  dir.create(cache)
  cacheFoundationModel(
    "moment",
    model_id = "physioml/foundation-tiny",
    revision = tiny_revision,
    manifest_sha256 = tiny_manifest,
    cache_dir = cache,
    offline = TRUE
  )
  root <- .ml_cache_root(cache)
  entry <- .ml_cache_entry(
    root,
    "moment",
    "physioml/foundation-tiny",
    tiny_revision
  )
  writeBin(as.raw(c(1, 2, 3)), file.path(entry, "assets", "config.json"))
  expect_error(foundationCacheInfo(cache_dir = cache), "mismatch")

  unlink(entry, recursive = TRUE)
  cacheFoundationModel(
    "moment",
    model_id = "physioml/foundation-tiny",
    revision = tiny_revision,
    manifest_sha256 = tiny_manifest,
    cache_dir = cache,
    offline = TRUE
  )
  writeLines(paste0("0", substring(tiny_manifest, 2L)), file.path(entry, "READY"))
  expect_error(foundationCacheInfo(cache_dir = cache), "READY")
})

test_that("manifest validation rejects traversal and case collisions", {
  fixture <- .ml_read_foundation_manifest(
    file.path(.ml_foundation_fixture_dir(), "manifest.json")
  )
  traversal <- fixture
  traversal$files[[1L]]$path <- "../escape"
  expect_error(
    .ml_validate_foundation_manifest(traversal),
    "traverses"
  )
  collision <- fixture
  collision$files[[2L]]$path <- toupper(collision$files[[1L]]$path)
  expect_error(
    .ml_validate_foundation_manifest(collision),
    "case-colliding"
  )
  pickle <- fixture
  pickle$checkpoint_format <- "pickle"
  expect_error(
    .ml_validate_foundation_manifest(pickle),
    "safetensors"
  )
  unsafe <- tempfile()
  dir.create(unsafe)
  writeLines("value", file.path(unsafe, "asset"))
  alias <- file.path(unsafe, "alias")
  created <- suppressWarnings(file.symlink(file.path(unsafe, "asset"), alias))
  # Some build filesystems (e.g. the Windows r-universe binary builder) cannot
  # create symlinks; file.symlink() then fails silently and no unsafe condition
  # exists to reject. Skip there rather than assert the check on a directory
  # that never held a symlink.
  skip_if_not(
    isTRUE(created) && nzchar(Sys.readlink(alias)),
    "platform cannot create symlinks"
  )
  expect_error(.ml_foundation_inventory(unsafe), "symlinks")
})

test_that("foundation cache metadata never exposes host paths", {
  cache <- tempfile("secret-user-cache-")
  dir.create(cache)
  cacheFoundationModel(
    "moment",
    model_id = "physioml/foundation-tiny",
    revision = tiny_revision,
    manifest_sha256 = tiny_manifest,
    cache_dir = cache,
    offline = TRUE
  )
  info <- foundationCacheInfo(cache_dir = cache)
  expect_false(any(grepl(cache, unlist(info), fixed = TRUE)))
  expect_false(any(grepl(Sys.info()[["nodename"]], unlist(info), fixed = TRUE)))
})

test_that("explicit overwrite retains a complete verified cache entry", {
  cache <- tempfile("foundation-cache-")
  dir.create(cache)
  arguments <- list(
    model = "moment",
    model_id = "physioml/foundation-tiny",
    revision = tiny_revision,
    manifest_sha256 = tiny_manifest,
    cache_dir = cache,
    offline = TRUE
  )
  do.call(cacheFoundationModel, arguments)
  do.call(cacheFoundationModel, c(arguments, list(overwrite = TRUE)))
  info <- foundationCacheInfo(cache_dir = cache)
  expect_equal(nrow(info), 1L)
  expect_identical(info$manifest_sha256, tiny_manifest)
  expect_true(info$verified)
})
