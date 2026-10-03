test_that("package metadata and exports follow the ecosystem contract", {
  description_path <- test_path("..", "..", "DESCRIPTION")
  if (!file.exists(description_path)) {
    description_path <- system.file("DESCRIPTION", package = "PhysioML")
  }
  description <- read.dcf(description_path)
  expect_identical(unname(description[1L, "Package"]), "PhysioML")
  expect_identical(unname(description[1L, "Version"]), "0.4.1")
  expect_match(description[1L, "Authors@R"], "Yusuke")
  expect_true(all(c(
    "catch22", "rocket", "minirocket", "peReducedFeatures",
    "peDataset", "peModel", "trainModel", "predictModel",
    "exportONNX", "importONNX", "onnxPredict",
    "freezeLayers", "fineTune", "domainAdapt",
    "cacheFoundationModel", "foundationCacheInfo", "foundationEmbed"
  ) %in% getNamespaceExports("PhysioML")))
})

test_that("PhysioML is registered in the canonical public inventory", {
  root <- normalizePath(test_path("..", "..", ".."), mustWork = FALSE)
  skip_if_not(file.exists(file.path(root, "install_ecosystem.R")))
  install_text <- readLines(file.path(root, "install_ecosystem.R"))
  check_text <- readLines(file.path(root, "publishing", "check-ecosystem.yml"))
  sync_text <- readLines(file.path(root, "publishing", "sync_public.sh"))
  packages_text <- readLines(file.path(root, "publishing", "packages.json"))

  expect_true(any(grepl('"PhysioML"', install_text, fixed = TRUE)))
  expect_true(any(grepl('"package": "PhysioML"', packages_text, fixed = TRUE)))
  expect_true(any(grepl("build_ecosystem.R", check_text, fixed = TRUE)))
  expect_true(any(grepl("packages.json", sync_text, fixed = TRUE)))
  expect_true(any(grepl("ALL_PACKAGES", sync_text, fixed = TRUE)))
})

test_that("absent optional backends fail only when requested", {
  pe <- ml_fixture(matrix(sin(seq_len(20L)), ncol = 1L), labels = "signal")
  if (!requireNamespace("Rcatch22", quietly = TRUE)) {
    expect_error(catch22(pe), "optional R package `Rcatch22`")
  } else {
    succeed()
  }
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    expect_error(rocket(pe, n_kernels = 1L), "optional R package `reticulate`")
  } else {
    succeed()
  }
})

test_that("torch backend guards distinguish package and runtime absence", {
  local_mocked_bindings(
    .ml_with_envvar = function(name, value, code) FALSE,
    .package = "PhysioML"
  )
  expect_error(
    PhysioML:::.ml_require_torch(),
    "optional R package `torch`"
  )

  local_mocked_bindings(
    .ml_with_envvar = function(name, value, code) TRUE,
    .package = "PhysioML"
  )
  if (requireNamespace("torch", quietly = TRUE)) {
    local_mocked_bindings(
      torch_is_installed = function() FALSE,
      .package = "torch"
    )
    expect_error(
      PhysioML:::.ml_require_torch(),
      "LibTorch is unavailable"
    )
  } else {
    expect_false(requireNamespace("torch", quietly = TRUE))
  }
})

test_that("runtime source never calls a torch installer", {
  package_root <- normalizePath(test_path("..", ".."), mustWork = TRUE)
  source_files <- list.files(
    file.path(package_root, "R"),
    pattern = "[.]R$",
    full.names = TRUE
  )
  symbols <- unique(unlist(lapply(source_files, function(path) {
    all.names(parse(path), functions = TRUE, unique = TRUE)
  })))
  expect_false("install_torch" %in% symbols)

  previous <- Sys.getenv("TORCH_INSTALL", unset = NA_character_)
  observed <- PhysioML:::.ml_with_envvar(
    "TORCH_INSTALL",
    "0",
    Sys.getenv("TORCH_INSTALL")
  )
  expect_identical(observed, "0")
  expect_identical(Sys.getenv("TORCH_INSTALL", unset = NA_character_), previous)
})
