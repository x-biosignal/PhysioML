.ml_read_onnx_payload <- function(path) {
  if (!file.exists(path) || dir.exists(path)) {
    .ml_stop("ONNX graph must be a readable regular file")
  }
  size <- file.info(path)$size
  if (!is.finite(size) || size < 1 || size > .ml_onnx_max_bytes) {
    .ml_stop("ONNX graph is empty or exceeds the 512 MiB size ceiling")
  }
  readBin(path, what = "raw", n = size)
}

.ml_write_raw_temp <- function(payload) {
  path <- tempfile("physioml-", fileext = ".onnx")
  stream <- file(path, open = "wb")
  on.exit(close(stream), add = TRUE)
  writeBin(payload, stream)
  path
}

#' Import a governed PhysioML ONNX graph
#'
#' Validates a graph against its canonical, hash-bound PhysioML model card and
#' returns raw persistent bytes rather than a live Python runtime session.
#'
#' @param path Path to an ONNX graph.
#' @param metadata_path Path to its canonical JSON model card.
#' @param providers Execution provider; currently CPU only.
#' @param verify Whether to run the ONNX checker and graph/card introspection.
#'
#' @return A persistent `physio_onnx_model`.
#' @export
#' @examples
#' \donttest{
#' # Reads a governed graph plus its canonical JSON model card (ONNX Runtime).
#' path <- tempfile(fileext = ".onnx")
#' # `path` and paste0(path, ".json") are produced by exportONNX().
#' model <- importONNX(path)
#' }
importONNX <- function(
    path,
    metadata_path = paste0(path, ".json"),
    providers = "CPUExecutionProvider",
    verify = TRUE) {
  path <- .ml_onnx_output_path(path, "path", ".onnx")
  metadata_path <- .ml_onnx_output_path(metadata_path, "metadata_path", ".json")
  providers <- .ml_exact_enum(
    providers,
    "CPUExecutionProvider",
    "providers"
  )
  verify <- .ml_flag(verify, "verify")
  card <- .ml_read_card(metadata_path)
  payload <- .ml_read_onnx_payload(path)
  hash <- .ml_sha256_raw(payload)
  if (!identical(hash, card$onnx_sha256) ||
      length(payload) != as.numeric(card$onnx_size_bytes)) {
    .ml_stop("ONNX graph bytes do not match the model card hash and size")
  }
  versions <- NULL
  if (verify) {
    bridge <- .ml_onnx_python(c("onnx", "onnxruntime"))
    python_versions <- bridge$versions()
    if (!providers %in% unlist(python_versions$providers)) {
      .ml_stop("ONNX Runtime CPUExecutionProvider is unavailable")
    }
    summary <- bridge$inspect_model(path, TRUE)
    .ml_validate_graph_card(summary, card)
    versions <- python_versions
  }
  output <- list(
    onnx_payload = payload,
    onnx_sha256 = hash,
    metadata = card,
    providers = providers,
    import_versions = versions,
    citations = card$citations
  )
  class(output) <- "physio_onnx_model"
  output
}
