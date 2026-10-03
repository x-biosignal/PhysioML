# Obtain governed foundation-model embeddings

The exact model ID, immutable revision, complete manifest hash, ordered
channels, sampling rate, units, and window contract are recorded. Python
and all provider packages remain caller-managed. A first upstream
download is explicit; package code never installs software or selects an
environment.

## Usage

``` r
foundationEmbed(
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
  offline = FALSE
)
```

## Arguments

- x:

  A `PhysioExperiment`.

- model:

  Exact adapter name.

- model_id:

  Optional exact upstream model identifier.

- revision:

  Immutable 40-character commit hash.

- manifest_sha256:

  Expected canonical cache-manifest SHA-256.

- assay_name, cases, channels:

  Exact input selections.

- window_samples, stride_samples:

  Complete-window settings.

- reduction:

  Exact documented embedding reduction.

- batch_size:

  Positive whole-number batch size.

- device:

  Currently exactly `"cpu"`.

- cache_dir:

  Caller-owned cache root.

- allow_download:

  Explicit first-download consent.

- offline:

  Whether all network access is forbidden.

## Value

A path-free, serializable `physio_foundation_embedding`.

## Details

Stable dimensions are promised only for one exact model, revision, and
preprocessing contract. The output is not a clinical prediction and does
not establish fairness, causal invariance, or domain invariance.

## Examples

``` r
arr <- array(stats::rnorm(400 * 2 * 2), dim = c(400, 2, 2))
pe <- PhysioExperiment::PhysioExperiment(
  assays = list(raw = arr),
  colData = S4Vectors::DataFrame(label = c("C3", "C4"), unit = "uV"),
  samplingRate = 200
)
# \donttest{
# Needs a pinned revision, the expected manifest hash, a caller-managed
# Python env, and (on first use) explicit download consent.
emb <- foundationEmbed(
  pe,
  model = "moment",
  revision = "411e288267f82cce86296dbe4d6c8bc533cc162f",
  manifest_sha256 = strrep("0", 64), # replace with the real manifest hash
  allow_download = TRUE
)
#> Error: MOMENT official adapter is not CPU-verified at revision 411e288267f82cce86296dbe4d6c8bc533cc162f; run the recorded provider preflight before enabling it
# }
```
