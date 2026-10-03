# Cache an immutable physiological foundation model

A first network download requires explicit consent, a fixed 40-character
revision, a predeclared canonical manifest hash, and license acceptance.
Cache hits revalidate the complete inventory and perform no download.

## Usage

``` r
cacheFoundationModel(
  model = c("moment", "chronos2", "labram", "bendr"),
  model_id = NULL,
  revision,
  manifest_sha256,
  cache_dir = tools::R_user_dir("PhysioML", "cache"),
  allow_download = FALSE,
  offline = FALSE,
  license_accepted = FALSE,
  overwrite = FALSE
)
```

## Arguments

- model:

  Exact adapter name.

- model_id:

  Optional exact upstream model identifier.

- revision:

  Immutable 40-character commit hash.

- manifest_sha256:

  Expected canonical manifest SHA-256.

- cache_dir:

  Caller-owned cache root.

- allow_download:

  Explicit first-download consent.

- offline:

  Whether all network access is forbidden.

- license_accepted:

  Whether the upstream license was accepted.

- overwrite:

  Whether an existing valid entry may be replaced.

## Value

Invisibly, path-free cache provenance.

## Examples

``` r
# \donttest{
# A first download needs explicit consent, the exact pinned revision, and the
# expected canonical manifest hash; it also needs a caller-managed Python env.
cacheFoundationModel(
  model = "moment",
  revision = "411e288267f82cce86296dbe4d6c8bc533cc162f",
  manifest_sha256 = strrep("0", 64), # replace with the real manifest hash
  cache_dir = tempfile("physioml-cache"),
  allow_download = TRUE,
  license_accepted = TRUE
)
#> Downloading uv...
#> Done!
#> Error: the Python module `huggingface_hub` is required in the active environment
# }
```
