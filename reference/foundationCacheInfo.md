# Inspect immutable foundation-model cache entries

Inspect immutable foundation-model cache entries

## Usage

``` r
foundationCacheInfo(
  model = NULL,
  cache_dir = tools::R_user_dir("PhysioML", "cache"),
  verify = TRUE
)
```

## Arguments

- model:

  Optional exact adapter filter.

- cache_dir:

  Caller-owned cache root.

- verify:

  Whether to revalidate every manifest and asset.

## Value

A path-free data frame, one row per READY entry.

## Examples

``` r
# An empty cache root returns a zero-row inventory, offline.
foundationCacheInfo(cache_dir = tempfile("physioml-cache"))
#> [1] adapter         model_id        revision        manifest_sha256
#> [5] asset_count     asset_bytes     verified       
#> <0 rows> (or 0-length row.names)
```
