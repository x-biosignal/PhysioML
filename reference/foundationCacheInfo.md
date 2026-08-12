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
