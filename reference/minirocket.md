# Fit or apply the MiniRocket transform

Uses `aeon.transformations.collection.convolution_based.MiniRocket`. A
fitted model records the realized feature width because aeon rounds the
requested kernel count according to the MiniRocket kernel family.

## Usage

``` r
minirocket(
  x,
  model = NULL,
  assay_name = NULL,
  channels = NULL,
  cases = NULL,
  n_kernels = 10000L,
  seed = 1L,
  deterministic = TRUE,
  backend = "aeon"
)
```

## Arguments

- x:

  A `PhysioExperiment`.

- model:

  `NULL` or a fitted `physio_rocket_model`.

- assay_name, channels, cases:

  Input selection; see
  [`catch22()`](https://x-biosignal.github.io/PhysioML/reference/catch22.md).

- n_kernels:

  One finite whole number of kernels.

- seed:

  One whole-number backend seed in `[0, 2^31 - 1]`.

- deterministic:

  One non-missing logical. `TRUE` requires the fixed-seed, single-worker
  aeon path and records that contract. Determinism is scoped to the
  recorded backend versions and platform.

- backend:

  Exactly `"aeon"`.

## Value

A `physio_rocket_transform`; see
[`rocket()`](https://x-biosignal.github.io/PhysioML/reference/rocket.md).

## References

Dempster A, Schmidt DF, Webb GI (2021). MiniRocket: A Very Fast (Almost)
Deterministic Transform for Time Series Classification.
[doi:10.1145/3447548.3467231](https://doi.org/10.1145/3447548.3467231)
