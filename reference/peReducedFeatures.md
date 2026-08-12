# Create a case-by-feature design matrix

Adapts
[`catch22()`](https://x-biosignal.github.io/PhysioML/reference/catch22.md),
[`rocket()`](https://x-biosignal.github.io/PhysioML/reference/rocket.md),
or
[`minirocket()`](https://x-biosignal.github.io/PhysioML/reference/minirocket.md)
output without adding an outcome or changing the feature transform. For
convolution methods, pass a model fitted only on training cases when
transforming validation/test cases.

## Usage

``` r
peReducedFeatures(
  x,
  method = c("catch22", "rocket", "minirocket"),
  model = NULL,
  assay_name = NULL,
  channels = NULL,
  cases = NULL,
  ...
)
```

## Arguments

- x:

  A `PhysioExperiment`.

- method:

  Exactly one of `"catch22"`, `"rocket"`, or `"minirocket"`.

- model:

  `NULL` or a fitted convolution-transform model. Must be `NULL` for
  catch22.

- assay_name, channels, cases:

  Input selection; see
  [`catch22()`](https://x-biosignal.github.io/PhysioML/reference/catch22.md).

- ...:

  Method-specific arguments.

## Value

A numeric matrix with class `physio_feature_matrix`. Attributes preserve
exact case identity, feature provenance, typed diagnostics, and the
fitted model when applicable.
