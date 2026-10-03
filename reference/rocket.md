# Fit or apply the ROCKET transform

Uses `aeon.transformations.collection.convolution_based.Rocket` through
an optional, caller-configured
[reticulate](https://rstudio.github.io/reticulate/reference/reticulate.html)
backend. With `model = NULL`, fits only on the selected cases and
returns a reusable persistent model. With a model, validates the
complete input contract and transforms without refitting.

## Usage

``` r
rocket(
  x,
  model = NULL,
  assay_name = NULL,
  channels = NULL,
  cases = NULL,
  n_kernels = 10000L,
  seed = 1L,
  normalize = TRUE,
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

- normalize:

  One non-missing logical passed to aeon's `normalise`.

- backend:

  Exactly `"aeon"`.

## Value

A `physio_rocket_transform` containing a finite case-by-feature matrix,
exact case identities, a persistent fitted model, settings, and typed
diagnostics.

## Details

Input assays are time x channel or time x channel x case and must
contain finite, equal-length series of at least nine samples. The fitted
model binds exact channel labels/order and time length. Requested kernel
count and realized feature width are recorded separately because backend
transforms may adjust the width.

Models persist the backend with Python pickle in a hashed raw vector. A
matching SHA-256 detects corruption but does not make deserialization
safe: only reuse model objects from trusted sources. PhysioML validates
class, metadata, hash, input identity, and backend version before
unpickling.

## References

Dempster A, Petitjean F, Webb GI (2020). ROCKET: Exceptionally fast and
accurate time series classification using random convolutional kernels.
*Data Mining and Knowledge Discovery*, 34, 1454-1495.
[doi:10.1007/s10618-020-00701-z](https://doi.org/10.1007/s10618-020-00701-z)

## Examples

``` r
arr <- array(stats::rnorm(60 * 2 * 4), dim = c(60, 2, 4))
pe <- PhysioExperiment::PhysioExperiment(
  assays = list(raw = arr),
  colData = S4Vectors::DataFrame(label = c("C3", "C4")),
  samplingRate = 100
)
# \donttest{
# The transform needs a caller-managed Python environment with NumPy and aeon.
fit <- rocket(pe, n_kernels = 50)
#> Error: the active reticulate Python lacks required module(s): aeon
dim(fit$features)
#> Error: object 'fit' not found
# }
```
