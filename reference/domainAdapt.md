# Align source embeddings to an unlabeled target domain with CORAL

Fit CORAL only with training-source observations and an explicitly
designated unlabeled target adaptation set. Held-out target validation
and test observations must not be supplied. CORAL matches first- and
second-order statistics; it does not establish causal,
class-conditional, fairness, or clinical invariance.

## Usage

``` r
domainAdapt(
  source,
  target,
  method = c("coral"),
  regularization = 1e-06,
  return_model = TRUE
)
```

## Arguments

- source, target:

  Finite observation-by-feature matrices, or foundation embedding
  results, with identical exact feature names and order.

- method:

  Currently exactly `"coral"`.

- regularization:

  Positive covariance regularization.

- return_model:

  Whether to return the fitted transform and diagnostics.

## Value

A matrix, or a plain list containing aligned source data and model.

## Examples

``` r
set.seed(1)
feat <- c("f1", "f2", "f3")
source <- matrix(stats::rnorm(30), ncol = 3, dimnames = list(NULL, feat))
target <- matrix(stats::rnorm(30, mean = 1), ncol = 3,
                 dimnames = list(NULL, feat))
fit <- domainAdapt(source, target)
fit$diagnostics$source_mean_residual
#> [1] 0
```
