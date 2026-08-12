# Freeze named parts of a fitted torch model

The fitted model is cloned before `requires_grad` is changed. Exact
matching is the default; prefix matching is component-aware.

## Usage

``` r
freezeLayers(
  model,
  trainable = NULL,
  frozen = NULL,
  match = c("exact", "prefix"),
  strict = TRUE
)
```

## Arguments

- model:

  A fitted `physio_dl_model`.

- trainable, frozen:

  Optional unique exact parameter names or prefixes.

- match:

  Exact selection mode.

- strict:

  Whether unmatched rules are errors.

## Value

A plain `physio_transfer_model`.
