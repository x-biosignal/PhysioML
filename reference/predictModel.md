# Predict with a trained deep-learning model

Inference uses evaluation mode and disabled gradients. The Dataset must
exactly match the training channel, time, task, class, dtype, and frozen
normalization contracts.

## Usage

``` r
predictModel(
  model,
  newdata,
  batch_size = 64L,
  type = c("class", "probability", "response", "logit"),
  num_workers = 0L
)
```

## Arguments

- model:

  A fitted `physio_dl_model`.

- newdata:

  A compatible target-free or targeted Dataset.

- batch_size:

  Positive whole-number inference batch size.

- type:

  Exact output type.

- num_workers:

  Currently exactly zero.

## Value

A numeric matrix or identity-preserving data frame.
