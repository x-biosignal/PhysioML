# Construct a reference torch model

Builds compact EEGNet, CNN1D, causal TCN, or
unidirectional/bidirectional LSTM modules. Classification models return
logits; regression models return raw values. Input is always batch by
channel by time.

## Usage

``` r
peModel(
  architecture = c("eegnet", "cnn1d", "tcn", "lstm"),
  n_channels,
  n_time,
  n_outputs,
  task = c("classification", "regression"),
  sampling_rate = NULL,
  dropout = 0.5,
  hidden = 32L,
  seed = 1L,
  ...
)
```

## Arguments

- architecture:

  Exact architecture name.

- n_channels, n_time, n_outputs:

  Positive whole dimensions.

- task:

  Exact task name.

- sampling_rate:

  Optional positive sampling rate.

- dropout:

  Dropout probability in `[0, 1)`.

- hidden:

  Positive hidden width.

- seed:

  CPU initialization seed.

- ...:

  Architecture-specific named settings.

## Value

A `physio_torch_module`.
