# Attach explicit subject identity to signal cases or feature rows

Establishes a case-to-subject mapping without inferring subjects from
case names. Row order is reconciled by exact case ID. A cycle is a case,
not an independent subject. Use cohort-derived subject/session keys when
available.

## Usage

``` r
withCaseData(x, case_data)
```

## Arguments

- x:

  A `PhysioExperiment` or finite numeric case-by-feature matrix with
  unique nonempty row names used as case IDs.

- case_data:

  Data frame with unique `case_id` and repeated `subject_id`. Optional
  character identity columns are `participant_id`, `session_id`,
  `trial_id`, `side`, `cycle_id`. All supplied columns must be complete.
  An optional `input_case_index` from a PhysioML transform is accepted
  as unique positive whole numbers; signal selection computes its own
  indices. The table must cover all input cases exactly, before any case
  selection.

## Value

The input with validated identity metadata. Experiments store it in
`metadata(x)$ml_case_data`; matrices store it in attribute `case_data`.
PhysioML input transforms and datasets preserve this information.
Ordinary external matrix manipulation is not guaranteed to preserve
attributes; reattach the corresponding table after subsetting feature
rows.

## See also

[`validateSubjectSplit()`](https://x-biosignal.github.io/PhysioML/reference/validateSubjectSplit.md),
[`peDataset()`](https://x-biosignal.github.io/PhysioML/reference/peDataset.md),
[`peReducedFeatures()`](https://x-biosignal.github.io/PhysioML/reference/peReducedFeatures.md)
