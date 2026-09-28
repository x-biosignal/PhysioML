# Validate a split intended to generalize to unseen subjects

Checks explicit identifiers only; this does not audit upstream
preprocessing fit scopes, verify biological identity, or prevent leakage
in external model software unless called before fitting. No backend is
needed.

## Usage

``` r
validateSubjectSplit(train, valid)
```

## Arguments

- train, valid:

  Case identity data frames as in
  [`withCaseData()`](https://x-biosignal.github.io/PhysioML/reference/withCaseData.md).

## Value

Invisibly `TRUE`, or an error for shared cases or subjects, invalid
labels, or inconsistent participant aliases.
