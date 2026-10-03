# Leakage-aware features and subject-safe splits

PhysioML turns
[`PhysioExperiment`](https://x-biosignal.github.io/PhysioExperiment/)
objects into model-ready feature matrices while keeping the two things
that are easy to get wrong in physiological machine learning explicit
and auditable:

- **case-to-subject identity**, so a split generalises to *unseen
  people* rather than to unseen trials of people the model already saw,
  and
- **transform provenance**, so a convolution transform fitted on
  training cases is reused unchanged on held-out cases.

This vignette walks the fully offline parts of that workflow. The
deep-learning (torch/luz), ROCKET (Python/aeon), ONNX, and
foundation-model paths are backend-gated and are shown as (unevaluated)
code at the end.

## A small experiment

We start from a `PhysioExperiment` whose assay is a
`time x channel x case` array. Each case is one labelled trial; several
cases may come from the same subject.

``` r

library(PhysioML)
#> Loading required package: PhysioExperiment
library(PhysioExperiment)
library(S4Vectors)
#> Loading required package: stats4
#> Loading required package: BiocGenerics
#> Loading required package: generics
#> 
#> Attaching package: 'generics'
#> The following objects are masked from 'package:base':
#> 
#>     as.difftime, as.factor, as.ordered, intersect, is.element, setdiff,
#>     setequal, union
#> 
#> Attaching package: 'BiocGenerics'
#> The following object is masked from 'package:PhysioExperiment':
#> 
#>     as.data.frame
#> The following objects are masked from 'package:stats':
#> 
#>     IQR, mad, sd, var, xtabs
#> The following objects are masked from 'package:base':
#> 
#>     anyDuplicated, aperm, append, as.data.frame, basename, cbind,
#>     colnames, dirname, do.call, duplicated, eval, evalq, Filter, Find,
#>     get, grep, grepl, is.unsorted, lapply, Map, mapply, match, mget,
#>     order, paste, pmax, pmax.int, pmin, pmin.int, Position, rank,
#>     rbind, Reduce, rownames, sapply, saveRDS, table, tapply, unique,
#>     unsplit, which.max, which.min
#> 
#> Attaching package: 'S4Vectors'
#> The following object is masked from 'package:utils':
#> 
#>     findMatches
#> The following objects are masked from 'package:base':
#> 
#>     expand.grid, I, unname

set.seed(42)
n_case <- 8
arr <- array(stats::rnorm(60 * 2 * n_case), dim = c(60, 2, n_case))
pe <- PhysioExperiment(
  assays = list(raw = arr),
  colData = DataFrame(label = c("C3", "C4")),
  samplingRate = 100
)
pe
#> class: PhysioExperiment
#> dim: 60 x 2 x 8 
#> assays(1): raw
#> samplingRate: 100 Hz
#> channels(2): C3, C4
#> colData names(1): label
```

## Attach subject identity, then split on subjects

A cycle or trial is a *case*, not an independent subject.
[`withCaseData()`](https://x-biosignal.github.io/PhysioML/reference/withCaseData.md)
records an explicit case-to-subject mapping instead of guessing subjects
from case names. Cases in this assay without names are addressed as
`case_1`, `case_2`, …

``` r

case_data <- data.frame(
  case_id    = paste0("case_", seq_len(n_case)),
  subject_id = paste0("subj_", rep(1:4, each = 2))
)
pe <- withCaseData(pe, case_data)
```

[`validateSubjectSplit()`](https://x-biosignal.github.io/PhysioML/reference/validateSubjectSplit.md)
then refuses a split that shares a case *or a subject* between training
and validation – the silent leak that inflates reported accuracy.

``` r

train <- case_data[case_data$subject_id %in% c("subj_1", "subj_2"), ]
valid <- case_data[case_data$subject_id %in% c("subj_3", "subj_4"), ]
validateSubjectSplit(train, valid)
```

## Feature extraction

[`peReducedFeatures()`](https://x-biosignal.github.io/PhysioML/reference/peReducedFeatures.md)
produces a case-by-feature design matrix. The canonical catch22
characteristics are computed through the official `Rcatch22` backend, so
that step runs only when the optional package is installed:

``` r

if (requireNamespace("Rcatch22", quietly = TRUE)) {
  features <- peReducedFeatures(pe, method = "catch22")
  dim(features)
} else {
  message("Install Rcatch22 to compute catch22 features.")
}
#> [1]  8 44
```

The convolution transforms
([`rocket()`](https://x-biosignal.github.io/PhysioML/reference/rocket.md),
[`minirocket()`](https://x-biosignal.github.io/PhysioML/reference/minirocket.md))
follow the same contract but use a caller-managed Python/aeon backend.
When transforming held-out cases, pass the model fitted only on training
cases so no information leaks from validation into the transform.

## Aligning two domains with CORAL

[`domainAdapt()`](https://x-biosignal.github.io/PhysioML/reference/domainAdapt.md)
matches the first- and second-order statistics of a labelled source
domain to an unlabelled target domain. It is pure R and runs offline.
Fit it with training-source observations and a designated unlabelled
adaptation set only – never with held-out target data.

``` r

feat <- c("f1", "f2", "f3")
source <- matrix(stats::rnorm(30), ncol = 3, dimnames = list(NULL, feat))
target <- matrix(stats::rnorm(30, mean = 1), ncol = 3,
                 dimnames = list(NULL, feat))
fit <- domainAdapt(source, target)
fit$diagnostics$source_mean_residual
#> [1] 0
```

The residual is ~0 because aligned source means now match the target
means.

## Backend-gated workflows

The remaining workflows need optional backends and are shown here for
orientation only (not evaluated in this vignette):

``` r

# Deep learning (needs torch + luz)
train_ds <- peDataset(
  pe, targets = rep(c("a", "b"), 4),
  task = "classification", class_levels = c("a", "b")
)
model <- trainModel(train_ds, model = "cnn1d", epochs = 10L)
predictModel(model, train_ds, type = "class")

# Governed ONNX round-trip (needs the torch -> ONNX bridge + ONNX Runtime)
path <- tempfile(fileext = ".onnx")
exportONNX(model, path)
onnx_model <- importONNX(path)

# Governed foundation-model embeddings (needs a Python env + pinned revision)
foundationEmbed(
  pe, model = "moment",
  revision = "411e288267f82cce86296dbe4d6c8bc533cc162f",
  manifest_sha256 = "<expected manifest hash>",
  allow_download = TRUE
)
```

None of these mechanics establish clinical validity; they establish
*reproducibility* and *leakage discipline*. See
[`?peReducedFeatures`](https://x-biosignal.github.io/PhysioML/reference/peReducedFeatures.md),
[`?validateSubjectSplit`](https://x-biosignal.github.io/PhysioML/reference/validateSubjectSplit.md),
and
[`?trainModel`](https://x-biosignal.github.io/PhysioML/reference/trainModel.md)
for the per-function contracts.
