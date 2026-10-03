# Calculate canonical catch22 features

Calls the official C-backed
[`Rcatch22::catch22_all()`](https://rdrr.io/pkg/Rcatch22/man/catch22_all.html)
implementation for every selected case-channel series. Input samples are
passed unchanged.

## Usage

``` r
catch22(x, assay_name = NULL, channels = NULL, cases = NULL, catch24 = FALSE)
```

## Arguments

- x:

  A `PhysioExperiment`.

- assay_name:

  `NULL` for
  [`PhysioExperiment::defaultAssay()`](https://x-biosignal.r-universe.dev/PhysioExperiment/reference/defaultAssay.html)
  or an exact assay name.

- channels:

  `NULL` or exact unique channel labels in requested order.

- cases:

  `NULL`, exact unique case labels, or exact unique positive case
  indices in requested order.

- catch24:

  One non-missing logical. When `TRUE`, append upstream mean and
  standard-deviation features after the canonical 22.

## Value

An
[`S4Vectors::DataFrame()`](https://rdrr.io/pkg/S4Vectors/man/DataFrame-class.html)
with one row per case-channel pair, identity columns followed by 22 or
24 numeric feature columns. Stable transform provenance and typed
diagnostics are stored as attributes.

## Details

A 2-D assay is interpreted as time x channel for one case. A 3-D assay
is interpreted as time x channel x case. Series must be finite, equal
length within an assay, and contain at least 10 samples. PhysioML does
not impute, resample, normalize, detrend, or collapse channels before
calling Rcatch22.

## References

Lubba CH, Sethi SS, Knaute P, Schultz SR, Fulcher BD, Jones NS (2019).
catch22: CAnonical Time-series CHaracteristics. *Data Mining and
Knowledge Discovery*, 33, 1821-1852.
[doi:10.1007/s10618-019-00647-x](https://doi.org/10.1007/s10618-019-00647-x)

Henderson T (2026). Rcatch22: Calculation of 22 Canonical Time-Series
Characteristics. R package.

## Examples

``` r
if (requireNamespace("Rcatch22", quietly = TRUE)) {
  values <- matrix(sin(seq(0, 8 * pi, length.out = 100)), 100, 1)
  pe <- PhysioExperiment::PhysioExperiment(
    assays = list(raw = values),
    colData = S4Vectors::DataFrame(label = "signal"),
    samplingRate = 100
  )
  catch22(pe)
}
#> DataFrame with 1 row and 26 columns
#>       case_id input_case_index  channel_id input_channel_index
#>   <character>        <integer> <character>           <integer>
#> 1      case_1                1      signal                   1
#>   DN_HistogramMode_5 DN_HistogramMode_10 CO_f1ecac CO_FirstMin_ac
#>            <numeric>           <numeric> <numeric>      <numeric>
#> 1                  0                   0   5.77817             12
#>   CO_HistogramAMI_even_2_5 CO_trev_1_num MD_hrv_classic_pnn40
#>                  <numeric>     <numeric>            <numeric>
#> 1                 0.806405  -6.72862e-18             0.919192
#>   SB_BinaryStats_mean_longstretch1 SB_TransitionMatrix_3ac_sumdiagcov
#>                          <numeric>                          <numeric>
#> 1                               14                          0.0102041
#>   PD_PeriodicityWang_th0_01 CO_Embed2_Dist_tau_d_expfit_meandiff
#>                   <numeric>                            <numeric>
#> 1                        24                               0.7845
#>   IN_AutoMutualInfoStats_40_gaussian_fmmi FC_LocalSimple_mean1_tauresrat
#>                                 <numeric>                      <numeric>
#> 1                                       5                              1
#>   DN_OutlierInclude_p_001_mdrmd DN_OutlierInclude_n_001_mdrmd
#>                       <numeric>                     <numeric>
#> 1                        -0.115                          0.13
#>   SP_Summaries_welch_rect_area_5_1 SB_BinaryStats_diff_longstretch0
#>                          <numeric>                        <numeric>
#> 1                         0.988596                               14
#>   SB_MotifThree_quantile_hh SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1
#>                   <numeric>                                   <numeric>
#> 1                   1.58547                                    0.542857
#>   SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1 SP_Summaries_welch_rect_centroid
#>                                <numeric>                        <numeric>
#> 1                               0.228571                         0.245437
#>   FC_LocalSimple_mean3_stderr
#>                     <numeric>
#> 1                    0.494793
```
