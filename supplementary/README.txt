SUPPLEMENTARY TABLES - ECG denoising comparison (computed by scripts/analyze.py from results/test_results.csv and results/dl_results.csv)

FILES
S1   mean ± SD over the 36 test records of delta-SNR (dB), RMSE (mV) and PRD (%) for every method x noise type x input SNR.
S2   QRS-detection scores for every condition: F1, Se, PPV, median R-peak localisation error (ms), relative error (%) of the RR-interval
     RMSSD, and F1 on the noise-free reference (F1_ref).
S3   all paired tests (families below) with raw and Holm-adjusted p-values.
S4   the same tests with test record 201 excluded (n = 35); 201 and development record 202 are from one subject.
S5   PRD (%) of each filter on the noise-free reference ECG: mean, median and maximum over the 36 test records.
S6   record-level Spearman correlation between delta-SNR gain and F1 gain (summary; S6b = per record and noise type).
S7   sensitivity to the reference ECG: mean delta-SNR of every conventional and adaptive method x noise type x input SNR (112 rows)
     for the main reference and three alternatives (spline only = spline baseline removed without the 0.3 Hz high-pass;
     highpass = 0.5 Hz zero-phase high-pass of the raw record; raw = raw record, mean removed). The frozen parameters were
     re-scored without re-tuning; the U-Net is not included. Rank correlations use all methods in the table; the statistics on the
     best blind method use six candidate methods (butter_zp, median_bl, notch, dwt, lms_blind, rls_blind): butter_causal and
     dwt_legacy are reference configurations and are not candidates.

CODES
Noise types   bw = baseline wander, em = electrode motion, ma = muscle artifact (MIT-BIH NST Database records); pli = synthetic power-line interference
Methods       none = no filtering; butter_zp = Butterworth band-pass 0.5-40 Hz, zero-phase; butter_causal = the same, causal;
              median_bl = median-filter baseline removal (BW only); notch = 50 Hz notch (PLI only); dwt = tuned wavelet configuration;
              dwt_legacy = level-5 wavelet keeping the approximation; lms_blind / rls_blind = adaptive filters with a 50 + 150 Hz quadrature
              reference (PLI only); lms_oracle / rls_oracle = adaptive filters with an oracle reference (true noise + white noise, 10 dB);
              dl_unet = 1D U-Net (4-fold cross-validation over records)
Metrics       dSNR = output SNR - input SNR (dB); RMSE (mV); PRD (%); all after mean removal over the evaluation window (first and last 10 s excluded).
              rmssd_err_pct = relative error of the RMSSD of RR intervals (detected vs annotated beats; ectopic beats included, no artefact correction).

TEST FAMILIES (S3, S4). Contrast "A - B" = mean of the per-record difference A minus B; Holm correction is applied within each family.
P1  planned      lms_oracle - rls_oracle, delta-SNR; 4 noise types x 4 SNR = 16 tests
P2  secondary    lms_blind - butter_zp, delta-SNR, PLI; 4 tests (comparator chosen after the classical test results had been inspected)
P3  secondary    dl_unet - butter_zp, delta-SNR; 16 tests (comparator chosen after the classical test results had been inspected)
P4  planned      every method - none, F1; 128 tests (8 + 7 + 7 + 10 methods for BW, EM, MA, PLI x 4 SNR)
E1  exploratory  PLI, delta-SNR: lms_oracle - lms_blind, rls_oracle - rls_blind, lms_blind - notch, rls_blind - notch, lms_oracle - notch; 20 tests
E4  exploratory  every method - none, "excess" RMSSD error (error of the filtered signal minus error on the noise-free reference); 128 tests
E5  exploratory  dl_unet - dwt (all noise types), - median_bl (BW), - notch / lms_blind / rls_blind (PLI), - best oracle method
                 (rls_oracle for BW/EM/MA, lms_oracle for PLI); delta-SNR; 48 tests
Columns       n = number of paired records; mean_diff, median_diff = in the units of the metric; ci_lo, ci_hi = 95% percentile bootstrap interval
              of the mean difference (10,000 resamples of the paired per-record differences); rank_biserial = matched-pairs rank-biserial
              correlation (positive when A > B); p = two-sided Wilcoxon signed-rank test (SciPy default method, zero differences discarded;
              minimum attainable p for n = 36 is 2.9e-11); p_holm = Holm-adjusted p within the family.
