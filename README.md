# Testing for heavy tails in financial returns

Parameter-free tests of whether the tails of daily returns follow a power law, a lognormal or an exponential, using constrained surrogates. Applied to both tails of 22 assets in six asset classes.

## Summary

Large daily losses happen far more often than a normal distribution predicts. For risk management it matters *which* heavy-tailed shape describes them, because the shape decides how likely a loss larger than any seen so far is. The usual tests first fit the shape and then test with the fitted parameters plugged in. Tails are small samples, so these tests are unreliable: they reject too rarely, and they overstate the probability of extremes beyond the data.

Constrained surrogates (Moore & Altmann, 2026; Moore, Yan & Altmann, 2022) avoid this. The idea is like shuffling. A shuffled time series is just as likely as the original under any model of independent draws, so comparing data with shuffles tests independence without assuming a model. In the same way, the fit of a tail shape depends on the data only through a simple summary (for a power law, the sum of the logarithms of the tail values). Rearranging the tail values while keeping that summary fixed gives new samples that are just as likely as the data under *every* member of the family. Comparing the data with these surrogates tests the whole family at once, without estimating a parameter.

This repository re-implements the method in Python, independently of the authors' code, checks it on simulated data, and asks four questions of real returns:

1. **Are the tails heavier than exponential?** Mostly for losses. After removing volatility clustering, the exponential is rejected for 13 of 22 loss tails but only 5 of 22 gain tails.
2. **Does a single power law describe the tail?** Only its far end. At the usual data-chosen threshold (typically the top 2–3% of days) the power law survives for 41 of 44 tails, but it is rejected for all 44 once the tail is the top 10% of days. The lognormal, tested on the same tails, mostly survives.
3. **Are losses heavier-tailed than gains?** They tend to be: the loss tail is heavier for 19 of 22 assets. This is a pattern across assets, not a firm result for any single one.
4. **Do the surrogates give better value-at-risk forecasts?** No. At 99–99.5% one-day VaR they match the usual forecast from the same shape. Their value is in checking tail models, not in forecasting.

## How it works

* **Remove volatility clustering.** Large moves cluster in time, but the tests assume independent draws. Each asset's returns are divided by their predicted volatility from a GARCH model. Three standard versions are fitted and the best by BIC is chosen before any test; a GJR-GARCH with Student-t shocks wins for 21 of 22 assets. Results are about these standardised shocks.
* **Three shapes.** Above a threshold, the tail is compared with a power law (heaviest), a lognormal and an exponential (thinnest). A power law is a limiting case of a lognormal, so no test can reject a power law *as* a lognormal.
* **Surrogates.** For the power law and exponential, a surrogate repeatedly picks two tail values and splits their total at random, which keeps the sum fixed. For the lognormal it moves three values at a time, keeping both the sum and the sum of squares fixed. Each surrogate has exactly the same fitted parameters as the data. The samplers are in [`src/tailsurr/surrogates.py`](src/tailsurr/surrogates.py) (about 200 lines).
* **Three statistics**, each asking a different question: the **largest value** (is the biggest move too big or too small for this shape?), the **far-tail slope** (does the very end of the tail fall off like the rest?) and the **overall fit** (Kolmogorov–Smirnov distance). The p-value is the share of surrogates at least as extreme as the data.
* **Where the tail starts.** The threshold is chosen by the standard rule of Clauset, Shalizi & Newman (2009), which favours the power law. Every test is therefore repeated with the tail set to the top 1, 2, 5, 10 and 20% of days.
* **Many tests.** The false discovery rate is controlled at 5% (Benjamini–Hochberg). Evidence is pooled across assets with a test that remains valid when assets move together (Cauchy combination), and tail indices are pooled with a random-effects model.

## Checking the method

On simulated tails with a known shape ([`scripts/01_validate_synthetic.py`](scripts/01_validate_synthetic.py)):

* **False alarms.** When the data come from the tested shape, the surrogate tests reject about 5% of the time at the 5% level, as they should (2.5–7.3% across all shapes, statistics and sample sizes). The standard approach of testing against a fitted distribution rejects too rarely, most of all for the lognormal.
* **Telling shapes apart.** An exponential is usually rejected for heavy-tailed data even with 50 tail values. Separating a lognormal from a power law needs several hundred.
* **Probabilities of new extremes.** For a loss beyond the sample maximum, plugging in the fitted parameter overstates its probability by about a quarter; the surrogate estimate is close to unbiased.
* **Agreement with the authors' code.** Surrogates from this implementation and the authors' reference code are statistically indistinguishable ([`scripts/06_crosscheck_reference.py`](scripts/06_crosscheck_reference.py); the reference code is not included).

![power](figures/validation_power.png)

**One important caveat.** In real data the threshold is chosen from the data. The tests then reject *less* often than they should when the shape is right, so a rejection is strong evidence but a non-rejection is weak. An attempt to correct for this, re-selecting the threshold inside each surrogate, over-rejects (about 15% at nominal 5%; [`scripts/10_check_selection_aware.py`](scripts/10_check_selection_aware.py)), so it is not used.

## Results

**Data.** Free daily closes ending 12 October 2026: seven equity indices, five large single stocks, five currency pairs, gold, Brent crude, the 10-year Treasury yield, Bitcoin and Ether (Yahoo Finance and FRED). A data-quality screen removed a handful of bad prints, stale prices and multi-week gaps; every flagged event is listed in [`results/breadth_data_events.csv`](results/breadth_data_events.csv).

**Thin tails are rejected, mainly for losses.** The exponential is rejected for 13 of 22 loss tails and 5 of 22 gain tails after filtering, against 17 and 16 on raw returns. Part of the raw heavy tail therefore comes from volatility clustering. Pooled across assets, the exponential is rejected in every asset class.

**A power law fits only the far tail.** It is rejected for every asset once the tail reaches the top 10% of days, while the lognormal on the same tails is rejected for only 3 of 44. Pooled across assets there is modest evidence that the far tail bends down, as a lognormal would, but no single asset separates the two shapes.

![threshold sensitivity](figures/breadth_sensitivity.png)

**Losses tend to be heavier-tailed than gains.** The pooled tail index (smaller means heavier) is 3.71 [3.50, 3.92] for losses and 4.37 [4.05, 4.70] for gains, and the loss tail is heavier for 19 of 22 assets. For no single asset do the two intervals separate, and the pooled intervals treat correlated markets as independent, so they are too narrow.

![tail index](figures/breadth_tail_index.png)

**No forecasting gain.** In a rolling test on the S&P 500 (2000–2026) and Bitcoin (2019–2026), surrogate VaR forecasts stay within about 2% of the usual forecast from the same shape on 90% of days ([`scripts/04_backtest_var.py`](scripts/04_backtest_var.py)). At 99–99.5% with a 100-point tail the forecast hardly reaches beyond the data, so unbiasedness has little room to help. The shape matters more: a power law fitted to the top 10% of shocks is too cautious, with 17–18 breaches at 99.5% for the S&P 500 where 34 are expected.

![S&P 500 backtest](figures/backtest_SP500.png)

The per-asset table (tail indices with intervals, verdicts, threshold patterns and clustering flags) is in [`results/breadth_summary_table.md`](results/breadth_summary_table.md).

## Limitations

* **Where the tail starts.** The threshold is chosen from the data, so non-rejections are weak evidence. A valid correction remains an open problem.
* **Power.** Tails of a few hundred values cannot separate a power law from a lognormal. I do not claim the tails *are* power laws.
* **The volatility model.** The tests are applied to GARCH residuals fitted on the full sample. 7 of 44 tails still show some clustering, and one tail shape is assumed over decades.
* **Multiple testing.** Most rejections sit near the smallest p-value that 999 surrogates can give, so each rejection count is uncertain by about one. Stricter family-wise control would need about 10,000 surrogates.
* **The surrogate chains start at the data**, so the tests are exact only once the chains have mixed. Quadrupling the chain length changed nothing ([`scripts/05_mixing_check.py`](scripts/05_mixing_check.py)).
* **The forecast test is narrow:** two assets at 99–99.5%, where forecasts barely extrapolate.
* **Data.** Free data can contain errors and may be revised. The five single stocks are long-lived survivors. One very large loss (IBM, −29.0% on 2026-07-14) could not be independently verified and was kept; it does not decide IBM's result.

## Reproduce

Market data are **not included** (Yahoo Finance data may not be redistributed). The download scripts fetch them from Yahoo Finance (via `yfinance`) and FRED and cache them in `data/`, which is git-ignored. The sample end date is fixed (`END_DATE` in [`src/tailsurr/data.py`](src/tailsurr/data.py) and [`universe.py`](src/tailsurr/universe.py)), and all randomness is seeded.

```bash
git clone <this-repo-url> tail-risk-surrogates
cd tail-risk-surrogates
conda env create -f environment.yml
conda activate tail-risk-surrogates

# synthetic validation (no data needed)
python scripts/01_validate_synthetic.py      # ~8 min -> results/validation_*, figures/validation_*.png
python scripts/05_mixing_check.py            # MCMC mixing diagnostic, ~1 min
python scripts/10_check_selection_aware.py   # threshold re-selection check, ~1 min

# first pass on three assets (S&P 500, Bitcoin, EUR/USD) and the VaR test
python scripts/02_download_data.py           # download + cache data/*.csv, prints large moves
python scripts/03_pilot_tail_tests.py        # ~3 min -> results/pilot_*, figures/pilot_*.png
python scripts/04_backtest_var.py SP500      # rolling VaR/ES backtest, ~4 min
python scripts/04_backtest_var.py BTC        # ~1.5 min

# main study (22 assets)
python scripts/07_breadth_data.py            # download + data-quality screen -> data/universe/, results/breadth_data_*.csv
python scripts/08_breadth_tail_tests.py      # filters, clustering, tail index, all surrogate tests, ~7 min
python scripts/09_breadth_summary.py         # BH, pooled tests, summary table, figures/breadth_*.png (seconds)

# optional: cross-check against the authors' reference implementation (not included here; no licence)
git clone https://github.com/JackMurdochMoore/surrogates-for-distributions ../surrogates-for-distributions
python scripts/06_crosscheck_reference.py ../surrogates-for-distributions/src/constrained_likelihood_surrogates.py
```

Notes:

* `01` and `03` accept `--replot` to redraw figures from saved results. `02` and `07` accept `--refresh` to re-download.
* `09` runs from the shipped `results/breadth_*.csv`, so the breadth summary and figures can be regenerated without downloading any data.
* To keep downloaded data outside the repository, set `TAILSURR_DATA_DIR=/path/to/data` (the breadth universe goes in its `universe/` subfolder).
* The backtest writes its daily forecasts and losses to `results/backtest_<asset>.csv.gz`. These files are git-ignored because they contain the return series.
* Run times are wall-clock on a 16-core Linux machine; scripts 01, 03 and 08 use all cores.

## Repository layout

```
src/tailsurr/   surrogates.py  constrained MCMC samplers (power law, lognormal, exponential)
                fitting.py     MLE, CSN threshold, Vuong test, UMVU tail probabilities
                stats.py       tail statistics, surrogate p-values, waiting-time clustering
                selection.py   threshold grid, bootstrap CIs, selection-aware KS (not used; see caveats)
                filters.py     GARCH / GJR-GARCH / Student-t filters, BIC choice
                multitest.py   Benjamini–Hochberg, Holm, Cauchy and Fisher combination
                backtest.py    VaR/ES forecasts, Kupiec and Christoffersen tests
                data.py, universe.py, quality.py   data download, caching and screening
                plotting.py    figure style
scripts/        01–06 validation, first pass and VaR test; 07–10 main study (22 assets)
results/        summary tables, test results and run logs (CSV / text)
figures/        PNG figures used above
```

## References

* J. M. Moore and E. G. Altmann, "Constrained surrogates for arbitrary families of continuous probability distributions", arXiv:2609.31695 (2026).
* J. M. Moore, G. Yan and E. G. Altmann, "Nonparametric power-law surrogates", *Physical Review X* **12**, 021056 (2022). doi:[10.1103/PhysRevX.12.021056](https://doi.org/10.1103/PhysRevX.12.021056)
* A. Clauset, C. R. Shalizi and M. E. J. Newman, "Power-law distributions in empirical data", *SIAM Review* **51**, 661–703 (2009).
* A. J. McNeil and R. Frey, "Estimation of tail-related risk measures for heteroscedastic financial time series: an extreme value approach", *Journal of Empirical Finance* **7**, 271–300 (2000).
* P. H. Kupiec, "Techniques for verifying the accuracy of risk measurement models", *Journal of Derivatives* **3**, 73–84 (1995).
* P. F. Christoffersen, "Evaluating interval forecasts", *International Economic Review* **39**, 841–862 (1998).
* Q. H. Vuong, "Likelihood ratio tests for model selection and non-nested hypotheses", *Econometrica* **57**, 307–333 (1989).
* L. R. Glosten, R. Jagannathan and D. E. Runkle, "On the relation between the expected value and the volatility of the nominal excess return on stocks", *Journal of Finance* **48**, 1779–1801 (1993).
* Y. Benjamini and Y. Hochberg, "Controlling the false discovery rate", *Journal of the Royal Statistical Society B* **57**, 289–300 (1995).
* S. Holm, "A simple sequentially rejective multiple test procedure", *Scandinavian Journal of Statistics* **6**, 65–70 (1979).
* Y. Liu and J. Xie, "Cauchy combination test", *Journal of the American Statistical Association* **115**, 393–402 (2020).
* R. DerSimonian and N. Laird, "Meta-analysis in clinical trials", *Controlled Clinical Trials* **7**, 177–188 (1986).

---

Developed with AI assistance (Claude Code).
