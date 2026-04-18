"""In-place markdown patch for the error-analysis cell. Preserves all executed code outputs."""
import json
import sys
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "preliminary_visualizations.ipynb"
OLD_MARKER = "Why does sentiment help Linear Regression on AAPL but not tree models?"
NEW_MARKDOWN = """# [sentiment-comparison]
## Error Analysis & Discussion

### What hyperparameter tuning revealed

The earlier version of this notebook reported that **tree models were indifferent or hurt by sentiment**, and attributed it to the sparse-feature problem. Running a per-ticker 4-fold chronological-CV grid search on Random Forest (`max_depth \u2208 {5, 8, None}`, `min_samples_leaf \u2208 {5, 10, 20}`, plus `n_estimators` and `max_features`) and XGBoost (`max_depth \u2208 {3, 5, 8}`, `learning_rate \u2208 {0.05, 0.1}`, `reg_lambda \u2208 {1.0, 5.0}`) overturned that conclusion:

- Default RF (`min_samples_leaf=1`, unbounded depth) was overfitting training-period volatility clusters, particularly on TSLA where the 2014\u20132015 China-correction and Model X launch periods created heavy-tailed RV spikes that don't recur in the 2017 test window. The default tree memorized those spikes as feature-space rules; tuned RF (`min_samples_leaf \u2208 {10, 20}`, `max_depth \u2264 8`) cut TSLA MSE by **95%** (1.66e-05 \u2192 8.03e-07).
- Once the trees were properly regularized, sentiment features became additive rather than adversarial. The largest single win is **AAPL XGBoost with sentiment: \u221219.7% MSE** (1.38e-07 \u2192 1.11e-07). Sentiment now improves tuned tree models on 5 of 6 (ticker \u00d7 tree-model) combinations.

**Takeaway:** the \"sparse sentiment hurts trees\" intuition was correct in direction but misattributed. The failure mode was unregularized trees extracting spurious splits from both volatility features and sentiment features simultaneously. Regularize the trees and sentiment helps.

### Why does GARCH(1,1) win on TSLA?

GARCH models two things ML has to learn from scratch: volatility clustering (past shocks predict current volatility) and mean reversion (volatility drifts toward a long-run level). TSLA in 2013\u20132017 has both effects pronounced \u2014 the 2014 and 2015 spike regimes are followed by explicit mean reversion toward ~0.0005 daily RV. GARCH's parametric form captures this with 3 parameters; Random Forest has to learn it from ~1000 training rows across 7+ features. On the tickers with cleaner mean reversion (AAPL, NKE), the ML models have enough data to catch up. On TSLA, the econometric prior wins.

This is a finding worth keeping in the report: ML does not dominate classical financial econometrics here. It is competitive, beats GARCH on 2 of 3 tickers, but on the most volatile ticker GARCH still wins.

### Feature importance

Across tuned RF and XGBoost on AAPL (see the feature importance plot above), the ranking is:

1. `RV` (current volatility) \u2014 dominant (~40\u201360% of importance), as expected from volatility clustering.
2. `rolling_vol_5`, `rolling_vol_10` \u2014 secondary volatility-persistence signals.
3. `RSI_14` \u2014 modest contribution; useful as a regime indicator (oversold/overbought) even though it was designed for return prediction, not volatility.
4. `volume_change`, `SMA_10`, `SMA_20` \u2014 low but non-zero.
5. Sentiment features (`sentiment_mean`, `sentiment_mean_5d`, `news_count`, `news_count_5d`, `sentiment_std`) \u2014 lowest importance individually, but collectively non-negligible, and \u2014 critically \u2014 their presence changes the split choices of the other features enough to improve test MSE. Among the sentiment columns, `news_count_5d` usually ranks highest, consistent with the finance-research finding that the *volume* of news attention predicts volatility more reliably than sentiment polarity.

### What would improve these results further?

1. **Denser news coverage.** The single biggest remaining constraint. A multi-year Finnhub or Reuters archive would likely flip TSLA from GARCH-best to ML-best by pushing sentiment coverage past 50% of trading days.
2. **Walk-forward CV reporting.** The `walk_forward_evaluate` helper in `src/evaluate.py` runs 5-fold expanding-window evaluation and reports mean\u00b1std MSE; currently the notebook reports a single train/test split for readability. A paper-ready version would quote the walk-forward numbers.
3. **FinBERT sentiment** scoring the same headlines \u2014 financial-domain transformer vs. general-purpose VADER \u2014 worth trying once coverage is denser.
4. **Asymmetric features** (e.g. signed squared returns, semi-variances) \u2014 capture the leverage effect where downside moves predict more next-day volatility than upside moves of equal magnitude.
"""


def main():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    patched = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        src = "".join(cell.get("source", []))
        if OLD_MARKER in src:
            cell["source"] = NEW_MARKDOWN.splitlines(keepends=True)
            patched += 1
    NB_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Patched {patched} error-analysis cell(s) in {NB_PATH.name}")


if __name__ == "__main__":
    main()
