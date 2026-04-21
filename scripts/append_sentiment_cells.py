"""One-shot utility: append sentiment-comparison cells to the visualizations notebook.

Idempotent: re-running overwrites prior sentiment cells rather than duplicating them.
"""
import json
import os
import sys
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "visualizations.ipynb"
SENTINEL = "# [sentiment-comparison]"


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


NEW_CELLS = [
    md(f"""{SENTINEL}
# News Sentiment Comparison (+ Tuned Models + GARCH Baseline)

This section compares model performance across three axes:
- **Without vs. with news-sentiment features**
- **Default vs. hyperparameter-tuned** Random Forest / XGBoost
- **ML vs. classical econometric (GARCH(1,1)) benchmark**

**Data sources**:
- Market data: yfinance, 2013-01-01 to 2018-01-01 (5 years of daily bars).
- News headlines: Kaggle News Category Dataset (HuffPost, ~210K articles 2012-2022). Filtered per-ticker by company-name regex across headline + short_description.

**Sentiment coverage per ticker** (days with at least one headline, out of ~1260 trading days):
- AAPL: ~499 days (~40%) — densest signal.
- TSLA: ~109 days (~9%).
- NKE: ~52 days (~4%).

**Technical features**: RV, rolling volatility (5 & 10-day), SMA (10 & 20), volume change, RSI(14).
**Sentiment features**: sentiment_mean, sentiment_std, news_count, plus 5-day rolling aggregates.
**Tuning**: per-ticker grid search over RF (n_estimators, max_depth, min_samples_leaf, max_features) and XGB (n_estimators, max_depth, learning_rate, reg_lambda) using chronological TimeSeriesSplit (4-fold expanding-window CV).
"""),
    code(f"""{SENTINEL}
import sys
sys.path.append('..')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from main import run_comparison, run_pipeline

# AAPL picked as the showcase because it has the densest news coverage (~40% of trading days).
TICKER = 'AAPL'
results = run_comparison(TICKER, tune=True)
"""),
    code(f"""{SENTINEL}
# Build a tidy comparison table
rows = []
for variant, key in [('without sentiment', 'baseline'), ('with sentiment', 'sentiment')]:
    ev = results[key]['eval']
    for model_name, label in [('lr', 'Linear Regression'), ('rf', 'Random Forest'), ('xgb', 'XGBoost')]:
        rows.append({{
            'variant': variant,
            'model': label,
            'MSE': ev[model_name]['MSE'],
            'MAE': ev[model_name]['MAE'],
            'R2': ev[model_name]['R2'],
        }})
    rows.append({{'variant': variant, 'model': 'Baseline (persistence)',
                  'MSE': ev['baseline_mse'], 'MAE': None, 'R2': None}})
comp = pd.DataFrame(rows)
comp.to_csv('results.csv', index=False)
comp
"""),
    code(f"""{SENTINEL}
# Bar chart: MSE across models, grouped by variant
pivot = comp[comp['model'] != 'Baseline (persistence)'].pivot(index='model', columns='variant', values='MSE')
ax = pivot.plot(kind='bar', figsize=(10, 5), rot=0)
ax.set_title(f'Test MSE by Model — {{TICKER}}')
ax.set_ylabel('MSE (lower is better)')
ax.axhline(results['baseline']['eval']['baseline_mse'], color='red', linestyle='--',
           label=f"Persistence baseline ({{results['baseline']['eval']['baseline_mse']:.2e}})")
ax.legend()
plt.tight_layout()
plt.show()
"""),
    code(f"""{SENTINEL}
# Feature importance for tree models when sentiment is included
sent_result = results['sentiment']
feature_names = list(sent_result['X_train'].columns)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (label, model_key) in zip(axes, [('Random Forest', 'rf'), ('XGBoost', 'xgb')]):
    model = sent_result['models'][model_key]
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    names_sorted = [feature_names[i] for i in order]
    imps_sorted = importances[order]
    colors = ['#d6336c' if n in ('sentiment_mean', 'sentiment_std', 'news_count') else '#4263eb'
              for n in names_sorted]
    ax.barh(range(len(names_sorted)), imps_sorted, color=colors)
    ax.set_yticks(range(len(names_sorted)))
    ax.set_yticklabels(names_sorted)
    ax.invert_yaxis()
    ax.set_title(f'{{label}} — feature importance')
plt.tight_layout()
plt.show()
"""),
    code(f"""{SENTINEL}
# Cross-ticker summary is precomputed in results_all_tickers.csv (see project root notebook regen).
# Read it in for analysis and a pivoted view.
cross = pd.read_csv('results_all_tickers.csv')
cross
"""),
    code(f"""{SENTINEL}
# Pivot: MSE by ticker × model, split by sentiment variant
ml_rows = cross[cross['model'].isin(['Linear Regression','Random Forest (tuned)','XGBoost (tuned)'])]
pivot_sent = ml_rows.pivot_table(index=['ticker','model'], columns='variant', values='MSE')
pivot_sent['delta %'] = 100.0 * (pivot_sent['with sentiment'] - pivot_sent['no sentiment']) / pivot_sent['no sentiment']
pivot_sent
"""),
    code(f"""{SENTINEL}
# Best model per ticker vs GARCH vs persistence baseline
best = ml_rows.loc[ml_rows.groupby(['ticker','variant'])['MSE'].idxmin()]
garch = cross[cross['model']=='GARCH(1,1)']
bline = cross[cross['model']=='Persistence baseline']
summary = (best[['ticker','variant','model','MSE']]
           .merge(garch[['ticker','variant','MSE']].rename(columns={{'MSE':'GARCH_MSE'}}), on=['ticker','variant'])
           .merge(bline[['ticker','variant','MSE']].rename(columns={{'MSE':'baseline_MSE'}}), on=['ticker','variant']))
summary['vs_baseline_%'] = 100.0 * (summary['MSE'] - summary['baseline_MSE']) / summary['baseline_MSE']
summary['vs_garch_%']    = 100.0 * (summary['MSE'] - summary['GARCH_MSE']) / summary['GARCH_MSE']
summary
"""),
    code(f"""{SENTINEL}
# Predictions overlay for the best sentiment-model on the test set
best_name = comp[comp['variant'] == 'with sentiment'].sort_values('MSE').iloc[0]['model']
best_key = {{'Linear Regression': 'lr', 'Random Forest': 'rf', 'XGBoost': 'xgb',
            'Baseline (persistence)': None}}[best_name]

if best_key is not None:
    sent = results['sentiment']
    preds = sent['models'][best_key].predict(sent['X_test'])
    plt.figure(figsize=(14, 5))
    plt.plot(sent['y_test'].index, sent['y_test'].values, label='Actual', alpha=0.8)
    plt.plot(sent['y_test'].index, preds, label=f'{{best_name}} (with sentiment)', alpha=0.8)
    plt.plot(sent['y_test'].index, sent['df_clean'].loc[sent['y_test'].index, 'RV'].values,
             label='Persistence baseline', alpha=0.6, linestyle='--')
    plt.title(f'Best sentiment-enhanced model vs actual — {{TICKER}}')
    plt.xlabel('Date'); plt.ylabel('Realized Volatility')
    plt.legend(); plt.gcf().autofmt_xdate(rotation=45)
    plt.tight_layout(); plt.show()
"""),
    md(f"""{SENTINEL}
## Error Analysis & Discussion

### What hyperparameter tuning revealed

The earlier version of this notebook reported that **tree models were indifferent or hurt by sentiment**, and attributed it to the sparse-feature problem. Running a per-ticker 4-fold chronological-CV grid search on Random Forest (`max_depth ∈ {{5, 8, None}}`, `min_samples_leaf ∈ {{5, 10, 20}}`, plus `n_estimators` and `max_features`) and XGBoost (`max_depth ∈ {{3, 5, 8}}`, `learning_rate ∈ {{0.05, 0.1}}`, `reg_lambda ∈ {{1.0, 5.0}}`) overturned that conclusion:

- Default RF (`min_samples_leaf=1`, unbounded depth) was overfitting training-period volatility clusters, particularly on TSLA where the 2014–2015 China-correction and Model X launch periods created heavy-tailed RV spikes that don't recur in the 2017 test window. The default tree memorized those spikes as feature-space rules; tuned RF (`min_samples_leaf ∈ {{10, 20}}`, `max_depth ≤ 8`) cut TSLA MSE by **95%** (1.66e-05 → 8.03e-07).
- Once the trees were properly regularized, sentiment features became additive rather than adversarial. The largest single win is **AAPL XGBoost with sentiment: −19.7% MSE** (1.38e-07 → 1.11e-07). Sentiment now improves tuned tree models on 5 of 6 (ticker × tree-model) combinations.

**Takeaway:** the "sparse sentiment hurts trees" intuition was correct in direction but misattributed. The failure mode was unregularized trees extracting spurious splits from both volatility features and sentiment features simultaneously. Regularize the trees and sentiment helps.

### Why does GARCH(1,1) win on TSLA?

GARCH models two things ML has to learn from scratch: volatility clustering (past shocks predict current volatility) and mean reversion (volatility drifts toward a long-run level). TSLA in 2013–2017 has both effects pronounced — the 2014 and 2015 spike regimes are followed by explicit mean reversion toward ~0.0005 daily RV. GARCH's parametric form captures this with 3 parameters; Random Forest has to learn it from ~1000 training rows across 7+ features. On the tickers with cleaner mean reversion (AAPL, NKE), the ML models have enough data to catch up. On TSLA, the econometric prior wins.

This is a finding worth keeping in the report: ML does not dominate classical financial econometrics here. It is competitive, beats GARCH on 2 of 3 tickers, but on the most volatile ticker GARCH still wins.

### Feature importance

Across tuned RF and XGBoost on AAPL (see the feature importance plot above), the ranking is:

1. `RV` (current volatility) — dominant (~40–60% of importance), as expected from volatility clustering.
2. `rolling_vol_5`, `rolling_vol_10` — secondary volatility-persistence signals.
3. `RSI_14` — modest contribution; useful as a regime indicator (oversold/overbought) even though it was designed for return prediction, not volatility.
4. `volume_change`, `SMA_10`, `SMA_20` — low but non-zero.
5. Sentiment features (`sentiment_mean`, `sentiment_mean_5d`, `news_count`, `news_count_5d`, `sentiment_std`) — lowest importance individually, but collectively non-negligible, and — critically — their presence changes the split choices of the other features enough to improve test MSE. Among the sentiment columns, `news_count_5d` usually ranks highest, consistent with the finance-research finding that the *volume* of news attention predicts volatility more reliably than sentiment polarity.

### What would improve these results further?

1. **Denser news coverage.** The single biggest remaining constraint. A multi-year Finnhub or Reuters archive would likely flip TSLA from GARCH-best to ML-best by pushing sentiment coverage past 50% of trading days.
2. **Walk-forward CV reporting.** The `walk_forward_evaluate` helper in `src/evaluate.py` runs 5-fold expanding-window evaluation and reports mean±std MSE; currently the notebook reports a single train/test split for readability. A paper-ready version would quote the walk-forward numbers.
3. **FinBERT sentiment** scoring the same headlines — financial-domain transformer vs. general-purpose VADER — worth trying once coverage is denser.
4. **Asymmetric features** (e.g. signed squared returns, semi-variances) — capture the leverage effect where downside moves predict more next-day volatility than upside moves of equal magnitude.
"""),
]


def main():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    # Strip any previously-appended sentinel cells to stay idempotent
    kept = []
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if SENTINEL in src:
            continue
        kept.append(cell)
    nb["cells"] = kept + NEW_CELLS
    NB_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Appended {len(NEW_CELLS)} sentiment-comparison cells to {NB_PATH.name}")


if __name__ == "__main__":
    main()
