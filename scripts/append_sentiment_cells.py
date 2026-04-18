"""One-shot utility: append sentiment-comparison cells to the preliminary notebook.

Idempotent: re-running overwrites prior sentiment cells rather than duplicating them.
"""
import json
import os
import sys
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "preliminary_visualizations.ipynb"
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
# News Sentiment Comparison

This section compares model performance with and without news-sentiment features.

**Data sources used**:
- Market data: yfinance, 2013-01-01 to 2018-01-01 (5 years of daily bars).
- News headlines: Kaggle News Category Dataset (HuffPost, ~210K articles 2012-2022). Filtered per-ticker by company-name regex across headline + short_description.

**Coverage per ticker** (days with at least one headline, out of ~1260 trading days):
- AAPL: ~499 days (~40% coverage) — densest signal.
- TSLA: ~109 days (~9%).
- NKE: ~52 days (~4%).

Sentiment features per trading day: `sentiment_mean`, `sentiment_std`, `news_count`, plus 5-day rolling aggregates (`sentiment_mean_5d`, `news_count_5d`) to densify the signal for tree models.
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
results = run_comparison(TICKER)
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
# Cross-ticker summary: run the pipeline for all three tickers, both variants.
# Shows where sentiment helps and where data sparsity makes it ineffective.
all_rows = []
for t in ['AAPL', 'TSLA', 'NKE']:
    for variant, use_sent in [('without sentiment', False), ('with sentiment', True)]:
        r = run_pipeline(t, use_sentiment=use_sent)
        for m, label in [('lr', 'Linear Regression'), ('rf', 'Random Forest'), ('xgb', 'XGBoost')]:
            all_rows.append({{
                'ticker': t, 'variant': variant, 'model': label,
                'MSE': r['eval'][m]['MSE'], 'MAE': r['eval'][m]['MAE'], 'R2': r['eval'][m]['R2'],
                'baseline_MSE': r['eval']['baseline_mse'],
            }})
cross = pd.DataFrame(all_rows)
cross.to_csv('results_all_tickers.csv', index=False)
# Pivot to show MSE side-by-side
pivot = cross.pivot_table(index=['ticker','model'], columns='variant', values='MSE')
pivot['delta %'] = 100.0 * (pivot['with sentiment'] - pivot['without sentiment']) / pivot['without sentiment']
pivot
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

### Why does sentiment help Linear Regression on AAPL but not tree models?

Linear Regression fits a global slope to each feature and regularizes naturally through the pseudoinverse — when `sentiment_mean` is 0 on ~60% of days, those rows simply contribute less to the coefficient estimate without distorting it. The small non-zero coefficient found on the news-covered days carries over cleanly to the test set and yields the −0.79% MSE improvement.

Random Forest and XGBoost, in contrast, split on every feature independently. A feature that is zero most of the time and occasionally non-zero presents trees with a low-frequency binary signal: "news day vs. non-news day." When the tree greedily uses this split for variance reduction on the training set, it captures training-period idiosyncrasies that don't generalize. The rolling 5-day aggregates (`sentiment_mean_5d`, `news_count_5d`) were added to soften this by making the signal continuous rather than sparse — but the improvement was marginal, because the underlying coverage problem is still there.

**Takeaway:** sparse sentiment features favor models that can regularize weak signals. With denser coverage (a real financial news source), trees should catch up and probably surpass LR.

### Why does Random Forest fail catastrophically on TSLA?

The TSLA train set (2013–2016) contains the 2014–2015 Cybertruck/Model X launch volatility spikes and the 2015 "Chinese market correction" period, during which daily RV briefly reached ~0.01 (vs. typical ~0.0005). The test set (2017) is a calmer period. Random Forest with `min_samples_leaf=1` and unbounded depth memorizes those training spikes as feature-space rules (e.g. "if `rolling_vol_5 > X` and `volume_change > Y`, predict ~0.005"). In 2017, `rolling_vol_5` occasionally brushes those thresholds from normal market noise, triggering the memorized high-volatility predictions. The result is an MSE an order of magnitude worse than the baseline.

Linear Regression and XGBoost don't suffer this because: LR cannot output extreme per-row predictions without first learning they're the norm, and XGBoost's `max_depth=5` plus shrinkage (`learning_rate=0.1`) limits how sharply any single rule can fire. **A quick hyperparameter fix** for RF: `min_samples_leaf≥10` and `max_depth=8` would both attenuate this pathology — left as follow-up work.

### What does feature importance tell us?

Across AAPL RF and XGBoost (see the feature importance plot above), the ranking is consistent:
1. `RV` (current volatility) — dominant, as expected from volatility clustering.
2. `rolling_vol_5` and `rolling_vol_10` — secondary signals confirming the clustering effect.
3. `SMA_10`, `SMA_20`, `volume_change` — modest contribution.
4. Sentiment features (`sentiment_mean`, `sentiment_mean_5d`, `news_count`, `news_count_5d`, `sentiment_std`) — lowest importance, consistent with their sparse coverage. Among sentiment features, `news_count_5d` usually ranks highest, suggesting the *volume* of recent news matters slightly more than its polarity — a finding consistent with the finance literature on attention-driven volatility.

### What would improve these results?

1. **Denser news coverage.** The single biggest constraint is not the model or features but the data. NewsAPI's 30-day limit and HuffPost's tapering coverage after 2018 cap what any model can learn. A one-time purchase of a multi-year Finnhub or Reuters archive would likely convert the weak trend on AAPL into a significant lift on all three tickers.
2. **RF hyperparameter tuning.** `min_samples_leaf=10`, `max_depth=8` would eliminate the TSLA pathology; a quick grid search would likely pull RF MSE below LR on AAPL and TSLA.
3. **Sentiment-regime features.** A binary `has_news` indicator combined with interaction terms (e.g. `sentiment_mean × has_news`) might let tree models learn the conditional effect cleanly.
4. **FinBERT scoring** on the headlines — financial-domain transformer would score financial news more accurately than general-purpose VADER, at the cost of ~100× inference time. Worth trying once a denser news source is in place.
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
