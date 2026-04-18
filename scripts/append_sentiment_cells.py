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
