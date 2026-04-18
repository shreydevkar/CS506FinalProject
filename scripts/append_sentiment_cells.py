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

**Data constraint**: NewsAPI's free tier returns roughly the last 30 days of headlines, so sentiment features are populated only for the most recent ~month of the price series. Earlier dates receive neutral sentiment (mean=0, std=0, news_count=0). This limits the signal available to the models but the framework is correct and will scale to a longer-history sentiment source.
"""),
    code(f"""{SENTINEL}
import sys
sys.path.append('..')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from main import run_comparison

TICKER = 'TSLA'
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
    nb = json.loads(NB_PATH.read_text())
    # Strip any previously-appended sentinel cells to stay idempotent
    kept = []
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if SENTINEL in src:
            continue
        kept.append(cell)
    nb["cells"] = kept + NEW_CELLS
    NB_PATH.write_text(json.dumps(nb, indent=1))
    print(f"Appended {len(NEW_CELLS)} sentiment-comparison cells to {NB_PATH.name}")


if __name__ == "__main__":
    main()
