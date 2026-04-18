"""Extract convergence curve summary at key bar counts (1, 20, 40, 80, 120, 200)."""
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

p = Path('runs/2026-04-17_diagnostic_deep_dive/check4_convergence_results.pkl')
with open(p, 'rb') as f:
    results = pickle.load(f)

rows = []
checkpoints = [1, 10, 20, 40, 60, 80, 120, 160, 200]
for name, r in results.items():
    if r is None:
        continue
    med = r['median_curve']
    p90 = r['p90_curve']
    row = {'contract': name, 'n_sessions': r['n_sessions']}
    for cp in checkpoints:
        idx = cp - 1
        row[f'med_{cp}'] = med[idx] if idx < len(med) else np.nan
        row[f'p90_{cp}'] = p90[idx] if idx < len(p90) else np.nan
    rows.append(row)

df = pd.DataFrame(rows)
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 40)
print(df.to_string(index=False))
df.to_csv(p.parent / 'check4_convergence_table.csv', index=False)
print(f"\nSaved: {p.parent / 'check4_convergence_table.csv'}")
