# -*- coding: utf-8 -*-
"""miner_3 2027-10-07: quick data availability check."""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
import miner3_lib as L

C, V, H, Lw, O = L.load_close_panel(4000)
R = C.pct_change()
print("Panel dates:", C.index.min().date(), "->", C.index.max().date(), "| n_dates:", len(C), "| n_assets:", C.shape[1])
print("Last 5 close dates:")
print(C.index[-5:])
print("\nPer-asset last close / n_obs / has_volume / has_highlow:")
for s in C.columns:
    v_ok = (V[s] > 0).sum() > 10
    hl_ok = H[s].notna().sum() > 10 and Lw[s].notna().sum() > 10
    print(f"  {s:10s} last={str(C[s].dropna().index[-1].date())} n={C[s].notna().sum():5d} vol={v_ok} hl={hl_ok}")

# macro index files
import os
for name in ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']:
    p = f'../persistent/index_data/{name}.csv'
    if os.path.exists(p):
        df = pd.read_csv(p, parse_dates=['date'])
        df['date'] = pd.to_datetime(df['date']).dt.normalize()
        print(f"macro {name}: {df['date'].min().date()} -> {df['date'].max().date()} n={len(df)}")
    else:
        print(f"macro {name}: MISSING")

# library factors with artifacts
import glob, json
print("\nLibrary factors:")
for p in sorted(glob.glob('factors/*.json')):
    if os.path.basename(p) == 'factor_ensemble.json':
        continue
    d = json.load(open(p))
    art = d.get('validation', {}).get('signal_artifact')
    print(f"  {d.get('factor_id'):25s} status={d.get('validation',{}).get('status'):10s} artifact={'Y' if art else 'N'} last_validated={d.get('last_validated')}")
