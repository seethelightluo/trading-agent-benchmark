"""miner_2 2035-08-30 candidate: kurt_60 - rolling 60d excess kurtosis of daily returns.
Hypothesis: recent tail-heaviness (rare big moves) signals stress/regime change and may
predict lower forward returns (fat-tailed assets mean-revert)."""
import sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_common import load_prices, factor_to_panel, validate_factor, canonical_grid
import miner2_common as m2

prices = load_prices(days=4200)
grid = canonical_grid(prices)

def factor_fn(df, s):
    r = df['close'].pct_change()
    return r.rolling(60, min_periods=30).kurt()

panel = factor_to_panel(factor_fn, prices)
m = validate_factor('kurt_60', panel, prices)
if m is None:
    print('kurt_60: insufficient data'); sys.exit()
artifacts = m2.load_effective_artifacts()
rho, fid, details = m2.max_library_correlation(panel, artifacts, grid)
m['max_abs_library_correlation'] = rho
m['max_corr_library_id'] = fid
print(f"kurt_60 panel {panel.shape} range {panel.index.min().date()}..{panel.index.max().date()}")
print(json.dumps({k: v for k, v in m.items() if k != 'decay_ic_by_horizon'}, indent=1, default=str))
print('decay:', json.dumps(m['decay_ic_by_horizon'], default=str))
ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084 and rho < 0.5
print(f"ADMISSION: |IC|={abs(m['ic']):.4f}>=0.007 {abs(m['ic'])>=0.007} | |ICIR|={abs(m['icir']):.4f}>=0.084 {abs(m['icir'])>=0.084} | rho={rho:.3f}<0.5 {rho<0.5} -> {'PASS' if ok else 'FAIL'}")
print('top corr details:', {k: round(v, 3) for k, v in sorted(details.items(), key=lambda kv: -abs(kv[1]))[:4]})
