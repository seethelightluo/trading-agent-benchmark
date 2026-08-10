"""miner_1 2026-07-30: full validation of top candidate factors.

Uses the simulator-API dense panel (factor_validation_lib) so that rolling
statistics see each asset's own dense history (no macro-calendar reindex gaps).
Macro inputs (VIX) are cut at visible_through to avoid lookahead.

Prints full metric reports + pairwise correlations + library correlations.
"""
import sys
sys.path.insert(0, 'scripts')
import json
import numpy as np
import pandas as pd
from pathlib import Path
from factor_validation_lib import (load_panel, load_macro, ic_analysis,
                                   print_report, library_corr, rank_ic_series,
                                   align_fwd_returns, MIN_INSTR)

VISIBLE = '2026-07-29'  # visible_through from date.json

panel = load_panel(max_date=VISIBLE)
ret = panel.pct_change()
vix = load_macro('VIX', max_date=VISIBLE)
vixr = vix.pct_change()

# ---------------- candidate factor signals ----------------
C = {}

# momentum family
for lb in (10, 20, 120, 180, 250):
    C[f'mom_{lb}d_skip5'] = panel.shift(5) / panel.shift(5 + lb) - 1.0
C['risk_adj_mom_20d'] = (panel.shift(5) / panel.shift(25) - 1.0) / ret.rolling(20).std()
C['dist_sma_20d'] = panel / panel.rolling(20).mean() - 1.0
C['price_vs_52w_high'] = panel / panel.rolling(250).max() - 1.0

# vol family
vol20 = ret.rolling(20).std()
vol60 = ret.rolling(60).std()
C['inv_vol_20d'] = -vol20
C['vol_of_vol20x60'] = vol20.rolling(60).std()
C['vol_ratio_20x60'] = vol20 / vol60
C['downside_vol_ratio_20x60'] = ret.clip(upper=0).rolling(20).std() / ret.clip(upper=0).rolling(60).std()
C['skew_60d'] = ret.rolling(60).skew()
C['max_drawdown_60d'] = panel / panel.rolling(60).max() - 1.0

# macro-conditional
def beta_of(a, m, win):
    return a.rolling(win).cov(m) / m.rolling(win).var()
C['vix_beta_cond_60x20'] = -beta_of(ret, vixr, 60) * (vix / vix.shift(20) - 1.0)
C['vix_beta_60d'] = beta_of(ret, vixr, 60)

# liquidity / illiquidity
C['amihud_20d'] = (ret.abs() / panel).rolling(20).mean()

# ---------------- library signals (existing effective factor defs) ----------------
lib = {}
lib['mom_10d_skip5'] = panel.shift(5) / panel.shift(15) - 1.0
lib['mom_120d_skip5'] = panel.shift(5) / panel.shift(125) - 1.0
lib['vol_of_vol20x60'] = ret.rolling(20).std().rolling(60).std()
lib['vix_beta_cond_60x20'] = -beta_of(ret, vixr, 60) * (vix / vix.shift(20) - 1.0)

# ---------------- full IC/ICIR/decay/turnover/coverage reports ----------------
print('=' * 100)
print(f'FULL VALIDATION  (visible_through={VISIBLE}, panel {panel.shape}, '
      f'assets={panel.shape[1]}, dates={panel.shape[0]})')
print('=' * 100)
results = {}
for name, f in C.items():
    res = ic_analysis(f, panel, horizon=10, label=name)
    results[name] = res
    print_report(res)
    # sub-period robustness
    ic10 = rank_ic_series(f, align_fwd_returns(panel, 10))
    ic10 = ic10.dropna()
    for lo, hi, tag in [('2020-01-01', '2022-12-31', '20-22'), ('2023-01-01', '2026-07-29', '23-26')]:
        sub = ic10[(ic10.index >= lo) & (ic10.index <= hi)]
        if len(sub):
            print(f'    subperiod {tag}: ic={sub.mean():.4f} n={len(sub)}')
    print()

# ---------------- library correlation audit ----------------
print('--- max |rho| vs existing library signals (pairwise, per-date avg) ---')
for name in C:
    rho = library_corr(C[name], lib)
    print(f'  {name:<28} max_abs_library_corr={rho:.4f}')

# ---------------- pairwise correlation among candidates ----------------
print('\n--- pairwise |rho| among candidates (avg per-date rank corr) ---')
names = list(C.keys())
corr = pd.DataFrame(index=names, columns=names, dtype=float)
for i, a in enumerate(names):
    for j, b in enumerate(names):
        if j <= i:
            continue
        fa, fb = C[a].rank(axis=1, pct=True), C[b].rank(axis=1, pct=True)
        rhos = []
        for d in fa.index.intersection(fb.index):
            x, y = fa.loc[d], fb.loc[d]
            m = x.notna() & y.notna()
            if m.sum() >= MIN_INSTR:
                r, _ = __import__('scipy.stats').spearmanr(x[m].values, y[m].values)
                rhos.append(r)
        corr.loc[a, b] = corr.loc[b, a] = float(np.mean(rhos)) if rhos else np.nan
np.fill_diagonal(corr.values, 1.0)
print(corr.round(3).to_string())
