"""miner_1 2035-05-04: exploration screen of candidate factor families.
Computes factor values for ~12 novel candidates and evaluates 1d/5d/10d IC vs library correlation.
Not a persistence validation; used to prioritize deep dives.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import sys
sys.path.insert(0, 'scripts')
from miner3_eval_lib import load_panel, make_library_factors_full, eval_factor, print_eval

panel = load_panel() if False else pd.read_pickle('scripts/panel_cache_20350504.pkl')
px = panel['close']; ret = panel['ret']
hi = panel['high']; lo = panel['low']; op = panel['open']; vol = panel['vol']
mac = panel['macro']

lib = make_library_factors_full(panel)

cands = {}

# 1) Kaufman efficiency ratio 20d (trendiness): |C - C[20]| / sum|dC|
n = 20
cands['trend_eff_20'] = (px - px.shift(n)).abs() / ret.abs().rolling(n).sum()

# 2) realized skewness 60d (crash-risk)
def rskew(x):
    m = x.mean(); s = x.std()
    return ((x - m) ** 3).mean() / (s ** 3) if s > 0 else np.nan
cands['skew_60'] = ret.rolling(60).apply(rskew, raw=True)

# 3) DXY beta conditional 60x20 (parallel to vix_beta_cond_60x20)
dxy = mac['DXY'].reindex(px.index).ffill()
dxy_ret = dxy.pct_change()
betas = pd.DataFrame(index=px.index, columns=px.columns, dtype=float)
for i in range(60, len(ret)):
    a = ret.iloc[i-60:i]; b = dxy_ret.iloc[i-60:i]
    m = a.notna() & b.notna()
    if int(m.sum().sum()) < 10:
        continue
    aa = a[m]; bb = b[m]
    cov = (aa * bb).mean() - aa.mean() * bb.mean()
    var = bb.var()
    if var > 0:
        betas.iloc[i] = cov / var
dxy_trend = dxy_ret.rolling(20).mean()
cands['dxy_beta_cond_60x20'] = betas * np.sign(dxy_trend).values[:, None]

# 4) volume z-score 20d
volm = vol.rolling(20).mean(); vols = vol.rolling(20).std()
cands['volz_20'] = (vol - volm) / vols.replace(0, np.nan)

# 5) close location value avg 5d: mean((close-low)/(high-low))
rng = (hi - lo).replace(0, np.nan)
clv = (px - lo) / rng
cands['clv_avg_5'] = clv.rolling(5).mean()

# 6) overnight gap reversal 1d: -(open/prev_close - 1)
prev_close = px.shift(1)
gap = op / prev_close - 1.0
cands['gap_rev_1d'] = -gap

# 7) drawdown from 60d max (negative dd)
cands['maxdd_60'] = px / px.rolling(60).max() - 1.0

# 8) 20d time-series momentum with 5d skip
cands['mom_20d_skip5'] = px.shift(5) / px.shift(25) - 1.0

# 9) cross-sectional relative strength 20d (demeaned log return)
cands['relstrength_20'] = (np.log(px) - np.log(px.shift(20))) - (np.log(px) - np.log(px.shift(20))).mean(axis=1).values[:, None]

# 10) efficiency ratio 60d
cands['trend_eff_60'] = (px - px.shift(60)).abs() / ret.abs().rolling(60).sum()

# 11) realized kurtosis 60d
def rkurt(x):
    m = x.mean(); s = x.std()
    return ((x - m) ** 4).mean() / (s ** 4) if s > 0 else np.nan
cands['kurt_60'] = ret.rolling(60).apply(rkurt, raw=True)

# 12) downside semideviation ratio 60d (downside vol / total vol, inverted: low downside risk good)
def dsr(x):
    m = x.mean()
    dn = x[x < m]
    dv = np.sqrt((dn * dn).mean()) if len(dn) > 0 else 0.0
    tv = x.std()
    return dv / tv if tv > 0 else np.nan
cands['downside_ratio_60'] = ret.rolling(60).apply(dsr, raw=True)

print("=" * 100)
for name, f in cands.items():
    res = eval_factor(f, px, horizons=(1, 5, 10), min_valid=8, lib=lib)
    print_eval(name, res)
    print()
