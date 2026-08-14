"""miner_2 2035-03-09: scan new candidate factor families on the 15-name cross-asset panel.
Focus: families NOT already in library (reversal/nclv/mom120/vov/vixbeta).
Candidates:
  A. dxy_beta_60x20    - rolling 60d beta of asset daily ret vs DXY daily ret
  B. downside_skew_20d - rolling 20d skewness of daily returns (crash-risk dimension)
  C. vol_adj_mom_60x20 - 60d return / 20d realized vol (skip 5) risk-adjusted momentum
  D. eff_ratio_60d     - Kaufman efficiency ratio |ret_60| / sum(|daily ret|,60)
  E. drawdown_60d      - (close - rolling_max(close,60))/rolling_max(close,60)
  F. upper_wick_5d     - (2*close-high-low)/(high-low) averaged over 5d (upper shadow)
  G. amihud_20d        - mean(|ret|/volume) over 20d (illiquidity)
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'scripts')
from miner2_val_lib import fwd_ret, daily_rank_ic, summarize, WATCH

panel = pd.read_pickle('scripts/panel_cache_20350309.pkl')
close, high, low, vol, macro = panel['close'], panel['high'], panel['low'], panel['vol'], panel['macro']
ret = close.pct_change()

dxy = macro['DXY'].reindex(close.index).ffill()
dxy_ret = dxy.pct_change()

def roll_beta(y, x, w=60):
    out = pd.DataFrame(index=y.index, columns=y.columns, dtype=float)
    for s in y.columns:
        ys, xs = y[s], x
        c = ys.rolling(w).corr(xs)
        vx = xs.rolling(w).var()
        out[s] = c * (ys.rolling(w).std() / np.sqrt(vx))
    return out

signals = {}
# A. DXY beta 60x20
beta_dxy = roll_beta(ret, dxy_ret, 60)
signals['dxy_beta_60x20'] = beta_dxy
# B. downside skew 20d (negative skew => crash-prone)
skew = ret.rolling(20).skew()
signals['downside_skew_20d'] = -skew   # negate so high value = crash-prone? keep raw skew for scan both
signals['skew_20d_raw'] = skew
# C. vol-adjusted momentum
mom60 = close.shift(5) / close.shift(65) - 1.0
vol20 = ret.rolling(20).std()
signals['vol_adj_mom_60x20'] = mom60 / vol20
# D. efficiency ratio 60d
abs_sum = ret.abs().rolling(60).sum()
signals['eff_ratio_60d'] = (close / close.shift(60) - 1.0).abs() / abs_sum
# E. drawdown 60d
signals['drawdown_60d'] = close / close.rolling(60).max() - 1.0
# F. upper wick 5d (average close-location-in-bar inverted => high value = upper shadow)
bar = (2*close - high - low) / (high - low)
signals['upper_wick_5d'] = bar.rolling(5).mean()
# G. amihud illiquidity 20d
signals['amihud_20d'] = (ret.abs() / vol).rolling(20).mean()

print("="*100)
for name, sig in signals.items():
    res = {}
    for h in (1, 2, 3, 5, 10):
        fwd = fwd_ret(close, h)
        ics, dates = daily_rank_ic(sig, fwd, min_n=8)
        if len(ics):
            res[h] = dict(ic=float(np.mean(ics)), icir=float(np.mean(ics)/np.std(ics, ddof=1)),
                          hit=float(np.mean(ics>0)), n=len(ics))
    cov = float(sig.notna().mean().mean())
    print(f"[{name}] cov={cov:.3f} | " + " | ".join(
        f"h{h}: IC={res[h]['ic']:.4f} ICIR={res[h]['icir']:.3f} hit={res[h]['hit']:.3f} n={res[h]['n']}"
        for h in (1,2,3,5,10) if h in res))
