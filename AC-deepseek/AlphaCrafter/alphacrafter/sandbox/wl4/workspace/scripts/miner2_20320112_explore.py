"""miner2_20320112: explore candidate factor ideas (screen).
Tests novel cross-asset factor constructions NOT already in library:
1. trend_strength_60d  : (close - sma60)/ (atr-like range) -> trend persistence
2. down_vol_ratio_20d  : downside deviation / upside deviation (vol asymmetry)
3. roll_skew_20d       : rolling skewness of 20d returns (crash-risk)
4. hi_lo_pos_20d       : (close - min20)/(max20 - min20) - stochastic position
5. us10y_mom_60d       : US10Y yield momentum as conditioning on assets? -> instead per-asset close vs 60d ago for rates
6. xret_vs_median_20d  : excess return vs cross-sectional median (relative momentum)
7. vol_ratio_5_60      : short vol / long vol (regime)
8. dxy_beta_60d        : rolling beta to DXY (USD sensitivity)
9. vix_level_z_60d     : VIX z-score conditioned? -> vix beta instead
10. ret_autocorr_10d   : serial correlation of returns (reversal/continuation)
"""
import sys, json
sys.path.insert(0, "scripts")
from miner2_20320112_validator import load_panel, forward_returns, daily_ic, full_metrics
import numpy as np
import pandas as pd

px, mx = load_panel()
ret = px.pct_change()
fwd = forward_returns(px)

def ic_summary(factor_df, label):
    m = full_metrics(factor_df, fwd, min_valid=8)
    h10 = m["horizons"]["10"]
    h5 = m["horizons"]["5"]
    print(f"{label:28s} h5 ic={h5['ic']:+.4f} icir={h5['icir']:+.3f} | h10 ic={h10['ic']:+.4f} icir={h10['icir']:+.3f} hit={h10['hit']:.2f} n={h10['n']:4d} cov_ad={m['coverage_asset_days']:.3f} cov_d8={m['coverage_dates_ge8']:.3f} to={m['turnover_10d_rank']:.2f}")
    return m

cands = {}

# 1. trend strength: (close - sma60)/std60
sma60 = px.rolling(60).mean()
std60 = px.rolling(60).std()
cands["trend_strength_60d"] = (px - sma60) / std60

# 2. downside vol ratio: downside dev / upside dev over 20d
pos = ret.clip(lower=0)
neg = ret.clip(upper=0)
upside = pos.rolling(20).std()
downside = neg.rolling(20).std()
cands["down_vol_ratio_20d"] = downside / upside

# 3. rolling skewness 20d
cands["roll_skew_20d"] = ret.rolling(20).skew()

# 4. hi-lo position 20d (stochastic)
min20 = px.rolling(20).min()
max20 = px.rolling(20).max()
cands["hi_lo_pos_20d"] = (px - min20) / (max20 - min20)

# 5. rate momentum: yield close momentum 20d (rates as tradable, momentum continuation)
cands["rate_mom_20d"] = px.pct_change(20)

# 6. excess return vs cross-sectional median 20d
med20 = px.rolling(20).median().median(axis=1)
cands["xret_vs_median_20d"] = (px / px.shift(20) - 1) - (px.rolling(20).median().median(axis=1) / px.shift(20).rolling(20).median().median(axis=1) - 1).to_frame().values if False else (px / px.shift(20) - 1)
# simpler: demeaned momentum
mom20 = px.pct_change(20)
cands["xret_vs_median_20d"] = mom20.sub(mom20.median(axis=1), axis=0)

# 7. vol ratio 5/60
vol5 = ret.rolling(5).std()
vol60 = ret.rolling(60).std()
cands["vol_ratio_5_60"] = vol5 / vol60

# 8. DXY beta 60d: rolling regression slope of asset ret on DXY ret
dxy_ret = mx["DXY"].pct_change()
beta = {}
for s in px.columns:
    df = pd.concat([ret[s], dxy_ret], axis=1).dropna()
    cov = df[s].rolling(60).cov(df["DXY"])
    var = df["DXY"].rolling(60).var()
    beta[s] = cov / var
cands["dxy_beta_60d"] = pd.DataFrame(beta, index=px.index)

# 9. VIX beta 60d (risk sensitivity)
vix_ret = mx["VIX"].pct_change()
beta_v = {}
for s in px.columns:
    df = pd.concat([ret[s], vix_ret], axis=1).dropna()
    cov = df[s].rolling(60).cov(df["VIX"])
    var = df["VIX"].rolling(60).var()
    beta_v[s] = cov / var
cands["vix_beta_60d"] = pd.DataFrame(beta_v, index=px.index)

# 10. return autocorrelation 10d
def _acf(x):
    if len(x) < 5: return np.nan
    a, b = x[:-1], x[1:]
    if np.std(a) == 0 or np.std(b) == 0: return 0.0
    return float(np.corrcoef(a, b)[0, 1])
cands["ret_autocorr_10d"] = ret.rolling(10).apply(_acf, raw=True)

print(f"Panel: {px.shape[0]} dates x {px.shape[1]} assets (through 2032-01-12)")
print("=" * 120)
results = {}
for name, f in cands.items():
    results[name] = ic_summary(f, name)
