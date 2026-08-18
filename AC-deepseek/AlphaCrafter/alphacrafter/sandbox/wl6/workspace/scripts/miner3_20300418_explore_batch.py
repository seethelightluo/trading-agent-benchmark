"""miner_3 exploration 2030-04-18: batch screen of novel factor candidates.
Regime context (from memory): risk-off equity down, VIX elevated ~61, WTI crash,
N225 crash, SOX/SX5E/BTC recovery lead. Focus: trend-quality, risk-adjusted
momentum, cross-asset betas, downside sensitivity, relative strength vs refs.
"""
import sys, time
sys.path.insert(0, "scripts")
from miner3_20300418_common import *

t0 = time.time()
px, rets = load_asset_panel()
print(f"Panel: {px.shape[0]} dates x {px.shape[1]} assets, {px.index[0].date()}..{px.index[-1].date()} [{time.time()-t0:.1f}s]")

t0 = time.time()
fwd = build_fwd_returns(rets, horizons=(5, 10, 20))
print(f"fwd rets [{time.time()-t0:.1f}s]")

def roll_beta(ret_asset, ret_ref, win, minp):
    cov = ret_asset.rolling(win, min_periods=minp).cov(ret_ref)
    var = ret_ref.rolling(win, min_periods=minp).var()
    return cov / var

t0 = time.time()
factors = {}
P = px[TRADABLE]
R = rets[TRADABLE]
vol20 = R.rolling(20, min_periods=10).std()
vol60 = R.rolling(60, min_periods=30).std()

# 1. pullback depth from 20d high (normalized drawdown)
rmax20 = P.rolling(20, min_periods=10).max()
factors["pullback_20d"] = P / rmax20 - 1.0

# 2. upside/downside capture ratio 20d (trend quality)
pos = R.clip(lower=0).rolling(20, min_periods=10).sum()
neg = R.clip(upper=0).rolling(20, min_periods=10).sum()
factors["updown_capture_20d"] = pos / (neg.abs() + 1e-12)

# 3. vol-adjusted momentum 20d (Sharpe-like)
factors["vol_adj_mom_20d"] = R.rolling(20, min_periods=10).sum() / vol20

# 4. vol-adjusted momentum 60d
factors["vol_adj_mom_60d"] = R.rolling(60, min_periods=30).sum() / vol60

# 5. momentum acceleration: 60d momentum minus 20d momentum
factors["mom_accel_60_20"] = R.rolling(60, min_periods=30).sum() - R.rolling(20, min_periods=10).sum()

# 6. beta to US10Y (rate sensitivity)
r_us10y = rets["US10Y"]
factors["us10y_beta_60d"] = roll_beta(R, r_us10y, 60, 30)

# 7. correlation with XAU (risk-off alignment)
factors["xau_corr_60d"] = R.rolling(60, min_periods=30).corr(rets["XAU"])

# 8. beta to BTC (crypto leadership alignment)
factors["btc_beta_60d"] = roll_beta(R, rets["BTC"], 60, 30)

# 9. max daily gain in 20d (upside burst)
factors["max_gain_20d"] = R.rolling(20, min_periods=10).max()

# 10. kaufman efficiency ratio 60d (trend linearity)
dist60 = (P - P.shift(60)).abs()
path60 = R.abs().rolling(60, min_periods=30).sum()
factors["kaufman_eff_60d"] = dist60 / (path60 + 1e-12)

# 11. downside beta to SPX (defensive character)
r_spx = rets["SPX"]
down = r_spx < 0
r_spx_d = r_spx.where(down, 0.0)
r_d = R.where(down, 0.0)
cov_d = (r_d * r_spx_d).rolling(60, min_periods=20).mean() - r_d.rolling(60, min_periods=20).mean() * r_spx_d.rolling(60, min_periods=20).mean()
var_d = (r_spx_d ** 2).rolling(60, min_periods=20).mean() - r_spx_d.rolling(60, min_periods=20).mean() ** 2
factors["downside_beta_spx_60d"] = cov_d / (var_d + 1e-12)

# 12. skewness 60d (tail asymmetry, longer window than lib skew_20d_neg)
factors["skew_60d_neg"] = -R.rolling(60, min_periods=30).skew()

# 13. relative strength vs BTC 20d
factors["relstr_20d_vs_btc"] = R.rolling(20, min_periods=10).sum() - rets["BTC"].rolling(20, min_periods=10).sum()

# 14. relative vol: asset vol20 / cross-sectional median vol20
med_vol20 = vol20.median(axis=1)
factors["relvol_20d"] = vol20.div(med_vol20, axis=0)

# 15. positive-day ratio 20d (trend consistency)
factors["pos_day_ratio_20d"] = (R > 0).rolling(20, min_periods=10).mean()

print(f"factors computed [{time.time()-t0:.1f}s]")

t0 = time.time()
for name, sig in factors.items():
    res = evaluate_factor(sig, fwd, label=name)
    h10 = res["h10"]
    if h10 is None:
        print(f"{name:24s} NO VALID IC")
        continue
    rec = res.get("recent_h10", {})
    last6 = res.get("last6m_h10", {})
    print(f"{name:24s} IC10={h10['ic']:+.4f} ICIR10={h10['icir']:+.4f} hit={h10['hit']:.3f} n={h10['n_dates']:4d} "
          f"cov={res['coverage_asset_days']:.2f}/{res['coverage_dates_ge8']:.2f} | "
          f"IC5={res.get('h5',{}).get('ic',float('nan')):+.4f} IC20={res.get('h20',{}).get('ic',float('nan')):+.4f} | "
          f"rec28+ IC={rec.get('ic',float('nan')):+.4f} ICIR={rec.get('icir',float('nan')):+.4f} | "
          f"last6m IC={last6.get('ic',float('nan')):+.4f} ICIR={last6.get('icir',float('nan')):+.4f}")
print(f"eval done [{time.time()-t0:.1f}s]")
