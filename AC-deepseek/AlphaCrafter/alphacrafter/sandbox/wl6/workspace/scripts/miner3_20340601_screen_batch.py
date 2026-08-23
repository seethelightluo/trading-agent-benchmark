"""miner_3 screen batch (2034-06-01), VIS 2034-05-31. Fresh interpretable candidates
for the 15-instrument cross-asset universe. Admission gate |IC|>=0.0070 and
|ICIR|>=0.0840 at horizon 10. Report recent-window instability. Prints dates/instruments.
"""
import sys, os
sys.path.insert(0, 'scripts')
from factor_validation_lib import rank_ic_series, align_fwd_returns, load_macro, TRADABLE
import pandas as pd, numpy as np, math

VIS = "2034-05-31"
closes = {}
for sym in TRADABLE:
    df = pd.read_csv(f"../persistent/stock_data/{sym}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= pd.Timestamp(VIS)].sort_values("date")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    closes[sym] = df.set_index("date")["close"]
px = pd.DataFrame(closes).ffill().dropna(how="all").dropna(axis=1, how="all")
ret = px.pct_change()
print("panel shape:", px.shape, "n assets:", px.shape[1])

def evalc(f, label):
    ic = rank_ic_series(f, align_fwd_returns(px, 10))
    if len(ic) == 0:
        print(f"[{label}] NO IC DATES"); return
    icm = float(ic.mean()); icstd = float(ic.std(ddof=1)) if len(ic)>1 else np.nan
    icir = icm/icstd if icstd and math.isfinite(icstd) and icstd>0 else np.nan
    hit = float((ic>0).mean())
    recent = ic[ic.index >= "2033-05-01"]
    ricm = float(recent.mean()) if len(recent) else np.nan
    ricir = ricm/recent.std(ddof=1) if len(recent)>2 and recent.std(ddof=1)>0 else np.nan
    cov = float(f.notna().mean().mean())
    gate = (abs(icm) >= 0.0070) and (abs(icir) >= 0.0840)
    print(f"[{label}] n_ic={len(ic)} IC={icm:+.4f} ICIR={icir:+.4f} hit={hit:.3f} "
          f"recent1y_IC={ricm:+.4f} recent_ICIR={ricir:+.4f} cov={cov:.3f} GATE={'PASS' if gate else 'fail'}")

cands = {}

# A. Dollar beta (sensitivity to DXY): low-downside carry proxy.
dxy = load_macro("DXY", VIS).reindex(px.index).ffill()
dxret = dxy.pct_change()
dxybeta = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
for a in px.columns:
    dxybeta[a] = ret[a].rolling(60).corr(dxret)
cands["dxy_beta_60"] = dxybeta

# B. Rate beta to US10Y (tradable series): growth/duration sensitivity
us10 = px["US10Y"]
tenret = us10.pct_change()
ratebeta = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
for a in px.columns:
    ratebeta[a] = ret[a].rolling(60).corr(tenret)
cands["rate_beta_us10y_60"] = ratebeta

# C. Trend distance from 200d mean
ma200 = px.rolling(200).mean()
cands["dist_ma200"] = px/ma200 - 1.0

# D. 52-week high proximity
hi = px.rolling(252).max()
cands["dist_high_252"] = px/hi - 1.0

# E. Vol-scaled momentum: 20d mom / 60d vol
mom20 = px/px.shift(20) - 1.0
rv60 = ret.rolling(60).std()
cands["mom20_vol_scaled"] = mom20 / rv60.replace(0, np.nan)

# F. Downside deviation ratio of 60d
down_vol = ret.copy(); down_vol[down_vol > 0] = 0.0
dv = down_vol.rolling(60).std(); tot_v = ret.rolling(60).std()
cands["downside_vol_ratio"] = dv/tot_v.replace(0, np.nan)

# G. Cross-sectional demeaned 20d relative momentum
rel_mom = mom20 - mom20.median(axis=1)
cands["rel_mom_20_demean"] = rel_mom

# H. 5d momentum skip 3d
cands["mom_5d_skip3"] = px.shift(3)/px.shift(8) - 1.0

for name, f in cands.items():
    evalc(f, name)