"""miner_2 screen batch (2034-03-28), VIS 2034-03-27. Fresh candidate factors
for the 15-instrument cross-asset universe plus macro observation series.
Admission gate (benchmark-wide): |IC|>=0.0070 and |ICIR|>=0.0840 at horizon 10.
Prints n dates/instruments used; reports recent-2y instability too.
"""
import sys
sys.path.insert(0, 'scripts')
from factor_validation_lib import rank_ic_series, align_fwd_returns, load_macro, TRADABLE
import pandas as pd, numpy as np, math

VIS = "2034-03-27"
closes = {}
for sym in TRADABLE:
    df = pd.read_csv(f"../persistent/stock_data/{sym}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= pd.Timestamp(VIS)].sort_values("date")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    closes[sym] = df.set_index("date")["close"]
px = pd.DataFrame(closes).ffill()
px = px.dropna(how="all").dropna(axis=1, how="all")
ret = px.pct_change()
print("panel shape:", px.shape, "n assets:", px.shape[1], "n dates:", px.shape[0])

def evalc(f, label):
    ic = rank_ic_series(f, align_fwd_returns(px, 10))
    if len(ic) == 0:
        print(f"[{label}] NO IC DATES"); return
    icm = float(ic.mean()); icstd = float(ic.std(ddof=1)) if len(ic)>1 else np.nan
    icir = icm/icstd if icstd and math.isfinite(icstd) and icstd>0 else np.nan
    hit = float((ic>0).mean())
    recent = ic[ic.index >= "2032-03-28"]
    ricm = float(recent.mean()) if len(recent) else np.nan
    ricir = ricm/recent.std(ddof=1) if len(recent)>2 and recent.std(ddof=1)>0 else np.nan
    cov = float(f.notna().mean().mean())
    gate = (abs(icm) >= 0.0070) and (abs(icir) >= 0.0840)
    print(f"[{label}] n_ic={len(ic)} IC={icm:+.4f} ICIR={icir:+.4f} hit={hit:.3f} "
          f"recent2y_IC={ricm:+.4f} recent_ICIR={ricir:+.4f} cov={cov:.3f} GATE={'PASS' if gate else 'fail'}")
    return (label, icm, icir)

cands = {}
spx = px["SPX"]

# A. Cross-asset momentum relative to cross-sectional average (broad-based momentum tilt)
mom10 = px/px.shift(10)-1
cs_avg = mom10.mean(axis=1)
cands["rel_mom10"] = mom10.sub(cs_avg, axis=0)

# B. Crash risk: max drawdown within last 20d (negated = high when benign)
def maxdd20(s):
    return (s / s.cummax() - 1.0).rolling(20).min()
md = pd.DataFrame({a: maxdd20(px[a]) for a in px.columns})
cands["crash_20d_neg"] = -md

# C. Sharpe momentum: 40d mean / 40d std (risk-adjusted trend)
mom40 = px/px.shift(40)-1
vol40 = ret.rolling(40).std().replace(0,np.nan)
cands["sharpe_mom40"] = mom40/vol40

# D. Vol of vol change: rising vol regime
cands["vol_of_vol_20d"] = ret.rolling(20).std().rolling(60).std()

# G. Range position: where close sits in 20d high/low range (trend strength)
rng = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
for a in px.columns:
    hi = px[a].rolling(20).max(); lo = px[a].rolling(20).min()
    rng[a] = (px[a]-lo)/(hi-lo).replace(0,np.nan)
cands["range_pos_20"] = rng

for name, f in cands.items():
    assert isinstance(f, pd.DataFrame), name
    evalc(f, name)