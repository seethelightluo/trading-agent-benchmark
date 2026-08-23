"""miner_3 screen batch (2034-05-01), VIS 2034-04-24. Fresh candidates for the
15-instrument cross-asset universe plus macro observation series.
Admission gate: |IC|>=0.0070 and |ICIR|>=0.0840 at horizon 10.
Prints n dates/instruments used; reports recent-2y instability too.
"""
import sys, os
sys.path.insert(0, 'scripts')
from factor_validation_lib import rank_ic_series, align_fwd_returns, load_macro, TRADABLE
import pandas as pd, numpy as np, math

VIS = "2034-04-24"
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
    recent = ic[ic.index >= "2032-05-01"]
    ricm = float(recent.mean()) if len(recent) else np.nan
    ricir = ricm/recent.std(ddof=1) if len(recent)>2 and recent.std(ddof=1)>0 else np.nan
    cov = float(f.notna().mean().mean())
    gate = (abs(icm) >= 0.0070) and (abs(icir) >= 0.0840)
    print(f"[{label}] n_ic={len(ic)} IC={icm:+.4f} ICIR={icir:+.4f} hit={hit:.3f} "
          f"recent2y_IC={ricm:+.4f} recent_ICIR={ricir:+.4f} cov={cov:.3f} GATE={'PASS' if gate else 'fail'}")

cands = {}

spx = px["SPX"]
# A. Residual momentum after removing SPX beta (idiosyncratic 20d, skip5)
beta_spx = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
for a in px.columns:
    beta_spx[a] = ret[a].rolling(60).cov(ret["SPX"]) / ret["SPX"].rolling(60).var()
resid = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
for a in px.columns:
    resid[a] = ret[a] - beta_spx[a] * ret["SPX"]
cands["resid_mom_20_skip5"] = resid.rolling(20).sum() - resid.shift(5).rolling(20).sum()

# B. Vol regime ratio 5d/60d (mean-reversion when short vol spikes)
rv5 = ret.rolling(5).std(); rv60 = ret.rolling(60).std()
cands["vol_ratio_5x60"] = rv5/rv60.replace(0,np.nan)

# C. Downside mean return over 60d (negated)
down = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
for a in px.columns:
    down[a] = ret[a].rolling(60).apply(lambda x: np.mean(x[x<0]) if (x<0).sum()>0 else 0, raw=True)
cands["downside_ret_60_neg"] = -down

# D. Momentum-consistent quality: 120d momentum divided by 60d vol (risk-adjusted trend)
mom120 = px/px.shift(120)-1
vol60 = ret.rolling(60).std().replace(0,np.nan)
cands["mom120_div_vol60"] = mom120/vol60

# E. Upward capture ratio: mean positive over 120d / mean negative over 120d
upcap = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
for a in px.columns:
    def cap(x):
        pos = x[x>0]
        neg = x[x<0]
        mp = pos.mean() if len(pos) else np.nan
        mn = -neg.mean() if len(neg) else np.nan
        return mp/mn if mn and mp==mp else np.nan
    upcap[a] = ret[a].rolling(120).apply(cap, raw=True)
cands["updown_capture_120"] = upcap

# F. DXY beta (inverse dollar carry) - assets rising when USD weak
dxy = load_macro("DXY", VIS).reindex(px.index).ffill()
dxy_ret = dxy.pct_change()
bdy = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
for a in px.columns:
    bdy[a] = ret[a].rolling(60).corr(dxy_ret)
cands["dxy_beta_60_neg"] = -bdy

# G. Correlation to CN10Y (yield-sensitivity), high yield beta
cny = px["CN10Y"].pct_change()
bcn = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
for a in px.columns:
    bcn[a] = ret[a].rolling(60).corr(cny)
cands["cn10y_beta_60"] = bcn

# H. Cross-section persistence: avg sign of last 10d returns (trend consistency)
sign10 = np.sign(ret).rolling(10).mean().fillna(0)
cands["sign_persistence_10"] = sign10

for name, f in cands.items():
    evalc(f, name)