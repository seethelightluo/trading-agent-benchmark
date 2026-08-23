"""miner_3 screen batch (2034-09-14), VIS 2034-09-13. Fresh candidates for the
15-instrument cross-asset universe plus macro observation series.
Admission gate: |IC|>=0.0070 and |ICIR|>=0.0840 at horizon 10.
Report recent (last 2y) instability too. Prints n dates/instruments used.
"""
import sys, os
sys.path.insert(0, 'scripts')
from factor_validation_lib import rank_ic_series, align_fwd_returns, load_macro, TRADABLE
import pandas as pd, numpy as np, math

VIS = "2034-09-13"
# load tradable panel directly from CSVs (like simulator reads)
closes = {}
for sym in TRADABLE:
    df = pd.read_csv(f"../persistent/stock_data/{sym}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= pd.Timestamp(VIS)].sort_values("date")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    closes[sym] = df.set_index("date")["close"]
px = pd.DataFrame(closes).ffill()
# drop leading all-NaN
px = px.dropna(how="all").dropna(axis=1, how="all")
ret = px.pct_change()
print("panel shape:", px.shape, "n assets:", px.shape[1])

def evalc(f, label):
    ic = rank_ic_series(f, align_fwd_returns(px, 10))
    if len(ic) == 0:
        print(f"[{label}] NO IC DATES"); return
    icm = float(ic.mean()); icstd = float(ic.std(ddof=1)) if len(ic)>1 else np.nan
    icir = icm/icstd if icstd and math.isfinite(icstd) and icstd>0 else np.nan
    hit = float((ic>0).mean())
    recent = ic[ic.index >= "2032-09-01"]
    ricm = float(recent.mean()) if len(recent) else np.nan
    ricir = ricm/recent.std(ddof=1) if len(recent)>2 and recent.std(ddof=1)>0 else np.nan
    cov = float(f.notna().mean().mean())
    gate = (abs(icm) >= 0.0070) and (abs(icir) >= 0.0840)
    print(f"[{label}] n_ic={len(ic)} IC={icm:+.4f} ICIR={icir:+.4f} hit={hit:.3f} "
          f"recent2y_IC={ricm:+.4f} recent_ICIR={ricir:+.4f} cov={cov:.3f} GATE={'PASS' if gate else 'fail'}")

cands = {}

# A. Residual momentum after removing SPX beta (idiosyncratic 20d)
spx = px["SPX"]
beta_spx = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
for a in px.columns:
    bx = ret[a].rolling(60).cov(ret["SPX"]) / ret["SPX"].rolling(60).var()
    beta_spx[a] = bx
res = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
for a in px.columns:
    res[a] = ret[a] - beta_spx[a] * ret["SPX"]
res_mom = res.rolling(20).sum() - res.shift(5).rolling(20).sum()  # skip last 5
cands["resid_mom_20_skip5"] = res_mom

# B. Volatility regime: vol ratio 5d/60d (mean-reversion when short vol spikes)
rv5 = ret.rolling(5).std(); rv60 = ret.rolling(60).std()
cands["vol_ratio_5x60"] = rv5/rv60.replace(0,np.nan)

# C. 60/40 style quality: asset vs SPX rolling spread Sharpe / downside capture
# downside beta neg vs SPX
down = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
for a in px.columns:
    m = ret[a].rolling(60).apply(lambda x: np.mean(x[x<0]), raw=True)
    down[a] = m
cands["downside_ret_60_neg"] = -down

# D. Range reversal: (high-low)/close percentile short-mean
# approximate using close range
hlr = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
for a in px.columns:
    d = px[a]
    rng = d.rolling(20).max()/d.rolling(20).min() - 1.0
    hlr[a] = rng
cands["range_20_neg"] = -hlr  # low recent range => expect continuation? test mean-rev

# E. Cross-section dispersion timing: per asset deviation from SPX in rank-trend
# Asset-relative momentum vs cross-section mean (long winners relative)
cs_mom = px.px if False else None
mom10 = px/px.shift(10)-1
cs_rank = mom10.rank(axis=1, pct=True)
cands["rel_rank_mom_10"] = cs_rank

# F. EURUSD beta (weak-dollar carry): assets rising when EUR rises
eur = load_macro("EURUSD", VIS).reindex(px.index).ffill()
emom = (eur/eur.shift(20)-1).reindex(px.index).ffill()
ebeta = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
for a in px.columns:
    ebeta[a] = ret[a].rolling(60).corr(emom)
cands["eur_beta_60"] = ebeta

for name, f in cands.items():
    evalc(f, name)