"""miner3 screen batch 2035-05-08, VIS 2035-05-08. Fresh candidates for low-VIX risk-on regime.
Admission gate: |IC|>=0.0070 and |ICIR|>=0.0840 at horizon 10.
Also report recent-2y IC drift, coverage, turnover.
"""
import sys, os, math
sys.path.insert(0, 'scripts')
from factor_validation_lib import rank_ic_series, align_fwd_returns, load_macro, TRADABLE
import pandas as pd, numpy as np

VIS = "2035-06-05"
closes = {}
for sym in TRADABLE:
    df = pd.read_csv(f"../persistent/stock_data/{sym}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= pd.Timestamp(VIS)].sort_values("date")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    closes[sym] = df.set_index("date")["close"]
px = pd.DataFrame(closes).ffill().dropna(how="all").dropna(axis=1, how="all")
ret = px.pct_change()
print("panel shape:", px.shape, "n assets:", px.shape[1], "date:", px.index.min().date(), "->", px.index.max().date())


def evalc(f, label):
    ic = rank_ic_series(f, align_fwd_returns(px, 10))
    if len(ic) == 0:
        print(f"[{label}] NO IC DATES"); return None
    icm = float(ic.mean()); icstd = float(ic.std(ddof=1)) if len(ic) > 1 else np.nan
    icir = icm / icstd if icstd and math.isfinite(icstd) and icstd > 0 else np.nan
    hit = float((ic > 0).mean())
    recent = ic[ic.index >= "2033-01-01"]
    ricm = float(recent.mean()) if len(recent) else np.nan
    ricir = ricm / recent.std(ddof=1) if len(recent) > 2 and recent.std(ddof=1) > 0 else np.nan
    cov_ad = float(f.notna().mean().mean())
    cov_d8 = float((f.notna().sum(axis=1) >= 8).mean())
    turn = float(f.rank(axis=1, pct=True).diff().abs().mean().mean()) if len(f) > 2 else np.nan
    gate = (abs(icm) >= 0.0070) and (abs(icir) >= 0.0840)
    flag = "PASS" if gate else "fail"
    print(f"[{label}] n_ic={len(ic)} IC={icm:+.4f} ICIR={icir:+.4f} hit={hit:.3f} "
          f"recentIC={ricm:+.4f} ricir={ricir:+.4f} covAD={cov_ad:.3f} covD8={cov_d8:.3f} "
          f"turn={turn:.4f} GATE={flag}")
    return dict(ic=icm, icir=icir, hit=hit, ricm=ricm, ricir=ricir, cov=cov_ad, turn=turn)


cands = {}

vix = load_macro("VIX", VIS).reindex(px.index).ffill()
vix_d20 = (vix.shift(5)/vix.shift(25) - 1).reindex(px.index)  # VIX change 25->5d ago (falling = negative)
mom20 = px/px.shift(20) - 1

# A. VIX-decline-conditional momentum (risk-on carry)
cands["vixdecl_mom20_gate"] = mom20 * (-np.clip(vix_d20, -1, 1))
cands["vixdecl_mom40_gate"] = (px/px.shift(40)-1) * (-np.clip(vix_d20, -1, 1))

# B. Momentum normalized by own vol (risk-adjusted carry)
mom30 = px/px.shift(30) - 1
vol30 = ret.rolling(30).std()
cands["mom30_div_vol"] = mom30 / vol30.shift(1)

# C. Idiosyncratic residual 30d momentum after removing SPX beta
spxr = ret["SPX"]
beta = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
for a in px.columns:
    beta[a] = ret[a].rolling(60).cov(spxr) / spxr.rolling(60).var()
res = ret - beta * spxr
cands["resid_mom_30"] = res.rolling(30).sum()

# D. Vol-of-vol ratio 40x10
rv10 = ret.rolling(10).std(); rv40 = ret.rolling(40).std()
cands["vol_ratio_40x10"] = (rv40.shift(1) / rv10.shift(1)).replace(0, np.nan)

# E. Liquidity flow: 20d log-volume vs 60d
volpanel = {}
for sym in TRADABLE:
    df = pd.read_csv(f"../persistent/stock_data/{sym}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= pd.Timestamp(VIS)].sort_values("date")
    s = df.set_index("date")["volume"].astype(float)
    volpanel[sym] = s
vp = pd.DataFrame(volpanel).reindex(px.index).ffill()
lv20 = np.log(vp.replace(0, np.nan)).rolling(20).mean()
lv60 = np.log(vp.replace(0, np.nan)).rolling(60).mean()
cands["liq_flow_20x60"] = lv20 - lv60

# F. Yield momentum 60d neg (bond declining = risk-on environment)
us10y = px["US10Y"]; cn10y = px["CN10Y"]
cands["us10y_mom60_neg"] = -((us10y/us10y.shift(60)-1) * (cn10y/cn10y.shift(60)-1)).reindex(px.index)
cands["us10y_mom20_neg"] = -((us10y/us10y.shift(20)-1) * (cn10y/cn10y.shift(20)-1)).reindex(px.index)

for name, f in cands.items():
    evalc(f, name)