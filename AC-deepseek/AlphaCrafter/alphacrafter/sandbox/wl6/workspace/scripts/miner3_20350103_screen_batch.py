"""miner3 screen batch 2035-01-03, VIS 2035-01-02. Fresh candidates.
Admission gate: |IC|>=0.0070 and |ICIR|>=0.0840 at horizon 10.
Also reports recent-2y IC drift and per-date instrument count (>=8 rule).
"""
import sys, os
sys.path.insert(0, 'scripts')
from factor_validation_lib import rank_ic_series, align_fwd_returns, load_macro, TRADABLE
import pandas as pd, numpy as np, math

VIS = "2035-01-02"
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

# A. Residual 40d momentum after removing SPX beta, skip last 5 (idiosyncratic carry)
spxr = ret["SPX"]
beta = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
for a in px.columns:
    beta[a] = ret[a].rolling(60).cov(spxr) / spxr.rolling(60).var()
res = ret - beta * spxr
cands["resid_mom_40_skip5"] = res.rolling(40).sum() - res.shift(5).rolling(40).sum()

# B. Intra-asset vol-of-vol change 40d vs 10d (regime shift)
rv10 = ret.rolling(10).std(); rv40 = ret.rolling(40).std()
vov = (rv40.shift(1) / rv10.shift(1)).replace(0, np.nan)
cands["vol_ratio_40x10"] = vov

# C. 5d reversal (mean reversion short-term)
cands["reversal_5"] = -(px / px.shift(5) - 1)

# D. Cross-section relative momentum: asset 20d - cross median 20d (winner continuation)
mom20 = px / px.shift(20) - 1
cands["rel_mom_20_vs_median"] = mom20.sub(mom20.median(axis=1), axis=0)

# E. Asset distance-from-peak: 1 - close/rolling_max(close,120) (pull-to-mech mean rev)
cands["dist_peak_120_neg"] = -(px / px.rolling(120, min_periods=40).max() - 1.0)

# F. USDCNY beta 90d (CNY-risk linked assets lag)
cny = load_macro("USDCNY", VIS).reindex(px.index).ffill()
cm = (cny / cny.shift(20) - 1).reindex(px.index).ffill()
cbeta = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
for a in px.columns:
    cbeta[a] = ret[a].rolling(90).corr(cm)
cands["cny_beta_90_neg"] = -cbeta

# G. Cross-section skew of 40d ret normalized (asymmetric payoff timing)
sk = ret.rolling(40).skew()
cands["skew_60"] = ret.rolling(60).skew()

for name, f in cands.items():
    evalc(f, name)