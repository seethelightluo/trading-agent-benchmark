"""miner3 screen v2 2035-06-06. Fresh candidates for low-VIX risk-on recovery regime.
Admission: |paper IC|>=0.0070 and |ICIR|>=0.0840 at horizon 10. Report recency drift too.
Builds factors as full-index DataFrames (assets x dates) so rank_ic_series works.
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
px = pd.DataFrame(closes).sort_index().ffill()
# restrict to common window starting 2021 for factor standardization
px = px[px.index >= "2021-01-01"]
ret = px.pct_change()
print("panel shape:", px.shape, "assets:", px.shape[1],
      "date:", px.index.min().date(), "->", px.index.max().date())


def build(f, label):
    """Ensure f is a DataFrame indexed like px with same columns."""
    if isinstance(f, pd.Series):
        f = pd.DataFrame({c: f for c in px.columns}, index=f.index)
    return f.reindex(index=px.index, columns=px.columns)


def evalc(f, label):
    f = build(f, label)
    ic = rank_ic_series(f, align_fwd_returns(px, 10))
    if len(ic) == 0:
        print(f"[{label}] NO IC DATES")
        return None
    icm = float(ic.mean()); icstd = float(ic.std(ddof=1)) if len(ic) > 1 else np.nan
    icir = icm / icstd if icstd and math.isfinite(icstd) and icstd > 0 else np.nan
    hit = float((ic > 0).mean())
    recent = ic[ic.index >= "2033-01-01"]
    ricm = float(recent.mean()) if len(recent) else np.nan
    ricir = ricm / recent.std(ddof=1) if len(recent) > 2 and recent.std(ddof=1) > 0 else np.nan
    cov_ad = float(f.notna().mean().mean())
    cov_d8 = float((f.notna().sum(axis=1) >= 8).mean())
    turn = float(f.rank(axis=1, pct=True).diff().abs().mean(axis=1).mean() * 10) if len(f) > 2 else np.nan
    gate = (abs(icm) >= 0.0070) and (abs(icir) >= 0.0840)
    flag = "PASS" if gate else "fail"
    print(f"[{label}] n_ic={len(ic)} IC={icm:+.4f} ICIR={icir:+.4f} hit={hit:.3f} "
          f"recentIC={ricm:+.4f} ricir={ricir:+.4f} covAD={cov_ad:.3f} covD8={cov_d8:.3f} "
          f"turn={turn:.3f} GATE={flag}")
    return dict(ic=icm, icir=icir, hit=hit, ricm=ricm, ricir=ricir, cov=cov_ad, turn=turn)


vix = load_macro("VIX", VIS).reindex(px.index).ffill()
vix_chg = vix / vix.shift(60) - 1  # 60d VIX change
vix_20 = vix / vix.shift(20) - 1

mom20 = px / px.shift(20) - 1
mom40 = px / px.shift(40) - 1
mom60 = px / px.shift(60) - 1
vol20 = ret.rolling(20).std()
vol60 = ret.rolling(60).std()
rv10 = ret.rolling(10).std()
rv40 = ret.rolling(40).std()

candidates = {}

# A. VIX-decline (risk-on) conditional long momentum: VIX down => magnify carry
candidates["vixdecl_cond_mom20"] = mom20 * (-np.clip(vix_chg, -1, 1))
candidates["vixdecl_cond_mom40"] = mom40 * (-np.clip(vix_chg, -1, 1))

# B. Risk-adjusted momentum: mom / own trailing vol (carry per unit risk)
candidates["mom30_div_vol20"] = (px / px.shift(30) - 1) / vol20.shift(1)
candidates["mom20_div_vol60"] = mom20 / vol60.shift(1)

# C. Long-vol regime momentum (low realized vol => trend carry persists)
candidates["mom40_lowvol_gate"] = mom40 * np.clip(1.0 - vol20 / vol60.shift(1), 0, np.inf)
candidates["mom20_lowvol_gate"] = mom20 * np.clip(1.0 - rv10 / rv40.shift(1), 0, np.inf)

# D. Diversification/co-movement mean-reversion: asset vs SPX correlation drop => idiosyncratic opp.
spxr = ret["SPX"]
roll_corr = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
for a in px.columns:
    roll_corr[a] = ret[a].rolling(60).corr(spxr)
candidates["corr_drop_60"] = -roll_corr  # falling co-movement = cross-asset dispersion

# E. Yield-curve 60d change conditional momentum (bond yields falling => risk-on)
us10y = (px["US10Y"] / px["US10Y"].shift(60) - 1)
cn10y = (px["CN10Y"] / px["CN10Y"].shift(60) - 1)
yield_down = -np.clip(us10y + cn10y, -0.5, 0.5)
candidates["yield_down_x_mom20"] = mom20 * yield_down.reindex(px.index)

# F. VIX level-gated risk-on carry (VIX<threshold regime)
candidates["lowvix_gate_mom20"] = mom20.mul(pd.Series(np.where(vix < 25, 1.0, 0.5), index=vix.index), axis=0)

for name, f in candidates.items():
    evalc(f, name)