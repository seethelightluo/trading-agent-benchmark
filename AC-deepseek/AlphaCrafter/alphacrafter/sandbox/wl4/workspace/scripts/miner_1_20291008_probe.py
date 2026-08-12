"""miner_1 2029-10-08 data probe + library factor re-validation.

Loads panels through the previous completed trading day (no lookahead),
prints data coverage, regime snapshot, and re-validates the three currently
effective library factors against the admission gate (|IC|>=0.007, |ICIR|>=0.084 @h=10).
"""
import sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
from factor_research_lib import (
    load_panels, close_panel, ret_panel, forward_returns, rank_ic_series,
    summarize_ic, coverage_metrics, turnover_rank, decay_profile,
    library_signals, max_library_corr, full_eval,
)

panels = load_panels(days=3000)
px = close_panel(panels)
rets = px.pct_change()
assets = px.columns.tolist()
print("assets:", len(assets), "dates:", len(px), px.index.min().date(), "->", px.index.max().date())

# per-asset coverage
for a in assets:
    print(f"  {a:8s} rows={len(px[a].dropna()):5d} last={px[a].dropna().index[-1].date()}")

# volume coverage
vol_ok = {}
for a in assets:
    df = panels.get(a)
    if df is not None and "volume" in df.columns:
        v = df["volume"].dropna()
        vol_ok[a] = round(len(v) / len(df), 2)
print("volume coverage:", vol_ok)

# regime snapshot
macro = {m: panels[m]["close"].astype(float) for m in ["VIX", "DXY", "USDJPY", "USDCNY", "EURUSD"] if m in panels}
vix = macro.get("VIX")
mkt_ret = rets.mean(axis=1)
for w in [20, 60, 120]:
    r = (1 + mkt_ret).rolling(w).apply(np.prod, raw=True) - 1
    v = mkt_ret.rolling(w).std() * np.sqrt(252)
    print(f"mkt(live) {w:3d}d cum: {r.iloc[-1]*100:+.2f}%  vol_ann: {v.iloc[-1]*100:.1f}%")
if vix is not None:
    print("VIX last:", round(float(vix.iloc[-1]), 2), " 20d ago:", round(float(vix.iloc[-21]), 2), " 60d ago:", round(float(vix.iloc[-61]), 2))
for m in ["DXY", "USDJPY", "USDCNY", "EURUSD"]:
    if m in macro:
        s = macro[m]
        print(f"{m} last: {float(s.iloc[-1]):.3f} 60d ago: {float(s.iloc[-61]):.3f}")

# recent 10d asset returns
print("\nlast 10d asset returns:")
print((px.iloc[-1] / px.iloc[-11] - 1.0).sort_values().round(4).to_string())

# ---- re-validate library factors ----
print("\n=== library factor re-validation (h=10 admission gate) ===")
mkt = rets.mean(axis=1)
dn_x = mkt.clip(upper=0.0)
def rolling_beta_fast(y, x, win, minp):
    z = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    b = z["y"].rolling(win, min_periods=minp).cov(z["x"]) / z["x"].rolling(win, min_periods=minp).var()
    return b
cn10y_ret = px["CN10Y"].pct_change()
def vol_adj_mom_accel(c, fast=20, slow=60, vol_win=20):
    r = c.pct_change()
    return (c / c.shift(fast) - 1.0 - (c / c.shift(slow) - 1.0)) / r.rolling(vol_win).std()

lib = {
    "dn_mkt_beta_60d": px.apply(lambda c: rolling_beta_fast(c.pct_change(), dn_x, 60, 40)),
    "rate_beta_cn10y_60d": px.apply(lambda c: rolling_beta_fast(c.pct_change(), cn10y_ret, 60, 40)),
    "vol_adj_mom_accel_20x60": px.apply(vol_adj_mom_accel),
}

fwd = {h: forward_returns(px, h) for h in [1, 2, 3, 5, 10, 20]}
for fid, fv in lib.items():
    ic10 = rank_ic_series(fv, fwd[10])
    s = summarize_ic(ic10, expected_sign=1)
    s.update(coverage_metrics(fv))
    s["turnover_10d_rank"] = turnover_rank(fv, 10)
    s["decay_ic_by_horizon"] = decay_profile(fv, px, (1, 2, 3, 5, 10, 20), 8, 1)
    gate_ic = abs(s["ic"]) >= 0.007
    gate_icir = abs(s["icir"]) >= 0.084
    print(json.dumps({"factor": fid, "n": s["n_ic_dates"], "ic": s["ic"], "icir": s["icir"],
                      "hit": s["ic_hit_ratio"], "cov": s["coverage_asset_days"],
                      "dates_ge8": s["coverage_dates_ge8"], "turn": s["turnover_10d_rank"],
                      "decay": s["decay_ic_by_horizon"],
                      "GATE": "PASS" if (gate_ic and gate_icir) else "fail"}))
    # split-half stability
    half = len(ic10) // 2
    for lab, sub in [("first_half", ic10.iloc[:half]), ("second_half", ic10.iloc[half:])]:
        if len(sub) > 5:
            print(f"    {lab}: ic={sub.mean():.4f} icir={sub.mean()/sub.std(ddof=1):.3f} n={len(sub)}")
