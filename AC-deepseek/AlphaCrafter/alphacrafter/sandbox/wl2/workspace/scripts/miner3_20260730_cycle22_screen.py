"""miner_3 2026-07-30 cycle 22: screen novel close-only candidate factors.

Strict gate namespace: eval(expr, {'__builtins__':{}}, {'pd':pd,'np':np,'close':panel}).
Admission: |IC| >= 0.0070 and |ICIR| >= 0.0840 at h=10 on the 15-asset cross-section
(dates with >=8 valid instruments).  Also reports decay, coverage, turnover, regime
stability and pairwise |rho| among passers (redundancy gate).
"""
import sys, json
import numpy as np
import pandas as pd
from miner3_lib import (build_panel, forward_returns, spearman_ic,
                        mean_rank_turnover, ADMISSION_HORIZON, HORIZONS,
                        MIN_ASSETS)

prices = build_panel()
panel = pd.DataFrame(prices)
env = {"pd": pd, "np": np, "close": panel}
RET = "close.pct_change()"

EXPRS = {
    # --- novel volatility / distributional ---
    "skew_60":        f"{RET}.rolling(60, min_periods=30).skew()",
    "kurt_60":        f"{RET}.rolling(60, min_periods=30).kurt()",
    "downside_vol_60": f"{RET}.where({RET} < 0, 0.0).rolling(60, min_periods=15).std()",
    "vol_ratio_5_60": f"{RET}.rolling(5, min_periods=3).std() / ({RET}.rolling(60, min_periods=15).std() + 1e-9)",
    "vol_ratio_20_120": f"{RET}.rolling(20, min_periods=10).std() / ({RET}.rolling(120, min_periods=30).std() + 1e-9)",
    "vol_persist_60":  f"{RET}.rolling(60, min_periods=15).std() / ({RET}.rolling(60, min_periods=15).std().shift(60) + 1e-9)",
    # --- trend / drawdown ---
    "drawdown_60":    f"close / close.rolling(60, min_periods=15).max() - 1.0",
    "drawdown_120":   f"close / close.rolling(120, min_periods=30).max() - 1.0",
    "eff_ratio_120":  f"(close - close.shift(120)).abs() / {RET}.abs().rolling(120, min_periods=60).sum()",
    "rsi_like_14":    f"{RET}.clip(lower=0).rolling(14, min_periods=7).mean() / ({RET}.abs().rolling(14, min_periods=7).mean() + 1e-9)",
    # --- risk-adjusted momentum (new lookbacks) ---
    "mom90_vol60":    f"(close.shift(5)/close.shift(95)-1.0) / ({RET}.rolling(60, min_periods=15).std() + 1e-9)",
    "mom120_vol90":   f"(close.shift(5)/close.shift(125)-1.0) / ({RET}.rolling(90, min_periods=20).std() + 1e-9)",
    # --- cross-sectional relative ---
    "rel_mom20_uni":  f"(close.shift(5)/close.shift(25)-1.0) - (close.shift(5)/close.shift(25)-1.0).mean(axis=1)",
    "rel_mom60_uni":  f"(close.shift(5)/close.shift(65)-1.0) - (close.shift(5)/close.shift(65)-1.0).mean(axis=1)",
    # --- short-term autocorrelation / trending ---
    "acorr_10":       f"({RET} * {RET}.shift(1)).rolling(10, min_periods=5).mean()",
    "acorr_20":       f"({RET} * {RET}.shift(1)).rolling(20, min_periods=10).mean()",
}

print(f"panel: {panel.shape[0]} dates x {panel.shape[1]} assets  visible_through=2026-07-29")
print("=== strict-namespace eval (close+pd+np only) ===")
signals = {}
for fid, exp in EXPRS.items():
    try:
        sig = eval(exp, {"__builtins__": {}}, env)
        ok = isinstance(sig, pd.DataFrame) and sig.shape == panel.shape and sig.notna().sum().sum() > 100
        if ok:
            signals[fid] = sig
        print(f"  {fid:16s} eval={'OK' if ok else 'BAD_SIGNAL'}")
    except Exception as e:
        print(f"  {fid:16s} eval=FAIL {type(e).__name__}: {str(e)[:80]}")

fwd10 = forward_returns(prices, ADMISSION_HORIZON)
print(f"\n=== validation (admission h={ADMISSION_HORIZON}, strict gate view) ===")
rows = {}
for fid, sig in signals.items():
    ics = spearman_ic(sig, fwd10)
    if len(ics) == 0:
        print(f"  {fid:16s} NO IC DATES"); continue
    ic = float(ics.mean())
    icir = float(ics.mean() / ics.std()) if ics.std() > 0 else 0.0
    hit = float((ics > 0).mean()) if ic >= 0 else float((ics < 0).mean())
    decay = {str(h): round(float(spearman_ic(sig, forward_returns(prices, h)).mean()), 4) for h in HORIZONS}
    cov = float(sig.notna().sum().sum()) / sig.size
    n_ge8 = sum(1 for d in sig.index if sig.loc[d].notna().sum() >= MIN_ASSETS)
    turn = mean_rank_turnover(sig)
    regime = {}
    for b0, b1 in [("2020-01-01", "2021-12-31"), ("2022-01-01", "2022-12-31"),
                   ("2023-01-01", "2024-12-31"), ("2025-01-01", "2026-12-31")]:
        sub = ics[(ics.index >= b0) & (ics.index <= b1)]
        if len(sub) >= 30:
            regime[f"{b0[:4]}-{b1[:4]}"] = {"ic": round(float(sub.mean()), 4),
                                            "icir": round(float(sub.mean() / sub.std()), 4) if sub.std() > 0 else 0.0,
                                            "n_dates": int(len(sub))}
    gate = (abs(ic) >= 0.0070) and (abs(icir) >= 0.0840)
    rows[fid] = dict(ic=ic, icir=icir, hit=hit, n=len(ics), cov=cov, dates_ge8=n_ge8 / len(sig),
                     turn=turn, decay=decay, regime=regime, gate=gate)
    print(f"  {fid:16s} n={len(ics):5d} ic={ic:+.4f} icir={icir:+.4f} hit={hit:.3f} "
          f"cov={cov:.3f} dates_ge8={n_ge8/len(sig):.3f} turn={turn:.3f} "
          f"gate={'PASS' if gate else 'no '} decay10={decay['10']:+.4f}")

print("\n=== passing candidates ===")
passers = {k: v for k, v in rows.items() if v["gate"]}
for k, v in sorted(passers.items(), key=lambda kv: -abs(kv[1]["ic"]) * abs(kv[1]["icir"])):
    print(f"  {k:16s} ic={v['ic']:+.4f} icir={v['icir']:+.4f} hit={v['hit']:.3f} "
          f"turn={v['turn']:.3f} cov={v['cov']:.3f} regime={json.dumps(v['regime'])}")

print("\n=== pairwise pooled |rho| among passers (redundancy gate) ===")
names = sorted(passers.keys())
if len(names) > 1:
    rho = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            if a == b:
                rho.loc[a, b] = 1.0
                continue
            both = pd.concat([signals[a].stack().rename("x"), signals[b].stack().rename("y")], axis=1).dropna()
            rho.loc[a, b] = float(both["x"].corr(both["y"])) if len(both) > 100 else np.nan
    print(rho.round(3).to_string())
else:
    print("  (0 or 1 passers)")

with open("scripts/miner3_cycle22_results.json", "w") as f:
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "decay"} | {"decay": v["decay"]}
               for k, v in rows.items()}, f, indent=1, default=str)
print("\nresults saved to scripts/miner3_cycle22_results.json")
