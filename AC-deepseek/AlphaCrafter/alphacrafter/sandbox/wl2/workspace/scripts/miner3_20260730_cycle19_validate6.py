"""miner_3 2026-07-30 cycle 19 full validation.

Validates ALL cycle-19 candidates in the STRICT gate namespace
(eval(expr, {'__builtins__':{}}, {'pd':pd,'np':np,'close':panel})):
  - 6 gate-recoverable masked candidates (corr_wti60, corr_xau60, corr_us10y60,
    corr_btc60, rel_mom20_spx, rel_mom60_spx) with self-column masked NaN
  - 5 screen passers (gain_loss_20, mom30_vol60, mom10_vol20, mom60_vol20,
    zscore_252)
Computes IC / ICIR (h=10 admission), hit, decay, coverage, turnover, regime
split, and the full pairwise |rho| matrix among passers (redundancy gate).
"""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner3_lib import (build_panel, forward_returns, spearman_ic,
                        mean_rank_turnover, ADMISSION_HORIZON, HORIZONS,
                        MIN_ASSETS)

prices = build_panel()
panel = pd.DataFrame(prices)
env = {"pd": pd, "np": np, "close": panel}

MASK = "pd.DataFrame(np.tile({cond}, (len(close), 1)), index=close.index, columns=close.columns)"
RET = "close.pct_change()"

EXPRS = {
    # ---- masked cross-asset correlation family ----
    "corr_wti60":   f"{RET}.rolling(60, min_periods=15).corr(close['WTI'].pct_change()).where({MASK.format(cond='close.columns != \\\"WTI\\\"')})",
    "corr_xau60":   f"{RET}.rolling(60, min_periods=15).corr(close['XAU'].pct_change()).where({MASK.format(cond='close.columns != \\\"XAU\\\"')})",
    "corr_us10y60": f"{RET}.rolling(60, min_periods=15).corr(close['US10Y'].pct_change()).where({MASK.format(cond='close.columns != \\\"US10Y\\\"')})",
    "corr_btc60":   f"{RET}.rolling(60, min_periods=15).corr(close['BTC'].pct_change()).where({MASK.format(cond='close.columns != \\\"BTC\\\"')})",
    # ---- masked relative strength vs SPX ----
    "rel_mom20_spx": f"(close.shift(5)/close.shift(25)-1.0) - (close['SPX'].shift(5)/close['SPX'].shift(25)-1.0)",  # noqa: E501
    "rel_mom60_spx": f"(close.shift(5)/close.shift(65)-1.0) - (close['SPX'].shift(5)/close['SPX'].shift(65)-1.0)",  # noqa: E501
    # ---- screen passers (already gate-OK in cycle19 screen) ----
    "gain_loss_20":  "close.pct_change().clip(lower=0).rolling(20, min_periods=10).mean() / (close.pct_change().clip(upper=0).rolling(20, min_periods=10).mean().abs() + 1e-9)",
    "mom30_vol60":   "(close.shift(5)/close.shift(35)-1.0) / close.pct_change().rolling(60, min_periods=15).std()",
    "mom10_vol20":   "(close.shift(5)/close.shift(15)-1.0) / close.pct_change().rolling(20, min_periods=5).std()",
    "mom60_vol20":   "(close.shift(5)/close.shift(65)-1.0) / close.pct_change().rolling(20, min_periods=5).std()",
    "zscore_252":    "(close - close.rolling(252, min_periods=30).mean()) / close.rolling(252, min_periods=30).std()",
}

print(f"panel: {panel.shape[0]} dates x {panel.shape[1]} assets  visible_through=2026-07-29")
signals = {}
print("=== strict-namespace eval ===")
for fid, exp in EXPRS.items():
    try:
        sig = eval(exp, {"__builtins__": {}}, env)
        ok = isinstance(sig, pd.DataFrame) and sig.shape == panel.shape
        if ok:
            signals[fid] = sig
        print(f"  {fid:14s} eval={'OK' if ok else 'BAD_SHAPE'}")
    except Exception as e:
        print(f"  {fid:14s} eval=FAIL {type(e).__name__}: {str(e)[:80]}")

fwd10 = forward_returns(prices, ADMISSION_HORIZON)
print(f"\n=== validation (admission h={ADMISSION_HORIZON}) ===")
rows = {}
for fid, sig in signals.items():
    ics = spearman_ic(sig, fwd10)
    if len(ics) == 0:
        print(f"  {fid:14s} NO IC DATES"); continue
    ic = float(ics.mean())
    icir = float(ics.mean() / ics.std()) if ics.std() > 0 else 0.0
    hit = float((ics > 0).mean()) if ic >= 0 else float((ics < 0).mean())
    decay = {str(h): round(float(spearman_ic(sig, forward_returns(prices, h)).mean()), 4)
             for h in HORIZONS}
    cov = float(sig.notna().sum().sum()) / sig.size
    n_ge8 = sum(1 for d in sig.index if sig.loc[d].notna().sum() >= MIN_ASSETS)
    turn = mean_rank_turnover(sig)
    gate = abs(ic) >= 0.007 and abs(icir) >= 0.084
    regime = {}
    for b0, b1 in [("2020-01-01", "2021-12-31"), ("2022-01-01", "2022-12-31"),
                   ("2023-01-01", "2024-12-31"), ("2025-01-01", "2026-12-31")]:
        sub = ics[(ics.index >= b0) & (ics.index <= b1)]
        if len(sub) >= 30:
            regime[f"{b0[:4]}-{b1[:4]}"] = {"ic": round(float(sub.mean()), 4),
                                            "icir": round(float(sub.mean() / sub.std()), 4) if sub.std() > 0 else 0.0,
                                            "n_dates": int(len(sub))}
    rows[fid] = dict(ic=ic, icir=icir, hit=hit, n=len(ics), cov=cov,
                     dates_ge8=n_ge8 / len(sig), turn=turn, decay=decay,
                     regime=regime, gate=gate)
    print(f"  {fid:14s} n={len(ics):5d} ic={ic:+.4f} icir={icir:+.4f} hit={hit:.3f} "
          f"cov={cov:.3f} dates_ge8={n_ge8/len(sig):.3f} turn={turn:.3f} "
          f"gate={'PASS' if gate else 'no '} decay10={decay['10']:+.4f}")

names = list(signals.keys())
rho = pd.DataFrame(index=names, columns=names, dtype=float)
for i, a in enumerate(names):
    for j, b in enumerate(names):
        if j <= i:
            continue
        both = pd.concat([signals[a].stack().rename("x"), signals[b].stack().rename("y")], axis=1).dropna()
        rho.loc[a, b] = float(both["x"].corr(both["y"])) if len(both) > 100 else np.nan
        rho.loc[b, a] = rho.loc[a, b]

print("\n=== pairwise |rho| among candidates (pooled, gate view) ===")
print("        " + "".join(f"{b[:6]:>8s}" for b in names))
for i, a in enumerate(names):
    print(f"  {a:14s}" + "".join(f"{abs(rho.loc[a,b]):>8.3f}" if pd.notna(rho.loc[a, b]) else f"{'-':>8s}" for b in names))

print("\n=== passers & max |rho| vs other passers ===")
passers = [fid for fid, r in rows.items() if r["gate"]]
for a in passers:
    mx = max((abs(rho.loc[a, b]) for b in passers if b != a and pd.notna(rho.loc[a, b])), default=0.0)
    red = "OK" if mx < 0.5 else "REDUNDANT"
    print(f"  {a:14s} ic={rows[a]['ic']:+.4f} icir={rows[a]['icir']:+.4f} "
          f"max_rho_vs_passers={mx:.3f} [{red}]")
    print(f"           regime={json.dumps(rows[a]['regime'])}")
    print(f"           decay={json.dumps(rows[a]['decay'])}")

json.dump(rows, open("scripts/_cycle19_validate_results.json", "w"), indent=1, default=float)
print("\nsaved scripts/_cycle19_validate_results.json")
