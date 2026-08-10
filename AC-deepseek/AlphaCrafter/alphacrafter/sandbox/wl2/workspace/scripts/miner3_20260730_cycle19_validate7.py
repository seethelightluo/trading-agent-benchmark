"""miner_3 2026-07-30 cycle 19 final validation.

Strict gate namespace: eval(expr, {'__builtins__':{}}, {'pd':pd,'np':np,'close':panel}).
Masked candidates use `.assign(REF=np.nan)` - works in strict namespace without `len`.
All candidates are evaluated and validated in the strict namespace.
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
    "corr_wti60":   f"{RET}.rolling(60, min_periods=15).corr(close['WTI'].pct_change()).assign(WTI=np.nan)",
    "corr_xau60":   f"{RET}.rolling(60, min_periods=15).corr(close['XAU'].pct_change()).assign(XAU=np.nan)",
    "corr_us10y60": f"{RET}.rolling(60, min_periods=15).corr(close['US10Y'].pct_change()).assign(US10Y=np.nan)",
    "corr_btc60":   f"{RET}.rolling(60, min_periods=15).corr(close['BTC'].pct_change()).assign(BTC=np.nan)",
    "rel_mom20_spx": "(close.shift(5)/close.shift(25)-1.0) - (close['SPX'].shift(5)/close['SPX'].shift(25)-1.0)",
    "rel_mom60_spx": "(close.shift(5)/close.shift(65)-1.0) - (close['SPX'].shift(5)/close['SPX'].shift(65)-1.0)",
    "eff_ratio_20":  f"(close - close.shift(20)).abs() / {RET}.abs().rolling(20, min_periods=10).sum()",
    "eff_ratio_60":  f"(close - close.shift(60)).abs() / {RET}.abs().rolling(60, min_periods=30).sum()",
    "upday_ratio_20": f"({RET} > 0).rolling(20, min_periods=10).mean()",
    "gain_loss_20":  f"{RET}.clip(lower=0).rolling(20, min_periods=10).mean() / ({RET}.clip(upper=0).rolling(20, min_periods=10).mean().abs() + 1e-9)",
    "kurt20":        f"{RET}.rolling(20, min_periods=10).kurt()",
    "mom30_vol60":   f"(close.shift(5)/close.shift(35)-1.0) / {RET}.rolling(60, min_periods=15).std()",
    "mom10_vol20":   f"(close.shift(5)/close.shift(15)-1.0) / {RET}.rolling(20, min_periods=5).std()",
    "mom60_vol20":   f"(close.shift(5)/close.shift(65)-1.0) / {RET}.rolling(20, min_periods=5).std()",
    "zscore_252":    f"(close - close.rolling(252, min_periods=30).mean()) / close.rolling(252, min_periods=30).std()",
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
        print(f"  {fid:14s} eval={'OK' if ok else 'BAD_SIGNAL'}")
    except Exception as e:
        print(f"  {fid:14s} eval=FAIL {type(e).__name__}: {str(e)[:80]}")

fwd10 = forward_returns(prices, ADMISSION_HORIZON)
print(f"\n=== validation (admission h={ADMISSION_HORIZON}, strict gate view) ===")
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

print("\n=== pairwise |rho| among candidates (pooled, strict gate view) ===")
print("        " + "".join(f"{b[:7]:>9s}" for b in names))
for i, a in enumerate(names):
    print(f"  {a:14s}" + "".join(f"{abs(rho.loc[a,b]):>9.3f}" if pd.notna(rho.loc[a, b]) else f"{'-':>9s}" for b in names))

print("\n=== passers & max |rho| vs other passers (threshold 0.5) ===")
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
