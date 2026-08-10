"""miner_3 2026-07-30 cycle 19 screen: cross-asset correlation/beta, relative
strength, and trend-quality factor families.

Key lesson from prior cycles: the post-Miner gate recovers signals by evaluating
the persisted `calculation.expression` in a restricted namespace
eval(expr, {'__builtins__':{}}, {'pd':pd,'np':np,'close':panel}) on the UNION
panel. Rolling statistics are gate-recoverable when `min_periods` is set.
For reference-correlation factors, the reference asset's own column is masked
(out-of-sample: an asset's correlation with itself is undefined), so WTI/XAU/
US10Y/BTC/SPX columns get NaN for their own-reference factors.

Admission gates (shared benchmark contract): |IC|>=0.0070 (h=10), |ICIR|>=0.0840,
pairwise |rho|<0.5 vs other passers, coverage with >=8 valid assets per date.
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

RET = "close.pct_change()"
EXPRS = {
    # ---- cross-asset correlation family (rolling corr with a reference asset) ----
    "corr_wti60":  f"{RET}.rolling(60, min_periods=15).corr(close['WTI'].pct_change()).where(close.columns != 'WTI')",
    "corr_xau60":  f"{RET}.rolling(60, min_periods=15).corr(close['XAU'].pct_change()).where(close.columns != 'XAU')",
    "corr_us10y60": f"{RET}.rolling(60, min_periods=15).corr(close['US10Y'].pct_change()).where(close.columns != 'US10Y')",
    "corr_btc60":  f"{RET}.rolling(60, min_periods=15).corr(close['BTC'].pct_change()).where(close.columns != 'BTC')",
    # ---- relative strength vs SPX (alpha vs global equity benchmark) ----
    "rel_mom20_spx": "(close.shift(5)/close.shift(25)-1.0) - (close['SPX'].shift(5)/close['SPX'].shift(25)-1.0)",
    "rel_mom60_spx": "(close.shift(5)/close.shift(65)-1.0) - (close['SPX'].shift(5)/close['SPX'].shift(65)-1.0)",
    # ---- trend quality / efficiency ----
    "eff_ratio_20": "(close - close.shift(20)).abs() / close.pct_change().abs().rolling(20, min_periods=10).sum()",
    "eff_ratio_60": "(close - close.shift(60)).abs() / close.pct_change().abs().rolling(60, min_periods=30).sum()",
    "upday_ratio_20": "(close.pct_change() > 0).rolling(20, min_periods=10).mean()",
    "gain_loss_20": "close.pct_change().clip(lower=0).rolling(20, min_periods=10).mean() / (close.pct_change().clip(upper=0).rolling(20, min_periods=10).mean().abs() + 1e-9)",
    "kurt20": "close.pct_change().rolling(20, min_periods=10).kurt()",
    # ---- risk-adjusted momentum variants (mom20/vol60 passed in cycle 18) ----
    "mom30_vol60": "(close.shift(5)/close.shift(35)-1.0) / close.pct_change().rolling(60, min_periods=15).std()",
    "mom10_vol20": "(close.shift(5)/close.shift(15)-1.0) / close.pct_change().rolling(20, min_periods=5).std()",
    "mom60_vol20": "(close.shift(5)/close.shift(65)-1.0) / close.pct_change().rolling(20, min_periods=5).std()",
    # ---- yield-rate level z-score (carry-style for US10Y/CN10Y) ----
    "zscore_252": "(close - close.rolling(252, min_periods=30).mean()) / close.rolling(252, min_periods=30).std()",
}

env = {"pd": pd, "np": np, "close": panel}
signals = {}
print("=== restricted-namespace eval (gate view: close+pd+np only) ===")
for fid, exp in EXPRS.items():
    try:
        sig = eval(exp, {"__builtins__": {}}, env)
        ok = isinstance(sig, pd.DataFrame) and sig.shape == panel.shape and sig.notna().sum().sum() > 100
        if ok:
            signals[fid] = sig
        print(f"  {fid:16s} eval={'OK' if ok else 'BAD'}")
    except Exception as e:
        print(f"  {fid:16s} eval=FAIL {type(e).__name__}: {str(e)[:70]}")

fwd10 = forward_returns(prices, ADMISSION_HORIZON)
print(f"\n=== validation (admission h={ADMISSION_HORIZON}, gate view signal) ===")
rows = {}
for fid, sig in signals.items():
    ic_series = spearman_ic(sig, fwd10)
    if len(ic_series) == 0:
        rows[fid] = dict(ic=np.nan, icir=np.nan, hit=np.nan, n=0, cov=0.0,
                         dates_ge8=0.0, turn=np.nan, decay={}, gate=False)
        print(f"  {fid:16s} NO IC DATES")
        continue
    ic = float(ic_series.mean())
    icir = float(ic_series.mean() / ic_series.std()) if ic_series.std() > 0 else 0.0
    hit = float((ic_series > 0).mean()) if ic >= 0 else float((ic_series < 0).mean())
    decay = {str(h): round(float(spearman_ic(sig, forward_returns(prices, h)).mean()), 4)
             for h in HORIZONS}
    valid = float(sig.notna().sum().sum())
    cov = valid / sig.size
    n_ge8 = sum(1 for d in sig.index if sig.loc[d].notna().sum() >= MIN_ASSETS)
    turn = mean_rank_turnover(sig)
    gate = abs(ic) >= 0.007 and abs(icir) >= 0.084
    regime = {}
    for b0, b1 in [('2020-01-01', '2021-12-31'), ('2022-01-01', '2022-12-31'),
                   ('2023-01-01', '2024-12-31'), ('2025-01-01', '2026-12-31')]:
        sub = ic_series[(ic_series.index >= b0) & (ic_series.index <= b1)]
        if len(sub) >= 30:
            regime[f'{b0[:4]}-{b1[:4]}'] = {'ic': round(float(sub.mean()), 4),
                                            'icir': round(float(sub.mean() / sub.std()), 4) if sub.std() > 0 else 0.0,
                                            'n_dates': int(len(sub))}
    rows[fid] = dict(ic=ic, icir=icir, hit=hit, n=len(ic_series), cov=cov,
                     dates_ge8=n_ge8 / len(sig), turn=turn, decay=decay,
                     regime=regime, gate=gate)
    print(f"  {fid:16s} n={len(ic_series):5d} ic={ic:+.4f} icir={icir:+.4f} "
          f"hit={hit:.3f} cov={cov:.3f} dates_ge8={n_ge8/len(sig):.3f} turn={turn:.3f} "
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
for i, a in enumerate(names):
    row = "".join(f"{abs(rho.loc[a,b]):>7.3f}" if j < i else "       " for j, b in enumerate(names))
    print(f"  {a:16s}{row}")

print("\n=== passers & max |rho| vs other passers ===")
passers = [fid for fid, r in rows.items() if r["gate"]]
for a in passers:
    mx = max((abs(rho.loc[a, b]) for b in passers if b != a and pd.notna(rho.loc[a, b])), default=0.0)
    print(f"  {a:16s} ic={rows[a]['ic']:+.4f} icir={rows[a]['icir']:+.4f} "
          f"max_rho_vs_passers={mx:.3f} regime={json.dumps(rows[a]['regime'])}")

json.dump({k: v for k, v in rows.items()},
          open("scripts/_cycle19_screen_results.json", "w"), indent=1, default=float)
print("\nsaved scripts/_cycle19_screen_results.json")
