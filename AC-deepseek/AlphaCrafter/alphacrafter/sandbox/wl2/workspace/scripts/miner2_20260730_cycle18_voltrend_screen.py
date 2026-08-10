"""miner_2 2026-07-30 cycle 18: volatility / trend-strength / skewness /
cross-asset correlation factor screen.

Key lesson from cycle 17 (miner_3): the post-Miner gate recovers signals by
evaluating the persisted `calculation.expression` in a restricted namespace
(close + pd + np) on the union panel. Rolling statistics were previously
assumed to collapse on the union calendar; a follow-up test showed they are
gate-recoverable with adequate coverage when `min_periods` is set
(vol20 mp=5 -> cov 0.996; vol60 mp=15 -> 0.988; max252 mp=30 -> 0.982;
skew60 mp=10 -> 0.992).

This script validates every candidate EXACTLY as the gate would recover it:
eval(expression, {'__builtins__':{}}, {'pd':pd,'np':np,'close':panel}).
Admission gates (shared benchmark contract): |IC|>=0.0070 (h=10),
|ICIR|>=0.0840, pairwise |rho|<0.5 vs other passers, coverage with >=8 valid
assets per date across a majority of dates.
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

EXPRS = {
    # --- realized volatility (defensive/quality; low vol tends to outperform) ---
    "vol20":          "close.pct_change().rolling(20, min_periods=5).std()",
    "vol60":          "close.pct_change().rolling(60, min_periods=15).std()",
    "vol_of_vol20x60": "close.pct_change().rolling(20, min_periods=5).std().rolling(60, min_periods=15).std()",
    "downside_vol20": "close.pct_change().clip(upper=0).rolling(20, min_periods=5).std()",
    "vol_ratio_20_60": "close.pct_change().rolling(20, min_periods=5).std() / close.pct_change().rolling(60, min_periods=15).std()",
    # --- trend strength / distance from highs / range position ---
    "dist_high_60":   "close / close.rolling(60, min_periods=10).max() - 1.0",
    "dist_high_252":  "close / close.rolling(252, min_periods=30).max() - 1.0",
    "range_pos_252":  "(close - close.rolling(252, min_periods=30).min()) / (close.rolling(252, min_periods=30).max() - close.rolling(252, min_periods=30).min())",
    # --- skewness ---
    "skew60":         "close.pct_change().rolling(60, min_periods=10).skew()",
    # --- risk-adjusted momentum (20d momentum / 60d vol) ---
    "mom20_vol60":    "(close.shift(5)/close.shift(25)-1.0) / close.pct_change().rolling(60, min_periods=15).std()",
    # --- cross-asset: rolling correlation with SPX returns ---
    "spx_corr60":     "close.pct_change().rolling(60, min_periods=15).corr(close['SPX'].pct_change())",
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
        print(f"  {fid:18s} eval={'OK' if ok else 'BAD'}")
    except Exception as e:
        print(f"  {fid:18s} eval=FAIL {type(e).__name__}: {str(e)[:70]}")

fwd10 = forward_returns(prices, ADMISSION_HORIZON)
print(f"\n=== validation (admission h={ADMISSION_HORIZON}, gate view signal) ===")
rows = {}
for fid, sig in signals.items():
    ic_series = spearman_ic(sig, fwd10)
    if len(ic_series) == 0:
        print(f"  {fid:18s} NO IC DATES")
        rows[fid] = dict(ic=np.nan, icir=np.nan, hit=np.nan, n=0, cov=0.0,
                         dates_ge8=0.0, turn=np.nan, decay={}, gate=False)
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
    # regime split
    regime = {}
    blocks = [('2020-01-01', '2021-12-31'), ('2022-01-01', '2022-12-31'),
              ('2023-01-01', '2024-12-31'), ('2025-01-01', '2026-12-31')]
    for b0, b1 in blocks:
        sub = ic_series[(ic_series.index >= b0) & (ic_series.index <= b1)]
        if len(sub) >= 30:
            regime[f'{b0[:4]}-{b1[:4]}'] = {
                'ic': round(float(sub.mean()), 4),
                'icir': round(float(sub.mean() / sub.std()), 4) if sub.std() > 0 else 0.0,
                'n_dates': int(len(sub))}
    rows[fid] = dict(ic=ic, icir=icir, hit=hit, n=len(ic_series), cov=cov,
                     dates_ge8=n_ge8 / len(sig), turn=turn, decay=decay,
                     regime=regime, gate=gate)
    print(f"  {fid:18s} n={len(ic_series):5d} ic={ic:+.4f} icir={icir:+.4f} "
          f"hit={hit:.3f} cov={cov:.3f} dates_ge8={n_ge8/len(sig):.3f} turn={turn:.3f} "
          f"gate={'PASS' if gate else 'no '} decay10={decay['10']:+.4f}")

# pairwise pooled rho among cycle-18 candidates (gate view)
names = list(signals.keys())
rho = pd.DataFrame(index=names, columns=names, dtype=float)
for i, a in enumerate(names):
    for j, b in enumerate(names):
        if j <= i:
            continue
        both = pd.concat([signals[a].stack().rename("x"), signals[b].stack().rename("y")], axis=1).dropna()
        rho.loc[a, b] = float(both["x"].corr(both["y"])) if len(both) > 100 else np.nan
        rho.loc[b, a] = rho.loc[a, b]

print("\n=== pairwise |rho| among cycle-18 candidates (pooled) ===")
for i, a in enumerate(names):
    row = "".join(f"{abs(rho.loc[a,b]):>7.3f}" if j < i else "       " for j, b in enumerate(names))
    print(f"  {a:18s}{row}")

# correlation vs cycle-17 momentum passers (library candidates)
mom17 = {
    "mom_10d_skip5": "close.shift(5) / close.shift(15) - 1.0",
    "mom_20d_skip5": "close.shift(5) / close.shift(25) - 1.0",
    "mom_30d_skip5": "close.shift(5) / close.shift(35) - 1.0",
    "mom_180d_skip5": "close.shift(5) / close.shift(185) - 1.0",
    "mom_250d_skip5": "close.shift(5) / close.shift(255) - 1.0",
    "mom20d_damp_rev5": "(close.shift(5)/close.shift(25)-1.0) - 0.5*(close/close.shift(5)-1.0)",
}
print("\n=== max |rho| of cycle-18 candidates vs cycle-17 momentum passers ===")
for a in names:
    mx = 0.0; who = None
    for b, exp in mom17.items():
        try:
            other = eval(exp, {"__builtins__": {}}, env)
        except Exception:
            continue
        both = pd.concat([signals[a].stack().rename("x"), other.stack().rename("y")], axis=1).dropna()
        if len(both) > 100:
            r = abs(float(both["x"].corr(both["y"])))
            if r > mx:
                mx, who = r, b
    print(f"  {a:18s} max_rho_vs_mom17={mx:.3f} (vs {who})")

print("\n=== passers (cycle-18, gate) ===")
passers = [fid for fid, r in rows.items() if r["gate"]]
for a in passers:
    print(f"  {a:18s} ic={rows[a]['ic']:+.4f} icir={rows[a]['icir']:+.4f} "
          f"regime={json.dumps(rows[a]['regime'])}")

out = {k: {kk: vv for kk, vv in v.items()} for k, v in rows.items()}
json.dump(out, open("scripts/_cycle18_results.json", "w"), indent=1)
print("\nsaved scripts/_cycle18_results.json")
