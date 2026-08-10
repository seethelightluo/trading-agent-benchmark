"""miner_3 2026-07-30 cycle 16: validate candidate factors with self-contained
expressions (namespace: close, pct_change, pd, np) and embed signal artifacts.
Admission gates: |IC|>=0.007 (h=10d), |ICIR|>=0.084, pairwise rho < 0.5."""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner3_lib import (build_panel, forward_returns, spearman_ic,
                        mean_rank_turnover, ADMISSION_HORIZON, HORIZONS, MIN_ASSETS)

prices = build_panel()
panel = pd.DataFrame(prices)
ret = panel.pct_change()

EXPRS = {
    "cs_beta_20d": "close.pct_change().rolling(20).cov(close.pct_change().mean(axis=1)) / close.pct_change().mean(axis=1).rolling(20).var()",
    "cs_beta_60d": "close.pct_change().rolling(60).cov(close.pct_change().mean(axis=1)) / close.pct_change().mean(axis=1).rolling(60).var()",
    "price_zscore_60d": "(close - close.rolling(60).mean()) / close.rolling(60).std()",
    "upday_ratio_20d": "(close.pct_change() > 0).rolling(20).mean()",
    "max_ret_20d": "close.pct_change().rolling(20).max()",
    "ret_skew_20d": "close.pct_change().rolling(20).skew()",
    "dist_high_120d": "close / close.rolling(120).max() - 1.0",
    "vol_ratio_5_60": "close.pct_change().rolling(5).std() / close.pct_change().rolling(60).std()",
    "mom_30d_skip5": "close.shift(5) / close.shift(35) - 1.0",
    "zscore_rev_20d": "-(close - close.rolling(20).mean()) / close.rolling(20).std()",
}

env = {"pd": pd, "np": np, "close": panel, "pct_change": ret}
signals = {}
print("=== expression eval in restricted namespace ===")
for fid, exp in EXPRS.items():
    try:
        sig = eval(exp, {"__builtins__": {}}, env)
        ok = isinstance(sig, pd.DataFrame) and sig.shape == panel.shape and sig.notna().sum().sum() > 0
        signals[fid] = sig
        print(f"  {fid:18s} eval={'OK' if ok else 'BAD_SHAPE'}")
    except Exception as e:
        print(f"  {fid:18s} eval=FAIL {type(e).__name__}: {str(e)[:80]}")

fwd = forward_returns(prices, ADMISSION_HORIZON)
print("\n=== validation (admission horizon h=10) ===")
rows = {}
for fid, sig in signals.items():
    ic_series = spearman_ic(sig, fwd)
    ic = float(ic_series.mean())
    icir = float(ic_series.mean() / ic_series.std()) if ic_series.std() > 0 else 0.0
    hit = float((ic_series > 0).mean()) if ic >= 0 else float((ic_series < 0).mean())
    decay = {}
    for h in HORIZONS:
        s = spearman_ic(sig, forward_returns(prices, h))
        decay[str(h)] = round(float(s.mean()), 4)
    valid = sig.notna().sum().sum()
    cov = valid / (sig.shape[0] * sig.shape[1])
    n_ge8 = sum(1 for d in sig.index if sig.loc[d].notna().sum() >= MIN_ASSETS)
    turn = mean_rank_turnover(sig)
    gate = abs(ic) >= 0.007 and abs(icir) >= 0.084
    rows[fid] = dict(ic=ic, icir=icir, hit=hit, decay=decay, cov=cov,
                     dates_ge8=n_ge8 / len(sig), turn=turn, gate=gate)
    print(f"{fid:18s} n={len(ic_series):5d} ic={ic:+.4f} icir={icir:+.4f} "
          f"hit={hit:.3f} cov={cov:.3f} dates_ge8={n_ge8/len(sig):.3f} turn={turn:.3f} "
          f"gate={'PASS' if gate else 'no'} decay10={decay['10']:.4f}")

# pairwise pooled rho among candidates (self-containment sanity + diversity)
names = list(signals.keys())
rho = pd.DataFrame(index=names, columns=names, dtype=float)
for i, a in enumerate(names):
    for j, b in enumerate(names):
        if j >= i:
            continue
        both = pd.concat([signals[a].stack().rename("x"), signals[b].stack().rename("y")], axis=1).dropna()
        rho.loc[a, b] = float(both["x"].corr(both["y"])) if len(both) > 100 else np.nan

print("\n=== pairwise rho (lower triangle) ===")
for i, a in enumerate(names):
    row = "".join(f"{rho.loc[a,b]:>8.3f}" if j <= i else "        " for j, b in enumerate(names))
    print(f"{a:18s}{row}")

print("\n=== passers & max pairwise |rho| vs other passers ===")
passers = [fid for fid, r in rows.items() if r["gate"]]
for a in passers:
    mx = max((abs(rho.loc[a, b]) for b in passers if b != a and pd.notna(rho.loc[a, b])), default=0.0)
    print(f"  {a:18s} max_rho_vs_passers={mx:.3f}")

json.dump({k: {kk: vv for kk, vv in v.items() if kk != "decay"} | {"decay": v["decay"]}
           for k, v in rows.items()}, open("scripts/_cycle16_results.json", "w"), indent=1)
print("\nsaved scripts/_cycle16_results.json")
