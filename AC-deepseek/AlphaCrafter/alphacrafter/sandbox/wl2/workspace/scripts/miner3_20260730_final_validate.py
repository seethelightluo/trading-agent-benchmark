"""miner_3 2026-07-30 FINAL validation: all factors expressed SELF-CONTAINED
using only `close` (+pd/np) so the post-Miner gate can recover signal artifacts.
Validates IC/ICIR (h=10 admission), coverage, turnover, decay, regime stability,
and pairwise pooled correlation among the final factor set.
"""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner3_lib import build_panel, forward_returns, spearman_ic, mean_rank_turnover, ADMISSION_HORIZON, MIN_ASSETS

prices = build_panel()
panel = pd.DataFrame(prices)

# ---- self-contained expressions (namespace: close + pd + np only) ----
EXPRS = {
    # my candidates
    "price_zscore_60d": "(close - close.rolling(60).mean()) / close.rolling(60).std()",
    "mkt_beta_20d": "close.pct_change().rolling(20).cov(close.pct_change().mean(axis=1)) / close.pct_change().mean(axis=1).rolling(20).var()",
    "max_ret_20d": "close.pct_change().rolling(20).max()",
    # library factors re-validated (corrected self-contained form)
    "mom_10d_skip5": "close.shift(5) / close.shift(15) - 1.0",
    "mom_120d_skip5": "close.shift(5) / close.shift(125) - 1.0",
    "vol_of_vol20x60": "close.pct_change().rolling(20).std().rolling(60).std()",
}
REF = {  # reference implementations computed outside the restricted namespace
    "price_zscore_60d": lambda s: (s - s.rolling(60).mean()) / s.rolling(60).std(),
    "mkt_beta_20d": lambda s: None,  # handled below
    "max_ret_20d": lambda s: s.pct_change().rolling(20).max(),
    "mom_10d_skip5": lambda s: s.shift(5) / s.shift(15) - 1.0,
    "mom_120d_skip5": lambda s: s.shift(5) / s.shift(125) - 1.0,
    "vol_of_vol20x60": lambda s: s.pct_change().rolling(20).std().rolling(60).std(),
}

# reference for mkt_beta (per-asset rolling cov with equal-weight market)
ret = panel.pct_change()
market = ret.mean(axis=1)
mktbeta_ref = pd.DataFrame(index=panel.index, columns=panel.columns, dtype=float)
for a in panel.columns:
    z = pd.concat([ret[a].rename("a"), market.rename("m")], axis=1)
    mktbeta_ref[a] = z["a"].rolling(20).cov(z["m"]) / z["m"].rolling(20).var()

env = {"pd": pd, "np": np, "close": panel}
signals = {}
print("=== expression eval in restricted namespace (close+pd+np) ===")
for fid, exp in EXPRS.items():
    try:
        sig = eval(exp, {"__builtins__": {}}, env)
        assert isinstance(sig, pd.DataFrame) and sig.shape == panel.shape
        signals[fid] = sig
        print(f"  {fid:18s} eval=OK")
    except Exception as e:
        print(f"  {fid:18s} eval=FAIL {type(e).__name__}: {str(e)[:70]}")

# rho vs reference implementations (sanity that restricted eval == intended factor)
print("\n=== rho_vs_reference (restricted-eval vs intended factor) ===")
for fid, sig in signals.items():
    if fid == "mkt_beta_20d":
        ref = mktbeta_ref
    else:
        ref = pd.DataFrame({a: REF[fid](panel[a].dropna()).reindex(panel.index) for a in panel.columns})
    both = pd.concat([sig.stack().rename("x"), ref.stack().rename("y")], axis=1).dropna()
    r = float(both["x"].corr(both["y"]))
    print(f"  {fid:18s} rho_vs_ref={r:.4f}  n={len(both)}")

print("\n=== pooled pairwise |rho| among final signals ===")
names = list(signals.keys())
rho = pd.DataFrame(index=names, columns=names, dtype=float)
for a in names:
    for b in names:
        both = pd.concat([signals[a].stack().rename("x"), signals[b].stack().rename("y")], axis=1).dropna()
        rho.loc[a, b] = float(both["x"].corr(both["y"]))
for i, a in enumerate(names):
    row = "".join(f"{abs(rho.loc[a,b]):>7.3f}" if j <= i else "       " for j, b in enumerate(names))
    print(f"  {a:18s}{row}")

print("\n=== max |rho| of each vs every other ===")
for a in names:
    others = [abs(rho.loc[a, b]) for b in names if b != a]
    print(f"  {a:18s} max_other={max(others):.3f}")

print("\n=== admission metrics (h=10, direction=+1) ===")
fwd = forward_returns(prices, ADMISSION_HORIZON)
for fid, sig in signals.items():
    ics = spearman_ic(sig, fwd)
    ic = float(ics.mean())
    icir = float(ics.mean() / ics.std()) if ics.std() > 0 else 0.0
    hit = float((ics > 0).mean())
    turn = mean_rank_turnover(sig)
    valid = sig.notna().sum().sum(); total = sig.size
    cov = valid / total
    nge8 = sum(1 for d in sig.index if sig.loc[d].notna().sum() >= MIN_ASSETS)
    decay = {}
    for h in [1, 2, 3, 5, 10, 20]:
        s = spearman_ic(sig, forward_returns(prices, h))
        decay[str(h)] = round(float(s.mean()), 4)
    print(f"  {fid:18s} ic={ic:+.4f} icir={icir:+.4f} hit={hit:.3f} n={len(ics)} "
          f"cov={cov:.3f} dates_ge8={nge8/len(sig):.3f} turn={turn:.3f} decay={decay}")
