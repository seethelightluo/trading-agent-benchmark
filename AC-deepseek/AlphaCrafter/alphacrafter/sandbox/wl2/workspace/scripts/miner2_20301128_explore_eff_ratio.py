"""miner_2 2030-11-28 exploration #1: Trend Efficiency Ratio (Kaufman ER) family.
Motivation: memory flags momentum whipsaw (WTI re-add near top, ETH trim regret).
ER = |close_t - close_{t-n}| / sum(|daily ret| over n) measures trend smoothness /
directionality independent of sign of the move. Hypothesis: assets with clean,
efficient trends (high ER) continue; choppy assets mean-revert. Also test
ER-conditioned momentum (momentum x ER) and ER x downside-vol damping.
One idea family per script.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
import miner2_20301128_lib as lib

px, ret = lib.load_prices()
print("prices:", px.shape, "dates:", px.index.min().date(), "->", px.index.max().date())
print("instruments:", len(px.columns))

def eff_ratio(px, n):
    """Kaufman efficiency ratio over n days (shift 0, computed on close)."""
    path = ret.abs().rolling(n).sum()
    net = (px - px.shift(n)).abs()
    return net / path

cands = {}
for n in (10, 20, 60, 120):
    cands[f"er_{n}"] = eff_ratio(px, n)

# ER-conditioned momentum: sign(mom20) * ER20 (directionality-weighted momentum)
mom20 = px.pct_change(20)
cands["er_mom20xer20"] = np.sign(mom20) * cands["er_20"]
cands["mom20_x_er20"] = mom20 * cands["er_20"]
cands["mom20_x_er60"] = mom20 * cands["er_60"]

# ER * (1/vol) trend-quality-scaled momentum
vol20 = ret.rolling(20).std()
cands["mom20_x_er20_div_vol20"] = mom20 * cands["er_20"] / (vol20 + 1e-9)

print("\n=== Candidate IC/ICIR evaluation (horizon=10) ===")
results = {}
for name, f in cands.items():
    r = lib.eval_factor(f, ret, horizon=10, name=name)
    results[name] = r
    print(f"{name:24s} ic={r['ic']:+.4f} icir={r['icir']:+.3f} hit={r['hit']:.3f} "
          f"turn={r['turnover']:.3f} cov={r['coverage']:.3f} n={r['n_dates']} ok={r['ok']}")

print("\n=== Decay (IC by horizon) for top candidates ===")
for name in ["er_20", "er_60", "er_120", "mom20_x_er20", "mom20_x_er60"]:
    dec = lib.decay_analysis(cands[name], ret, name=name)
    print(f"{name:20s} decay(1,2,3,5,10,20) = {dec}")

print("\n=== Regime IC for er_20 and mom20_x_er20 ===")
for name in ["er_20", "er_60", "mom20_x_er20", "mom20_x_er60"]:
    reg = lib.regime_ic(cands[name], ret, horizon=10)
    print(name, reg)

print("\n=== Corr with library factors (rank cross-section) ===")
for name in ["er_20", "er_60", "mom20_x_er20", "mom20_x_er60"]:
    rho, best = lib.corr_with_library(cands[name], px, ret)
    print(f"{name:20s} max_abs_lib_corr={rho:+.4f} with {best}")
