"""Exploration: relative volatility expansion (RV spread) factor family.

Hypothesis: volatility clustering is leading — when an asset's short-horizon
realized vol expands relative to its medium-run regime (vol expansion), it
predicts cross-asset dispersion and changes in forward risk premium. We test
several signed variants on the ret-shifted factor (factor computed with past
data only, predictive relation via IC).

Variants:
  V1: rv_spread  = short_rv / long_rv - 1        (vol expansion ratio)
  V2: rv_zscore  = (short_rv - rolling_mean(long_rv)) / rolling_std(long_rv)
  V3: vol_slope  = linear slope of short_rv over k days
  V4: down_vol_up = ratio of downside vol expansion to total (asymmetry)
Use only data <= 2026-07-29 (visible); validation window 2020-01-01..2026-07-15.
"""
import sys
sys.path.insert(0, "scripts")
from miner3_20260730_harness import load_closes, evaluate, VISIBLE_END, VALID_START, VALID_END


def close_array(s, **kw):
    # subset to validation cap on the visible series
    return s[(s.index >= VALID_START) & (s.index <= VISIBLE_END)].rolling(**kw)


def rv(close, window):
    r = close.pct_change()
    return r.rolling(window).std() * (252 ** 0.5)


def build_variants(closes):
    out = {}
    for a, s in closes.items():
        c = s[(s.index >= VALID_START) & (s.index <= VISIBLE_END)]
        short = rv(c, 5)
        long_rv = rv(c, 60)
        med = rv(c, 120)
        # V1: short vs medium-run vol ratio (forward 0-1 shifted later by IC semantics)
        out.setdefault("V1_rv_expand_5x60", {})[a] = (short / long_rv - 1.0)
        # V2: z-score of short rv vs its own 60d rolling loc/scale
        rvs = rv(c, 20)
        loc = rvs.rolling(60).mean()
        sca = rvs.rolling(60).std()
        out.setdefault("V2_rv_z20", {})[a] = (rvs - loc) / sca
        # V3: week-over-week change in short rv (slope proxy)
        out.setdefault("V3_rv_slope", {})[a] = short.diff(5)
        # V4: magnitude of recent daily |returns| vs 60d mean (amplitude fragility)
        mag = c.pct_change().abs()
        out.setdefault("V4_amp_vs_60", {})[a] = mag.rolling(10).mean() / (mag.rolling(60).mean() + 1e-9) - 1.0
    return out


if __name__ == "__main__":
    closes = load_closes()
    print(f"loaded {len(closes)} assets; window {VALID_START}..{VALID_END} (cap {VISIBLE_END})")
    results = {}
    variants = build_variants(closes)
    for label, vals in variants.items():
        res = evaluate(closes, vals, label)
        results[label] = res
        print()
    # summary table
    print("\n==== SUMMARY ====")
    for label, r in results.items():
        print(f"{label:20s} IC={r['ic']:+.4f} ICIR={r['icir']:+.4f} hit={r['hit']:.3f} "
              f"cov={r['coverage_asset_days']:.3f} to={r['turnover_10d_rank']:.3f} "
              f"max_lib={r['max_abs_library_correlation']:.3f} PASS={r['passed']}")