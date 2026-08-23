"""Exploration: risk-adjusted momentum and 52-week-high proximity families.

Motivation: raw momentum persists in this universe; conditioning it on realized
vol (risk-adjusted momentum) and on distance-to-52w-high (trend strength) may
extract incremental, cross-sectionally differentiated signal while remaining
interpretable. Data capped at VISIBLE_END.

Variants:
  L1: sharpe_60   = mom(60,skip5) / vol(60)             risk-adjusted momentum
  L2: sharpe_20   = mom(20,skip5) / vol(20)
  L3: sharpe_120  = mom(120,skip5) / vol(120)
  L4: near52w_high = close / rolling_max(close,260)     proximity to 52w high (0..1)
  L5: trendstrength = (close - rolling_min(close,120)) / (rolling_max(close,120)-rolling_min(close,120)+eps)
"""
import sys
sys.path.insert(0, "scripts")
from miner3_20260730_harness import load_closes, evaluate, VISIBLE_END

def build(closes):
    out = {}
    for a, s in closes.items():
        c = s[(s.index >= "2020-01-01") & (s.index <= VISIBLE_END)]
        r = c.pct_change()
        def vol(w): return r.rolling(w).std() * (252**0.5)
        def mom(begin, end): return c.shift(begin) / c.shift(end) - 1.0  # momentum over [begin,end]
        out.setdefault("L1_sharpe_60", {})[a] = mom(5, 65) / (vol(60) + 1e-6)
        out.setdefault("L2_sharpe_20", {})[a] = mom(5, 25) / (vol(20) + 1e-6)
        out.setdefault("L3_sharpe_120", {})[a] = mom(5, 125) / (vol(120) + 1e-6)
        out.setdefault("L4_near52w", {})[a] = c / c.rolling(260).max()
        hi = c.rolling(120).max(); lo = c.rolling(120).min()
        out.setdefault("L5_range_pos_120", {})[a] = (c - lo) / (hi - lo + 1e-9)
    return out

if __name__ == "__main__":
    closes = load_closes()
    print(f"loaded {len(closes)} assets; cap {VISIBLE_END}")
    results = {}
    for label, vals in build(closes).items():
        r = evaluate(closes, vals, label)
        results[label] = r
        print()
    print("\n==== SUMMARY ====")
    for label, r in results.items():
        print(f"{label:18s} IC={r['ic']:+.4f} ICIR={r['icir']:+.4f} hit={r['hit']:.3f} "
              f"cov={r['coverage_asset_days']:.3f} to={r['turnover_10d_rank']:.3f} "
              f"max_lib={r['max_abs_library_correlation']:.3f} PASS={r['passed']}")