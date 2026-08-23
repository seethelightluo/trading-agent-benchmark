"""miner_2 (2026-09-24): screen fresh factor families on the 15-asset tradable universe.

Validation window: factor dates 2020-01-01..2026-09-23, data visible through
2026-09-23 (previous completed trading day relative to current date 2026-09-24).
Forward returns per-asset own-calendar shift(-h).

Gates (shared, 15-asset universe): abs(IC)>=0.0070 and abs(ICIR)>=0.0840 at h=10.
Also report turnover/coverage/decay and max_abs_library_correlation.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import miner3_20260730_harness as mm

# Extend research window to the current live date (visible through 2026-09-23).
mm.VISIBLE_END = "2026-09-23"
mm.VALID_END = "2026-09-09"

from miner3_20260730_harness import ASSETS, load_closes, evaluate  # noqa: E402


def mom_pct(s, w=5):
    """Fraction of trailing w-day window (incl today) that is <= current close."""
    return s.rolling(w, min_periods=3).apply(lambda x: (x.iloc[-1] >= x).mean(), raw=False)


def load_ohlc_asset(a):
    f = Path(f"../persistent/stock_data/{a}.csv")
    if not f.exists():
        return None
    df = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
    df = df[df["date"] <= pd.Timestamp(mm.VISIBLE_END)]
    return df.set_index("date")


def main():
    closes = load_closes()
    print("assets loaded:", len(closes))
    print("visible close data through:", max(s.index.max() for s in closes.values()).date())

    cands = {}
    # A: momentum percentile (current close rank in trailing window)
    for w in (10, 20, 30):
        cands[f"mom_pct_{w}"] = {a: mom_pct(s, w) for a, s in closes.items()}
    # D: drawdown depth - close below trailing local max, normalized (<=0)
    for w in (20, 60):
        cands[f"ddepth_{w}"] = {}
        for a, s in closes.items():
            rollmax = s.rolling(w, min_periods=10).max()
            cands[f"ddepth_{w}"][a] = (s / rollmax - 1.0)
    # E: relative volume (log volume vs trailing median)
    for w in (20, 60):
        cands[f"relvol_{w}"] = {}
        for a in ASSETS:
            d = hl_range_ohlc = hl_range_ohlc = None
    # F: HL range divided by trailing median of range
    cands["hl_r_norm_20"] = {}
    for a in ASSETS:
        d = hl_range_ohlc(a)
        if d is None:
            continue
        rng = d["high"] - d["low"]
        med = rng.rolling(20, min_periods=10).median()
        cands["hl_r_norm_20"][a] = rng / med

    results = []
    for name, vals in cands.items():
        try:
            res = evaluate(closes, vals, f"SCREEN {name}", horizon=10)
            results.append((name, res))
        except Exception as e:
            print(f"  ERROR {name}: {e}")

    print("\n=== SUMMARY (sorted by |IC|) ===")
    results.sort(key=lambda x: -abs(x[1]["ic"]) if np.isfinite(x[1]["ic"]) else 0)
    for name, r in results:
        flag = "<<" if r["passed"] else ""
        print(f"  {name:16s} IC={r['ic']:+.4f} ICIR={r['icir']:+.4f} "
              f"hit={r['hit']:.3f} to10={r['turnover_10d_rank']:.3f} "
              f"cov_d8={r['coverage_dates_ge8']:.3f} maxcorr={r['max_abs_library_correlation']:.4f} {flag}")


if __name__ == "__main__":
    main()