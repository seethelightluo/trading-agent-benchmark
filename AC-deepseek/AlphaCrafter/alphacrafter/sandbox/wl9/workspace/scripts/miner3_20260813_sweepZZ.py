"""miner_3 (2026-08-13): Sweep Z2 - refine promising orthogonal variants to clear ICIR gate."""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
import pathlib

sys.path.insert(0, "scripts")
from miner3_20260730_harness import ASSETS, evaluate, load_closes, load_macro


def load_ohlc():
    out = {}
    for a in ASSETS:
        f = pathlib.Path(f"../persistent/stock_data/{a}.csv")
        df = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
        df = df[df["date"] <= "2026-07-29"]
        out[a] = df.set_index("date")
    return out


def main():
    ohlc = load_ohlc()
    closes = load_closes()
    ret = {a: closes[a].pct_change() for a in closes}
    cand = {}

    # wick imbalance variants (different windows/normalization)
    for a in closes:
        hi = ohlc[a]["high"]; lo = ohlc[a]["low"]; cl = closes[a]; op = ohlc[a]["open"]
        upper = (hi - np.maximum(cl, op))
        lower = (np.minimum(cl, op) - lo)
        for w in (5, 10, 20):
            wick = (upper.rolling(w).sum() - lower.rolling(w).sum()) / \
                   (cl.rolling(w).mean()).replace(0, np.nan)
            cand.setdefault(f"wick_norm_{w}", {})[a] = wick
        # wick imbalance relative to intraday range (daily, then summed)
        tr = (hi - lo)
        wickr = (upper - lower) / tr.replace(0, np.nan)
        cand.setdefault("wick_ratio_10", {})[a] = wickr.rolling(10).sum()

    # range symmetry variants (position vs longer regime) with different windows
    for a in closes:
        for w in (10, 20):
            hi_w = ohlc[a]["high"].rolling(w).max()
            lo_w = ohlc[a]["low"].rolling(w).min()
            pos = (closes[a] - lo_w) / (hi_w - lo_w).replace(0, np.nan)
            mid = (closes[a].rolling(3 * w).mean() - lo_w) / (hi_w - lo_w).replace(0, np.nan)
            cand.setdefault(f"range_symm_{w}", {})[a] = pos - mid

    # vol persistence: ratio of 20d vol to trailing 120d median vol (vol regime)
    for a in closes:
        rv = ret[a].rolling(20).std()
        base = ret[a].rolling(120).median()
        cand.setdefault("vol_persist_20_120", {})[a] = (rv / base.replace(0, np.nan))

    for name, vals in cand.items():
        try:
            evaluate(closes, vals, name, horizon=10)
        except Exception as e:
            print(name, "ERROR:", repr(e))
        print()


if __name__ == "__main__":
    main()