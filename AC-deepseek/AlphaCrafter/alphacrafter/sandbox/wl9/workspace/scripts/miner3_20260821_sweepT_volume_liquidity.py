"""miner_3 (2026-08-21): Sweep T - volume / liquidity dimensions.

Library has volume z-score (vol_z_20d) and price-volume EWM had been explored.
Probe fresh volume-trend and liquidity-factor candidates likely orthogonal to
the existing vol_z_20d and momentum library:
  - vol_ratio_short20_long60 : mean(vol,20)/mean(vol,60) (short-term volume surge relative to baseline)
  - vol_ratio_5_20           : mean(vol,5)/mean(vol,20)
  - price_vol_div_20         : 20d ewm price momentum minus 20d ewm volume momentum (divergence)
  - volatility_volume_div    : 20d realized vol z-score minus 20d volume z-score (price-vol vs volume divergence)
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner3_20260730_harness import ASSETS, evaluate, load_closes


def load_ohlc():
    out = {}
    for a in ASSETS:
        import pathlib
        f = pathlib.Path(f"../persistent/stock_data/{a}.csv")
        df = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
        df = df[df["date"] <= "2026-07-29"]
        out[a] = df.set_index("date")
    return out


ohlc = load_ohlc()

def main():
    closes = load_closes()
    print("assets:", len(closes))
    ohlc = load_ohlc()
    vol = {a: ohlc[a]["volume"].astype(float) for a in closes}
    ret = {a: closes[a].pct_change() for a in closes}

    cand = {
        "vol_ratio_20_60": {a: vol[a].rolling(20).mean() / vol[a].rolling(60).mean() for a in closes},
        "vol_ratio_5_20": {a: vol[a].rolling(5).mean() / vol[a].rolling(20).mean() for a in closes},
        "price_vol_div_20": {a: (
            (closes[a] / closes[a].ewm(span=20).mean() - 1.0)
            - (vol[a] / vol[a].ewm(span=20).mean() - 1.0)
        ) for a in closes},
    }
    # realized vol z minus volume z
    rvz = {}
    vz = {}
    for a in closes:
        rv = ret[a].rolling(20).std()
        rvz[a] = (rv - rv.rolling(60).mean()) / rv.rolling(60).std()
        vm = vol[a].rolling(20).mean()
        vz[a] = (vm - vm.rolling(60).mean()) / vm.rolling(60).std()
    cand["volvol_div_20_60"] = {a: rvz[a] - vz[a] for a in closes}

    for name, vals in cand.items():
        try:
            evaluate(closes, vals, name, horizon=10)
        except Exception as e:
            print(name, "ERROR:", repr(e))
        print()


if __name__ == "__main__":
    main()