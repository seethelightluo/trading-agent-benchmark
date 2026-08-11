"""miner_3 2026-07-30 — Exploration round 2: screen fresh factor families.
Motivation: library holds price momentum (10/120d) and vol-of-vol + VIX-beta
conditionals. Candidate families NOT yet covered by library or prior miner
screens: intraday buying pressure (close location in day range), upper-shadow
supply proxy, Amihud illiquidity, RSI overbought/oversold, 120d drawdown depth,
realized-vol acceleration, range-based volatility, up-day coherence, and
bond(US10Y)-return correlation regime characteristic.

This is a SCREEN only (no persistence). One final deep validation script will
be written for the most promising candidate.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner_3_20260730_common import (
    get_watchlist, load_data, align_panel, factor_ic_table,
    coverage_stats, rank_turnover, max_library_corr, library_factors,
)

data = load_data(days=3200)
closes = {a: d["close"].astype(float) for a, d in data.items()}
opens = {a: d["open"].astype(float) for a, d in data.items()}
highs = {a: d["high"].astype(float) for a, d in data.items()}
lows = {a: d["low"].astype(float) for a, d in data.items()}
vols = {a: d["volume"].astype(float).replace(0, np.nan) for a, d in data.items()}
print(f"Panel assets: {len(data)}; closes range {min(c.index.min() for c in closes.values()).date()}..{max(c.index.max() for c in closes.values()).date()}")


def buy_pressure_20(c, o, h, l, v, params):
    rng = (h - l).replace(0, np.nan)
    return ((c - l) / rng).rolling(params["win"]).mean()


def upper_shadow_20(c, o, h, l, v, params):
    rng = (h - l).replace(0, np.nan)
    return ((h - np.maximum(o, c)) / rng).rolling(params["win"]).mean()


def amihud_20(c, o, h, l, v, params):
    ill = (c.pct_change().abs() / (v / 1e6))
    return np.log(ill.rolling(params["win"]).mean())


def rsi_14(c, o, h, l, v, params):
    d = c.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    au = up.ewm(alpha=1.0 / params["win"], adjust=False).mean()
    ad = dn.ewm(alpha=1.0 / params["win"], adjust=False).mean()
    rs = au / ad.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def dist_high_120(c, o, h, l, v, params):
    return c / c.rolling(params["win"]).max() - 1.0


def rv_accel_20x60(c, o, h, l, v, params):
    r = c.pct_change()
    rv20 = r.rolling(20).std()
    rv60 = r.rolling(60).std()
    return rv20 / rv60 - 1.0


def range_vol_20(c, o, h, l, v, params):
    return ((h - l) / c).rolling(params["win"]).mean()


def up_day_ratio_60(c, o, h, l, v, params):
    up = (c.diff() > 0).astype(float)
    return up.rolling(params["win"]).mean() - 0.5


def bond_corr_60(c, o, h, l, v, params, bond=None):
    r = c.pct_change()
    br = bond.pct_change()
    return r.rolling(60).corr(br)


CANDIDATES = [
    ("buy_pressure_20", "intraday close location (buying pressure)", {"win": 20}),
    ("upper_shadow_20", "upper shadow supply proxy", {"win": 20}),
    ("amihud_20", "Amihud illiquidity (log)", {"win": 20}),
    ("rsi_14", "Wilder RSI(14)", {"win": 14}),
    ("dist_high_120", "drawdown depth vs 120d high", {"win": 120}),
    ("rv_accel_20x60", "realized vol acceleration", {"win": 20}),
    ("range_vol_20", "range-based volatility", {"win": 20}),
    ("up_day_ratio_60", "up-day coherence (trend consistency)", {"win": 60}),
    ("bond_corr_60", "60d corr with US10Y returns", {"win": 60}),
]

us10y = closes.get("US10Y")
lib = library_factors(data)

for fid, desc, params in CANDIDATES:
    panel = {}
    for a in closes:
        try:
            if fid == "bond_corr_60":
                if a == "US10Y" or us10y is None:
                    continue
                panel[a] = bond_corr_60(closes[a], opens[a], highs[a], lows[a], vols[a], params, bond=us10y)
            else:
                fn = globals()[fid]
                panel[a] = fn(closes[a], opens[a], highs[a], lows[a], vols[a], params)
        except Exception as e:
            print(f"[{fid}] {a} ERROR: {e}")
    tbl = factor_ic_table(panel, data, horizons=(1, 3, 5, 10, 20))
    prim = tbl[10]
    if prim is None:
        print(f"[{fid}] {desc}: DEGENERATE (no IC dates)")
        continue
    cov = coverage_stats(panel, data)
    to = rank_turnover(panel)
    maxrho, rho_map = max_library_corr(panel, data)
    gate = abs(prim["ic"]) >= 0.0070 and abs(prim["icir"]) >= 0.0840
    decay = {str(h): (round(v["ic"], 4) if v else None) for h, v in tbl.items()}
    print(f"[{fid}] {desc} | IC10={prim['ic']:.4f} ICIR10={prim['icir']:.4f} hit={prim['ic_hit']:.3f} "
          f"n={prim['n_dates']} cov={cov['coverage_asset_days']:.3f} ge8={prim['dates_ge8']:.3f} "
          f"turn={to:.2f} maxLibCorr={maxrho:.3f} decay={decay} -> {'PASS' if gate else 'fail'}")
