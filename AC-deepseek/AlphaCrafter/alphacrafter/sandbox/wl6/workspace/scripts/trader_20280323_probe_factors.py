"""Trader probe 2028-03-23: validate new factor computations (read-only)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import pandas as pd
from strategy import (stock, index, compute_raw_factors, rank_series,
                      load_ensemble, frozen_set, regime_from_market)

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]

closes = {}
for a in ASSETS:
    f = stock(a)
    closes[a] = f.close.astype(float) if f is not None and "close" in f else None

vf = index("VIX")
vix_close = vf.close.astype(float) if vf is not None and "close" in vf else None

raw = compute_raw_factors(closes, vix_close, assets=ASSETS)
ens = load_ensemble()
print("ensemble:", [(f["factor_id"], f["weight"], f["direction"]) for f in ens])
print()
for fid in ["sign_ewma_60d", "vol_beta_spx_60d", "mom_120d_skip5",
            "beta_vix_60d_neg", "low_vol_20d", "down_vol_ratio_20x120"]:
    vals = raw.get(fid, {})
    nv = sum(1 for v in vals.values() if v is not None)
    r = rank_series(vals, ASSETS)
    top = sorted(ASSETS, key=lambda a: r[a], reverse=True)[:4]
    bot = sorted(ASSETS, key=lambda a: r[a])[:4]
    print(f"{fid}: nvalid={nv}/15")
    print(f"  top  : {top}")
    print(f"  bot  : {bot}")

# frozen + regime sanity
usable = [c.rename(a) for a, c in closes.items() if c is not None and len(c) >= 140]
panel = pd.concat(usable, axis=1, join="inner")
print("\nfrozen:", sorted(frozen_set(closes, ASSETS)))
print("regime:", regime_from_market(panel), "n_assets_usable:", len(usable))
