"""miner_1 (2026-08-11 revisit): sweep P - RATES CARRY and MACRO-BETA dimensions.

Fresh dimensions vs current library (dominated by price-return, vol, skew, DXY/VIX
correlation):
  - carry_yield_spread: US10Y - CN10Y yield differential, mapped to all assets
    (global rates carry regime).
  - us10y_mom_60: US10Y level momentum (rising rates = tight financial conditions).
  - dxy_mom_5: DXY level momentum (dollar regime) mapped cross-sectionally.
  - vix_mom_5: VIX direction (risk regime) - distinct from static beta_VIX.
  - rates_asset_beta: per-asset rolling beta to US10Y changes (rates sensitivity).
  - usdjpy_asset_beta: per-asset rolling beta to USDJPY (global carry-trade proxy).

Gate: abs(IC)>=0.0070 & abs(ICIR)>=0.0840 at h=10; prefer max lib corr<0.5.
US10Y/CN10Y live in stock_data (tradable); DXY/USDJPY/VIX are macro obs series.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner3_20260730_harness import (
    ASSETS, evaluate, load_closes, load_macro, STOCK_DIR, VISIBLE_END,
)
from pathlib import Path


def load_uscn():
    out = {}
    for a in ["US10Y", "CN10Y"]:
        f = STOCK_DIR / f"{a}.csv"
        df = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
        df = df[df["date"] <= VISIBLE_END]
        out[a] = df.set_index("date")["close"].astype(float)
    return out


uscn = load_uscn()
macro = load_macro()
closes = load_closes()
print("closes:", len(closes), "uscn:", list(uscn), "macro:", list(macro))

us10y = uscn["US10Y"]
cn10y = uscn["CN10Y"]

# --- 1. carry yield spread (level), mapped to all assets ---
carry = (us10y - cn10y).rename("carry")
carry_map = {a: carry for a in closes}
print("\n[carry_yield_spread] US10Y - CN10Y level (rate differential regime)")

# --- 2. US10Y momentum ---
us10y_mom = us10y.pct_change(60)
us10y_mom_map = {a: us10y_mom for a in closes}

# --- 3. DXY momentum ---
dxy_mom = macro["DXY"].pct_change(5)
dxy_mom_map = {a: dxy_mom for a in closes}

# --- 4. VIX momentum ---
vix_mom = macro["VIX"].pct_change(5)
vix_mom_map = {a: vix_mom for a in closes}

# --- 5/6. per-asset rolling beta to US10Y / USDJPY changes ---
def asset_beta(close, macro_series, n=60):
    r = close.pct_change()
    m = macro_series.pct_change()
    aligned = pd.concat([r.rename("a"), m.reindex(r.index).rename("m")], axis=1).dropna()
    cov = aligned["a"].rolling(n, min_periods=30).cov(aligned["m"])
    var = aligned["m"].rolling(n, min_periods=30).var().replace(0, np.nan)
    return (cov / var).reindex(r.index)


rates_beta = {a: asset_beta(closes[a], us10y, 60) for a in closes}
usdjpy_beta = {a: asset_beta(closes[a], macro["USDJPY"], 60) for a in closes}


for name, vals in [
    ("carry_yield_spread", carry_map),
    ("us10y_mom_60", us10y_mom_map),
    ("dxy_mom_5", dxy_mom_map),
    ("vix_mom_5", vix_mom_map),
    ("rates_beta_60", rates_beta),
    ("usdjpy_beta_60", usdjpy_beta),
]:
    try:
        evaluate(closes, vals, name, horizon=10)
    except Exception as e:
        print(name, "ERROR:", repr(e))
    print()