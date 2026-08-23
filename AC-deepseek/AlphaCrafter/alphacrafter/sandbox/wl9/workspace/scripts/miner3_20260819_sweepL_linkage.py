"""Exploration sweep L (miner_3, 2026-08-19): cross-asset linkage / beta-to-benchmark dims.

Library now covers momentum, vol, range, skew, VIX beta, vol_z, days_since_high,
kaufman_eff, kurt, streak, dxy_corr_change. We target FRESH linkage dimensions:
  - sox_beta_60  : rolling beta of each asset on SOX (semiconductor/tech risk-on)
  - gold_beta_60 : rolling beta of each asset on XAU (safe-haven correlation)
  - eurusd_beta_60: rolling beta on EURUSD (risk-on fx / dollar weakness)
  - cny_beta_60  : rolling beta on USDCNY (China currency / renminbi stress)
  - dxy_vol_20   : rolling correlation of asset return with DXY return (dollar linkage)
  - us10y_beta_60: rolling beta of each RISK asset on US10Y (rate sensitivity)
  - corr_ndx_btc_60: rolling corr of asset with BTC (crypto risk appetite)

Gate: abs(IC)>=0.0070 & abs(ICIR)>=0.0840 at h=10; persistence needs max lib corr <0.5.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner3_20260730_harness import ASSETS, evaluate, load_closes, load_macro

closes = load_closes()
macro = load_macro()


def rolling_beta(asset_r, mkt_r, w, minp=40):
    df = pd.concat([asset_r.rename("a"), mkt_r.rename("m")], axis=1)
    beta = df["a"].rolling(w, min_periods=minp).cov(df["m"]) / df["m"].rolling(w, min_periods=minp).var()
    return beta


def rolling_corr(asset_r, mkt_r, w, minp=40):
    df = pd.concat([asset_r.rename("a"), mkt_r.rename("m")], axis=1)
    return df["a"].rolling(w, min_periods=minp).corr(df["m"])


def ret_of(s):
    return s.pct_change()


# --- price returns for risk assets ---
def rt(a):
    return ret_of(closes[a])


def mrt(name):
    return macro[name].pct_change()


# reference series available from closes
mkt = {a: rt(a) for a in closes}

candidates = {
    "sox_beta_60": {a: rolling_beta(rt(a), mkt["SOX"], 60) for a in closes},
    "gold_beta_60": {a: rolling_beta(rt(a), mkt["XAU"], 60) for a in closes},
    "eurusd_beta_60": {a: rolling_beta(rt(a), mrt("EURUSD"), 60) for a in closes},
    "cny_beta_60": {a: rolling_beta(rt(a), mrt("USDCNY"), 60) for a in closes},
    "dxy_corr_20": {a: rolling_corr(rt(a), mrt("DXY"), 20, minp=10) for a in closes},
    "us10y_beta_60": {a: rolling_beta(rt(a), mkt["US10Y"], 60) for a in closes},
    "corr_btc_60": {a: rolling_corr(rt(a), mkt["BTC"], 60) for a in closes},
}

print("assets:", len(closes), "macro:", len(macro))
for name, vals in candidates.items():
    try:
        evaluate(closes, vals, name, horizon=10)
    except Exception as e:
        print(name, "ERROR:", repr(e))
    print()