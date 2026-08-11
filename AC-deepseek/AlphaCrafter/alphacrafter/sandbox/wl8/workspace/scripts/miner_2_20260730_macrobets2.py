"""miner_2 2026-07-30 -- macro-beta family v2 (ffill fix).
v1 had ~10% coverage because rolling(60).cov() returns NaN whenever ANY of the
60 window rows is NaN (macro calendars don't perfectly align with asset
calendars, e.g. crypto 7d/wk vs macro 5d/wk). Fix: reindex macro onto the
asset's dense calendar and ffill so beta windows are complete.
Also expands family: raw betas + conditional (beta x 20d macro move) variants
for DXY, USDJPY, EURUSD, USDCNY, VIX, and US10Y (rate) on 60d windows.
"""
import sys
import json
import base64
import zlib
import io

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from factor_validation_lib import (load_closes, load_index, validate_factor,
                                   load_library_panels, max_library_corr,
                                   print_result, IC_GATE, ICIR_GATE, artifact_b64)

close, vol, open_, high, low = load_closes()
macro = {k: load_index(k) for k in ["VIX", "DXY", "USDCNY", "USDJPY", "EURUSD"]}
macro["US10Y"] = close["US10Y"].dropna()
print(f"Panel {close.index[0].date()}..{close.index[-1].date()}, {len(close)} rows x {close.shape[1]} assets")
lib = load_library_panels()
print(f"Library: {list(lib.keys())}")


def _beta(asset_ret, macro_ret, win):
    cov = asset_ret.rolling(win).cov(macro_ret)
    var = macro_ret.rolling(win).var()
    return cov / var


def _m(c, m, key):
    """Reindex macro onto asset calendar and ffill gaps."""
    return m[key].reindex(c.index).ffill()


def make_raw(key, desc, win=60, use_diff=False):
    def fn(c, v, o, h, l, m, win=win):
        ms = _m(c, m, key)
        mr = ms.diff() if use_diff else ms.pct_change()
        return _beta(c.pct_change(), mr, win)
    fn.__name__ = f"{key.lower()}_beta_{win}"
    return fn, f"{win}d beta to {key} returns"


def make_cond(key, desc, win=60, look=20, use_diff=False):
    def fn(c, v, o, h, l, m, win=win, look=look):
        ms = _m(c, m, key)
        mr = ms.diff() if use_diff else ms.pct_change()
        b = _beta(c.pct_change(), mr, win)
        move = ms / ms.shift(look) - 1.0 if not use_diff else ms.diff(look)
        return b * move
    fn.__name__ = f"{key.lower()}_beta_cond_{win}x{look}"
    return fn, f"{win}d beta to {key} x {look}d {key} move"


cands = [
    ("dxy_beta_60", *make_raw("DXY", "USD index beta")),
    ("usdjpy_beta_60", *make_raw("USDJPY", "JPY carry beta")),
    ("eurusd_beta_60", *make_raw("EURUSD", "EUR beta")),
    ("usdcny_beta_60", *make_raw("USDCNY", "CNY beta")),
    ("vix_beta_60", *make_raw("VIX", "VIX beta")),
    ("rate_beta_60", *make_raw("US10Y", "US10Y yield beta", use_diff=True)),
    ("dxy_beta_cond_60x20", *make_cond("DXY", "DXY beta x move")),
    ("usdjpy_beta_cond_60x20", *make_cond("USDJPY", "USDJPY beta x move")),
    ("vix_beta_cond_120x20", *make_cond("VIX", "VIX beta x move", win=120)),
    ("rate_beta_cond_60x20", *make_cond("US10Y", "yield beta x move", use_diff=True)),
]

results = {}
for name, fn, desc in cands:
    res = validate_factor(fn, close, vol, open_, high, low, macro)
    res["max_abs_library_correlation"] = round(max_library_corr(res["panel"], lib), 4)
    results[name] = res
    print_result(f"{name} [{desc}]", res)
    print(f"  max_abs_library_correlation: {res['max_abs_library_correlation']}")

print("\n===== SUMMARY =====")
for name, res in results.items():
    ok = abs(res["ic"]) >= IC_GATE and abs(res["icir"]) >= ICIR_GATE
    print(f"{name:26s} IC={res['ic']:+.4f} ICIR={res['icir']:+.4f} hit={res['ic_hit_ratio']:.3f} "
          f"cov_ad={res['coverage_asset_days']:.3f} cov8={res['coverage_dates_ge8']:.3f} "
          f"to={res['turnover_10d_rank']:.2f} n={res['n_ic_dates']} libcorr={res['max_abs_library_correlation']:.3f} -> {'PASS' if ok else 'fail'}")
