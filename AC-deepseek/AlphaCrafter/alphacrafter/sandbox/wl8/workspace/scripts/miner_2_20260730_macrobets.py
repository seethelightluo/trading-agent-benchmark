"""miner_2 2026-07-30 — Idea: cross-asset macro-beta factors.
Motivation: in a cross-asset universe, each asset's sensitivity to global macro
drivers (USD, risk-on/off JPY carry, rates, equity vol) encodes its risk regime
role. Rolling beta to a macro return is unit-free and cross-sectionally comparable.
Variants: DXY beta, USDJPY beta, US10Y-rate beta, raw VIX beta (library has a
conditional VIX-beta; raw beta is tested for incremental value).
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
from factor_validation_lib import (load_closes, load_index, validate_factor,
                                   load_library_panels, max_library_corr,
                                   print_result, IC_GATE, ICIR_GATE, artifact_b64)

close, vol, open_, high, low = load_closes()
macro = {k: load_index(k) for k in ["VIX", "DXY", "USDCNY", "USDJPY", "EURUSD"]}
print(f"Panel {close.index[0].date()}..{close.index[-1].date()}, {len(close)} rows x {close.shape[1]} assets")
lib = load_library_panels()
# add newly persisted pv_corr_60 to the library for correlation screening
import json, base64, zlib, io
try:
    d = json.load(open("factors/pv_corr_60.json"))
    raw = base64.b64decode(d["validation"]["signal_artifact"]["data"])
    lib["pv_corr_60"] = pd.read_csv(io.StringIO(zlib.decompress(raw).decode()), index_col=0, parse_dates=True)
except Exception as e:
    print("warn: no pv_corr_60 artifact", e)
print(f"Library: {list(lib.keys())}")


def _beta(asset_ret, macro_ret, win):
    cov = asset_ret.rolling(win).cov(macro_ret)
    var = macro_ret.rolling(win).var()
    return cov / var


def f_dxy_beta_60(c, v, o, h, l, m):
    dxy = m["DXY"].reindex(c.index).pct_change()
    return _beta(c.pct_change(), dxy, 60)


def f_usdjpy_beta_60(c, v, o, h, l, m):
    jpy = m["USDJPY"].reindex(c.index).pct_change()
    return _beta(c.pct_change(), jpy, 60)


def f_rate_beta_60(c, v, o, h, l, m):
    # US10Y is a tradable asset; yield changes drive rates-sensitive assets
    us10 = close["US10Y"].reindex(c.index).diff()
    return _beta(c.pct_change(), us10, 60)


def f_vix_beta_60(c, v, o, h, l, m):
    vix = m["VIX"].reindex(c.index).pct_change()
    return _beta(c.pct_change(), vix, 60)


def f_dxy_beta_cond_60x20(c, v, o, h, l, m):
    """Beta to DXY times 20d DXY move: hedge-demand conditional signal."""
    dxy = m["DXY"].reindex(c.index)
    b = _beta(c.pct_change(), dxy.pct_change(), 60)
    return b * (dxy / dxy.shift(20) - 1.0)


results = {}
for name, fn, desc in [
    ("dxy_beta_60", f_dxy_beta_60, "60d beta to DXY returns"),
    ("usdjpy_beta_60", f_usdjpy_beta_60, "60d beta to USDJPY returns (risk-on)"),
    ("rate_beta_60", f_rate_beta_60, "60d beta to US10Y yield change"),
    ("vix_beta_60", f_vix_beta_60, "60d raw beta to VIX returns"),
    ("dxy_beta_cond_60x20", f_dxy_beta_cond_60x20, "60d DXY-beta x 20d DXY move"),
]:
    res = validate_factor(fn, close, vol, open_, high, low, macro)
    res["max_abs_library_correlation"] = round(max_library_corr(res["panel"], lib), 4)
    results[name] = res
    print_result(f"{name} [{desc}]", res)
    print(f"  max_abs_library_correlation: {res['max_abs_library_correlation']}")

print("\n===== SUMMARY =====")
for name, res in results.items():
    ok = abs(res["ic"]) >= IC_GATE and abs(res["icir"]) >= ICIR_GATE
    print(f"{name:22s} IC={res['ic']:+.4f} ICIR={res['icir']:+.4f} hit={res['ic_hit_ratio']:.3f} "
          f"cov_ad={res['coverage_asset_days']:.3f} cov8={res['coverage_dates_ge8']:.3f} "
          f"to={res['turnover_10d_rank']:.2f} libcorr={res['max_abs_library_correlation']:.3f} -> {'PASS' if ok else 'fail'}")
