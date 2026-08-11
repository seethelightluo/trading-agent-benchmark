"""miner_2 2026-07-30 — Idea: volume participation / volume-price interaction.
Motivation: in a cross-asset universe, volume surges indicate attention/positioning.
Volume participation trend (20d/60d volume ratio) and price-volume correlation are
unit-free, cross-sectionally comparable, and orthogonal to price-only library factors.
Variants: vol_ratio_20x60 (participation trend), vol_z_20 (short-term volume z-score),
pv_corr_60 (correlation of daily return and log-volume over 60d).
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
from factor_validation_lib import (load_closes, load_index, validate_factor,
                                   load_library_panels, max_library_corr,
                                   print_result, IC_GATE, ICIR_GATE)

close, vol, open_, high, low = load_closes()
macro = {
    "VIX": load_index("VIX"),
    "DXY": load_index("DXY"),
    "USDCNY": load_index("USDCNY"),
    "USDJPY": load_index("USDJPY"),
    "EURUSD": load_index("EURUSD"),
}
print(f"Panel {close.index[0].date()}..{close.index[-1].date()}, {len(close)} rows x {close.shape[1]} assets")
lib = load_library_panels()
print(f"Library: {list(lib.keys())}")


def f_vol_ratio_20x60(c, v, o, h, l, m):
    v = v.replace(0, np.nan)
    return v.rolling(20).mean() / v.rolling(60).mean()


def f_vol_z_20(c, v, o, h, l, m):
    v = v.replace(0, np.nan)
    mu = v.rolling(60).mean()
    sd = v.rolling(60).std()
    return (v.rolling(20).mean() - mu) / sd.replace(0, np.nan)


def f_pv_corr_60(c, v, o, h, l, m):
    v = v.replace(0, np.nan)
    r = c.pct_change()
    lv = np.log(v)
    return r.rolling(60).corr(lv)


results = {}
for name, fn, desc in [
    ("vol_ratio_20x60", f_vol_ratio_20x60, "volume participation trend 20v/60v"),
    ("vol_z_20", f_vol_z_20, "20d avg volume z-score vs 60d"),
    ("pv_corr_60", f_pv_corr_60, "60d corr(daily ret, log volume)"),
]:
    res = validate_factor(fn, close, vol, open_, high, low, macro)
    res["max_abs_library_correlation"] = round(max_library_corr(res["panel"], lib), 4)
    results[name] = res
    print_result(f"{name} [{desc}]", res)
    print(f"  max_abs_library_correlation: {res['max_abs_library_correlation']}")

print("\n===== SUMMARY =====")
for name, res in results.items():
    ok = abs(res["ic"]) >= IC_GATE and abs(res["icir"]) >= ICIR_GATE
    print(f"{name:16s} IC={res['ic']:+.4f} ICIR={res['icir']:+.4f} hit={res['ic_hit_ratio']:.3f} "
          f"cov_ad={res['coverage_asset_days']:.3f} cov8={res['coverage_dates_ge8']:.3f} "
          f"to={res['turnover_10d_rank']:.2f} libcorr={res['max_abs_library_correlation']:.3f} -> {'PASS' if ok else 'fail'}")
