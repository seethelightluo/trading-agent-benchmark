"""miner_2 2026-07-30 — Idea: realized skewness (lottery effect).
Motivation: positive-skew assets behave like lottery tickets (retail demand,
overpricing) and should earn lower forward returns => negative IC expected.
Cross-sectional validation on the 15-asset tradable universe, window to 2026-07-30.
Variants: 20d and 60d realized skewness of daily returns.
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


def f_skew_20(c, v, o, h, l, m):
    r = c.pct_change()
    return r.rolling(20).skew()


def f_skew_60(c, v, o, h, l, m):
    r = c.pct_change()
    return r.rolling(60).skew()


results = {}
for name, fn in [("skew_20", f_skew_20), ("skew_60", f_skew_60)]:
    res = validate_factor(fn, close, vol, open_, high, low, macro)
    res["max_abs_library_correlation"] = round(max_library_corr(res["panel"], lib), 4)
    results[name] = res
    print_result(name, res)
    print(f"  max_abs_library_correlation: {res['max_abs_library_correlation']}")

print("\n===== SUMMARY =====")
for name, res in results.items():
    ok = abs(res["ic"]) >= IC_GATE and abs(res["icir"]) >= ICIR_GATE
    print(f"{name:8s} IC={res['ic']:+.4f} ICIR={res['icir']:+.4f} hit={res['ic_hit_ratio']:.3f} "
          f"cov_ad={res['coverage_asset_days']:.3f} cov8={res['coverage_dates_ge8']:.3f} "
          f"to={res['turnover_10d_rank']:.2f} libcorr={res['max_abs_library_correlation']:.3f} -> {'PASS' if ok else 'fail'}")
