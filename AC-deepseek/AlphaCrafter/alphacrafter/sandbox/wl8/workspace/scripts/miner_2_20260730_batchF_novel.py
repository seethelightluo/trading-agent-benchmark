"""miner_2 2026-07-30 — Batch F: novel dynamics / cross-asset ratio factors.
Motivation: the library is concentrated in USDCNY-beta (sparse, EM/China FX
exposure). We need factor families that (a) pass IC/ICIR gates and (b) are
orthogonal to usdcny_beta_60 (pooled |rho| < 0.5). Ideas here are genuinely
novel relative to prior batches (structure/orthogonal/dynamics/macrobets):
  1. vr_ratio_10       - Lo-MacKinlay variance ratio (trend persistence vs MR)
  2. gk_cc_ratio_20    - Garman-Klass intraday vol / close-to-close vol
                         (overnight information share)
  3. yld_spread_beta_60- beta of asset returns to (US10Y - CN10Y) spread moves
  4. csd_beta_60       - beta of asset returns to cross-sectional dispersion
  5. xau_cop_beta_60   - beta of asset returns to XAU/COPPER ratio moves
  6. btc_eth_beta_60   - beta of asset returns to BTC/ETH ratio moves
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
                                   max_library_corr, print_result,
                                   IC_GATE, ICIR_GATE)

close, vol, open_, high, low = load_closes()
macro = {k: load_index(k) for k in ["VIX", "DXY", "USDCNY", "USDJPY", "EURUSD"]}
# cross-sectional dispersion state variable (union panel)
ret_panel = close.pct_change()
csd = ret_panel.std(axis=1, ddof=0)
macro["CSD"] = csd
print(f"Panel {close.index[0].date()}..{close.index[-1].date()}, "
      f"{len(close)} rows x {close.shape[1]} assets")


def load_effective_library():
    """Load currently EFFECTIVE factor signal artifacts from factors/ (non-bak)."""
    lib = {}
    for f in ["usdcny_beta_60"]:
        try:
            d = json.load(open(f"factors/{f}.json"))
            if d.get("validation", {}).get("status") != "EFFECTIVE":
                continue
            raw = base64.b64decode(d["validation"]["signal_artifact"]["data"])
            panel = pd.read_csv(io.StringIO(zlib.decompress(raw).decode()),
                                index_col=0, parse_dates=True)
            panel.index = pd.DatetimeIndex(panel.index)
            lib[f] = panel
        except Exception as e:
            print(f"  [warn] library load {f}: {e}")
    return lib


lib = load_effective_library()
print(f"Effective library: {list(lib.keys())}")


def _beta(asset_ret, driver_ret, win):
    cov = asset_ret.rolling(win).cov(driver_ret)
    var = driver_ret.rolling(win).var()
    return cov / var


def f_vr_ratio_10(c, v, o, h, l, m, win=60, hz=10):
    """Variance ratio Var(hz)/ (hz*Var(1)): >1 trending, <1 mean-reverting."""
    ret1 = c.pct_change()
    v1 = ret1.rolling(win).var()
    rh = c.pct_change(hz)
    vh = rh.rolling(win).var()
    return vh / (hz * v1)


def f_gk_cc_ratio_20(c, v, o, h, l, m, win=20):
    """Garman-Klass (OHLC) variance / close-to-close variance."""
    lc = np.log(c)
    lo_ = np.log(o)
    lh = np.log(h)
    ll = np.log(l)
    gk = 0.5 * (lh - ll) ** 2 - (2 * np.log(2) - 1) * (lc - lo_) ** 2
    gk_var = gk.rolling(win).mean()
    cc_var = lc.diff().rolling(win).var()
    return gk_var / cc_var


def f_yld_spread_beta_60(c, v, o, h, l, m, win=60):
    """Beta of asset returns to US10Y - CN10Y yield spread changes."""
    us10 = close["US10Y"].reindex(c.index)
    cn10 = close["CN10Y"].reindex(c.index)
    dspread = (us10 - cn10).diff()
    return _beta(c.pct_change(), dspread, win)


def f_csd_beta_60(c, v, o, h, l, m, win=60):
    """Beta of asset returns to cross-sectional return dispersion changes."""
    dcsd = m["CSD"].reindex(c.index).diff()
    return _beta(c.pct_change(), dcsd, win)


def f_xau_cop_beta_60(c, v, o, h, l, m, win=60):
    """Beta of asset returns to XAU/COPPER ratio moves (risk-on commodity)."""
    xau = close["XAU"].reindex(c.index)
    cop = close["COPPER"].reindex(c.index)
    r = (xau / cop).pct_change()
    return _beta(c.pct_change(), r, win)


def f_btc_eth_beta_60(c, v, o, h, l, m, win=60):
    """Beta of asset returns to BTC/ETH ratio moves (crypto internal risk)."""
    btc = close["BTC"].reindex(c.index)
    eth = close["ETH"].reindex(c.index)
    r = (btc / eth).pct_change()
    return _beta(c.pct_change(), r, win)


results = {}
for name, fn, desc in [
    ("vr_ratio_10", f_vr_ratio_10, "variance ratio 10d/1d (trend persistence)"),
    ("gk_cc_ratio_20", f_gk_cc_ratio_20, "GK intraday vol / close-close vol"),
    ("yld_spread_beta_60", f_yld_spread_beta_60, "beta to US-CN yield spread"),
    ("csd_beta_60", f_csd_beta_60, "beta to cross-sectional dispersion"),
    ("xau_cop_beta_60", f_xau_cop_beta_60, "beta to XAU/COPPER ratio"),
    ("btc_eth_beta_60", f_btc_eth_beta_60, "beta to BTC/ETH ratio"),
]:
    res = validate_factor(fn, close, vol, open_, high, low, macro)
    res["max_abs_library_correlation"] = round(max_library_corr(res["panel"], lib), 4)
    results[name] = res
    print_result(f"{name} [{desc}]", res)
    print(f"  max_abs_library_correlation: {res['max_abs_library_correlation']}")

print("\n===== SUMMARY =====")
for name, res in results.items():
    ok = abs(res["ic"]) >= IC_GATE and abs(res["icir"]) >= ICIR_GATE
    ok2 = res["max_abs_library_correlation"] < 0.5
    print(f"{name:20s} IC={res['ic']:+.4f} ICIR={res['icir']:+.4f} hit={res['ic_hit_ratio']:.3f} "
          f"cov_ad={res['coverage_asset_days']:.3f} cov8={res['coverage_dates_ge8']:.3f} "
          f"to={res['turnover_10d_rank']:.2f} libcorr={res['max_abs_library_correlation']:.3f} "
          f"-> {'PASS' if (ok and ok2) else 'fail'}")
