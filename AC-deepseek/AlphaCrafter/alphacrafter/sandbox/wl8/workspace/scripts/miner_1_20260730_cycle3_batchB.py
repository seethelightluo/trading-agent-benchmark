"""miner_1 factor mining cycle 3 (2026-07-30) batch B.
More NEW factor families with LOW expected correlation to the current library:
 - return autocorrelation (persistence/reversal)
 - intraday shadow/wick asymmetry (buying pressure)
 - volume/liquidity trend (Amihud ratio)
 - cross-sectional dispersion / relative reversal
 - conditional volatility regime factor (vol-trend interaction)
 - momentum curve slope (term structure of momentum)
 - VWAP deviation (volume-weighted price anchor)
"""
import sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from factor_validation_lib import (load_closes, load_index, validate_factor,
                                   max_library_corr, print_result, IC_GATE, ICIR_GATE)

close, vol, open_, high, low = load_closes()
macro = {
    "VIX": load_index("VIX"), "DXY": load_index("DXY"), "USDCNY": load_index("USDCNY"),
    "USDJPY": load_index("USDJPY"), "EURUSD": load_index("EURUSD"),
}
print(f"Panel dates {close.index[0].date()}..{close.index[-1].date()}, {len(close)} rows, {close.shape[1]} assets")


def load_lib_all():
    import base64, zlib, io
    lib = {}
    for fid in ["mom_10d_skip5", "vix_beta_cond_60x20", "yield_beta_cond_60x20"]:
        try:
            d = json.load(open(f"factors/{fid}.json"))
            raw = base64.b64decode(d["validation"]["signal_artifact"]["data"])
            panel = pd.read_csv(io.StringIO(zlib.decompress(raw).decode()),
                                index_col=0, parse_dates=True)
            panel.index = pd.DatetimeIndex(panel.index)
            lib[fid] = panel
        except Exception as e:
            print(f"  [warn] could not load {fid}: {e}")
    return lib


lib = load_lib_all()
print(f"Library panels loaded: {list(lib.keys())}")

CS_MEAN_5 = close.pct_change(5).mean(axis=1)
EW_RET = close.pct_change().mean(axis=1)

# ---------------- candidate definitions (per-asset dense series) ----------------
def _autocorr(x, lag):
    x = x.dropna()
    if len(x) < lag + 5:
        return pd.Series(np.nan, index=x.index)
    a = x.iloc[lag:]
    b = x.iloc[:-lag]
    ma = a.rolling(60).mean()
    mb = b.rolling(60).mean()
    cov = ((a - ma) * (b - mb)).rolling(60).mean()
    va = a.rolling(60).var().clip(lower=1e-12)
    vb = b.rolling(60).var().clip(lower=1e-12)
    out = (cov / np.sqrt(va * vb))
    return out.reindex(x.index)

def f_autocorr_1_60(c, v, o, h, l, m):
    r = c.pct_change()
    return _autocorr(r, 1)

def f_autocorr_5_60(c, v, o, h, l, m):
    r = c.pct_change()
    return _autocorr(r, 5)

def f_upper_shadow_20(c, v, o, h, l, m):
    rng = (h - l).replace(0, np.nan)
    shadow = (h - np.maximum(o, c)) / rng
    return shadow.rolling(20).mean()

def f_lower_shadow_20(c, v, o, h, l, m):
    rng = (h - l).replace(0, np.nan)
    shadow = (np.minimum(o, c) - l) / rng
    return shadow.rolling(20).mean()

def f_net_shadow_20(c, v, o, h, l, m):
    rng = (h - l).replace(0, np.nan)
    up = (h - np.maximum(o, c)) / rng
    dn = (np.minimum(o, c) - l) / rng
    return (up - dn).rolling(20).mean()

def f_amihud_ratio_20x60(c, v, o, h, l, m):
    if v is None:
        return pd.Series(np.nan, index=c.index)
    am = (c.pct_change().abs() / v.replace(0, np.nan))
    a20 = am.rolling(20).mean()
    a60 = am.rolling(60).mean()
    return a20 / a60.clip(lower=1e-12)

def f_vol_trend_20x60(c, v, o, h, l, m):
    if v is None:
        return pd.Series(np.nan, index=c.index)
    v20 = v.rolling(20).mean()
    v60 = v.rolling(60).mean()
    return v20 / v60.clip(lower=1e-12)

def f_dispersion_5(c, v, o, h, l, m):
    """Asset's 5d return deviation from cross-sectional mean (divergence)."""
    r5 = c / c.shift(5) - 1.0
    cs = CS_MEAN_5.reindex(c.index)
    return (r5 - cs).abs()

def f_rel_rev_5(c, v, o, h, l, m):
    """Relative short-term reversal: -(asset 5d ret - cross-sectional mean)."""
    r5 = c / c.shift(5) - 1.0
    cs = CS_MEAN_5.reindex(c.index)
    return -(r5 - cs)

def f_mom_slope_60x20(c, v, o, h, l, m):
    """Term-structure of momentum: 60d momentum minus 20d momentum."""
    return (c / c.shift(60) - 1.0) - (c / c.shift(20) - 1.0)

def f_vwap_dev_20(c, v, o, h, l, m):
    if v is None:
        return pd.Series(np.nan, index=c.index)
    tp = (h + l + c) / 3.0
    vwap = (tp * v).rolling(20).sum() / v.rolling(20).sum().clip(lower=1e-12)
    return c / vwap - 1.0

def f_vol_skew_20(c, v, o, h, l, m):
    """Up-move vs down-move volatility asymmetry (vol skew proxy)."""
    r = c.pct_change()
    up = r.clip(lower=0).rolling(20).std()
    dn = (-r.clip(upper=0)).rolling(20).std()
    return (up - dn) / (up + dn).clip(lower=1e-12)

CANDIDATES = [
    ("autocorr_1_60", f_autocorr_1_60, "60d autocorrelation of daily returns at lag 1"),
    ("autocorr_5_60", f_autocorr_5_60, "60d autocorrelation of daily returns at lag 5"),
    ("upper_shadow_20", f_upper_shadow_20, "20d mean upper wick / daily range (selling pressure)"),
    ("lower_shadow_20", f_lower_shadow_20, "20d mean lower wick / daily range (buying support)"),
    ("net_shadow_20", f_net_shadow_20, "20d mean (upper-lower) wick / range"),
    ("amihud_ratio_20x60", f_amihud_ratio_20x60, "20d/60d Amihud illiquidity ratio"),
    ("vol_trend_20x60", f_vol_trend_20x60, "20d/60d raw volume trend"),
    ("dispersion_5", f_dispersion_5, "|asset 5d ret - cross-sectional mean| (divergence)"),
    ("rel_rev_5", f_rel_rev_5, "relative 5d reversal vs cross-sectional mean"),
    ("mom_slope_60x20", f_mom_slope_60x20, "60d mom minus 20d mom (momentum term slope)"),
    ("vwap_dev_20", f_vwap_dev_20, "close vs 20d VWAP deviation"),
    ("vol_skew_20", f_vol_skew_20, "20d up-vol vs down-vol asymmetry"),
]

results = {}
for name, fn, desc in CANDIDATES:
    try:
        res = validate_factor(fn, close, vol, open_, high, low, macro)
        res["max_abs_library_correlation"] = round(max_library_corr(res["panel"], lib), 4)
        results[name] = res
        print_result(f"{name} [{desc}]", res)
        print(f"  max_abs_library_correlation: {res['max_abs_library_correlation']}")
    except Exception as e:
        print(f"[ERROR] {name}: {type(e).__name__}: {e}")

print("\n\n===== SUMMARY BATCH B =====")
for name, res in results.items():
    ok = abs(res["ic"]) >= IC_GATE and abs(res["icir"]) >= ICIR_GATE
    print(f"{name:20s} IC={res['ic']:+.4f} ICIR={res['icir']:+.4f} hit={res['ic_hit_ratio']:.3f} "
          f"n={res['n_ic_dates']:4d} cov_ad={res['coverage_asset_days']:.3f} cov8={res['coverage_dates_ge8']:.3f} "
          f"to={res['turnover_10d_rank']:.2f} libcorr={res['max_abs_library_correlation']:.3f} -> {'PASS' if ok else 'fail'}")

with open("scripts/_miner1_cycle3_batchB_results.json", "w") as f:
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "panel"} for k, v in results.items()},
              f, indent=1, default=str)
print("\nSaved scripts/_miner1_cycle3_batchB_results.json")
