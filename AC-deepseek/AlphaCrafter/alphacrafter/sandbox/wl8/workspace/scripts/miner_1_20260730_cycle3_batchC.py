"""miner_1 factor mining cycle 3 (2026-07-30) batch C.
Fixes autocorr implementation (integer-index rolling corr) and adds more
distinct families: return skewness, range/volatility trend, intraday
sentiment persistence, conditional mean-reversion (RSI x VIX regime),
and up/down move asymmetry.
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


# ---------------- candidate definitions (per-asset dense series) ----------------
def _autocorr(x, lag):
    x = x.dropna()
    if len(x) < lag + 65:
        return pd.Series(np.nan, index=x.index)
    r = pd.Series(x.pct_change().to_numpy())
    rl = r.shift(lag)
    num = (r * rl).rolling(60).mean() - r.rolling(60).mean() * rl.rolling(60).mean()
    den = r.rolling(60).std() * rl.rolling(60).std()
    ac = num / den.replace(0, np.nan)
    return pd.Series(ac.to_numpy(), index=x.index)


def f_autocorr_1_60(c, v, o, h, l, m):
    return _autocorr(c, 1)


def f_autocorr_5_60(c, v, o, h, l, m):
    return _autocorr(c, 5)


def f_ret_skew_60(c, v, o, h, l, m):
    r = c.pct_change()
    return r.rolling(60).skew()


def f_range_ratio_20x60(c, v, o, h, l, m):
    rng = (h - l) / c.replace(0, np.nan)
    return rng.rolling(20).mean() / rng.rolling(60).mean().clip(lower=1e-12)


def f_intraday_mom_20(c, v, o, h, l, m):
    idr = (c - o) / o.replace(0, np.nan)
    return idr.rolling(20).mean()


def f_rsi14_vix_cond(c, v, o, h, l, m):
    """Mean-reversion strength conditioned on VIX regime:
    RSI-14 (0..1 scale) x (VIX percentile rank over 252d - 0.5)."""
    delta = c.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    dn = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = up / (up + dn).clip(lower=1e-12)
    vix = m["VIX"].reindex(c.index)
    pct = vix.rolling(252).rank(pct=True)
    return (rsi - 0.5) * (pct - 0.5).fillna(0.0)


def f_up_down_ratio_20(c, v, o, h, l, m):
    r = c.pct_change()
    up = r.clip(lower=0).rolling(20).mean()
    dn = (-r.clip(upper=0)).rolling(20).mean()
    return up / (up + dn).clip(lower=1e-12)


CANDIDATES = [
    ("autocorr_1_60", f_autocorr_1_60, "60d lag-1 autocorrelation of daily returns"),
    ("autocorr_5_60", f_autocorr_5_60, "60d lag-5 autocorrelation of daily returns"),
    ("ret_skew_60", f_ret_skew_60, "60d skewness of daily returns"),
    ("range_ratio_20x60", f_range_ratio_20x60, "20d/60d mean daily range ratio (vol trend)"),
    ("intraday_mom_20", f_intraday_mom_20, "20d mean of (close-open)/open (intraday persistence)"),
    ("rsi14_vix_cond", f_rsi14_vix_cond, "RSI-14 x VIX-percentile regime conditioner"),
    ("up_down_ratio_20", f_up_down_ratio_20, "20d up-move share of |daily moves|"),
]

results = {}
for name, fn, desc in CANDIDATES:
    try:
        res = validate_factor(fn, close, vol, open_, high, low, macro)
        res["max_abs_library_correlation"] = round(max_library_corr(res["panel"], lib), 4)
        results[name] = res
        print_result(f"{name} [{desc}]", res)
        print(f"  max_abs_library_correlation: {res['max_abs_library_correlation']:.4f}")
    except Exception as e:
        print(f"=== {name} === ERROR: {e}")

print("\n===== SUMMARY BATCH C =====")
for name, fn, desc in CANDIDATES:
    r = results.get(name)
    if r is None:
        print(f"{name:22s} ERROR")
        continue
    ok = abs(r["ic"]) >= IC_GATE and abs(r["icir"]) >= ICIR_GATE
    print(f"{name:22s} IC={r['ic']:+.4f} ICIR={r['icir']:+.4f} hit={r['ic_hit_ratio']:.3f} "
          f"n={r['n_ic_dates']:4d} cov_ad={r['coverage_asset_days']:.3f} cov8={r['coverage_dates_ge8']:.3f} "
          f"to={r['turnover_10d_rank']:.2f} libcorr={r['max_abs_library_correlation']:.3f} "
          f"-> {'PASS' if ok else 'fail'}")

with open("scripts/_miner1_cycle3_batchC_results.json", "w") as f:
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "panel"} for k, v in results.items()},
              f, indent=1, default=str)
print("\nSaved scripts/_miner1_cycle3_batchC_results.json")
