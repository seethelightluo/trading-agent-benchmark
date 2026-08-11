"""miner_1 factor mining cycle 3 (2026-07-30) batch A.
Explore NEW factor families with LOW expected correlation to the current library
(mom_10d_skip5, vix_beta_cond_60x20, yield_beta_cond_60x20):
 - range/oscillator (mean reversion): stochastic %K, RSI, Bollinger %B
 - drawdown / distance from high
 - short-term reversal scaled by vol
 - extreme return (lottery) factors
Admission gates: |IC|>=0.0070, |ICIR|>=0.0840 at h=10, min 8 assets/date.
Library correlation threshold: 0.5 (deterministic gate recomputes from artifacts).
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
    """Load signal artifacts of the 3 current effective library factors."""
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
            print(f"  loaded library panel: {fid} {panel.shape}")
        except Exception as e:
            print(f"  [warn] could not load {fid}: {e}")
    return lib


lib = load_lib_all()

CS_MEAN_5 = close.pct_change(5).mean(axis=1)

# ---------------- candidate definitions (per-asset dense series) ----------------
def f_stoch_k_14(c, v, o, h, l, m):
    ll = l.rolling(14).min()
    hh = h.rolling(14).max()
    rng = (hh - ll).replace(0, np.nan)
    return (c - ll) / rng

def f_rsi_14(c, v, o, h, l, m):
    r = c.pct_change()
    gain = r.clip(lower=0).rolling(14).mean()
    loss = (-r.clip(upper=0)).rolling(14).mean()
    denom = (gain + loss).replace(0, np.nan)
    return gain / denom

def f_bb_pctb_20(c, v, o, h, l, m):
    r = c.pct_change()
    sma = c.rolling(20).mean()
    sd = r.rolling(20).std().clip(lower=1e-12)
    return (c - sma) / (2 * sd * c.rolling(20).mean().replace(0, np.nan)).replace(0, np.nan)

def f_bb_pctb_20_v2(c, v, o, h, l, m):
    sma = c.rolling(20).mean()
    sd = c.rolling(20).std().clip(lower=1e-12)
    return (c - sma) / (2 * sd)

def f_dist_high_20(c, v, o, h, l, m):
    return c / c.rolling(20).max() - 1.0

def f_dist_high_60(c, v, o, h, l, m):
    return c / c.rolling(60).max() - 1.0

def f_rev5_vol20(c, v, o, h, l, m):
    r = c.pct_change()
    rev = -(c / c.shift(5) - 1.0)
    return rev / r.rolling(20).std().clip(lower=1e-12)

def f_rev1_vol20(c, v, o, h, l, m):
    r = c.pct_change()
    rev = -r
    return rev / r.rolling(20).std().clip(lower=1e-12)

def f_max_ret_20(c, v, o, h, l, m):
    return c.pct_change().rolling(20).max()

def f_min_ret_20(c, v, o, h, l, m):
    return c.pct_change().rolling(20).min()

def f_extreme_spread_20(c, v, o, h, l, m):
    r = c.pct_change()
    return r.rolling(20).max() - r.rolling(20).min()

CANDIDATES = [
    ("stoch_k_14", f_stoch_k_14, "stochastic %K: close position in 14d high-low range"),
    ("rsi_14", f_rsi_14, "RSI(14): 14d mean gain / (gain+loss)"),
    ("bb_pctb_20", f_bb_pctb_20, "Bollinger %B: (close-sma20)/(2*20d price std)"),
    ("bb_pctb_20_v2", f_bb_pctb_20_v2, "Bollinger %B v2: (close-sma20)/(2*std of close)"),
    ("dist_high_20", f_dist_high_20, "distance from 20d rolling max (drawdown)"),
    ("dist_high_60", f_dist_high_60, "distance from 60d rolling max (drawdown)"),
    ("rev5_vol20", f_rev5_vol20, "-5d return / 20d vol (vol-scaled reversal)"),
    ("rev1_vol20", f_rev1_vol20, "-1d return / 20d vol (vol-scaled 1d reversal)"),
    ("max_ret_20", f_max_ret_20, "max daily return over 20d (lottery)"),
    ("min_ret_20", f_min_ret_20, "min daily return over 20d (crash indicator)"),
    ("extreme_spread_20", f_extreme_spread_20, "20d max - min daily return (range)"),
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

print("\n\n===== SUMMARY BATCH A =====")
for name, res in results.items():
    ok = abs(res["ic"]) >= IC_GATE and abs(res["icir"]) >= ICIR_GATE
    print(f"{name:18s} IC={res['ic']:+.4f} ICIR={res['icir']:+.4f} hit={res['ic_hit_ratio']:.3f} "
          f"n={res['n_ic_dates']:4d} cov_ad={res['coverage_asset_days']:.3f} cov8={res['coverage_dates_ge8']:.3f} "
          f"to={res['turnover_10d_rank']:.2f} libcorr={res['max_abs_library_correlation']:.3f} -> {'PASS' if ok else 'fail'}")

with open("scripts/_miner1_cycle3_batchA_results.json", "w") as f:
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "panel"} for k, v in results.items()},
              f, indent=1, default=str)
print("\nSaved scripts/_miner1_cycle3_batchA_results.json")
