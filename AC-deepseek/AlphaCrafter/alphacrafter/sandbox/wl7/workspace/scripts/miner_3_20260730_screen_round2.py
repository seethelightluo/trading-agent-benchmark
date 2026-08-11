"""miner_3 screening round 2: volume, overnight/intraday, high-proximity, beta-to-SPX."""
from __future__ import annotations
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
from miner3_lib import load_panel, load_macro, per_asset, validate_factor, WATCH

_VOL_CACHE = {}

def _vol_series(sym):
    if sym in _VOL_CACHE:
        return _VOL_CACHE[sym]
    df = get_stock_daily_data(sym, days=4000)
    vol = None
    if df is not None and "volume" in df.columns:
        vol = df.set_index("date")["volume"].astype(float)
        vol = vol[~vol.index.duplicated(keep="last")].sort_index()
    _VOL_CACHE[sym] = vol
    return vol

def with_volume(panel, fn):
    cols = {}
    for a in panel.columns:
        s = panel[a].dropna()
        v = _vol_series(a)
        if v is None:
            continue
        vv = v.reindex(s.index)
        cols[a] = fn(s, vv)
    return pd.DataFrame(cols, index=panel.index)

def cand_volume_trend_20_120(panel, macro):
    # 20d avg volume / 120d avg volume (volume expansion)
    def f(s, v):
        return v.rolling(20).mean() / v.rolling(120).mean()
    return with_volume(panel, f)

def cand_volume_z_20(panel, macro):
    def f(s, v):
        m = v.rolling(120).mean(); sd = v.rolling(120).std()
        return (v.rolling(20).mean() - m) / sd
    return with_volume(panel, f)

def cand_overnight_ret_20(panel, macro):
    # mean overnight return over 20d: open/prev_close - 1
    def f(s):
        o = _overnight(s)
        return o.rolling(20).mean()
    return per_asset(f)(panel, macro)

def _overnight(s):
    # approximate overnight using close and open from a fetched df is complex here;
    # fallback: use close-to-close without open -> skip in per_asset context.
    return None

def cand_high_prox_250(panel, macro):
    return per_asset(lambda s: s / s.rolling(250).max())(panel, macro)

def cand_beta_spx_60(panel, macro):
    spx = panel["SPX"].dropna().pct_change()
    def f(s):
        r = s.pct_change()
        z = pd.concat([r.rename("r"), spx.rename("m")], axis=1).dropna()
        beta = z["r"].rolling(60).cov(z["m"]) / z["m"].rolling(60).var()
        return beta
    return per_asset(f)(panel, macro)

def cand_skew_over_vol_20(panel, macro):
    def f(s):
        r = s.pct_change()
        return r.rolling(20).skew() / r.rolling(20).std()
    return per_asset(f)(panel, macro)

def cand_dd_vol_60x20(panel, macro):
    # drawdown depth scaled by recent vol (distance-from-high in vol units)
    def f(s):
        dd = s / s.rolling(60).max() - 1.0
        vol = s.pct_change().rolling(20).std()
        return dd / vol
    return per_asset(f)(panel, macro)

def cand_overnight_gap_ratio(panel, macro):
    # ratio of overnight-ish gap (abs open-to-prev-close) to intraday range — needs open; skip w/o open
    return None

if __name__ == "__main__":
    cands = {
        "volume_trend_20_120": cand_volume_trend_20_120,
        "volume_z_20": cand_volume_z_20,
        "high_prox_250": cand_high_prox_250,
        "beta_spx_60": cand_beta_spx_60,
        "skew_over_vol_20": cand_skew_over_vol_20,
        "dd_vol_60x20": cand_dd_vol_60x20,
    }
    summary = []
    for name, fn in cands.items():
        try:
            res = validate_factor(name, fn, horizons=(5, 10, 20), print_extra="")
            summary.append((name, res["ic_h10"], res["icir_h10"], res.get("max_abs_library_correlation", float("nan"))))
        except Exception as e:
            print(f"{name}: ERROR {e}")
            summary.append((name, float("nan"), float("nan"), float("nan")))
    print("\n===== SCREEN SUMMARY R2 (h=10) =====")
    for name, ic, icir, mc in summary:
        flag = "PASS" if (abs(ic) >= 0.007 and abs(icir) >= 0.084) else "fail"
        print(f"{name:24s} IC={ic:+.4f} ICIR={icir:+.4f} maxcorr={mc:.3f} -> {flag}")
