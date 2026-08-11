"""miner_3 focused validation: top candidates vs FULL library correlation + sub-period robustness."""
from __future__ import annotations
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner3_lib import (load_panel, load_macro, per_asset, fwd_returns,
                        rank_ic_series, WATCH, MIN_ASSETS_PER_DATE)

panel = load_panel()
macro = load_macro()


def build_library(panel, macro):
    lib = {}
    # rel_mom_20d_skip5 (cross-sectionally demeaned 20d mom skip 5)
    raw = per_asset(lambda s: s.shift(5) / s.shift(25) - 1.0)(panel, macro)
    lib["rel_mom_20d_skip5"] = raw.sub(raw.median(axis=1), axis=0)
    # beta_ew_60d: rolling 60d beta vs EW market
    ret = panel.pct_change()
    mkt = ret.mean(axis=1, min_count=8)
    cols = {}
    for a in panel.columns:
        s = ret[a].dropna()
        m = mkt.reindex(s.index)
        z = pd.concat([s.rename("r"), m.rename("m")], axis=1).dropna()
        cols[a] = z["r"].rolling(60).cov(z["m"]) / z["m"].rolling(60).var()
    lib["beta_ew_60d"] = pd.DataFrame(cols, index=panel.index)
    # mom_120d_skip5
    lib["mom_120d_skip5"] = per_asset(lambda s: s.shift(5) / s.shift(125) - 1.0)(panel, macro)
    # vol_of_vol20x60
    lib["vol_of_vol20x60"] = per_asset(lambda s: s.pct_change().rolling(20).std().rolling(60).std())(panel, macro)
    # max_ret_20d
    lib["max_ret_20d"] = per_asset(lambda s: s.pct_change().rolling(20).max())(panel, macro)
    # downside_vol_ratio_20: -(downside semi-vol / total vol)
    def dsvr(s):
        r = s.pct_change()
        tot = r.rolling(20).std()
        dn = r.where(r < 0, 0.0).rolling(20).std()
        return -(dn / tot)
    lib["downside_vol_ratio_20"] = per_asset(dsvr)(panel, macro)
    # mom_10d_skip5
    lib["mom_10d_skip5"] = per_asset(lambda s: s.shift(5) / s.shift(15) - 1.0)(panel, macro)
    # amihud_20: mean(|ret| / volume, 20)
    def amihud(panel):
        cols = {}
        for a in panel.columns:
            df = getattr(sys.modules[__name__], "_get_ohlcv")(a)
            s = panel[a].dropna()
            if df is None:
                continue
            v = df.reindex(s.index)["volume"].astype(float)
            r = s.pct_change().abs()
            cols[a] = (r / v.replace(0, np.nan)).rolling(20).mean()
        return pd.DataFrame(cols, index=panel.index)
    # vix_beta_cond_60x20
    vix = macro["VIX"].dropna()
    cols = {}
    for a in panel.columns:
        s = panel[a].dropna()
        r = s.pct_change()
        v = vix.pct_change().reindex(s.index)
        z = pd.concat([r.rename("r"), v.rename("v")], axis=1).dropna()
        beta = z["r"].rolling(60).cov(z["v"]) / z["v"].rolling(60).var()
        vix20 = (vix / vix.shift(20) - 1.0).reindex(s.index)
        cols[a] = -beta * vix20
    lib["vix_beta_cond_60x20"] = pd.DataFrame(cols, index=panel.index)
    return lib


def _get_ohlcv(sym):
    from alphacrafter.sim.utils import get_stock_daily_data
    df = get_stock_daily_data(sym, days=4000)
    if df is None:
        return None
    df = df.set_index("date")
    return df[~df.index.duplicated(keep="last")].sort_index()


def cand_dxy_beta_cond_60x20(panel, macro):
    dxy = macro["DXY"].dropna()
    def f(s):
        r = s.pct_change()
        v = dxy.pct_change().reindex(s.index)
        z = pd.concat([r.rename("r"), v.rename("v")], axis=1).dropna()
        beta = z["r"].rolling(60).cov(z["v"]) / z["v"].rolling(60).var()
        dxy20 = (dxy / dxy.shift(20) - 1.0).reindex(s.index)
        return -beta * dxy20
    return per_asset(f)(panel, macro)


def cand_beta_spx_60(panel, macro):
    spx = panel["SPX"].dropna().pct_change()
    def f(s):
        r = s.pct_change()
        z = pd.concat([r.rename("r"), spx.rename("m")], axis=1).dropna()
        beta = z["r"].rolling(60).cov(z["m"]) / z["m"].rolling(60).var()
        return beta
    return per_asset(f)(panel, macro)


def cand_dd_vol_60x20(panel, macro):
    def f(s):
        dd = s / s.rolling(60).max() - 1.0
        vol = s.pct_change().rolling(20).std()
        return dd / vol
    return per_asset(f)(panel, macro)


def corr_with_lib(factor, lib):
    fx = factor.stack().rename("f")
    out = {}
    for fid, lf in lib.items():
        both = pd.concat([fx, lf.stack().rename("l")], axis=1).dropna()
        if len(both) >= 200:
            out[fid] = float(np.corrcoef(both["f"], both["l"])[0, 1])
    return out


def subperiod_stats(factor, h=10, warm_end="2026-07-15"):
    fw = factor.loc[:warm_end]
    fr = fwd_returns(panel, h)
    ic = rank_ic_series(fw, fr)
    parts = {"2020-2022": (ic.loc["2020":"2022"]), "2023-2024": (ic.loc["2023":"2024"]),
             "2025-2026": (ic.loc["2025":"2026"])}
    out = {}
    for k, s in parts.items():
        if len(s) > 10:
            out[k] = (float(s.mean()), float(s.mean() / s.std()))
    return ic, out


lib = build_library(panel, macro)
print("library built:", list(lib.keys()))

for name, fn in [("dxy_beta_cond_60x20", cand_dxy_beta_cond_60x20),
                 ("beta_spx_60", cand_beta_spx_60),
                 ("dd_vol_60x20", cand_dd_vol_60x20)]:
    f = fn(panel, macro)
    corrs = corr_with_lib(f, lib)
    maxc = max((abs(v) for v in corrs.values()), default=float("nan"))
    ic, sub = subperiod_stats(f)
    print(f"\n=== {name} ===")
    print(f"  full-lib corrs: { {k: round(v,3) for k,v in corrs.items()} }")
    print(f"  max_abs_library_corr = {maxc:.3f}")
    print(f"  h10 IC={ic.mean():+.4f} ICIR={ic.mean()/ic.std():+.4f} n={len(ic)}")
    for k, (m, ir) in sub.items():
        print(f"    sub {k}: IC={m:+.4f} ICIR={ir:+.4f}")
