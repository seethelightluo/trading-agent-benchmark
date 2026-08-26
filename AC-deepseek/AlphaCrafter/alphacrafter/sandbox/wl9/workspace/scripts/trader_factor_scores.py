"""Trader factor scoring validation script (not a runner)."""
import json
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCH = [
    "000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
    "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y",
]
MACRO_SERIES = {
    "VIX": get_index_daily_data("VIX", days=200),
    "DXY": get_index_daily_data("DXY", days=200),
    "USDCNY": get_index_daily_data("USDCNY", days=200),
}
VOLUME_ASSETS = {"000300.SH", "000688.SH", "SPX", "HSI", "N225", "SX5E", "NDX", "BTC", "ETH"}


def macro_ret(name, window):
    s = MACRO_SERIES[name]
    if s is None:
        return None
    close = s.sort_values("date").reset_index(drop=True)["close"].astype(float)
    return close.pct_change()


def compute_factors(symbol):
    df = get_stock_daily_data(symbol, days=200)
    if df is None or len(df) < 130:
        return None
    df = df.sort_values("date").reset_index(drop=True)
    close = df["close"].astype(float)
    r = close.pct_change()
    out = {}

    # 1 beta_VIX_60 (dir -1)
    dvix = macro_series("VIX", 60)
    if dvix is not None:
        m = min(len(r), len(dvix))
        rr, dd = r.iloc[-m:], dvix.iloc[-m:]
        rr, dd = rr.tail(60), dd.tail(60)
        denom = dd.var()
        out["beta_VIX_60"] = (rr.cov(dd) / denom) if denom != 0 else np.nan

    # 2 kaufman_eff_20d (dir +1)
    diff = close.diff().abs()
    net = (close - close.shift(20)).abs()
    denom = diff.tail(20).sum()
    out["kaufman_eff_20d"] = (net.iloc[-1] / denom) if denom != 0 else np.nan

    # 3 mom_120d_skip5 (dir +1)
    n = len(close)
    out["mom_120d_skip5"] = close.iloc[-1 - 5] / close.iloc[-1 - 125] - 1 if n >= 126 else np.nan

    # 4 bb_width_20d (dir +1)
    mn = close.rolling(20).mean()
    sd = close.rolling(20).std()
    out["bb_width_20d"] = 4 * sd.iloc[-1] / mn.iloc[-1] if mn.iloc[-1] != 0 else np.nan

    # 5 cny_beta_60 (dir +1)
    dcny = macro_series("USDCNY", 60)
    if dcny is not None:
        rr, dd = r.tail(60), dcny.tail(60)
        denom = dd.var()
        out["cny_beta_60"] = (rr.cov(dd) / denom) if denom != 0 else np.nan

    # 6 vol_z_20d (dir +1)
    vol = df["volume"].astype(float)
    if symbol in VOLUME_ASSETS:
        vm = vol.rolling(20).mean()
        vs = vol.rolling(20).std()
        out["vol_z_20d"] = (vol.iloc[-1] - vm.iloc[-1]) / vs.iloc[-1] if vs.iloc[-1] != 0 else np.nan
    else:
        out["vol_z_20d"] = np.nan

    # 7 ac1_120d (dir -1)
    r120 = r.tail(120)
    denom = r120.var()
    out["ac1_120d"] = (r120.cov(r120.shift(1)) / denom) if denom != 0 else np.nan

    # 8 mom_10d_skip5 (dir +1)
    out["mom_10d_skip5"] = close.iloc[-1 - 5] / close.iloc[-1 - 15] - 1 if n >= 16 else np.nan

    # 9 dxy_corr_change_20_60 (dir +1)
    ddxy = macro_series("DXY", 60)
    if ddxy is not None:
        rr, dd = r, ddxy
        m = min(len(r), len(dd))
        rr, dd = rr.iloc[-m:], dd.iloc[-m:]
        if len(rr) >= 60:
            c20 = rr.tail(20).corr(dd.tail(20))
            c60 = rr.tail(60).corr(dd.tail(60))
            out["dxy_corr_change_20_60"] = (c20 - c60) if (c20 == c20 and c60 == c60) else np.nan
        else:
            out["dxy_corr_change_20_60"] = np.nan

    # 10 skew_20d (dir +1)
    skew = r.rolling(20).skew().iloc[-1]
    out["skew_20d"] = skew if skew == skew else np.nan

    return out


def main():
    ens = json.load(open("factor_ensemble.json"))
    factors = ens["selected_factors"]
    print("Active factors (%d):" % len(factors))
    scores = {}
    for i, s in enumerate(WATCH):
        f = compute_factors(s)
        if f is None:
            print(s, "NO DATA")
            continue
        sc = 0.0
        nf = 0
        detail = {}
        for fac in factors:
            fid = fac["factor_id"]
            w = fac["weight"]
            d = fac["direction"]
            v = f.get(fid)
            if v is None or (isinstance(v, float) and v != v):
                detail[fid] = "nan"
                continue
            sc += w * d * v
            nf += 1
            detail[fid] = round(float(v), 4)
        scores[s] = sc
        print(s, "score=%.4f nf=%d" % (sc, nf), detail)