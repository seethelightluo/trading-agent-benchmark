"""Trader: compute ensemble scores for all 15 assets using factor signals (research-only)."""
import json
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCH = [
    "000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
    "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y",
]
VOLUME_ASSETS = {"000300.SH", "000688.SH", "SPX", "HSI", "N225", "SX5E", "NDX", "BTC", "ETH"}


def _close(name, days=200):
    df = get_index_daily_data(name, days=days)
    if df is None or len(df) == 0:
        df = get_stock_daily_data(name, days=days)
    if df is None or len(df) == 0:
        return None
    return df.sort_values("date").reset_index(drop=True)["close"].astype(float)


def _mret(name, days=250):
    c = _close(name, days)
    if c is None:
        return None
    return c.pct_change()


M = {n: _mret(n) for n in ["VIX", "DXY", "USDCNY"]}


def compute_factors(symbol):
    df = get_stock_daily_data(symbol, days=200)
    if df is None or len(df) < 130:
        return None
    df = df.sort_values("date").reset_index(drop=True)
    close = df["close"].astype(float)
    r = close.pct_change()
    out = {}
    # 1 beta_VIX_60 (dir -1)
    dv = M["VIX"]
    if dv is not None and len(r) >= 60 and len(dv) >= 60:
        rr, dd = r.tail(60), dv.tail(60)
        out["beta_VIX_60"] = rr.cov(dd) / dd.var() if dd.var() != 0 else np.nan
    # 2 kaufman_eff_20d (dir +1)
    diff = close.diff().abs()
    net = (close - close.shift(20)).abs()
    denom = diff.tail(20).sum()
    out["kaufman_eff_20d"] = (net.iloc[-1] / denom) if denom and denom != 0 else np.nan
    # 3 mom_120d_skip5 (dir +1)
    n = len(close)
    out["mom_120d_skip5"] = (close.iloc[-1 - 5] / close.iloc[-1 - 125] - 1) if n >= 126 else np.nan
    # 4 bb_width_20d (dir +1)
    mn = close.rolling(20).mean(); sd = close.rolling(20).std()
    out["bb_width_20d"] = 4 * sd.iloc[-1] / mn.iloc[-1] if mn.iloc[-1] != 0 else np.nan
    # 5 cny_beta_60 (dir +1)
    dc = M["USDCNY"]
    if dc is not None and len(r) >= 60 and len(dc) >= 60:
        rr, dd = r.tail(60), dc.tail(60)
        out["cny_beta_60"] = rr.cov(dd) / dd.var() if dd.var() != 0 else np.nan
    # 6 vol_z_20d (dir +1)
    vol = df["volume"].astype(float) if "volume" in df else None
    if symbol in VOLUME_ASSETS and vol is not None and len(vol) >= 20:
        vm = vol.rolling(20).mean(); vs = vol.rolling(20).std()
        out["vol_z_20d"] = (vol.iloc[-1] - vm.iloc[-1]) / vs.iloc[-1] if vs.iloc[-1] != 0 else np.nan
    else:
        out["vol_z_20d"] = np.nan
    # 7 ac1_120d (dir -1)
    r120 = r.tail(120)
    out["ac1_120d"] = (r120.cov(r120.shift(1)) / r120.var()) if r120.var() != 0 else np.nan
    # 8 mom_10d_skip5 (dir +1)
    out["mom_10d_skip5"] = (close.iloc[-1 - 5] / close.iloc[-1 - 15] - 1) if n >= 16 else np.nan
    # 9 dxy_corr_change_20_60 (dir +1)
    dd = M["DXY"]
    if dd is not None and len(r) >= 60:
        m = min(len(r), len(dd))
        rr2, dd2 = r.iloc[-m:], dd.iloc[-m:]
        if len(rr2) >= 60:
            c20 = rr2.tail(20).corr(dd2.tail(20)); c60 = rr2.tail(60).corr(dd2.tail(60))
            out["dxy_corr_change_20_60"] = c20 - c60
        else:
            out["dxy_corr_change_20_60"] = np.nan
    else:
        out["dxy_corr_change_20_60"] = np.nan
    # 10 skew_20d (dir +1)
    for w in (20, 10):
        sk = r.rolling(w).skew().iloc[-1]
        if sk == sk:
            out["skew_20d"] = sk
            break
    else:
        out["skew_20d"] = np.nan
    return out


def main():
    ens = json.load(open("factor_ensemble.json"))
    factors = ens["selected_factors"]
    rows = []
    for s in WATCH:
        f = compute_factors(s)
        if f is None:
            rows.append((s, None, 0, {}))
            continue
        sc = 0.0; nf = 0
        for fac in factors:
            fid = fac["factor_id"]; w = fac["weight"]; d = fac["direction"]
            v = f.get(fid)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            sc += w * d * v; nf += 1
        rows.append((s, sc, nf, f))
    scored = [r for r in rows if r[1] is not None]
    scored.sort(key=lambda x: x[1], reverse=True)
    print("Ranked by composite score:")
    for s, sc, nf, f in scored:
        print(f"{s:12s} score={sc:9.4f} nf={nf}")
    print("\nRaw factor values:")
    for s, sc, nf, f in scored:
        print(s, {k: (None if (v is None or (isinstance(v, float) and np.isnan(v))) else round(float(v), 4)) for k, v in f.items()})


if __name__ == "__main__":
    main()