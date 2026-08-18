"""Screener: compute recent cross-sectional rank IC for the active factor library
using the strategy.py factor definitions, data gated at current visible date.
Usage: python scripts/screener_recent_ic.py
"""
import sys, json
import numpy as np
import pandas as pd
from pathlib import Path

CUT = pd.Timestamp("2031-08-20")
WARMUP = pd.Timestamp("2026-07-16")  # online start

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "VIX"]


def load_close(sym, dtype="stock"):
    p = Path("../persistent/%s_data/%s.csv" % (dtype, sym))
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    s = df.set_index("date")["close"].astype(float).sort_index()
    return s[s.index <= CUT]


def rank_map(vals):
    """cross-sectional rank in [0,1], ties -> 0.5"""
    out = {a: 0.5 for a in ASSETS}
    valid = sorted((float(v), a) for a, v in vals.items()
                   if v is not None and np.isfinite(float(v)))
    n = len(valid)
    if n >= 2:
        for i, (_, a) in enumerate(valid):
            out[a] = i / (n - 1)
    return out


def trend_r2(c):
    s = c.dropna().tail(30)
    if len(s) < 18:
        return None
    y = np.log(s.values.astype(float))
    x = np.arange(len(y))
    cov = float(np.cov(y, x)[0, 1])
    vy, vx = float(np.var(y)), float(np.var(x))
    if vy <= 0 or vx <= 0:
        return None
    return np.copysign(cov * cov / (vy * vx), cov)


def semi_down_ratio(r):
    s = r.dropna().tail(20)
    if len(s) < 10:
        return None
    down = float((s.clip(upper=0) ** 2).mean() ** 0.5)
    up = float((s.clip(lower=0) ** 2).mean() ** 0.5)
    if up < 1e-12:
        return None
    return down / up - 1.0


def mom_120(c):
    if len(c) < 126:
        return None
    p0 = float(c.iloc[-126])
    if p0 <= 0:
        return None
    return float(c.iloc[-6]) / p0 - 1.0


def mom_10(c):
    if len(c) < 17:
        return None
    p0 = float(c.iloc[-16])
    if p0 <= 0:
        return None
    return float(c.iloc[-6]) / p0 - 1.0


def underwater(c):
    s = c.dropna().tail(125)
    if len(s) < 60:
        return None
    w = s.tail(120).values.astype(float)
    roll = np.maximum.accumulate(w)
    mask = w == roll
    idx = np.flatnonzero(mask)
    return float(len(w) - 1 - idx[-1]) if len(idx) else float(len(w))


def vol_of_vol(r):
    s = r.dropna().tail(120)
    if len(s) < 90:
        return None
    v = s.rolling(20).std()
    out = v.rolling(60).std().iloc[-1]
    return None if not np.isfinite(out) else float(out)


def tail_ratio(r):
    s = r.dropna().tail(20)
    if len(s) < 10:
        return None
    q95 = float(np.percentile(s.values, 95))
    q05 = float(np.percentile(s.values, 5))
    if abs(q05) < 1e-12:
        return None
    return q95 / abs(q05)


def beta_60(r, m_r):
    z = pd.concat([r.rename("a"), m_r.rename("m")], axis=1).dropna().tail(60)
    if len(z) < 30:
        return None
    vm = float(z["m"].var())
    if vm < 1e-14:
        return None
    return float(z["a"].cov(z["m"]) / vm)


def vix_beta_cond(r, vix_r, vix_c):
    z = pd.concat([r.rename("a"), vix_r.rename("v")], axis=1).dropna().tail(60)
    if len(z) < 30:
        return None
    vv = float(z["v"].var())
    if vv < 1e-14:
        return None
    beta = float(z["a"].cov(z["v"]) / vv)
    if vix_c is None or len(vix_c) < 22:
        return None
    v0 = float(vix_c.iloc[-21])
    if v0 <= 0:
        return None
    vmove = float(vix_c.iloc[-1]) / v0 - 1.0
    return -beta * vmove


def main():
    closes = {a: load_close(a) for a in ASSETS}
    panel = pd.DataFrame(closes).sort_index()
    rets = panel.pct_change()
    dxy_c = load_close("DXY", "index")
    vix_c = load_close("VIX", "index")
    cny_c = load_close("USDCNY", "index")
    dxy_r = dxy_c.pct_change()
    vix_r = vix_c.pct_change()
    cny_r = cny_c.pct_change()

    fdefs = {
        "trend_r2_30_signed": (1, lambda c, r: trend_r2(c)),
        "semi_down_ratio_20": (-1, lambda c, r: semi_down_ratio(r)),
        "mom_120d_skip5": (1, lambda c, r: mom_120(c)),
        "mom_10d_skip5": (1, lambda c, r: mom_10(c)),
        "vol_of_vol20x60": (1, lambda c, r: vol_of_vol(r)),
        "time_under_water_120": (-1, lambda c, r: underwater(c)),
        "tail_ratio_20": (1, lambda c, r: tail_ratio(r)),
        "dxy_beta_60": (1, lambda c, r: beta_60(r, dxy_r) if dxy_r is not None else None),
        "cny_beta_60": (1, lambda c, r: beta_60(r, cny_r) if cny_r is not None else None),
        "vix_beta_cond_60x20": (-1, lambda c, r: vix_beta_cond(r, vix_r, vix_c) if vix_r is not None else None),
        "kurt_20": (1, lambda c, r: None),  # not in strategy default path
        "WTI_BETA_60": (1, lambda c, r: None),
    }

    # build daily factor panels (rank per date)
    dates = panel.index
    fwd10 = panel.shift(-10) / panel - 1.0  # 10d forward return

    results = {}
    for fid, (exp_dir, fn) in fdefs.items():
        rows = {}
        for dt in dates:
            vals = {}
            for a in ASSETS:
                c = closes.get(a)
                if c is None:
                    continue
                cl = c[c.index <= dt]
                r = rets[a][rets[a].index <= dt] if a in rets else None
                if cl is None or len(cl) < 20 or r is None:
                    continue
                try:
                    v = fn(cl, r)
                except Exception:
                    v = None
                vals[a] = v
            rows[dt] = rank_map(vals)
        fpanel = pd.DataFrame(rows).T  # dates x assets
        # align
        common = fpanel.index.intersection(fwd10.index)
        fp = fpanel.loc[common]
        fr = fwd10.loc[common]
        # cross-sectional Spearman rank IC per date
        ic_series = {}
        for dt in common:
            x = fp.loc[dt]
            y = fr.loc[dt]
            m = x.notna() & y.notna()
            if m.sum() >= 5:
                ic_series[dt] = x[m].corr(y[m], method="spearman")
        ic = pd.Series(ic_series).dropna()
        ic = ic[ic.index >= WARMUP]
        for label, win in [("all_online", None), ("last180d", 180), ("last90d", 90)]:
            w = ic if win is None else ic.iloc[-win:]
            if len(w) < 20:
                continue
            mean_ic = float(w.mean())
            icir = mean_ic / float(w.std(ddof=1)) if w.std(ddof=1) > 0 else 0.0
            results.setdefault(fid, {})[label] = dict(n=len(w), ic=mean_ic, icir=icir)
        results.setdefault(fid, {})["n_dates"] = len(ic)

    print("%-22s %6s %8s %8s | %6s %8s %8s | %6s %8s %8s" % (
        "factor", "n", "all_ic", "all_ir", "n", "180_ic", "180_ir", "n", "90_ic", "90_ir"))
    for fid, (exp_dir, _) in fdefs.items():
        r = results.get(fid, {})
        a = r.get("all_online", {}); b = r.get("last180d", {}); c = r.get("last90d", {})
        print("%-22s %6d %8.4f %8.3f | %6d %8.4f %8.3f | %6d %8.4f %8.3f" % (
            fid, r.get("n_dates", 0),
            a.get("ic", np.nan), a.get("icir", np.nan),
            b.get("n", 0), b.get("ic", np.nan), b.get("icir", np.nan),
            c.get("n", 0), c.get("ic", np.nan), c.get("icir", np.nan)))


if __name__ == "__main__":
    main()
