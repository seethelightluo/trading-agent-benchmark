"""SCREENER cycle 2027-03-25: regime assessment + factor IC refresh + ensemble build.

Reuses strategy.py factor math (imported by re-implementation, no alphacrafter dependency)
and the miner_2 validation framework conventions (rank IC, 10d forward, visible window).
"""
import json
import os
import numpy as np
import pandas as pd

ROOT = "../persistent"
STOCK_DIR = os.path.join(ROOT, "stock_data")
INDEX_DIR = os.path.join(ROOT, "index_data")
VISIBLE = "2027-03-25"

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]

# ---------------------------------------------------------------- data loading
def load_panel(symbols, source, visible_through=VISIBLE):
    d = STOCK_DIR if source == "stock" else INDEX_DIR
    out = {}
    for s in symbols:
        fp = os.path.join(d, s + ".csv")
        if not os.path.exists(fp):
            continue
        df = pd.read_csv(fp, parse_dates=["date"])
        df = df[df["date"] <= pd.Timestamp(visible_through)].reset_index(drop=True)
        if "close" in df.columns:
            out[s] = df
    return out

closes = {s: df.set_index("date")["close"].astype(float)
          for s, df in load_panel(WATCH, "stock").items()}
panel = pd.DataFrame(closes).sort_index()
rets = panel.pct_change()
macro_c = {s: df.set_index("date")["close"].astype(float)
           for s, df in load_panel(MACRO, "index").items()}
macro_r = {s: c.pct_change() for s, c in macro_c.items()}

print("panel dates:", panel.index.min().date(), "->", panel.index.max().date(),
      "| assets:", len(panel.columns))

# ---------------------------------------------------------------- regime metrics
mkt = rets.mean(axis=1)  # equal-weight cross-asset daily return
eq_px = (1.0 + mkt).cumprod()

trend20 = float(mkt.tail(20).mean())
trend60 = float(mkt.tail(60).mean())
ma20 = float(eq_px.tail(20).mean())
ma60 = float(eq_px.tail(60).mean())
dd = float(eq_px / eq_px.cummax() - 1.0)

vol20 = rets.tail(20).std().mean() * np.sqrt(252)   # avg annualized vol
vol60 = rets.tail(60).std().mean() * np.sqrt(252)
disp20 = float(rets.tail(20).std(axis=1).mean())    # cross-sectional dispersion

# average pairwise correlation (last 60d)
c60 = rets.tail(60).corr()
tri = c60.where(np.triu(np.ones(c60.shape), k=1).astype(bool))
avg_corr = float(tri.stack().mean())

vix = macro_c.get("VIX")
vix_level = float(vix.iloc[-1]) if vix is not None else np.nan
vix_pctile = float((vix.tail(250) <= vix_level).mean()) if vix is not None else np.nan
vix_20d = float(vix.iloc[-1] / vix.iloc[-21] - 1.0) if vix is not None and len(vix) > 21 else np.nan

dxy = macro_c.get("DXY")
dxy_20d = float(dxy.iloc[-1] / dxy.iloc[-21] - 1.0) if dxy is not None and len(dxy) > 21 else np.nan

# per-asset 20d returns and 60d trend
r20 = rets.tail(20).apply(lambda s: (1.0 + s).prod() - 1.0)
r60 = rets.tail(60).apply(lambda s: (1.0 + s).prod() - 1.0)

regime = {
    "trend20_bps": round(trend20 * 1e4, 1),
    "trend60_bps": round(trend60 * 1e4, 1),
    "ma20_vs_ma60": round(ma20 / ma60 - 1.0, 4),
    "eq_drawdown": round(dd, 4),
    "vol20_ann": round(vol20, 4),
    "vol60_ann": round(vol60, 4),
    "dispersion20": round(disp20 * 1e4, 1),
    "avg_pair_corr_60": round(avg_corr, 3),
    "vix_level": round(vix_level, 2),
    "vix_250d_pctile": round(vix_pctile, 3),
    "vix_20d_chg": round(vix_20d, 4),
    "dxy_20d_chg": round(dxy_20d, 4),
}
print(json.dumps(regime, indent=1))
print("\n20d returns per asset:")
print(r20.sort_values(ascending=False).round(4).to_string())
print("\n60d returns per asset:")
print(r60.sort_values(ascending=False).round(4).to_string())

# ---------------------------------------------------------------- factor math (mirror strategy.py)
def _trend_r2(c):
    s = c.dropna().tail(30)
    if len(s) < 18:
        return None
    y = np.log(s.values.astype(float)); x = np.arange(len(y))
    cov = float(np.cov(y, x)[0, 1]); vy, vx = float(np.var(y)), float(np.var(x))
    if vy <= 0 or vx <= 0:
        return None
    return np.copysign(cov * cov / (vy * vx), cov)

def _semi_down_ratio(r):
    s = r.dropna().tail(20)
    if len(s) < 10:
        return None
    down = float((s.clip(upper=0) ** 2).mean() ** 0.5)
    up = float((s.clip(lower=0) ** 2).mean() ** 0.5)
    if up < 1e-12:
        return None
    return down / up - 1.0

def _mom_120(c):
    if len(c) < 126:
        return None
    p0 = float(c.iloc[-126])
    return None if p0 <= 0 else float(c.iloc[-6]) / p0 - 1.0

def _mom_10(c):
    if len(c) < 17:
        return None
    p0 = float(c.iloc[-16])
    return None if p0 <= 0 else float(c.iloc[-6]) / p0 - 1.0

def _underwater(c):
    s = c.dropna().tail(125)
    if len(s) < 60:
        return None
    w = s.tail(120).values.astype(float)
    roll = np.maximum.accumulate(w)
    mask = w == roll
    idx = np.flatnonzero(mask)
    return float(len(w) - 1 - idx[-1]) if len(idx) else float(len(w))

def _vol_of_vol(r):
    s = r.dropna().tail(120)
    if len(s) < 90:
        return None
    v = s.rolling(20).std()
    out = v.rolling(60).std().iloc[-1]
    return None if not np.isfinite(out) else float(out)

def _kurt_20(r):
    s = r.dropna().tail(40)
    if len(s) < 20:
        return None
    k = s.rolling(20, min_periods=8).kurt().iloc[-1]
    return None if not np.isfinite(k) else float(k)

def _tail_ratio(r):
    s = r.dropna().tail(20)
    if len(s) < 10:
        return None
    q95 = float(np.percentile(s.values, 95)); q05 = float(np.percentile(s.values, 5))
    if abs(q05) < 1e-12:
        return None
    return q95 / abs(q05)

def _dxy_beta(r, dxy_r):
    z = pd.concat([r.rename("a"), dxy_r.rename("d")], axis=1).dropna().tail(60)
    if len(z) < 30:
        return None
    vd = float(z["d"].var())
    if vd < 1e-14:
        return None
    return float(z["a"].cov(z["d"]) / vd)

def _vix_beta_cond(r, vix_r, vix_c):
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

def _wti_beta(r, wti_r):
    z = pd.concat([r.rename("a"), wti_r.rename("w")], axis=1).dropna().tail(60)
    if len(z) < 30:
        return None
    vw = float(z["w"].var())
    if vw < 1e-14:
        return None
    return float(z["a"].cov(z["w"]) / vw)

FACTORS = {
    "trend_r2_30_signed": {"dir": 1, "tags": "trend"},
    "semi_down_ratio_20": {"dir": -1, "tags": "risk"},
    "mom_120d_skip5": {"dir": 1, "tags": "momentum"},
    "dxy_beta_60": {"dir": 1, "tags": "macro-beta"},
    "time_under_water_120": {"dir": -1, "tags": "drawdown"},
    "mom_10d_skip5": {"dir": 1, "tags": "momentum"},
    "tail_ratio_20": {"dir": 1, "tags": "tail-risk"},
    "vix_beta_cond_60x20": {"dir": -1, "tags": "macro-beta"},
    "vol_of_vol20x60": {"dir": 1, "tags": "volatility"},
    "kurt_20": {"dir": 1, "tags": "quality"},
    "WTI_BETA_60": {"dir": 1, "tags": "beta"},
}

dxy_r = macro_r.get("DXY")
vix_c = macro_c.get("VIX")
vix_r = macro_r.get("VIX")
wti_c = closes.get("WTI")
wti_r = rets["WTI"] if "WTI" in rets else None

fvals = {}
for fid in FACTORS:
    fvals[fid] = {}
    for a in WATCH:
        c = closes.get(a); r = rets[a] if a in rets else None
        if c is None or r is None:
            continue
        try:
            if fid == "trend_r2_30_signed": v = _trend_r2(c)
            elif fid == "semi_down_ratio_20": v = _semi_down_ratio(r)
            elif fid == "mom_120d_skip5": v = _mom_120(c)
            elif fid == "mom_10d_skip5": v = _mom_10(c)
            elif fid == "vol_of_vol20x60": v = _vol_of_vol(r)
            elif fid == "time_under_water_120": v = _underwater(c)
            elif fid == "tail_ratio_20": v = _tail_ratio(r)
            elif fid == "kurt_20": v = _kurt_20(r)
            elif fid == "dxy_beta_60": v = _dxy_beta(r, dxy_r) if dxy_r is not None else None
            elif fid == "vix_beta_cond_60x20": v = _vix_beta_cond(r, vix_r, vix_c) if vix_r is not None else None
            elif fid == "WTI_BETA_60": v = _wti_beta(r, wti_r) if wti_r is not None else None
            else: v = None
        except Exception:
            v = None
        fvals[fid][a] = v

fdf = pd.DataFrame({fid: pd.Series({a: fvals[fid][a] for a in WATCH}) for fid in FACTORS})
# build per-date factor panel: for each date, factor value per asset (computed at that date's end)
factor_panels = {}
for fid in FACTORS:
    rows = {}
    for a in WATCH:
        c = closes.get(a); r = rets[a] if a in rets else None
        if c is None or r is None:
            continue
        s = pd.DataFrame({"close": c, "ret": r}).dropna()
        vals = {}
        for dt in s.index:
            cc = s["close"].loc[:dt]; rr = s["ret"].loc[:dt]
            try:
                if fid == "trend_r2_30_signed": v = _trend_r2(cc)
                elif fid == "semi_down_ratio_20": v = _semi_down_ratio(rr)
                elif fid == "mom_120d_skip5": v = _mom_120(cc)
                elif fid == "mom_10d_skip5": v = _mom_10(cc)
                elif fid == "vol_of_vol20x60": v = _vol_of_vol(rr)
                elif fid == "time_under_water_120": v = _underwater(cc)
                elif fid == "tail_ratio_20": v = _tail_ratio(rr)
                elif fid == "kurt_20": v = _kurt_20(rr)
                elif fid == "dxy_beta_60": v = _dxy_beta(rr, dxy_r.loc[:dt]) if dxy_r is not None else None
                elif fid == "vix_beta_cond_60x20": v = _vix_beta_cond(rr, vix_r.loc[:dt], vix_c.loc[:dt]) if vix_r is not None else None
                elif fid == "WTI_BETA_60": v = _wti_beta(rr, wti_r.loc[:dt]) if wti_r is not None else None
                else: v = None
            except Exception:
                v = None
            vals[dt] = v
        rows[a] = pd.Series(vals)
    factor_panels[fid] = pd.DataFrame(rows).sort_index()

fwd10 = panel.shift(-10) / panel - 1.0

def rank_ic_series(fpanel, fwd, min_valid=8):
    dates = fpanel.index.intersection(fwd.index)
    ics = {}
    for d in dates:
        f = fpanel.loc[d]; r = fwd.loc[d]
        pair = pd.concat([f.rename("f"), r.rename("r")], axis=1).dropna()
        if len(pair) < min_valid:
            continue
        if pair["f"].nunique() < 3 or pair["r"].nunique() < 2:
            continue
        ic = pair["f"].corr(pair["r"], method="spearman")
        if np.isfinite(ic):
            ics[d] = ic
    return pd.Series(ics, dtype=float)

print("\n=== factor IC refresh (10d forward rank IC) ===")
summary = {}
for fid in FACTORS:
    fp = factor_panels[fid]
    ics = rank_ic_series(fp, fwd10)
    ics = ics.dropna()
    if len(ics) < 30:
        print(f"{fid}: insufficient IC obs ({len(ics)})")
        summary[fid] = {"n": int(len(ics)), "skip": True}
        continue
    for win_name, win in [("w120", 120), ("w250", 250), ("all", len(ics))]:
        icw = ics.tail(win)
        icm = float(icw.mean()); icsd = float(icw.std(ddof=1)) if len(icw) > 1 else float("nan")
        icir = icm / icsd if icsd and np.isfinite(icsd) and icsd > 0 else float("nan")
        hit = float((icw > 0).mean())
        summary.setdefault(fid, {})[win_name] = {"ic": round(icm, 4), "icir": round(icir, 3) if np.isfinite(icir) else None, "hit": round(hit, 3), "n": int(len(icw))}
    print(f"{fid}: last_ic={round(float(ics.iloc[-1]),3)} "
          f"w120 ic={summary[fid]['w120']['ic']} icir={summary[fid]['w120']['icir']} hit={summary[fid]['w120']['hit']} | "
          f"w250 ic={summary[fid]['w250']['ic']} icir={summary[fid]['w250']['icir']} hit={summary[fid]['w250']['hit']}")

with open("scripts/screener_20270325_regime.json", "w") as f:
    json.dump({"regime": regime, "factor_summary": summary,
               "r20": r20.round(4).to_dict(), "r60": r60.round(4).to_dict()}, f, indent=1, default=str)
print("\nsaved regime json")
