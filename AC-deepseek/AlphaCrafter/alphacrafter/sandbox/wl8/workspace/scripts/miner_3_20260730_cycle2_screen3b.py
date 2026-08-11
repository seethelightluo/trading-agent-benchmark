"""miner_3 2026-07-30 -- Screen round 3b: corrected library correlation incl.
VIX (from ../persistent/index_data/VIX.csv) and US10Y yield-beta, and fixed
DXY alignment (reindexed to asset calendar). Re-screens the round-3 families.
SCREEN ONLY.
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, "scripts")
from miner_3_20260730_common import (
    load_data, factor_ic_table, coverage_stats, rank_turnover,
)

data = load_data(days=3200)
closes = {a: d["close"].astype(float) for a, d in data.items()}
opens = {a: d["open"].astype(float) for a, d in data.items()}
highs = {a: d["high"].astype(float) for a, d in data.items()}
lows = {a: d["low"].astype(float) for a, d in data.items()}
vols = {a: d["volume"].astype(float).replace(0, np.nan) for a, d in data.items()}
last_vis = max(d.index.max() for d in data.values())
cal = pd.Series(dtype=float)


def load_obs(name):
    df = pd.read_csv(f"../persistent/index_data/{name}.csv")
    df["date"] = pd.to_datetime(df["date"])
    s = df.set_index("date").sort_index()["close"].astype(float)
    return s[s.index <= last_vis]


VIX = load_obs("VIX")
DXY = load_obs("DXY")
print(f"[obs] VIX {VIX.index.min().date()}..{VIX.index.max().date()} n={len(VIX)}; "
      f"DXY {DXY.index.min().date()}..{DXY.index.max().date()} n={len(DXY)}")

# ---- full library panel (matches persisted active factors) ----
def full_library():
    lib = {}
    for a, c in closes.items():
        lib.setdefault("mom_10d_skip5", {})[a] = c.shift(5) / c.shift(15) - 1.0
        lib.setdefault("mom_120d_skip5", {})[a] = c.shift(5) / c.shift(125) - 1.0
        lib.setdefault("vol_of_vol20x60", {})[a] = c.pct_change().rolling(20).std().rolling(60).std()
        r = c.pct_change()
        beta_v = r.rolling(60).cov(VIX.pct_change()) / VIX.pct_change().rolling(60).var()
        lib.setdefault("vix_beta_cond_60x20", {})[a] = -beta_v * (VIX / VIX.shift(20) - 1.0)
        us10 = closes["US10Y"]
        yr = us10.pct_change()
        beta_y = r.rolling(60).cov(yr) / yr.rolling(60).var()
        lib.setdefault("yield_beta_cond_60x20", {})[a] = -beta_y * (us10 / us10.shift(20) - 1.0)
    return lib


LIB = full_library()
print("[lib] factors:", list(LIB.keys()))


def lib_corr(factor):
    fdf = pd.DataFrame(factor).stack()
    fdf = fdf[fdf.notna()]
    out = {}
    for fid, lf in LIB.items():
        ldf = pd.DataFrame(lf).stack()
        ldf = ldf[ldf.notna()]
        both = fdf.index.intersection(ldf.index)
        if len(both) < 100:
            out[fid] = float("nan")
            continue
        rho, _ = pearsonr(fdf.loc[both].values, ldf.loc[both].values)
        out[fid] = float(rho) if np.isfinite(rho) else float("nan")
    vals = [abs(v) for v in out.values() if np.isfinite(v)]
    return (max(vals) if vals else float("nan")), out


def _wild(series, win):
    return series.ewm(alpha=1.0 / win, adjust=False).mean()


def adx_14(c, o, h, l, v, win=14):
    up = h.diff()
    dn = -l.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=c.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=c.index)
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = _wild(tr, win)
    pdi = 100.0 * _wild(plus_dm, win) / atr.replace(0, np.nan)
    mdi = 100.0 * _wild(minus_dm, win) / atr.replace(0, np.nan)
    dx = 100.0 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return _wild(dx, win)


def risk_adj_mom_10x60(c, o, h, l, v, **kw):
    mom = c.shift(5) / c.shift(15) - 1.0
    rv = c.pct_change().rolling(60).std()
    return mom / rv.replace(0, np.nan)


def vol_mom_20x40(c, o, h, l, v, **kw):
    rv20 = c.pct_change().rolling(20).std()
    return rv20 / rv20.shift(20) - 1.0


def gain_loss_asym_60(c, o, h, l, v, **kw):
    r = c.pct_change()
    mu = r.clip(lower=0.0).rolling(60).mean()
    md = (-r).clip(lower=0.0).rolling(60).mean()
    return mu / md.replace(0, np.nan)


def macd_hist_12x26(c, o, h, l, v, **kw):
    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    macd = e12 - e26
    sig = macd.ewm(span=9, adjust=False).mean()
    return (macd - sig) / c


def mom_term_20x120(c, o, h, l, v, **kw):
    return (c / c.shift(20) - 1.0) - (c / c.shift(120) - 1.0)


def mfi_14(c, o, h, l, v, win=14):
    tp = (h + l + c) / 3.0
    raw = tp * v
    pos = raw.where(tp > tp.shift(1), 0.0)
    neg = raw.where(tp < tp.shift(1), 0.0)
    mr = pos.rolling(win).sum() / neg.rolling(win).sum().replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + mr)


def dxy_beta_cond_60x20(c, o, h, l, v, **kw):
    dxy = kw["dxy"].reindex(c.index).ffill()
    r = c.pct_change()
    dr = dxy.pct_change()
    beta = r.rolling(60).cov(dr) / dr.rolling(60).var()
    move = dxy / dxy.shift(20) - 1.0
    return -beta * move


CANDIDATES = [
    ("adx_14", "Wilder ADX(14) trend strength", {}),
    ("risk_adj_mom_10x60", "10d skip5 momentum / rv60", {}),
    ("vol_mom_20x40", "vol momentum rv20/rv20_shift20-1", {}),
    ("gain_loss_asym_60", "gain/loss asymmetry 60d", {}),
    ("macd_hist_12x26", "normalized MACD histogram", {}),
    ("mom_term_20x120", "TSMOM term structure 20-120", {}),
    ("mfi_14", "Money Flow Index(14)", {}),
    ("dxy_beta_cond_60x20", "-beta(asset,DXY,60d)*DXY20d move", {}),
]

for fid, desc, params in CANDIDATES:
    try:
        f = {}
        for a, c in closes.items():
            if fid == "dxy_beta_cond_60x20":
                f[a] = dxy_beta_cond_60x20(c, opens[a], highs[a], lows[a], vols[a], dxy=DXY)
            else:
                f[a] = globals()[fid](c, opens[a], highs[a], lows[a], vols[a], **params)
    except Exception as e:
        print(f"[screen3b] {fid}: BUILD ERROR {e}")
        continue
    f = {a: s.replace([np.inf, -np.inf], np.nan) for a, s in f.items()}
    tbl = factor_ic_table(f, data, horizons=(1, 3, 5, 10, 20), min_assets=8, primary_h=10)
    prim = tbl[10]
    if prim is None:
        print(f"[screen3b] {fid:22s} {desc:34s} DEGENERATE")
        continue
    cov = coverage_stats(f, data)
    to = rank_turnover(f)
    maxrho, rho_map = lib_corr(f)
    gate_ic = abs(prim["ic"]) >= 0.0070
    gate_icir = abs(prim["icir"]) >= 0.0840
    gate_rho = (np.isfinite(maxrho) and maxrho < 0.5)
    flag = "PASS" if (gate_ic and gate_icir and gate_rho) else "fail"
    print(f"[screen3b] {fid:22s} {desc:34s} ic10={prim['ic']:+.4f} icir10={prim['icir']:+.4f} "
          f"hit={prim['ic_hit']:.3f} n={prim['n_dates']:4d} cov={cov['coverage_asset_days']:.3f} "
          f"ge8={prim['dates_ge8']:.3f} turn={to:5.2f} maxrho={maxrho:.3f} -> {flag}")
    print(f"            rho={ {k: (round(v, 2) if np.isfinite(v) else None) for k, v in rho_map.items()} } "
          f"decay={ {h: (round(v['ic'], 4) if v else None) for h, v in tbl.items()} }")
