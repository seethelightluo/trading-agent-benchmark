"""miner_3 2026-07-30 -- Screen round 3: novel orthogonal factor families.

Library/active: mom_10d_skip5, vix_beta_cond_60x20, yield_beta_cond_60x20.
Prior screens covered: RSI, stoch, bb_pctb, drawdown depth, Amihud, OBV slope,
skew/kurt, intraday/overnight, vol term structure, BTC/XAU conditional beta,
Kaufman efficiency, buy pressure, up-day coherence, bond-corr.

This screen tests NEW families:
  adx_14            Wilder ADX trend-strength (direction-agnostic)
  risk_adj_mom_10x60 short momentum scaled by 1/realized vol
  vol_mom_20x40     vol momentum: rv20/rv20.shift(20)-1
  gain_loss_asym_60 mean(up)/|mean(down)| lottery/asymmetry proxy
  macd_hist_12x26   MACD histogram oscillator
  mom_term_20x120   TSMOM term structure: mom20 - mom120
  mfi_14            Money Flow Index (volume-weighted overbought/oversold)
  dxy_beta_cond_60x20 conditional USD (DXY) beta, macro risk dimension

SCREEN ONLY (no persistence). Best candidate(s) clearing gates get a dedicated
deep-validation script.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner_3_20260730_common import (
    load_data, factor_ic_table, coverage_stats, rank_turnover, library_factors,
)

data = load_data(days=3200)
closes = {a: d["close"].astype(float) for a, d in data.items()}
opens = {a: d["open"].astype(float) for a, d in data.items()}
highs = {a: d["high"].astype(float) for a, d in data.items()}
lows = {a: d["low"].astype(float) for a, d in data.items()}
vols = {a: d["volume"].astype(float).replace(0, np.nan) for a, d in data.items()}
print(f"[screen3] assets={len(data)} range={min(c.index.min() for c in closes.values()).date()}..{max(c.index.max() for c in closes.values()).date()}")


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
    up = r.clip(lower=0.0)
    dn = (-r).clip(lower=0.0)
    mu = up.rolling(60).mean()
    md = dn.rolling(60).mean()
    return mu / md.replace(0, np.nan)


def macd_hist_12x26(c, o, h, l, v, **kw):
    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    macd = e12 - e26
    sig = macd.ewm(span=9, adjust=False).mean()
    return (macd - sig) / c  # normalized histogram


def mom_term_20x120(c, o, h, l, v, **kw):
    m20 = c / c.shift(20) - 1.0
    m120 = c / c.shift(120) - 1.0
    return m20 - m120


def mfi_14(c, o, h, l, v, win=14):
    tp = (h + l + c) / 3.0
    raw = tp * v
    pos = raw.where(tp > tp.shift(1), 0.0)
    neg = raw.where(tp < tp.shift(1), 0.0)
    pf = pos.rolling(win).sum()
    nf = neg.rolling(win).sum()
    mr = pf / nf.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + mr)


def dxy_beta_cond_60x20(c, o, h, l, v, **kw):
    if "dxy" not in kw:
        return pd.Series(np.nan, index=c.index)
    dxy = kw["dxy"]
    r = c.pct_change()
    dr = dxy.pct_change()
    beta = r.rolling(60).cov(dr) / dr.rolling(60).var()
    move = dxy / dxy.shift(20) - 1.0
    return -beta * move


# load DXY (observation-only), truncate at the last tradable-visible date
dxy_df = pd.read_csv("../persistent/index_data/DXY.csv")
dxy_df["date"] = pd.to_datetime(dxy_df["date"])
dxy = dxy_df.set_index("date").sort_index()["close"].astype(float)
last_vis = max(d.index.max() for d in data.values())
dxy = dxy[dxy.index <= last_vis]
print(f"[screen3] DXY loaded {dxy.index.min().date()}..{dxy.index.max().date()} n={len(dxy)}")

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

LIB = library_factors(data)

def lib_corr(factor):
    from scipy.stats import pearsonr
    fdf = pd.DataFrame(factor).stack()
    fdf = fdf[fdf.notna()]
    out = {}
    for fid, lf in LIB.items():
        ldf = pd.DataFrame(lf).stack()
        both = fdf.index.intersection(ldf.index)
        if len(both) < 100:
            out[fid] = float("nan")
            continue
        rho, _ = pearsonr(fdf.loc[both].values, ldf.loc[both].values)
        out[fid] = rho
    vals = [abs(v) for v in out.values() if np.isfinite(v)]
    return (max(vals) if vals else float("nan")), out

for fid, desc, params in CANDIDATES:
    try:
        f = {}
        for a, c in closes.items():
            if fid == "dxy_beta_cond_60x20":
                f[a] = dxy_beta_cond_60x20(c, opens[a], highs[a], lows[a], vols[a], dxy=dxy)
            else:
                f[a] = globals()[fid](c, opens[a], highs[a], lows[a], vols[a], **params)
    except Exception as e:
        print(f"[screen3] {fid}: BUILD ERROR {e}")
        continue
    f = {a: s.replace([np.inf, -np.inf], np.nan) for a, s in f.items()}
    tbl = factor_ic_table(f, data, horizons=(1, 3, 5, 10, 20), min_assets=8, primary_h=10)
    prim = tbl[10]
    if prim is None:
        print(f"[screen3] {fid:22s} {desc:34s} DEGENERATE")
        continue
    cov = coverage_stats(f, data)
    to = rank_turnover(f)
    maxrho, rho_map = lib_corr(f)
    gate_ic = abs(prim["ic"]) >= 0.0070
    gate_icir = abs(prim["icir"]) >= 0.0840
    gate_rho = (np.isfinite(maxrho) and maxrho < 0.5)
    flag = "PASS" if (gate_ic and gate_icir and gate_rho) else "fail"
    print(f"[screen3] {fid:22s} {desc:34s} ic10={prim['ic']:+.4f} icir10={prim['icir']:+.4f} "
          f"hit={prim['ic_hit']:.3f} n={prim['n_dates']:4d} cov={cov['coverage_asset_days']:.3f} "
          f"ge8={prim['dates_ge8']:.3f} turn={to:5.2f} maxrho={maxrho:.3f} -> {flag}")
    print(f"          rho={ {k: round(v, 2) for k, v in rho_map.items()} } decay={ {h: (round(v['ic'], 4) if v else None) for h, v in tbl.items()} }")
