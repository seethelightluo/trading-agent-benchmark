"""miner_3 2026-07-30 -- Screen round 4: orthogonal factor families.

Context: active library = mom_10d_skip5, vix_beta_cond_60x20, yield_beta_cond_60x20.
Post-gate evicts candidates with pairwise SPEARMAN rho > 0.5 vs library (in
particular vs yield_beta_cond_60x20, which killed vol/beta/momentum-type factors).
Goal: find factor families with DIFFERENT cross-sectional structure.

Candidates (all new or re-tested with gate-aligned spearman check):
  seasonal_month_756  calendar month mean return per asset (trailing 3y)
  seasonal_month_252  calendar month mean return per asset (trailing 1y)
  vol_trend_20x60     volume momentum: 20d mean volume / 60d mean volume - 1
  chaikin_mf_20       Chaikin Money Flow (volume+close-location)
  vpt_slope_20        Volume-Price Trend 20d slope / close
  gap_ret_20          avg overnight gap (open/prev_close - 1) over 20d
  close_loc_5         5d close location in range (short-horizon stoch)
  dd_20h              drawdown from 20d high
  eff_ratio_120       Kaufman efficiency ratio 120d
  down_vol_ratio      downside semi-dev 20d / 60d (risk regime change)
  ret_atr_10          10d return / ATR14 (risk-scaled trend)
  hi_lo_pos_20        avg (c-l)/(h-l) over 20d (buying pressure)

SCREEN ONLY. Deep validation + persistence only for candidates passing both
the IC/ICIR gate and |spearman rho| < 0.5 vs ACTIVE library.
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

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
rets = {a: c.pct_change() for a, c in closes.items()}
last_vis = max(d.index.max() for d in data.values())
print(f"[screen4] assets={len(data)} range={min(c.index.min() for c in closes.values()).date()}..{last_vis.date()}")

# ---------- ACTIVE library replication (must match persisted definitions) ----------
def load_obs(name):
    df = pd.read_csv(f"../persistent/index_data/{name}.csv")
    df["date"] = pd.to_datetime(df["date"])
    s = df.set_index("date").sort_index()["close"].astype(float)
    return s[s.index <= last_vis]

VIX = load_obs("VIX")
US10Y = closes["US10Y"]

def active_library():
    lib = {}
    for a, c in closes.items():
        lib.setdefault("mom_10d_skip5", {})[a] = c.shift(5) / c.shift(15) - 1.0
        r = c.pct_change()
        vix_r = VIX.pct_change()
        beta_v = r.rolling(60).cov(vix_r) / vix_r.rolling(60).var()
        lib.setdefault("vix_beta_cond_60x20", {})[a] = -beta_v * (VIX / VIX.shift(20) - 1.0)
        yr = US10Y.pct_change()
        beta_y = r.rolling(60).cov(yr) / yr.rolling(60).var()
        lib.setdefault("yield_beta_cond_60x20", {})[a] = -beta_y * (US10Y / US10Y.shift(20) - 1.0)
    return lib

LIB = active_library()
print("[lib] active:", list(LIB.keys()))

def spearman_panel_corr(factor):
    """Spearman rho on stacked (date,asset) panel - matches post-gate check."""
    fdf = pd.DataFrame(factor).stack()
    fdf = fdf[fdf.notna()]
    out = {}
    for fid, lf in LIB.items():
        ldf = pd.DataFrame(lf).stack()
        both = fdf.index.intersection(ldf.index)
        if len(both) < 100:
            out[fid] = float("nan")
            continue
        rho, _ = spearmanr(fdf.loc[both].values, ldf.loc[both].values)
        out[fid] = float(rho)
    return out

# ---------- candidate builders ----------
def seasonal_month(c, win_days):
    """Asset-specific calendar-month mean daily return over trailing window."""
    r = c.pct_change()
    idx = c.index
    out = pd.Series(np.nan, index=idx)
    months = idx.month
    # precompute month-group mean returns over expanding trailing window using
    # rolling by date: for each date, use returns in [date-win, date)
    for m in range(1, 13):
        mask = (months == m)
        dates_m = idx[mask]
        # for each date in month m, mean of r over prior dates that are in month m
        vals = np.full(len(idx), np.nan)
        for i, dt in enumerate(idx):
            if not mask.iloc[i]:
                continue
            lo = dt - pd.Timedelta(days=win_days * 1.6)  # approx calendar window
            sel = r[(idx >= lo) & (idx < dt) & (months == m)]
            if len(sel) >= 5:
                vals[i] = sel.mean()
        out.iloc[mask] = vals[mask.values]
    return out

def vol_trend_20x60(v):
    return v.rolling(20).mean() / v.rolling(60).mean() - 1.0

def chaikin_mf_20(c, h, l, v, win=20):
    mfm = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
    mfv = mfm * v
    return mfv.rolling(win).sum() / v.rolling(win).sum().replace(0, np.nan)

def vpt_slope_20(c, v, win=20):
    vpt = (c.pct_change() * v).fillna(0.0).cumsum()
    return vpt.diff(win) / c

def gap_ret_20(c, o, win=20):
    gap = o / c.shift(1) - 1.0
    return gap.rolling(win).mean()

def close_loc_5(c, h, l):
    rng = (h.rolling(5).max() - l.rolling(5).min()).replace(0, np.nan)
    return (c - l.rolling(5).min()) / rng

def dd_20h(c):
    return c / c.rolling(20).max() - 1.0

def eff_ratio_120(c):
    return c.diff(120).abs() / c.diff().abs().rolling(120).sum()

def down_vol_ratio(c, s=20, l=60):
    r = c.pct_change()
    dn = r.clip(upper=0.0)
    sd_s = dn.rolling(s).std()
    sd_l = dn.rolling(l).std()
    return sd_s / sd_l.replace(0, np.nan) - 1.0

def ret_atr_10(c, h, l, win=14):
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / win, adjust=False).mean()
    return (c / c.shift(10) - 1.0) / atr.replace(0, np.nan)

def hi_lo_pos_20(c, h, l, win=20):
    rng = (h - l).replace(0, np.nan)
    return ((c - l) / rng).rolling(win).mean()

CANDIDATES = [
    ("seasonal_month_756", lambda c, o, h, l, v: seasonal_month(c, 756), "calendar month mean ret, trailing ~3y"),
    ("seasonal_month_252", lambda c, o, h, l, v: seasonal_month(c, 252), "calendar month mean ret, trailing ~1y"),
    ("vol_trend_20x60", lambda c, o, h, l, v: vol_trend_20x60(v), "volume momentum 20/60"),
    ("chaikin_mf_20", lambda c, o, h, l, v: chaikin_mf_20(c, h, l, v), "Chaikin Money Flow 20"),
    ("vpt_slope_20", lambda c, o, h, l, v: vpt_slope_20(c, v), "VPT 20d slope"),
    ("gap_ret_20", lambda c, o, h, l, v: gap_ret_20(c, o), "avg overnight gap 20d"),
    ("close_loc_5", lambda c, o, h, l, v: close_loc_5(c, h, l), "5d close location"),
    ("dd_20h", lambda c, o, h, l, v: dd_20h(c), "drawdown from 20d high"),
    ("eff_ratio_120", lambda c, o, h, l, v: eff_ratio_120(c), "Kaufman efficiency 120d"),
    ("down_vol_ratio", lambda c, o, h, l, v: down_vol_ratio(c), "downside semi-dev 20/60 - 1"),
    ("ret_atr_10", lambda c, o, h, l, v: ret_atr_10(c, h, l), "10d ret / ATR14"),
    ("hi_lo_pos_20", lambda c, o, h, l, v: hi_lo_pos_20(c, h, l), "buying pressure 20d"),
]

print("=" * 110)
for fid, fn, desc in CANDIDATES:
    panel = {}
    for a in closes:
        try:
            panel[a] = fn(closes[a], opens[a], highs[a], lows[a], vols[a])
        except Exception as e:
            print(f"  [{fid}] {a} ERROR: {e}")
    tbl = factor_ic_table(panel, data, horizons=(1, 3, 5, 10, 20))
    prim = tbl[10]
    if prim is None or prim["n_dates"] < 200:
        print(f"[{fid:18s}] {desc:42s} DEGENERATE n={prim['n_dates'] if prim else 0}")
        continue
    cov = coverage_stats(panel, data)
    to = rank_turnover(panel)
    rho_map = spearman_panel_corr(panel)
    maxrho = max((abs(v) for v in rho_map.values() if np.isfinite(v)), default=float("nan"))
    gate = abs(prim["ic"]) >= 0.0070 and abs(prim["icir"]) >= 0.0840
    ortho = np.isfinite(maxrho) and maxrho < 0.5
    decay = {str(h): (round(v["ic"], 4) if v else None) for h, v in tbl.items()}
    print(f"[{fid:18s}] {desc:42s} IC10={prim['ic']:+.4f} ICIR10={prim['icir']:+.4f} "
          f"hit={prim['ic_hit']:.3f} n={prim['n_dates']} cov={cov['coverage_asset_days']:.3f} "
          f"turn={to:.2f} spearman={ {k: round(v,2) for k,v in rho_map.items()} } "
          f"max={maxrho:.2f} | gate={'PASS' if gate else 'fail'} ortho={'OK' if ortho else 'RISK'} | {decay}")
