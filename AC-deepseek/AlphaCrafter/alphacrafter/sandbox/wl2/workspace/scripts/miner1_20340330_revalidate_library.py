"""miner_1 cycle: re-validate the ENTIRE factor library on fresh data through 2034-03-29.

All library factors were last validated 2026-07/08 (7.7y stale). This script recomputes
every factor signal from raw OHLC (per-asset own trading calendar, no lookahead),
then evaluates full-sample + recent-window IC/ICIR against the benchmark admission gate
(|IC| >= 0.0070, |ICIR| >= 0.0840, ICIR = mean/std per-date ratio, consistent with
persisted validation.metrics).

Data: ../persistent/stock_data/*.csv + ../persistent/index_data/*.csv (macro obs-only)
Visible through: 2034-03-29 (current sim date 2034-03-30).
"""
import numpy as np
import pandas as pd
from pathlib import Path

TRADABLES = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
             "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]
VISIBLE = pd.Timestamp("2034-03-29")
DATA_DIR = Path("../persistent/stock_data")
INDEX_DIR = Path("../persistent/index_data")

def load_asset(symbol):
    p = (INDEX_DIR if symbol in MACRO else DATA_DIR) / f"{symbol}.csv"
    df = pd.read_csv(p, parse_dates=["date"])
    df = df[df["date"] <= VISIBLE].copy().sort_values("date").reset_index(drop=True)
    return df

def panel(column="close"):
    frames = {}
    for a in TRADABLES:
        df = load_asset(a)
        frames[a] = pd.Series(df[column].astype(float).values,
                              index=pd.to_datetime(df["date"]), name=a)
    return pd.concat(frames, axis=1).sort_index()

close = panel("close")
open_ = panel("open")
high = panel("high")
low = panel("low")
vol = panel("volume")
rets = close.pct_change()
print(f"close panel: {close.shape[0]} dates x {close.shape[1]} assets, "
      f"last date {close.index[-1].date()}")

macro = {m: load_asset(m).set_index("date")["close"] for m in MACRO}

def per_asset(panel_df, func, *args, **kwargs):
    out = {}
    for a in panel_df.columns:
        s = panel_df[a].dropna()
        out[a] = func(s, *args, **kwargs).reindex(panel_df.index)
    return pd.DataFrame(out, index=panel_df.index)

# ---------------- factor definitions (per-asset own calendar) ----------------
def f_calmness(s, w=20, mp=10):
    r = s.pct_change()
    return r.abs().lt(0.5 * r.rolling(w, min_periods=mp).std()).rolling(w, min_periods=mp).mean()

def f_close_pos(s, w=20, mp=10):
    # needs high/low: handled below
    return None

def f_days_since_high(s, w=60, mp=40):
    roll_max = s.rolling(w, min_periods=mp).max()
    vals = s.values
    mx = roll_max.values
    n = len(vals)
    res = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(mx[i]):
            continue
        # window is [i-w+1, i]; find last j in window with close[j] == trailing max
        lo = max(0, i - w + 1)
        j = i
        while j >= lo and (np.isnan(vals[j]) or vals[j] != mx[i]):
            j -= 1
        res[i] = i - j if j >= lo and vals[j] == mx[i] else np.nan
    return pd.Series(res, index=s.index)

def f_downbeta(s, w=60, min_down=15):
    spx = close["SPX"].reindex(s.index).pct_change()
    a = s.pct_change()
    m = spx.notna() & a.notna() & (spx < 0)
    b = a.rolling(w, min_periods=1).apply(
        lambda x: np.polyfit(spx.loc[x.index], a.loc[x.index], 1)[0] if
        (spx.loc[x.index] < 0).sum() >= min_down and len(x) >= 2 else np.nan, raw=False)
    return b

def f_dxy_beta_cond(s, bw=60, bmp=30, ml=20):
    dxy = macro["DXY"].reindex(s.index)
    a = s.pct_change()
    b = dxy.pct_change()
    beta = a.rolling(bw, min_periods=bmp).corr(b) * (a.rolling(bw, min_periods=bmp).std() / b.rolling(bw, min_periods=bmp).std())
    return beta * (dxy / dxy.shift(ml) - 1.0)

def f_gain_loss(s, w=20, mp=10):
    r = s.pct_change()
    g = r.clip(lower=0).rolling(w, min_periods=mp).mean()
    l = r.clip(upper=0).rolling(w, min_periods=mp).mean().abs()
    return g / (l + 1e-9)

def f_intraday_drift(s, w=20, mp=10):
    o = open_[s.name].reindex(s.index)
    return (s / o - 1.0).rolling(w, min_periods=mp).mean()

def f_lagbeta(s, w=60, mp=30):
    spx = close["SPX"].reindex(s.index).pct_change().shift(1)
    a = s.pct_change()
    m = spx.notna() & a.notna()
    cov = a.rolling(w, min_periods=mp).cov(spx)
    var = spx.rolling(w, min_periods=mp).var()
    return cov / var

def f_max_consec_gain(s, w=20, mp=10):
    r = (s.pct_change() > 0).astype(float)
    out = pd.Series(np.nan, index=s.index)
    vals = r.values
    n = len(vals)
    res = np.full(n, np.nan)
    run = 0
    for i in range(n):
        if np.isnan(vals[i]):
            run = 0
            continue
        run = run + 1 if vals[i] == 1 else 0
        res[i] = run
    rs = pd.Series(res, index=s.index)
    return rs.rolling(w + 1, min_periods=mp).max()

def f_max_consec_loss(s, w=20, mp=10):
    r = (s.pct_change() < 0).astype(float)
    vals = r.values
    n = len(vals)
    res = np.full(n, np.nan)
    run = 0
    for i in range(n):
        if np.isnan(vals[i]):
            run = 0
            continue
        run = run + 1 if vals[i] == 1 else 0
        res[i] = run
    rs = pd.Series(res, index=s.index)
    return rs.rolling(w + 1, min_periods=mp).max()

def f_mom(s, lb, skip=5):
    return s.shift(skip) / s.shift(lb + skip) - 1.0

def f_mom20_volproxy60(s, lb=20, skip=5):
    return s.shift(skip) / s.shift(lb + skip) - 1.0

def f_mom30_vol60(s, lb=30, skip=5, vw=60, mp=15):
    mom = s.shift(skip) / s.shift(lb + skip) - 1.0
    v = s.pct_change().rolling(vw, min_periods=mp).std()
    return mom / v

def f_range_pos(s, w=252, mp=30):
    mn = s.rolling(w, min_periods=mp).min()
    mx = s.rolling(w, min_periods=mp).max()
    return (s - mn) / (mx - mn)

def f_spx_corr(s, w=60, mp=15):
    spx = close["SPX"].reindex(s.index).pct_change()
    return s.pct_change().rolling(w, min_periods=mp).corr(spx)

def f_usdjpy_beta_cond(s, bw=120, ml=60, mpb=60):
    uj = macro["USDJPY"].reindex(s.index)
    a = s.pct_change()
    b = uj.pct_change()
    beta = a.rolling(bw, min_periods=mpb).corr(b) * (a.rolling(bw, min_periods=mpb).std() / b.rolling(bw, min_periods=mpb).std())
    return beta * (uj / uj.shift(ml) - 1.0)

def f_vix_beta_cond(s, bw=60, vw=20):
    vix = macro["VIX"].reindex(s.index)
    a = s.pct_change()
    b = vix.pct_change()
    beta = a.rolling(bw, min_periods=15).corr(b) * (a.rolling(bw, min_periods=15).std() / b.rolling(bw, min_periods=15).std())
    return -beta * (vix / vix.shift(vw) - 1.0)

def f_vol_of_vol(s, sw=20, lw=60, mps=5, mpl=15):
    v = s.pct_change().rolling(sw, min_periods=mps).std()
    return v.rolling(lw, min_periods=mpl).std()

def f_volcluster(s, w=60, mp=40):
    av = s.pct_change().abs()
    return av.rolling(w, min_periods=mp).corr(av.shift(1))

def f_close_pos2(s, w=20, mp=10):
    h = high[s.name].reindex(s.index)
    l = low[s.name].reindex(s.index)
    rng = (h - l)
    cp = ((s - l) / rng.replace(0, np.nan))
    return cp.rolling(w, min_periods=mp).mean()

# ---------------- compute all signals ----------------
FACTORS = {
    "calmness_20": f_calmness,
    "close_pos_20": f_close_pos2,
    "days_since_high_60": f_days_since_high,
    "downbeta_spx_60": f_downbeta,
    "dxy_beta_cond_60x20": f_dxy_beta_cond,
    "gain_loss_20": f_gain_loss,
    "intraday_drift_20": f_intraday_drift,
    "lagbeta_spx_60": f_lagbeta,
    "max_consec_gain_20": f_max_consec_gain,
    "max_consec_loss_20": f_max_consec_loss,
    "mom20_volproxy60": f_mom20_volproxy60,
    "mom30_vol60": f_mom30_vol60,
    "mom_10d_skip5": lambda s: f_mom(s, 10),
    "mom_120d_skip5": lambda s: f_mom(s, 120),
    "mom_180d_skip5": lambda s: f_mom(s, 180),
    "mom_20d_skip5": lambda s: f_mom(s, 20),
    "range_pos_252": f_range_pos,
    "spx_corr60": f_spx_corr,
    "usdjpy_beta_cond_120x60": f_usdjpy_beta_cond,
    "vix_beta_cond_60x20": f_vix_beta_cond,
    "vol_of_vol20x60": f_vol_of_vol,
    "volcluster_60": f_volcluster,
}

print("computing signals...")
signals = {}
for name, fn in FACTORS.items():
    signals[name] = per_asset(close, fn)
    cov = signals[name].notna().mean().mean()
    print(f"  {name:32s} coverage_asset_days={cov:.3f}")

# ---------------- forward returns (per-asset own calendar, h=10) ----------------
def fwd_ret_series(s, h):
    return s.shift(-h) / s - 1.0

fwd10 = per_asset(close, fwd_ret_series, 10)

def compute_ic(factor_panel, ret_panel, min_assets=8):
    dates = factor_panel.index.intersection(ret_panel.index)
    F = factor_panel.loc[dates]
    R = ret_panel.loc[dates]
    Fr = F.rank(axis=1).values
    Rr = R.rank(axis=1).values
    m = (~np.isnan(Fr)) & (~np.isnan(Rr))
    valid = m.sum(axis=1) >= min_assets
    ics = np.full(len(dates), np.nan)
    idx = np.where(valid)[0]
    for i in idx:
        f = Fr[i, m[i]] - Fr[i, m[i]].mean()
        r = Rr[i, m[i]] - Rr[i, m[i]].mean()
        denom = np.sqrt((f * f).sum() * (r * r).sum())
        ics[i] = (f * r).sum() / denom if denom > 0 else np.nan
    return pd.Series(ics, index=dates, name="ic")

print("\ncomputing IC series...")
results = []
for name, fp in signals.items():
    ic = compute_ic(fp, fwd10)
    s = ic.dropna()
    full_ic = s.mean()
    full_icir = full_ic / s.std(ddof=1) if s.std(ddof=1) > 0 else np.nan
    last250 = s.tail(250)
    l250_ic = last250.mean()
    l250_icir = l250_ic / last250.std(ddof=1) if last250.std(ddof=1) > 0 else np.nan
    last500 = s.tail(500)
    l500_ic = last500.mean()
    l500_icir = l500_ic / last500.std(ddof=1) if last500.std(ddof=1) > 0 else np.nan
    post27 = s[s.index >= "2027-01-01"]
    p27_ic = post27.mean()
    p27_icir = p27_ic / post27.std(ddof=1) if post27.std(ddof=1) > 0 else np.nan
    results.append({
        "factor": name, "n_ic": len(s),
        "ic": full_ic, "icir": full_icir,
        "ic_last250": l250_ic, "icir_last250": l250_icir,
        "ic_last500": l500_ic, "icir_last500": l500_icir,
        "ic_post27": p27_ic, "icir_post27": p27_icir,
        "gate_full": (abs(full_ic) >= 0.007 and abs(full_icir) >= 0.084),
        "gate_l250": (abs(l250_ic) >= 0.007 and abs(l250_icir) >= 0.084),
        "gate_post27": (abs(p27_ic) >= 0.007 and abs(p27_icir) >= 0.084),
    })
    print(f"{name:32s} n={len(s):5d} full ic={full_ic:+.4f} icir={full_icir:+.3f} | "
          f"l250 ic={l250_ic:+.4f} icir={l250_icir:+.3f} | post27 ic={p27_ic:+.4f} icir={p27_icir:+.3f}")

print("\n=== GATE SUMMARY (full-sample) ===")
for r in results:
    print(f"{r['factor']:32s} PASS={r['gate_full']!s:5s} ic={r['ic']:+.4f} icir={r['icir']:+.3f}")

print("\n=== GATE SUMMARY (last250) ===")
for r in results:
    print(f"{r['factor']:32s} PASS={r['gate_l250']!s:5s} ic={r['ic_last250']:+.4f} icir={r['icir_last250']:+.3f}")

print("\n=== GATE SUMMARY (post-2027) ===")
for r in results:
    print(f"{r['factor']:32s} PASS={r['gate_post27']!s:5s} ic={r['ic_post27']:+.4f} icir={r['icir_post27']:+.3f}")

import json
with open("scripts/miner1_20340330_reval_results.json", "w") as f:
    json.dump(results, f, indent=1, default=str)
print("\nsaved results json")
