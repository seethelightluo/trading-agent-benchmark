"""miner_3 2035-11-08: explore new reversal/vol-regime factors + revalidate 5 ensemble factors.
Data through 2035-11-07 (visible). Universe: 15 cross-asset tradable instruments, >=8 valid per date.
Gates: |IC|>=0.0070, |ICIR|>=0.0840 at 10d horizon. Spearman rank IC via cross-sectional pct-ranks.
"""
import sys, json, time, os
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
from miner_3_20260813_lib import (
    ASSETS, GRID, to_grid, load_macro, safe_div,
    cross_sectional_rank, summarize, turnover_10d_rank,
    coverage_stats, HORIZON, MIN_ASSETS, row_pearson,
)

OUT = "scripts/miner_3_20351108_screen_results.json"
DAYS = 4900
T0 = time.time()


def load_asset_raw(sym, days=DAYS):
    df = get_stock_daily_data(sym, days=days)
    if df is None:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.set_index("date")
    for c in ["open", "close", "high", "low", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def asset_series_full():
    out = {}
    for s in ASSETS:
        df = load_asset_raw(s)
        if df is None or len(df) < 100:
            continue
        close = df["close"].astype(float)
        ret = close.pct_change()
        fwd = close.shift(-HORIZON) / close - 1.0
        d = pd.DataFrame({
            "close": close, "ret": ret, "fwd10": fwd,
            "open": df["open"].astype(float), "high": df["high"].astype(float),
            "low": df["low"].astype(float), "volume": df["volume"].astype(float),
        })
        out[s] = d
    return out


def streak_count(pos):
    cs = pos.cumsum()
    last_reset = cs.where(pos == 0).ffill().fillna(0.0)
    return (cs - last_reset).where(pos == 1, 0.0)


def roll_beta_cond_vec(x, y, w, minp, cond=None):
    xx = x.astype(float); yy = y.astype(float)
    valid = xx.notna() & yy.notna()
    if cond is not None:
        valid = valid & cond.astype(bool)
    m = valid.astype(float)
    sx = (xx * m).rolling(w, min_periods=minp).sum()
    sy = (yy * m).rolling(w, min_periods=minp).sum()
    sxy = (xx * yy * m).rolling(w, min_periods=minp).sum()
    sxx = (xx * xx * m).rolling(w, min_periods=minp).sum()
    n = m.rolling(w, min_periods=minp).sum()
    mx = sx / n; my = sy / n
    cov = sxy / n - mx * my
    varx = sxx / n - mx * mx
    beta = cov / varx
    return beta.where(varx > 1e-12)


def library_pairwise_corr_ex(factor_mat, fid):
    import glob
    ours = cross_sectional_rank(factor_mat)
    out = {}
    for f in sorted(glob.glob("factors/*.signal.npy")):
        name = os.path.basename(f).replace(".signal.npy", "")
        if name == fid:
            continue
        arr = np.load(f, allow_pickle=True)
        rows = min(arr.shape[0], ours.shape[0])
        a = ours[:rows]
        b = arr[:rows]
        rhos = []
        for t in range(rows):
            x = a[t]; y = b[t]
            ok = ~(np.isnan(x) | np.isnan(y))
            if ok.sum() >= MIN_ASSETS:
                xs = pd.Series(x[ok]).rank(); ys = pd.Series(y[ok]).rank()
                r = xs.corr(ys)
                if np.isfinite(r):
                    rhos.append(r)
        if rhos:
            out[name] = round(float(np.nanmean(rhos)), 4)
    if not out:
        return {}, "none", 0.0
    best = max(out.items(), key=lambda kv: abs(kv[1]))
    return out, best[0], abs(best[1])


series = asset_series_full()
print("assets loaded:", len(series), "elapsed", round(time.time() - T0, 1), flush=True)

spxr = series["SPX"]["ret"]
vix = load_macro("VIX")
dxy = load_macro("DXY")

factors = {}

# ---- ensemble revalidation (exact persisted definitions) ----
for s, df in series.items():
    pos = (df["ret"] > 0).astype(float)
    factors.setdefault("max_consec_gain_20", {})[s] = streak_count(pos).rolling(20, min_periods=10).max()
for s, df in series.items():
    a = df["ret"]; b = spxr.reindex(df.index)
    factors.setdefault("spx_corr60", {})[s] = a.rolling(60, min_periods=15).corr(b)
for s, df in series.items():
    factors.setdefault("mom_180d_skip5", {})[s] = df["close"].shift(5) / df["close"].shift(185) - 1.0
for s, df in series.items():
    lo = df["close"].rolling(252, min_periods=30).min()
    hi = df["close"].rolling(252, min_periods=30).max()
    factors.setdefault("range_pos_252", {})[s] = pd.Series(safe_div(df["close"] - lo, hi - lo), index=df.index)
for s, df in series.items():
    spxr_s = spxr.reindex(df.index)
    factors.setdefault("downbeta_spx_60", {})[s] = roll_beta_cond_vec(df["ret"], spxr_s, 60, 15, cond=spxr_s < 0)

# ---- new candidates ----
for s, df in series.items():
    r5 = df["close"] / df["close"].shift(5) - 1.0
    vol60 = df["ret"].rolling(60, min_periods=20).std()
    factors.setdefault("rev5_vol60", {})[s] = pd.Series(safe_div(-r5, vol60), index=df.index)  # oversold(bounce) score
for s, df in series.items():
    r20 = df["close"] / df["close"].shift(20) - 1.0
    vol60 = df["ret"].rolling(60, min_periods=20).std()
    factors.setdefault("rev20_vol60", {})[s] = pd.Series(safe_div(-r20, vol60), index=df.index)  # medium-horizon reversal
for s, df in series.items():
    vol20 = df["ret"].rolling(20, min_periods=10).std()
    factors.setdefault("rel_vol_20", {})[s] = vol20  # raw panel; demeaned cross-sectionally below
for s, df in series.items():
    r60 = df["close"] / df["close"].shift(60) - 1.0
    vol60 = df["ret"].rolling(60, min_periods=20).std()
    factors.setdefault("dd60_bounce", {})[s] = pd.Series(safe_div(-np.minimum(r60, 0.0), vol60), index=df.index)  # crash depth / vol
for s, df in series.items():
    vol5 = df["ret"].rolling(5, min_periods=3).std()
    vol60 = df["ret"].rolling(60, min_periods=20).std()
    factors.setdefault("vol_ratio_5_60", {})[s] = pd.Series(safe_div(vol5, vol60), index=df.index)  # short vol spike

# cross-sectional demean for rel_vol_20
rel_vol_panel = factors["rel_vol_20"]
tmp = to_grid(rel_vol_panel)
med = np.nanmedian(tmp, axis=1, keepdims=True)
rel_mat = tmp / med
# rebuild as panel
for j, s in enumerate(ASSETS):
    rel_vol_panel[s] = pd.Series(rel_mat[:, j], index=GRID)

print("all panels computed, elapsed", round(time.time() - T0, 1), flush=True)

dates = np.array(GRID)
fwd_ranks = {}
for h in [1, 2, 3, 5, 10, 20]:
    cols = {}
    for s, df in series.items():
        cols[s] = df["close"].shift(-h) / df["close"] - 1.0
    fwd_ranks[h] = cross_sectional_rank(to_grid(cols))

ENSEMBLE = ["max_consec_gain_20", "spx_corr60", "mom_180d_skip5", "range_pos_252", "downbeta_spx_60"]
NEW = ["rev5_vol60", "rev20_vol60", "rel_vol_20", "dd60_bounce", "vol_ratio_5_60"]

results = {}
for fid in ENSEMBLE + NEW:
    panel = factors[fid]
    mat = to_grid(panel)
    rank_mat = cross_sectional_rank(mat)
    ic10 = row_pearson(rank_mat, fwd_ranks[10])
    idx = np.where(np.isfinite(ic10))[0]
    if len(idx) == 0:
        print(fid, "NO IC DATES")
        continue
    ics = [(int(t), float(ic10[t])) for t in idx]
    s = summarize(ics, dates, fid, HORIZON)
    if s is None:
        continue
    decay = {}
    for h, fwd_r in fwd_ranks.items():
        ih = row_pearson(rank_mat, fwd_r)
        if np.any(np.isfinite(ih)):
            decay[str(h)] = round(float(np.nanmean(ih)), 4)
    s["decay"] = decay
    rho_dict, rho_name, max_rho = library_pairwise_corr_ex(mat, fid)
    s["max_abs_library_correlation"] = round(max_rho, 4)
    s["max_corr_with"] = rho_name
    s["turnover_10d_rank"] = round(turnover_10d_rank(rank_mat), 4)
    cov, dates_ge8 = coverage_stats(mat)
    s["coverage"] = round(cov, 4)
    s["dates_ge8_frac"] = round(dates_ge8, 4)
    s["ok"] = bool((abs(s["ic"]) >= 0.0070) and (abs(s["icir"]) >= 0.0840))
    s.pop("idx", None); s.pop("icv", None)
    results[fid] = s
    reg250 = s["regime"].get("last250", {})
    print(f"{fid:22s} ic={s['ic']:+.4f} icir={s['icir']:+.4f} hit={s['hit']:.3f} "
          f"turn={s['turnover_10d_rank']:.3f} cov={s['coverage']:.3f} maxrho={max_rho:.3f} "
          f"({rho_name}) l250_ic={reg250.get('ic','NA')} l250_icir={reg250.get('icir','NA')} ok={s['ok']}",
          flush=True)

json.dump(results, open(OUT, "w"), indent=1, default=str)
print("DONE. saved", OUT, "elapsed", round(time.time() - T0, 1))
