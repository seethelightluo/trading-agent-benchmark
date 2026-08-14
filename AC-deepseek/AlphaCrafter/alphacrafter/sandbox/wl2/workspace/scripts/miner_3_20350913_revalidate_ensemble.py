"""miner_3 2035-09-13: revalidate the 5 current ensemble factors exactly as persisted.
Data through 2035-09-12 (visible). Vectorized row-Pearson on pct-ranked matrices
(Spearman rank IC). Gates: |IC|>=0.0070, |ICIR|>=0.0840 (15-instrument cross-asset
universe, >=8 valid per date). Horizon 10d.
"""
import sys, json, time
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
from miner_3_20260813_lib import (
    GRID, ASSETS, to_grid, load_macro,
    safe_div, cross_sectional_rank,
    summarize, turnover_10d_rank, coverage_stats, HORIZON, MIN_ASSETS,
)

OUT = "scripts/miner_3_20350913_revalidate_results.json"
DAYS = 4300
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


def roll_beta_cond_vec(x, y, w, minp, cond=None):
    xx = x.astype(float)
    yy = y.astype(float)
    valid = xx.notna() & yy.notna()
    if cond is not None:
        valid = valid & cond.astype(bool)
    m = valid.astype(float)
    sx = (xx * m).rolling(w, min_periods=minp).sum()
    sy = (yy * m).rolling(w, min_periods=minp).sum()
    sxy = (xx * yy * m).rolling(w, min_periods=minp).sum()
    sxx = (xx * xx * m).rolling(w, min_periods=minp).sum()
    n = m.rolling(w, min_periods=minp).sum()
    mx = sx / n
    my = sy / n
    cov = sxy / n - mx * my
    varx = sxx / n - mx * mx
    beta = cov / varx
    return beta.where(varx > 1e-12)


def streak_count(pos):
    cs = pos.cumsum()
    last_reset = cs.where(pos == 0).ffill().fillna(0.0)
    return (cs - last_reset).where(pos == 1, 0.0)


def row_pearson(X, Y, minp=MIN_ASSETS):
    V = ~(np.isnan(X) | np.isnan(Y))
    Xm = np.where(V, X, np.nan)
    Ym = np.where(V, Y, np.nan)
    mx = np.nanmean(Xm, axis=1, keepdims=True)
    my = np.nanmean(Ym, axis=1, keepdims=True)
    dx = Xm - mx
    dy = Ym - my
    num = np.nansum(dx * dy, axis=1)
    den = np.sqrt(np.nansum(dx * dx, axis=1) * np.nansum(dy * dy, axis=1))
    ic = num / den
    ic[np.sum(V, axis=1) < minp] = np.nan
    return ic


def library_pairwise_corr_ex(factor_mat, fid):
    import glob, os
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
        rho = None
        for t in range(rows):
            x = a[t]; y = b[t]
            ok = ~(np.isnan(x) | np.isnan(y))
            if ok.sum() >= MIN_ASSETS:
                xs = pd.Series(x[ok]).rank(); ys = pd.Series(y[ok]).rank()
                c = xs.corr(ys)
                if np.isfinite(c):
                    rho = c
                    break
        if rho is not None:
            out[name] = round(float(rho), 4)
    if out:
        mx = max(out.items(), key=lambda kv: abs(kv[1]))
        return out, mx[0], abs(mx[1])
    return out, None, 0.0


series = asset_series_full()
print("assets with data:", sorted(series.keys()), "elapsed", round(time.time() - T0, 1))

fwd_ranks = {}
for h in (1, 2, 3, 5, 10, 20):
    d = {s: df["close"].shift(-h) / df["close"] - 1.0 for s, df in series.items()}
    fwd_ranks[h] = cross_sectional_rank(to_grid(d))

dates = np.array(GRID)
spx = series["SPX"]["close"]
spxr = spx.pct_change()
vix = load_macro("VIX")

factors = {}

# ============ THE 5 CURRENT ENSEMBLE FACTORS (exact persisted definitions) ============
for s, df in series.items():
    pos = (df["ret"] > 0).astype(float)
    factors.setdefault("max_consec_gain_20", {})[s] = streak_count(pos).rolling(20, min_periods=10).max()

for s, df in series.items():
    a = df["ret"]
    b = spxr.reindex(df.index)
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

print("ensemble factor panels computed, elapsed", round(time.time() - T0, 1), flush=True)

results = {}
for fid in ["max_consec_gain_20", "spx_corr60", "mom_180d_skip5", "range_pos_252", "downbeta_spx_60"]:
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
    print(f"{fid:26s} ic={s['ic']:+.4f} icir={s['icir']:+.4f} hit={s['hit']:.3f} "
          f"turn={s['turnover_10d_rank']:.3f} cov={s['coverage']:.3f} maxrho={max_rho:.3f} "
          f"l250_ic={reg250.get('ic','NA')} ok={s['ok']}", flush=True)

json.dump(results, open(OUT, "w"), indent=1, default=str)
print("DONE. saved", OUT, "elapsed", round(time.time() - T0, 1))
