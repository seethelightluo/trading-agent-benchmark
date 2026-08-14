"""miner_3 2035-04-12 cycle screen: NEW factor candidates on 15-instrument cross-asset
universe, data through 2035-04-11 (visible; macro clipped to VISIBLE to avoid look-ahead).
Gates: |IC|>=0.0070, |ICIR|>=0.0840 (10d horizon), >=8 valid assets per date.
Method: daily cross-sectional Spearman (row-Pearson on pct-ranks)."""
import sys, json, time, glob, os
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
from miner_3_20260813_lib import (
    GRID, ASSETS, to_grid, load_macro, VISIBLE,
    safe_div, cross_sectional_rank, summarize, turnover_10d_rank,
    coverage_stats, MIN_ASSETS,
)

OUT = "scripts/miner_3_20350412_screen_results.json"
DAYS = 2100
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
        d = pd.DataFrame({
            "close": close, "ret": ret,
            "open": df["open"].astype(float), "high": df["high"].astype(float),
            "low": df["low"].astype(float),
        })
        out[s] = d
    return out


def macro_clipped(name):
    m = load_macro(name)
    if m is None:
        return None
    return m[m.index <= VISIBLE]


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

# forward returns by horizon (own calendar -> grid)
fwd_ranks = {}
for h in (1, 2, 3, 5, 10, 20):
    d = {s: df["close"].shift(-h) / df["close"] - 1.0 for s, df in series.items()}
    fwd_ranks[h] = cross_sectional_rank(to_grid(d))

dates = np.array(GRID)
spx = series["SPX"]["close"]
spxr = spx.pct_change()
dxy = macro_clipped("DXY")
vix = macro_clipped("VIX")
btc = series["BTC"]["close"] if "BTC" in series else None

factors = {}

def put(fid, s, vals):
    factors.setdefault(fid, {})[s] = vals

# ---- C1 gap_avg_5: mean overnight gap (open/prev_close-1) over 5d
for s, df in series.items():
    gap = df["open"] / df["close"].shift(1) - 1.0
    put("gap_avg_5", s, gap.rolling(5, min_periods=3).mean())

# ---- C2 gap_avg_20: mean overnight gap over 20d
for s, df in series.items():
    gap = df["open"] / df["close"].shift(1) - 1.0
    put("gap_avg_20", s, gap.rolling(20, min_periods=10).mean())

# ---- C3 range_ratio_5_60: 5d avg (hi-lo)/close vs 60d avg (contraction/expansion)
for s, df in series.items():
    rng = (df["high"] - df["low"]) / df["close"]
    put("range_ratio_5_60", s, safe_div(rng.rolling(5, min_periods=3).mean(),
                                        rng.rolling(60, min_periods=30).mean()))

# ---- C4 vol_ratio_5_60: vol5 / vol60 term structure
for s, df in series.items():
    v5 = df["ret"].rolling(5, min_periods=3).std()
    v60 = df["ret"].rolling(60, min_periods=30).std()
    put("vol_ratio_5_60", s, safe_div(v5, v60))

# ---- C5 trend_accel_60: eff_ratio_60 - eff_ratio_20 (trend efficiency acceleration)
for s, df in series.items():
    path60 = df["ret"].abs().rolling(60, min_periods=30).sum()
    move60 = (df["close"] / df["close"].shift(60) - 1.0).abs()
    er60 = safe_div(move60, path60)
    path20 = df["ret"].abs().rolling(20, min_periods=10).sum()
    move20 = (df["close"] / df["close"].shift(20) - 1.0).abs()
    er20 = safe_div(move20, path20)
    put("trend_accel_60", s, er60 - er20)

# ---- C6 skew_change_20_60: ret skew 20d - skew 60d
for s, df in series.items():
    sk20 = df["ret"].rolling(20, min_periods=10).skew()
    sk60 = df["ret"].rolling(60, min_periods=30).skew()
    put("skew_change_20_60", s, sk20 - sk60)

# ---- C7 vix_fall_mom20: 20d momentum x I(VIX 20d change < 0)
if vix is not None:
    for s, df in series.items():
        vx = vix.reindex(df.index)
        vchg = (vx / vx.shift(20) - 1.0)
        mom = df["close"] / df["close"].shift(20) - 1.0
        cond = (vchg < 0).astype(float).replace(0, np.nan)
        put("vix_fall_mom20", s, mom * cond)

# ---- C8 dxy_beta_trend: dxy beta 60d - dxy beta 120d
if dxy is not None:
    for s, df in series.items():
        dy = dxy.reindex(df.index)
        b60 = roll_beta_cond_vec(df["ret"], dy.pct_change(), 60, 30)
        b120 = roll_beta_cond_vec(df["ret"], dy.pct_change(), 120, 60)
        put("dxy_beta_trend", s, b60 - b120)

# ---- C9 risk_adj_mom5: 5d return / vol20 (short risk-adjusted momentum)
for s, df in series.items():
    m5 = df["close"] / df["close"].shift(5) - 1.0
    v20 = df["ret"].rolling(20, min_periods=10).std()
    put("risk_adj_mom5", s, safe_div(m5, v20))

# ---- C10 corr_spx_change_60_20: spx corr 60d - spx corr 20d
for s, df in series.items():
    a = df["ret"]
    b = spxr.reindex(df.index)
    c60 = a.rolling(60, min_periods=30).corr(b)
    c20 = a.rolling(20, min_periods=10).corr(b)
    put("corr_spx_change_60_20", s, c60 - c20)

# ---- C11 sys_corr_60: mean pairwise return corr vs other assets over 60d
rets = pd.DataFrame({s: df["ret"] for s, df in series.items()})
names = list(series.keys())
for s in names:
    others = [o for o in names if o != s]
    cors = rets[s].rolling(60, min_periods=30).corr(rets[others[0]])
    for o in others[1:]:
        c = rets[s].rolling(60, min_periods=30).corr(rets[o])
        cors = cors.add(c, fill_value=0)
    put("sys_corr_60", s, cors / len(others))

# ---- C12 ew_beta_60: rolling beta vs equal-weight cross-sectional index
ew = rets.mean(axis=1)
for s in names:
    b = rets[s].rolling(60, min_periods=30).cov(ew) / ew.rolling(60, min_periods=30).var()
    put("ew_beta_60", s, b)

# ---- C13 tail_mag_20: max |ret| / vol20 over 20d (tail magnitude)
for s, df in series.items():
    mx = df["ret"].abs().rolling(20, min_periods=10).max()
    v20 = df["ret"].rolling(20, min_periods=10).std()
    put("tail_mag_20", s, safe_div(mx, v20))

# ---- C14 dd_depth_60: 60d drawdown depth (close / trailing max - 1)
for s, df in series.items():
    rl = df["close"].rolling(60, min_periods=30).max()
    put("dd_depth_60", s, df["close"] / rl - 1.0)

# ---- C15 intraday_gap_ratio_20: mean |gap| / mean |intraday move| 20d
for s, df in series.items():
    gap = (df["open"] / df["close"].shift(1) - 1.0).abs()
    intra = (df["close"] / df["open"] - 1.0).abs()
    put("intraday_gap_ratio_20", s, safe_div(gap.rolling(20, min_periods=10).mean(),
                                             intra.rolling(20, min_periods=10).mean()))

print("all factor panels computed, elapsed", round(time.time() - T0, 1), flush=True)

results = {}
for fid, panel in factors.items():
    mat = to_grid(panel)
    rank_mat = cross_sectional_rank(mat)
    ic10 = row_pearson(rank_mat, fwd_ranks[10])
    idx = np.where(np.isfinite(ic10))[0]
    if len(idx) == 0:
        print(fid, "NO IC DATES")
        continue
    ics = [(int(t), float(ic10[t])) for t in idx]
    s = summarize(ics, dates, fid, 10)
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
