"""SCREENER 2031-01-23: full library revalidation with data through visible_through (2031-01-22).
Recomputes every library factor from raw prices (truncated at VISIBLE to avoid lookahead),
reports IC/ICIR/hit/coverage/turnover/maxcorr, applies gates |IC|>=0.0070, |ICIR|>=0.0840,
and computes market-regime statistics for factor selection.
Conventions match miner_3_20260813_lib (per-asset own calendar, h=10 Spearman rank IC, >=8 assets).
"""
import json, os
import numpy as np
import pandas as pd

DATE_PATH = "../persistent/date.json"
date_state = json.load(open(DATE_PATH))
TRADING_DAYS = date_state["trading_days"]
VISIBLE = date_state["visible_through"]
ROW0 = TRADING_DAYS.index("2020-01-01")
ROW1 = TRADING_DAYS.index(VISIBLE)
GRID = TRADING_DAYS[ROW0:ROW1 + 1]
GIDX = {d: i for i, d in enumerate(GRID)}
N_GRID = len(GRID)
HORIZON = 10
MIN_ASSETS = 8
GATE_IC = 0.0070
GATE_ICIR = 0.0840

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU",
          "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]

DATA_DIR = "../persistent/stock_data"
MACRO_DIR = "../persistent/index_data"


def load_asset(sym):
    p = os.path.join(DATA_DIR, f"{sym}.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df[df["date"] <= VISIBLE]           # hard truncate at visible date
    df = df.set_index("date")
    for c in ["open", "close", "high", "low", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_macro(name):
    p = os.path.join(MACRO_DIR, f"{name}.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df[df["date"] <= VISIBLE]
    df = df.set_index("date")
    return df["close"].astype(float)


def safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(b) < 1e-12, np.nan, a / b)


def roll_beta_cond(asset_ret, ref_ret, w, minp, cond=None):
    out = pd.Series(np.nan, index=asset_ret.index)
    a = asset_ret.values.astype(float)
    b = ref_ret.reindex(asset_ret.index).values.astype(float)
    c = None if cond is None else cond.reindex(asset_ret.index).values.astype(bool)
    for i in range(w - 1, len(a)):
        seg = slice(i - w + 1, i + 1)
        x = b[seg]
        y = a[seg]
        if c is not None:
            m = c[seg]
            x = x[m]
            y = y[m]
        if len(x) < minp or np.std(x) < 1e-12:
            continue
        beta = np.cov(x, y)[0, 1] / np.var(x)
        if np.isfinite(beta):
            out.iloc[i] = beta
    return out


def to_grid(series_dict):
    mat = np.full((N_GRID, len(ASSETS)), np.nan)
    for j, s in enumerate(ASSETS):
        if s not in series_dict:
            continue
        vals = series_dict[s].reindex(GRID)
        mat[:, j] = vals.values
    return mat


def cross_sectional_rank(mat):
    T, n = mat.shape
    out = np.full_like(mat, np.nan, dtype=float)
    for t in range(T):
        row = mat[t]
        valid = ~np.isnan(row)
        if valid.sum() < MIN_ASSETS:
            continue
        ranks = pd.Series(row[valid]).rank(pct=True).values
        out[t, valid] = ranks
    return out


def spearman_ic_matrix(factor_mat, fwd_mat):
    T = factor_mat.shape[0]
    ics = []
    for t in range(T):
        f = factor_mat[t]
        r = fwd_mat[t]
        ok = ~(np.isnan(f) | np.isnan(r))
        if ok.sum() < MIN_ASSETS:
            continue
        fr = pd.Series(f[ok]).rank().corr(pd.Series(r[ok]).rank())
        if np.isfinite(fr):
            ics.append((t, fr))
    return ics


def summarize(ics, dates, label):
    if len(ics) == 0:
        return None
    idx = np.array([t for t, _ in ics])
    icv = np.array([v for _, v in ics])
    ic = float(np.nanmean(icv))
    sd = float(np.nanstd(icv))
    icir = float(ic / sd) if sd > 0 else 0.0
    hit = float(np.mean(icv > 0))
    n = len(icv)
    reg = {}
    segs = [("2020-2021", "2020-01-01", "2021-12-31"), ("2022", "2022-01-01", "2022-12-31"),
            ("2023-2024", "2023-01-01", "2024-12-31"), ("2025-2026", "2025-01-01", "2026-12-31"),
            ("2027-2028", "2027-01-01", "2028-12-31"), ("2029-2031", "2029-01-01", "2099-12-31")]
    for name, a, b in segs:
        m = (dates[idx] >= a) & (dates[idx] <= b)
        if m.sum() > 20:
            sd_m = float(np.std(icv[m]))
            reg[name] = {"ic": round(float(np.mean(icv[m])), 4),
                         "icir": round(float(np.mean(icv[m]) / sd_m), 3) if sd_m > 0 else 0.0,
                         "n": int(m.sum())}
    if n >= 250:
        m = idx >= len(dates) - 250
        sd_m = float(np.std(icv[m]))
        reg["last250"] = {"ic": round(float(np.mean(icv[m])), 4),
                          "icir": round(float(np.mean(icv[m]) / sd_m), 3) if sd_m > 0 else 0.0,
                          "n": int(m.sum())}
    if n >= 120:
        m = idx >= len(dates) - 120
        sd_m = float(np.std(icv[m]))
        reg["last120"] = {"ic": round(float(np.mean(icv[m])), 4),
                          "icir": round(float(np.mean(icv[m]) / sd_m), 3) if sd_m > 0 else 0.0,
                          "n": int(m.sum())}
    return {"label": label, "horizon": HORIZON, "n_ic_dates": n, "ic": ic, "icir": icir,
            "hit": hit, "regime": reg}


def turnover_10d_rank(rank_mat):
    d = np.abs(np.diff(rank_mat, axis=0))
    d = np.nanmean(d, axis=1)
    return float(np.nanmean(d[::10]))


def coverage_stats(mat):
    cov = float(np.nanmean(~np.isnan(mat)))
    dates_ge8 = float(np.mean(np.sum(~np.isnan(mat), axis=1) >= 8))
    return cov, dates_ge8


def library_pairwise_corr(mat):
    T, n = mat.shape
    valid = ~np.isnan(mat)
    keep = [j for j in range(n) if valid[:, j].sum() > 100]
    if len(keep) < 2:
        return None, None, 0.0
    sub = mat[:, keep]
    c = np.corrcoef(sub.T)
    c = np.nan_to_num(c, nan=0.0)
    iu = np.triu_indices(len(keep), 1)
    max_rho = float(np.max(np.abs(c[iu]))) if len(iu[0]) else 0.0
    return c, keep, max_rho


series = {}
for s in ASSETS:
    df = load_asset(s)
    if df is None or len(df) < 100:
        print("skip", s)
        continue
    close = df["close"].astype(float)
    ret = close.pct_change()
    fwd = close.shift(-HORIZON) / close - 1.0
    series[s] = pd.DataFrame({"close": close, "ret": ret, "fwd10": fwd,
                              "open": df["open"].astype(float), "high": df["high"].astype(float),
                              "low": df["low"].astype(float), "volume": df["volume"].astype(float)})

print("assets with data:", sorted(series.keys()))
fwd10 = to_grid({s: df["fwd10"] for s, df in series.items()})
dates = np.array(GRID)

spx = series["SPX"]["close"]
dxy = load_macro("DXY")
usdjpy = load_macro("USDJPY")
vix = load_macro("VIX")

factors = {}

# 1 calmness_20
for s, df in series.items():
    r = df["ret"]
    th = 0.5 * r.rolling(20, min_periods=10).std()
    factors.setdefault("calmness_20", {})[s] = (r.abs() < th).rolling(20, min_periods=10).mean()

# 2 close_pos_20
for s, df in series.items():
    rng = (df["high"] - df["low"])
    cp = (df["close"] - df["low"]) / rng.replace(0, np.nan)
    factors.setdefault("close_pos_20", {})[s] = cp.rolling(20, min_periods=10).mean()

# 3 days_since_high_60  (miner convention: -days since 60d high; high days => more negative)
for s, df in series.items():
    roll_max = df["close"].rolling(60, min_periods=20).max()
    dd = df["close"] / roll_max - 1.0
    days = (dd < -0.01).astype(float).rolling(60, min_periods=20).sum()
    factors.setdefault("days_since_high_60", {})[s] = -days

# 4 downbeta_spx_60  (negated: high = defensive)
for s, df in series.items():
    if s == "SPX":
        factors.setdefault("downbeta_spx_60", {})[s] = pd.Series(1.0, index=df.index)
        continue
    b = roll_beta_cond(df["ret"], spx.reindex(df.index).pct_change(), 60, 30,
                       cond=spx.reindex(df.index).pct_change() < 0)
    factors.setdefault("downbeta_spx_60", {})[s] = -b

# 5 dxy_beta_cond_60x20
for s, df in series.items():
    if dxy is None:
        continue
    beta = roll_beta_cond(df["ret"], dxy.reindex(df.index).pct_change(), 60, 30)
    m = dxy.reindex(df.index) / dxy.reindex(df.index).shift(20) - 1.0
    factors.setdefault("dxy_beta_cond_60x20", {})[s] = beta * m

# 6 gain_loss_20  (ratio per library JSON)
for s, df in series.items():
    r = df["ret"]
    g = r.clip(lower=0).rolling(20, min_periods=10).mean()
    l = (-r.clip(upper=0)).rolling(20, min_periods=10).mean()
    factors.setdefault("gain_loss_20", {})[s] = g / (l + 1e-9)

# 7 intraday_drift_20
for s, df in series.items():
    dr = df["close"] / df["open"] - 1.0
    factors.setdefault("intraday_drift_20", {})[s] = dr.rolling(20, min_periods=10).mean()

# 8 lagbeta_spx_60 (negated)
for s, df in series.items():
    if s == "SPX":
        factors.setdefault("lagbeta_spx_60", {})[s] = pd.Series(1.0, index=df.index)
        continue
    b = roll_beta_cond(df["ret"], spx.reindex(df.index).pct_change().shift(1), 60, 30)
    factors.setdefault("lagbeta_spx_60", {})[s] = -b

# 9 max_consec_gain_20
for s, df in series.items():
    r = df["ret"]
    pg = (r > 0).astype(int)
    grp = (pg.diff() != 0).cumsum()
    streak = pg.groupby(grp).cumsum()
    factors.setdefault("max_consec_gain_20", {})[s] = streak.rolling(20, min_periods=10).max()

# 10 max_consec_loss_20 (negated)
for s, df in series.items():
    r = df["ret"]
    pl = (r < 0).astype(int)
    grp = (pl.diff() != 0).cumsum()
    streak = pl.groupby(grp).cumsum()
    factors.setdefault("max_consec_loss_20", {})[s] = -streak.rolling(20, min_periods=10).max()

# 11-14 mom_*_skip5
for lb, fid in [(10, "mom_10d_skip5"), (20, "mom_20d_skip5"), (120, "mom_120d_skip5"), (180, "mom_180d_skip5")]:
    for s, df in series.items():
        factors.setdefault(fid, {})[s] = df["close"] / df["close"].shift(lb + 5) - 1.0

# 15 mom20_volproxy60  (20d mom ending 5d ago / 60d vol proxy per JSON)
for s, df in series.items():
    m = df["close"] / df["close"].shift(25) - 1.0
    v = df["close"].pct_change().rolling(60, min_periods=30).std()
    factors.setdefault("mom20_volproxy60", {})[s] = pd.Series(safe_div(m, v), index=df.index)

# 16 mom30_vol60
for s, df in series.items():
    m = df["close"] / df["close"].shift(35) - 1.0
    v = df["close"].pct_change().rolling(60, min_periods=30).std()
    factors.setdefault("mom30_vol60", {})[s] = pd.Series(safe_div(m, v), index=df.index)

# 17 range_pos_252
for s, df in series.items():
    c = df["close"]
    ll = c.rolling(252, min_periods=120).min()
    hh = c.rolling(252, min_periods=120).max()
    factors.setdefault("range_pos_252", {})[s] = pd.Series(safe_div(c - ll, hh - ll), index=df.index)

# 18 spx_corr60
for s, df in series.items():
    if s == "SPX":
        factors.setdefault("spx_corr60", {})[s] = pd.Series(1.0, index=df.index)
        continue
    r = df["ret"]
    sret = spx.reindex(df.index).pct_change()
    factors.setdefault("spx_corr60", {})[s] = r.rolling(60, min_periods=30).corr(sret)

# 19 usdjpy_beta_cond_120x60
for s, df in series.items():
    if usdjpy is None:
        continue
    beta = roll_beta_cond(df["ret"], usdjpy.reindex(df.index).pct_change(), 120, 60)
    m = usdjpy.reindex(df.index) / usdjpy.reindex(df.index).shift(60) - 1.0
    factors.setdefault("usdjpy_beta_cond_120x60", {})[s] = beta * m

# 20 vix_beta_cond_60x20
for s, df in series.items():
    if vix is None:
        continue
    beta = roll_beta_cond(df["ret"], vix.reindex(df.index).pct_change(), 60, 30)
    m = vix.reindex(df.index) / vix.reindex(df.index).shift(20) - 1.0
    factors.setdefault("vix_beta_cond_60x20", {})[s] = -beta * m

# 21 vol_of_vol20x60
for s, df in series.items():
    rv = df["ret"].rolling(20, min_periods=5).std()
    factors.setdefault("vol_of_vol20x60", {})[s] = rv.rolling(60, min_periods=15).std()

# 22 volcluster_60
for s, df in series.items():
    ar = df["ret"].abs()
    factors.setdefault("volcluster_60", {})[s] = ar.rolling(60, min_periods=40).corr(ar.shift(1))

results = {}
for fid, panel in factors.items():
    mat = to_grid(panel)
    rank_mat = cross_sectional_rank(mat)
    ics = spearman_ic_matrix(rank_mat, fwd10)
    if len(ics) == 0:
        continue
    s = summarize(ics, dates, fid)
    if s is None:
        continue
    c, keep, max_rho = library_pairwise_corr(mat)
    s["max_abs_library_correlation"] = round(max_rho, 4)
    s["turnover_10d_rank"] = round(turnover_10d_rank(rank_mat), 4)
    cov, dates_ge8 = coverage_stats(mat)
    s["coverage"] = round(cov, 4)
    s["dates_ge8_frac"] = round(dates_ge8, 4)
    s["ok"] = bool((abs(s["ic"]) >= GATE_IC) and (abs(s["icir"]) >= GATE_ICIR))
    results[fid] = s
    l250 = s["regime"].get("last250", {})
    l120 = s["regime"].get("last120", {})
    print(f"{fid:26s} ic={s['ic']:+.4f} icir={s['icir']:+.4f} hit={s['hit']:.3f} "
          f"turn={s['turnover_10d_rank']:.3f} cov={s['coverage']:.3f} maxrho={max_rho:.3f} "
          f"l250_ic={l250.get('ic','NA')} l120_ic={l120.get('ic','NA')} ok={s['ok']}", flush=True)

json.dump(results, open("scripts/screener_20310123_revalidate_results.json", "w"), indent=1, default=str)

# ---------------- regime statistics ----------------
print("\n=== REGIME STATS (visible 2031-01-22) ===")
ret_df = pd.DataFrame({s: df["ret"] for s, df in series.items()}).reindex(GRID)
close_df = pd.DataFrame({s: df["close"] for s, df in series.items()}).reindex(GRID)

def rets(days):
    return close_df.iloc[-1] / close_df.iloc[-1 - days] - 1.0

for days in [5, 10, 20, 60, 120]:
    r = rets(days)
    print(f"--- {days}d returns ---")
    for s in ASSETS:
        if s in r.index:
            print(f"   {s:10s} {r[s]*100:8.2f}%")

r20 = rets(20)
print("\navg |20d ret|: %.2f%%  (dispersion proxy)" % (r20.abs().mean() * 100))

# realized vol (annualized 20d)
rv = (ret_df.rolling(20).std() * np.sqrt(252)).iloc[-1]
print("\n20d realized vol (ann):")
for s in ASSETS:
    if s in rv.index:
        print(f"   {s:10s} {rv[s]*100:6.1f}%")

# mean pairwise correlation of daily returns (60d)
c60 = ret_df.iloc[-60:].corr()
vals = c60.values[np.triu_indices_from(c60.values, 1)]
vals = vals[np.isfinite(vals)]
print("\nmean pairwise corr 60d: %.3f  median %.3f" % (np.mean(vals), np.median(vals)))

if vix is not None:
    v = vix.reindex(GRID).dropna()
    print("\nVIX last: %.2f | 5d ago: %.2f | 20d ago: %.2f | 60d ago: %.2f" %
          (v.iloc[-1], v.iloc[-6], v.iloc[-21], v.iloc[-61] if len(v) > 61 else np.nan))
if dxy is not None:
    d = dxy.reindex(GRID).dropna()
    print("DXY last: %.2f | 20d ret: %.2f%% | 60d ret: %.2f%%" %
          (d.iloc[-1], (d.iloc[-1]/d.iloc[-21]-1)*100, (d.iloc[-1]/d.iloc[-61]-1)*100 if len(d) > 61 else np.nan))
if usdjpy is not None:
    u = usdjpy.reindex(GRID).dropna()
    print("USDJPY last: %.2f | 20d ret: %.2f%% | 60d ret: %.2f%%" %
          (u.iloc[-1], (u.iloc[-1]/u.iloc[-21]-1)*100, (u.iloc[-1]/u.iloc[-61]-1)*100 if len(u) > 61 else np.nan))

print("\nDONE")
