"""miner_1 2029-07-26: explore new factor families on 15-instrument cross-asset universe.
Data through 2029-07-25 (visible_through). Gate: |IC|>=0.0070 AND |ICIR|>=0.0840.
Horizon: forward 10d own-calendar return; daily cross-sectional Spearman IC.
"""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

# ---- master grid from date.json (read-only) ----
DATE_PATH = "../persistent/date.json"
date_state = json.load(open(DATE_PATH))
TRADING_DAYS = date_state["trading_days"]
VISIBLE = date_state["visible_through"]
ROW0 = TRADING_DAYS.index("2020-01-01")
ROW1 = TRADING_DAYS.index(VISIBLE)
GRID = TRADING_DAYS[ROW0:ROW1 + 1]
GIDX = {d: i for i, d in enumerate(GRID)}
N_GRID = len(GRID)
print("grid:", N_GRID, "days", GRID[0], "->", GRID[-1])

ASSETS = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
          'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
HORIZON = 10
MIN_ASSETS = 8


def load_asset(sym, days=3400):
    df = get_stock_daily_data(sym, days=days)
    if df is None:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.set_index("date")
    for c in ["open", "close", "high", "low", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_macro(name):
    p = f"../persistent/index_data/{name}.csv"
    import os
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.set_index("date")
    return df["close"].astype(float)


def to_grid(series_dict):
    mat = np.full((N_GRID, len(ASSETS)), np.nan)
    for j, s in enumerate(ASSETS):
        if s not in series_dict:
            continue
        vals = series_dict[s].reindex(GRID)
        mat[:, j] = vals.values
    return mat


def safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(b) < 1e-12, np.nan, a / b)


def roll_beta(asset_ret, ref_ret, w, minp):
    out = pd.Series(np.nan, index=asset_ret.index)
    a = asset_ret.values.astype(float)
    b = ref_ret.reindex(asset_ret.index).values.astype(float)
    for i in range(w - 1, len(a)):
        seg = slice(i - w + 1, i + 1)
        x = b[seg]; y = a[seg]
        ok = ~(np.isnan(x) | np.isnan(y))
        if ok.sum() < minp or np.std(x[ok]) < 1e-12:
            continue
        beta = np.cov(x[ok], y[ok])[0, 1] / np.var(x[ok])
        if np.isfinite(beta):
            out.iloc[i] = beta
    return out


def roll_corr(asset_ret, ref_ret, w, minp):
    out = pd.Series(np.nan, index=asset_ret.index)
    a = asset_ret.values.astype(float)
    b = ref_ret.reindex(asset_ret.index).values.astype(float)
    for i in range(w - 1, len(a)):
        seg = slice(i - w + 1, i + 1)
        x = b[seg]; y = a[seg]
        ok = ~(np.isnan(x) | np.isnan(y))
        if ok.sum() < minp or np.std(x[ok]) < 1e-12 or np.std(y[ok]) < 1e-12:
            continue
        c = np.corrcoef(x[ok], y[ok])[0, 1]
        if np.isfinite(c):
            out.iloc[i] = c
    return out


def spearman_ic_matrix(factor_mat, fwd_mat):
    T = factor_mat.shape[0]
    ics = []
    for t in range(T):
        f = factor_mat[t]; r = fwd_mat[t]
        ok = ~(np.isnan(f) | np.isnan(r))
        if ok.sum() < MIN_ASSETS:
            continue
        fs = pd.Series(f[ok]); rs = pd.Series(r[ok])
        fr = fs.rank().corr(rs.rank())
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
    years = {}
    for yr in ["2020", "2021", "2022", "2023", "2024", "2025", "2026", "2027", "2028", "2029"]:
        m = (dates[idx] >= yr + "-01-01") & (dates[idx] <= yr + "-12-31")
        if m.sum() >= 20:
            sd_m = float(np.std(icv[m]))
            years[yr] = {"ic": round(float(np.mean(icv[m])), 4),
                         "icir": round(float(np.mean(icv[m]) / sd_m), 3) if sd_m > 0 else 0.0,
                         "n": int(m.sum())}
    recent = {}
    if n >= 250:
        m = idx >= len(dates) - 250
        sd_m = float(np.std(icv[m]))
        recent["last250"] = {"ic": round(float(np.mean(icv[m])), 4),
                             "icir": round(float(np.mean(icv[m]) / sd_m), 3) if sd_m > 0 else 0.0,
                             "n": int(m.sum())}
    return {"label": label, "n_ic_dates": n, "ic": ic, "icir": icir, "hit": hit,
            "years": years, "recent": recent, "idx": idx, "icv": icv}


# ---------------- load data ----------------
series = {}
for s in ASSETS:
    df = load_asset(s)
    if df is None or len(df) < 100:
        print("SKIP", s)
        continue
    close = df["close"].astype(float)
    ret = close.pct_change()
    fwd = close.shift(-HORIZON) / close - 1.0
    d = pd.DataFrame({"close": close, "ret": ret, "fwd10": fwd,
                      "open": df["open"].astype(float), "high": df["high"].astype(float),
                      "low": df["low"].astype(float), "volume": df["volume"].astype(float)})
    d["prev_close"] = close.shift(1)
    series[s] = d

print("assets loaded:", len(series))
dates = np.array(GRID)
fwd10 = to_grid({s: df["fwd10"] for s, df in series.items()})

# macro
dxy = load_macro("DXY")

# reference return series
wti_ret = series["WTI"]["ret"]
ndx_ret = series["NDX"]["ret"]
xau_ret = series["XAU"]["ret"]
btc_ret = series["BTC"]["ret"]

# ---------------- candidate factors ----------------
F = {}

# 1 vol_asym_60 : std(neg ret)/std(pos ret) over 60d
for s, df in series.items():
    r = df["ret"]
    pos = r.clip(lower=0); neg = (-r).clip(lower=0)
    F.setdefault("vol_asym_60", {})[s] = safe_div(
        neg.rolling(60, min_periods=30).std(), pos.rolling(60, min_periods=30).std())

# 2 ser_corr_20 : lag-1 autocorrelation of daily returns (rolling corr)
for s, df in series.items():
    r = df["ret"]
    F.setdefault("ser_corr_20", {})[s] = r.rolling(20, min_periods=12).corr(r.shift(1))

# 3 rel_mom_20 : 20d return minus cross-sectional median (relative momentum)
tmp = {}
for s, df in series.items():
    tmp[s] = df["close"] / df["close"].shift(20) - 1.0
rel_mat = to_grid(tmp)
med = np.nanmedian(rel_mat, axis=1, keepdims=True)
rel_adj = rel_mat - med
for j, s in enumerate(ASSETS):
    F.setdefault("rel_mom_20", {})[s] = pd.Series(rel_adj[:, j], index=GRID)

# 4 sharpe_60 : mean/std of daily returns over 60d
for s, df in series.items():
    r = df["ret"]
    F.setdefault("sharpe_60", {})[s] = safe_div(r.rolling(60, min_periods=30).mean(),
                                                r.rolling(60, min_periods=30).std())

# 5 max_dd_60 : rolling max drawdown over 60d
for s, df in series.items():
    roll_max = df["close"].rolling(60, min_periods=30).max()
    dd = df["close"] / roll_max - 1.0
    F.setdefault("max_dd_60", {})[s] = dd.rolling(60, min_periods=30).min()

# 6 drawup_60 : close/rolling_min(close,60) - 1
for s, df in series.items():
    roll_min = df["close"].rolling(60, min_periods=30).min()
    F.setdefault("drawup_60", {})[s] = df["close"] / roll_min - 1.0

# 7 vol_ratio_5_20 : avg volume 5d / 20d
for s, df in series.items():
    v = df["volume"]
    F.setdefault("vol_ratio_5_20", {})[s] = safe_div(v.rolling(5, min_periods=3).mean(),
                                                     v.rolling(20, min_periods=10).mean())

# 8 gap_20 : mean |open/prev_close - 1| over 20d
for s, df in series.items():
    g = (df["open"] / df["prev_close"] - 1.0).abs()
    F.setdefault("gap_20", {})[s] = g.rolling(20, min_periods=10).mean()

# 9 range_skew_20 : skewness of intraday position over 20d
for s, df in series.items():
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    cp = (df["close"] - df["low"]) / rng
    F.setdefault("range_skew_20", {})[s] = cp.rolling(20, min_periods=12).skew()

# 10 wti_beta_60
for s, df in series.items():
    if s == "WTI":
        F.setdefault("wti_beta_60", {})[s] = pd.Series(1.0, index=df.index)
        continue
    F.setdefault("wti_beta_60", {})[s] = roll_beta(df["ret"], wti_ret, 60, 30)

# 11 ndx_beta_60
for s, df in series.items():
    if s == "NDX":
        F.setdefault("ndx_beta_60", {})[s] = pd.Series(1.0, index=df.index)
        continue
    F.setdefault("ndx_beta_60", {})[s] = roll_beta(df["ret"], ndx_ret, 60, 30)

# 12 xau_corr_60
for s, df in series.items():
    if s == "XAU":
        F.setdefault("xau_corr_60", {})[s] = pd.Series(1.0, index=df.index)
        continue
    F.setdefault("xau_corr_60", {})[s] = roll_corr(df["ret"], xau_ret, 60, 30)

# 13 btc_beta_60
for s, df in series.items():
    if s == "BTC":
        F.setdefault("btc_beta_60", {})[s] = pd.Series(1.0, index=df.index)
        continue
    F.setdefault("btc_beta_60", {})[s] = roll_beta(df["ret"], btc_ret, 60, 30)

# 14 rev_5_skip1 : negative of 5d return skipping 1 day (short-term reversal)
for s, df in series.items():
    r5 = df["close"] / df["close"].shift(5) - 1.0
    F.setdefault("rev_5_skip1", {})[s] = -r5

# 15 hi_lo_20 : mean (high-low)/close over 20d
for s, df in series.items():
    hl = (df["high"] - df["low"]) / df["close"]
    F.setdefault("hi_lo_20", {})[s] = hl.rolling(20, min_periods=10).mean()

# 16 price_vs_sma20 : close/SMA20 - 1
for s, df in series.items():
    sma = df["close"].rolling(20, min_periods=10).mean()
    F.setdefault("price_vs_sma20", {})[s] = df["close"] / sma - 1.0

# 17 days_since_low_20 : days since 20d rolling low (contrarian persistence)
for s, df in series.items():
    roll_min = df["close"].rolling(20, min_periods=10).min()
    above = df["close"] > roll_min
    count = (~above).astype(float).rolling(20, min_periods=10).sum()
    F.setdefault("days_since_low_20", {})[s] = count

# ---------------- evaluate ----------------
results = {}
for name, sdict in F.items():
    mat = to_grid(sdict)
    cov = float(np.mean(~np.isnan(mat)))
    ics = spearman_ic_matrix(mat, fwd10)
    s = summarize(ics, dates, name)
    if s is None:
        print(f"{name:18s} NO IC")
        continue
    results[name] = {"mat": mat, "sum": s, "coverage": cov}
    passed = abs(s["ic"]) >= 0.0070 and abs(s["icir"]) >= 0.0840
    flag = "*** PASS ***" if passed else ""
    print(f"\n{name:18s} ic={s['ic']:+.4f} icir={s['icir']:+.3f} hit={s['hit']:.3f} n={s['n_ic_dates']} cov={cov:.3f} {flag}")
    yrs = {k: f"{v['ic']:+.3f}/{v['icir']:+.2f}" for k, v in s["years"].items()}
    print("   years:", yrs)
    print("   recent:", s["recent"])

# save for follow-up
out = {k: {"ic": v["sum"]["ic"], "icir": v["sum"]["icir"], "hit": v["sum"]["hit"],
           "n": v["sum"]["n_ic_dates"], "coverage": v["coverage"],
           "years": v["sum"]["years"], "recent": v["sum"]["recent"]} for k, v in results.items()}
with open("scripts/miner_1_20290726_batch_explore_results.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nsaved results json")
