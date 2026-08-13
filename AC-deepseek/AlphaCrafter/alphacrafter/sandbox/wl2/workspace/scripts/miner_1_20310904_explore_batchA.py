"""miner_1 2031-09-04: explore batch A of novel factor candidates.

Data through 2031-09-03 (visible). Validation convention identical to library:
daily cross-sectional Spearman IC vs fwd-10d own-calendar return.
Gates: |IC|>=0.0070, |ICIR|>=0.0840. MIN_ASSETS=8.
Candidates (all interpretable, no existing library duplicate):
  A1 autocorr_10   : lag-1 autocorrelation of daily returns (trendiness vs MR)
  A2 pullback_5_60 : 60d trend minus 5d return (buy-the-dip in leaders)
  A3 days_low_60   : -days since 60d low (fresh recovery recency)
  A4 us10y_beta_60 : 60d beta of asset returns to US10Y daily change
  A5 gap_20        : 20d mean open/prev_close - 1 (overnight gap persistence)
  A6 vol_squeeze   : 20d mean (high-low)/close vs 60d baseline (range squeeze)
  A7 eth_beta_20   : 20d beta of asset returns to ETH daily returns
  A8 vol_slope     : vol20/vol60 - 1 (vol regime slope, compression vs expansion)
"""
import sys, json, os
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict

MIN_ASSETS = 8
HORIZON = 10
DAYS = 4500

acct = get_account_dict()
ASSETS = list(acct.get("watch_list", []))
print("ASSETS:", ASSETS)

def load_asset(sym, days=DAYS):
    df = get_stock_daily_data(sym, days=days)
    if df is None:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.set_index("date")
    for c in ["open", "close", "high", "low", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

series = {}
for s in ASSETS:
    df = load_asset(s)
    if df is None or len(df) < 200:
        print("skip (no data):", s)
        continue
    close = df["close"].astype(float)
    ret = close.pct_change()
    fwd = close.shift(-HORIZON) / close - 1.0
    series[s] = pd.DataFrame({
        "close": close, "ret": ret, "fwd10": fwd,
        "open": df["open"].astype(float), "high": df["high"].astype(float),
        "low": df["low"].astype(float), "volume": df["volume"].astype(float),
    })
print("assets with data:", sorted(series.keys()))

# master grid = union of dates (use first asset's index; they share the master grid)
GRID = series[ASSETS[0]].index
print("grid size:", len(GRID), GRID[0], "->", GRID[-1])

def to_grid(series_dict):
    mat = np.full((len(GRID), len(ASSETS)), np.nan)
    for j, s in enumerate(ASSETS):
        if s not in series_dict:
            continue
        mat[:, j] = series_dict[s].reindex(GRID).values
    return mat

fwd10 = to_grid({s: df["fwd10"] for s, df in series.items()})

def safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(b) < 1e-12, np.nan, a / b)

def roll_beta(a, b, w, minp):
    """rolling beta of a on b (aligned series, same index)."""
    out = pd.Series(np.nan, index=a.index)
    av = a.values.astype(float); bv = b.values.astype(float)
    for i in range(w - 1, len(a)):
        seg = slice(i - w + 1, i + 1)
        x = bv[seg]; y = av[seg]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < minp or np.std(x[ok]) < 1e-12:
            continue
        beta = np.cov(x[ok], y[ok])[0, 1] / np.var(x[ok])
        if np.isfinite(beta):
            out.iloc[i] = beta
    return out

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
    if not ics:
        return None
    idx = np.array([t for t, _ in ics]); icv = np.array([v for _, v in ics])
    ic = float(np.nanmean(icv)); sd = float(np.nanstd(icv))
    icir = float(ic / sd) if sd > 0 else 0.0
    hit = float(np.mean(icv > 0))
    segs = [("2020-2021", "2020-01-01", "2021-12-31"), ("2022", "2022-01-01", "2022-12-31"),
            ("2023-2024", "2023-01-01", "2024-12-31"), ("2025-2026", "2025-01-01", "2026-12-31"),
            ("2027-2028", "2027-01-01", "2028-12-31"), ("2029-2031", "2029-01-01", "2099-12-31")]
    reg = {}
    for name, a, b in segs:
        m = (dates[idx] >= a) & (dates[idx] <= b)
        if m.sum() > 20:
            sd_m = float(np.std(icv[m]))
            reg[name] = {"ic": round(float(np.mean(icv[m])), 4),
                         "icir": round(float(np.mean(icv[m]) / sd_m), 3) if sd_m > 0 else 0.0,
                         "n": int(m.sum())}
    if len(idx) >= 250:
        m = idx >= len(dates) - 250
        sd_m = float(np.std(icv[m]))
        reg["last250"] = {"ic": round(float(np.mean(icv[m])), 4),
                          "icir": round(float(np.mean(icv[m]) / sd_m), 3) if sd_m > 0 else 0.0,
                          "n": int(m.sum())}
    return {"label": label, "n_ic_dates": n, "ic": ic, "icir": icir, "hit": hit, "regime": reg}

# ---- recent regime snapshot (20d returns as of last date) ----
last = GRID[-1]
print("\n=== 20d / 60d return snapshot at", last, "===")
snap = {}
for s, df in series.items():
    c = df["close"]
    r20 = c.iloc[-1] / c.iloc[-21] - 1 if len(c) > 21 else np.nan
    r60 = c.iloc[-1] / c.iloc[-61] - 1 if len(c) > 61 else np.nan
    snap[s] = (r20, r60)
for s in ASSETS:
    if s in snap:
        print(f"{s:10s} 20d={snap[s][0]*100:7.2f}%  60d={snap[s][1]*100:7.2f}%")

# ---- candidate factors ----
factors = {}

# A1 autocorr_10: lag-1 autocorrelation of daily returns over 10d
for s, df in series.items():
    r = df["ret"]
    a = r; b = r.shift(1)
    num = (a * b).rolling(10, min_periods=6).mean() - a.rolling(10, min_periods=6).mean() * b.rolling(10, min_periods=6).mean()
    den = a.rolling(10, min_periods=6).std() * b.rolling(10, min_periods=6).std()
    factors.setdefault("autocorr_10", {})[s] = num / den.replace(0, np.nan)

# A2 pullback_5_60: 60d trend minus 5d return
for s, df in series.items():
    c = df["close"]
    r5 = c / c.shift(5) - 1.0
    r60 = c / c.shift(60) - 1.0
    factors.setdefault("pullback_5_60", {})[s] = r60 - r5

# A3 days_low_60: -days since 60d low
for s, df in series.items():
    c = df["close"]
    roll_min = c.rolling(60, min_periods=20).min()
    above = (c > roll_min * 1.001).astype(float)
    days = above.rolling(60, min_periods=20).sum()
    factors.setdefault("days_low_60", {})[s] = -days

# A4 us10y_beta_60: beta of asset returns to US10Y daily change
us10y_close = series["US10Y"]["close"]
us10y_ret = us10y_close.pct_change()
for s, df in series.items():
    b = roll_beta(df["ret"], us10y_ret.reindex(df.index), 60, 30)
    factors.setdefault("us10y_beta_60", {})[s] = b

# A5 gap_20: 20d mean open/prev_close - 1 (overnight gap)
for s, df in series.items():
    prev_close = df["close"].shift(1)
    gap = df["open"] / prev_close - 1.0
    factors.setdefault("gap_20", {})[s] = gap.rolling(20, min_periods=10).mean()

# A6 vol_squeeze: 20d mean intraday range / 60d mean intraday range - 1
for s, df in series.items():
    rng = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    f20 = rng.rolling(20, min_periods=10).mean()
    f60 = rng.rolling(60, min_periods=30).mean()
    factors.setdefault("vol_squeeze", {})[s] = f20 / f60 - 1.0

# A7 eth_beta_20: beta of asset returns to ETH daily returns
eth_ret = series["ETH"]["ret"]
for s, df in series.items():
    if s == "ETH":
        factors.setdefault("eth_beta_20", {})[s] = pd.Series(1.0, index=df.index)
        continue
    b = roll_beta(df["ret"], eth_ret.reindex(df.index), 20, 10)
    factors.setdefault("eth_beta_20", {})[s] = b

# A8 vol_slope: vol20/vol60 - 1
for s, df in series.items():
    v20 = df["ret"].rolling(20, min_periods=10).std()
    v60 = df["ret"].rolling(60, min_periods=30).std()
    factors.setdefault("vol_slope", {})[s] = v20 / v60 - 1.0

dates = np.array(GRID)
results = {}
for name, fd in factors.items():
    fmat = to_grid(fd)
    ics = spearman_ic_matrix(fmat, fwd10)
    res = summarize(ics, dates, name)
    if res is None:
        print(name, "NO IC DATES")
        continue
    rk = cross_sectional_rank(fmat)
    valid = ~np.isnan(fmat)
    cov = float(valid.mean())
    d_ge8 = float((valid.sum(axis=1) >= MIN_ASSETS).mean())
    # turnover over 10d
    tos = []
    for t in range(0, len(rk) - HORIZON):
        a, b = rk[t], rk[t + HORIZON]
        ok = ~(np.isnan(a) | np.isnan(b))
        if ok.sum() >= MIN_ASSETS:
            tos.append(np.nanmean(np.abs(a[ok] - b[ok])))
    res["coverage"] = round(cov, 4)
    res["dates_ge8"] = round(d_ge8, 4)
    res["turnover_10d"] = round(float(np.mean(tos)), 4) if tos else float("nan")
    results[name] = res
    print(f"\n=== {name} === ic={res['ic']:.4f} icir={res['icir']:.4f} hit={res['hit']:.3f} "
          f"n={res['n_ic_dates']} cov={res['coverage']:.3f} dates_ge8={res['dates_ge8']:.3f} "
          f"turn={res['turnover_10d']:.3f}")
    for k, v in res["regime"].items():
        print(f"   {k:10s} ic={v['ic']:+.4f} icir={v['icir']:+.3f} n={v['n']}")

print("\n=== GATE CHECK (|IC|>=0.0070, |ICIR|>=0.0840) ===")
for name, res in results.items():
    gate = abs(res["ic"]) >= 0.0070 and abs(res["icir"]) >= 0.0840
    print(f"{name:16s} ic={res['ic']:+.4f} icir={res['icir']:+.4f} -> {'PASS' if gate else 'fail'}")

with open("scripts/miner_1_20310904_batchA_results.json", "w") as f:
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "regime"} for k, v in results.items()}, f, indent=1, default=str)
