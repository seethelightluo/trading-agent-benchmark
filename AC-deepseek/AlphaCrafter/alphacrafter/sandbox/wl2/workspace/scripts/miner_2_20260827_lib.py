"""miner_2 shared lib (2026-08-27): per-asset own-calendar factor computation,
reindexed onto master grid. IC = daily cross-sectional Spearman vs fwd 10d (own-calendar) return.
Gates: |IC|>=0.0070, |ICIR|>=0.0840. MIN_ASSETS=8.
"""
import json, glob, os
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict

DATE_PATH = "../persistent/date.json"
date_state = json.load(open(DATE_PATH))
TRADING_DAYS = date_state["trading_days"]
VISIBLE = date_state["visible_through"]
ROW0 = TRADING_DAYS.index("2020-01-01")
ROW1 = TRADING_DAYS.index(VISIBLE)
GRID = TRADING_DAYS[ROW0:ROW1 + 1]
GIDX = {d: i for i, d in enumerate(GRID)}
N_GRID = len(GRID)

acct = get_account_dict()
ASSETS = list(acct.get("watch_list", []))
HORIZON = 10
MIN_ASSETS = 8


def load_asset(sym, days=2600):
    df = get_stock_daily_data(sym, days=days)
    if df is None:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.set_index("date")
    for c in ["open", "close", "high", "low", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def asset_series():
    """sym -> DataFrame indexed by own dates with close, ret, fwd10 (own calendar)."""
    out = {}
    for s in ASSETS:
        df = load_asset(s)
        if df is None or len(df) < 100:
            continue
        close = df["close"].astype(float)
        ret = close.pct_change()
        fwd = close.shift(-HORIZON) / close - 1.0
        d = pd.DataFrame({"close": close, "ret": ret, "fwd10": fwd})
        out[s] = d
    return out


def to_grid(series_dict, fillna=True):
    mat = np.full((N_GRID, len(ASSETS)), np.nan)
    for j, s in enumerate(ASSETS):
        if s not in series_dict:
            continue
        ser = series_dict[s]
        vals = ser.reindex(GRID)
        mat[:, j] = vals.values
    return mat


def load_macro(name):
    p = f"../persistent/index_data/{name}.csv"
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.set_index("date")
    return df["close"].astype(float)


def safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(b) < 1e-12, np.nan, a / b)


def roll_mean(x, w):
    c = np.cumsum(np.where(np.isnan(x), 0.0, x))
    n = np.cumsum(~np.isnan(x))
    out = np.full(len(x), np.nan)
    out[w:] = safe_div(c[w:] - c[:-w], n[w:] - n[:-w])
    return out


def roll_std(x, w):
    mu = roll_mean(x, w)
    sq = roll_mean(x * x, w)
    return np.sqrt(np.maximum(sq - mu * mu, 0.0))


def rolling_corr(x, y, w, minp):
    """pairwise rolling Pearson corr of aligned arrays x,y (same length)."""
    n = len(x)
    out = np.full(n, np.nan)
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    for i in range(w - 1, n):
        a = xa[i - w + 1:i + 1]
        b = ya[i - w + 1:i + 1]
        ok = ~(np.isnan(a) | np.isnan(b))
        if ok.sum() < minp:
            continue
        aa, bb = a[ok], b[ok]
        if aa.std() < 1e-12 or bb.std() < 1e-12:
            continue
        out[i] = np.corrcoef(aa, bb)[0, 1]
    return out


def rolling_beta(x, y, w, minp):
    """beta of x on y over trailing w obs (own calendar), min minp valid pairs."""
    n = len(x)
    out = np.full(n, np.nan)
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    for i in range(w - 1, n):
        a = xa[i - w + 1:i + 1]
        b = ya[i - w + 1:i + 1]
        ok = ~(np.isnan(a) | np.isnan(b))
        if ok.sum() < minp:
            continue
        aa, bb = a[ok], b[ok]
        if bb.std() < 1e-12:
            continue
        out[i] = np.cov(aa, bb)[0, 1] / np.var(bb)
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
        f = factor_mat[t]
        r = fwd_mat[t]
        ok = ~(np.isnan(f) | np.isnan(r))
        if ok.sum() < MIN_ASSETS:
            continue
        fs = pd.Series(f[ok]); rs = pd.Series(r[ok])
        fr = fs.rank().corr(rs.rank())
        if np.isfinite(fr):
            ics.append((t, fr))
    return ics


def summarize(ics, dates, label, horizon):
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
            ("2023-2024", "2023-01-01", "2024-12-31"), ("2025-2026", "2025-01-01", "2099-12-31")]
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
    return {"label": label, "horizon": horizon, "n_ic_dates": n, "ic": ic, "icir": icir,
            "hit": hit, "regime": reg, "idx": idx, "icv": icv}


def decay_curve(factor_mat, series_dict, horizons=(1, 2, 3, 5, 10, 20)):
    out = {}
    for h in horizons:
        d = {}
        for s, df in series_dict.items():
            close = df["close"]
            d[s] = close.shift(-h) / close - 1.0
        fwd = to_grid(d)
        ics = spearman_ic_matrix(factor_mat, fwd)
        if ics:
            out[str(h)] = round(float(np.mean([v for _, v in ics])), 4)
    return out


def turnover_10d_rank(rank_mat, step=10):
    T = rank_mat.shape[0]
    d = []
    for t in range(0, T - step):
        a, b = rank_mat[t], rank_mat[t + step]
        ok = ~(np.isnan(a) | np.isnan(b))
        if ok.sum() < MIN_ASSETS:
            continue
        d.append(np.nanmean(np.abs(a[ok] - b[ok])))
    return float(np.mean(d)) if d else float("nan")


def library_pairwise_corr(factor_mat):
    out = {}
    ours = cross_sectional_rank(factor_mat)
    for f in sorted(glob.glob("factors/*.signal.npy")):
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
            out[os.path.basename(f).replace(".signal.npy", "")] = round(float(rho), 4)
    if out:
        mx = max(out.items(), key=lambda kv: abs(kv[1]))
        return out, mx[0], abs(mx[1])
    return out, None, 0.0


def coverage_stats(factor_mat):
    valid = ~np.isnan(factor_mat)
    cov_asset_days = float(valid.mean())
    dates_ge8 = float((valid.sum(axis=1) >= MIN_ASSETS).mean())
    return cov_asset_days, dates_ge8
