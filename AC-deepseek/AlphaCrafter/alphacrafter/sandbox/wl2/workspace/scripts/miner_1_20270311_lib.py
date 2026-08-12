"""miner_1 shared lib 2027-03-11 cycle.

Conventions:
- Master grid from persistent/date.json (2020-01-01 .. visible_through=2027-03-10).
- Per-asset own-calendar daily data via get_stock_daily_data (>=2500 days back).
- IC = daily cross-sectional Spearman(factor, fwd 10d own-calendar return), min 8 valid assets.
- Admission gate: |IC| >= 0.0070 and |ICIR| >= 0.0840 (15-instrument universe).
- FLAT-masking: SX5E/BTC/US10Y/CN10Y are flat since 2026-07-17 (data artifact).
  Factor rows and fwd returns for flat assets are set to NaN so they drop out of
  cross-sectional ranking on affected dates (per prior convention).
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
GATE_IC = 0.0070
GATE_ICIR = 0.0840
FLAT_EPS = 1e-12


def load_asset(sym, days=2600):
    df = get_stock_daily_data(sym, days=days)
    if df is None:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.set_index("date")
    for c in ["open", "close", "high", "low", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    close = df["close"].astype(float)
    ret = close.pct_change()
    flat = (close.diff().abs() < FLAT_EPS).astype(float)
    d = pd.DataFrame({"close": close, "ret": ret, "flat": flat})
    for c in ["open", "high", "low", "volume"]:
        d[c] = df[c]
    return d


def asset_series():
    out = {}
    for s in ASSETS:
        df = load_asset(s)
        if df is None or len(df) < 200:
            continue
        out[s] = df
    return out


def to_grid(series_dict, mask_flat=True):
    """Map asset-level pd.Series (indexed by date string) to the master grid."""
    mat = np.full((N_GRID, len(ASSETS)), np.nan)
    for j, s in enumerate(ASSETS):
        if s not in series_dict:
            continue
        ser = series_dict[s]
        for dte, v in ser.items():
            i = GIDX.get(dte)
            if i is not None and np.isfinite(v):
                mat[i, j] = v
    if mask_flat:
        # zero out rows where asset is flat on that grid date
        for j, s in enumerate(ASSETS):
            if s not in series_dict:
                continue
            flat = series_dict[s]["flat"]
            for dte, fv in flat.items():
                i = GIDX.get(dte)
                if i is not None and fv > 0.5:
                    mat[i, j] = np.nan
    return mat


def fwd_by_horizon_dict(series_dict, horizons=(1, 2, 3, 5, 10, 20)):
    out = {}
    for h in horizons:
        d = {}
        for s, df in series_dict.items():
            close = df["close"]
            f = close.shift(-h) / close - 1.0
            # NaN forward returns where the asset is flat at t or any flat in (t, t+h]
            flat = df["flat"]
            fwd_flat = flat.rolling(h + 1, min_periods=1).max().shift(-h)
            f = f.where(fwd_flat < 0.5)
            d[s] = f
        out[h] = to_grid(d, mask_flat=False)
    return out


def cross_sectional_rank(factor_mat):
    T = factor_mat.shape[0]
    out = np.full_like(factor_mat, np.nan)
    for t in range(T):
        row = factor_mat[t]
        valid = ~np.isnan(row)
        if valid.sum() < MIN_ASSETS:
            continue
        out[t, valid] = pd.Series(row[valid]).rank(pct=True).values
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
        fs = pd.Series(f[ok])
        rs = pd.Series(r[ok])
        fr = fs.rank().corr(rs.rank())
        if np.isfinite(fr):
            ics.append((t, fr))
    return ics


def summarize(ics, label, horizon=HORIZON):
    if len(ics) == 0:
        return None
    idx = np.array([t for t, _ in ics])
    icv = np.array([v for _, v in ics])
    ic = float(np.nanmean(icv))
    sd = float(np.nanstd(icv))
    icir = float(ic / sd) if sd > 0 else 0.0
    hit = float(np.mean(icv > 0))
    n = len(icv)
    dates = np.array(GRID)
    reg = {}
    segs = [("2020-2021", "2020-01-01", "2021-12-31"), ("2022", "2022-01-01", "2022-12-31"),
            ("2023-2024", "2023-01-01", "2024-12-31"), ("2025-2026", "2025-01-01", "2026-12-31"),
            ("2027", "2027-01-01", "2099-12-31")]
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
    if n >= 60:
        m = idx >= len(dates) - 60
        sd_m = float(np.std(icv[m]))
        reg["last60"] = {"ic": round(float(np.mean(icv[m])), 4),
                         "icir": round(float(np.mean(icv[m]) / sd_m), 3) if sd_m > 0 else 0.0,
                         "n": int(m.sum())}
    return {"label": label, "horizon": horizon, "n_ic_dates": n, "ic": ic, "icir": icir,
            "hit": hit, "regime": reg, "idx": idx, "icv": icv}


def decay_curve(factor_mat, fwd_by_horizon):
    out = {}
    for h, fwd in fwd_by_horizon.items():
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


def coverage_stats(factor_mat):
    valid = ~np.isnan(factor_mat)
    cov_asset_days = float(valid.mean())
    dates_ge8 = float((valid.sum(axis=1) >= MIN_ASSETS).mean())
    return cov_asset_days, dates_ge8


def library_pairwise_corr(factor_mat):
    """Spearman rho vs every factors/*.signal.npy artifact (rank-aligned, first overlapping row)."""
    out = {}
    ours = cross_sectional_rank(factor_mat)
    for f in sorted(glob.glob("factors/*.signal.npy")):
        arr = np.load(f, allow_pickle=True)
        rows = min(arr.shape[0], ours.shape[0])
        a = ours[:rows]
        b = arr[:rows]
        rho = None
        for t in range(rows):
            x = a[t]
            y = b[t]
            ok = ~(np.isnan(x) | np.isnan(y))
            if ok.sum() >= MIN_ASSETS:
                xs = pd.Series(x[ok]).rank()
                ys = pd.Series(y[ok]).rank()
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


def full_report(name, cand, save_artifact=True, artifact_name=None):
    """cand: dict asset -> pd.Series (indexed by own dates) of factor values."""
    mat = to_grid(cand)
    rank_mat = cross_sectional_rank(mat)
    fwd10 = fwd_by_horizon_dict(series, horizons=(10,))[10]
    ics = spearman_ic_matrix(mat, fwd10)
    summ = summarize(ics, name)
    if summ is None:
        print(name, "NO VALID IC DATES")
        return None
    cov_ad, cov_d8 = coverage_stats(mat)
    to = turnover_10d_rank(rank_mat)
    dec = decay_curve(mat, fwd_by_horizon)
    lpc, lpc_name, lpc_max = library_pairwise_corr(mat)
    passed = abs(summ["ic"]) >= GATE_IC and abs(summ["icir"]) >= GATE_ICIR
    print("=" * 70)
    print("FACTOR:", name)
    print("  IC=%.4f ICIR=%.4f hit=%.3f n_ic_dates=%d" % (summ["ic"], summ["icir"], summ["hit"], summ["n_ic_dates"]))
    print("  coverage_asset_days=%.3f dates_ge8=%.3f turnover_10d=%.4f" % (cov_ad, cov_d8, to))
    print("  decay:", dec)
    print("  regime:", json.dumps(summ["regime"]))
    print("  max_lib_corr=%.4f (%s)" % (lpc_max, lpc_name))
    print("  GATE PASS:", passed, "(|IC|>=%.4f & |ICIR|>=%.4f)" % (GATE_IC, GATE_ICIR))
    if save_artifact and artifact_name and passed:
        fn = "factors/%s.signal.npy" % artifact_name
        np.save(fn, rank_mat)
        print("  artifact saved:", fn)
    return {"name": name, "passed": passed, "ic": summ["ic"], "icir": summ["icir"],
            "hit": summ["hit"], "n": summ["n_ic_dates"], "cov_ad": cov_ad, "cov_d8": cov_d8,
            "turnover": to, "decay": dec, "regime": summ["regime"],
            "max_lib_corr": lpc_max, "lib_corr_name": lpc_name, "lib_corr": lpc}


series = asset_series()
fwd_by_horizon = fwd_by_horizon_dict(series)
print("assets loaded: %d/%d" % (len(series), len(ASSETS)), sorted(series.keys()))
print("grid: %s .. %s (%d rows)" % (GRID[0], GRID[-1], N_GRID))
