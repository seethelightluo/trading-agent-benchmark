"""miner_2 shared lib (2027-05-06): framework + candidate factor calculators.

Validation framework mirrors library convention:
- per-asset own-calendar factor computation, reindexed onto master calendar grid
- IC = daily cross-sectional Spearman vs fwd 10d (own-calendar) return
- Gates: |IC| >= 0.0070, |ICIR| >= 0.0840 (15-instrument cross-asset universe)
- Data visible through 2027-05-05 (date.json visible_through).
"""
import json, glob, os, sys
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

DATE_PATH = "../persistent/date.json"
date_state = json.load(open(DATE_PATH))
TRADING_DAYS = date_state["trading_days"]
VISIBLE = date_state["visible_through"]
ROW0 = TRADING_DAYS.index("2020-01-01")
ROW1 = TRADING_DAYS.index(VISIBLE)
GRID = TRADING_DAYS[ROW0:ROW1 + 1]
GIDX = {d: i for i, d in enumerate(GRID)}
N_GRID = len(GRID)

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
HORIZON = 10
MIN_ASSETS = 8
GATE_IC = 0.0070
GATE_ICIR = 0.0840


def load_asset(sym, days=2100):
    df = get_stock_daily_data(sym, days=days)
    if df is None:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.set_index("date")
    for c in ["open", "close", "high", "low", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ret"] = df["close"].pct_change()
    return df


def load_macro(name):
    p = f"../persistent/index_data/{name}.csv"
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.set_index("date")
    return df["close"].astype(float)


def asset_series():
    out = {}
    for s in ASSETS:
        df = load_asset(s)
        if df is None or len(df) < 100:
            continue
        close = df["close"].astype(float)
        d = pd.DataFrame({"close": close, "ret": close.pct_change(),
                          "open": df["open"].astype(float),
                          "high": df["high"].astype(float),
                          "low": df["low"].astype(float),
                          "volume": df["volume"].astype(float)})
        out[s] = d
    return out


def to_grid(series_dict):
    mat = np.full((N_GRID, len(ASSETS)), np.nan)
    for j, s in enumerate(ASSETS):
        if s not in series_dict:
            continue
        mat[:, j] = series_dict[s].reindex(GRID).values
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
            ics.append((t, float(fr)))
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
            ("2023-2024", "2023-01-01", "2024-12-31"), ("2025-2027", "2025-01-01", "2099-12-31")]
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
            "hit": hit, "regime": reg}


def fwd_by_horizon_dict(series_dict, horizons=(1, 2, 3, 5, 10, 20)):
    out = {}
    for h in horizons:
        d = {}
        for s, df in series_dict.items():
            close = df["close"]
            d[s] = close.shift(-h) / close - 1.0
        out[h] = to_grid(d)
    return out


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
    return float(valid.mean()), float((valid.sum(axis=1) >= MIN_ASSETS).mean())


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
            x = a[t]
            y = b[t]
            ok = ~(np.isnan(x) | np.isnan(y))
            if ok.sum() >= MIN_ASSETS:
                xs = pd.Series(x[ok]).rank()
                ys = pd.Series(y[ok]).rank()
                c = xs.corr(ys)
                if np.isfinite(c):
                    rho = float(c)
                    break
        if rho is not None:
            out[os.path.basename(f).replace(".signal.npy", "")] = round(rho, 4)
    if out:
        mx = max(out.items(), key=lambda kv: abs(kv[1]))
        return out, mx[0], abs(mx[1])
    return out, None, 0.0


def validate_candidate(name, cand_series, series_dict, spx_close=None):
    """cand_series: dict sym -> pd.Series (own calendar). Runs full validation."""
    mat = to_grid(cand_series)
    rank_mat = cross_sectional_rank(mat)
    fwd = fwd_by_horizon_dict(series_dict)
    ics = spearman_ic_matrix(rank_mat, fwd[10])
    dates = np.array(GRID)
    summ = summarize(ics, dates, name, HORIZON)
    if summ is None:
        print(name, "NO VALID IC DATES", flush=True)
        return None
    cov_ad, cov_d8 = coverage_stats(mat)
    to = turnover_10d_rank(rank_mat)
    dec = decay_curve(rank_mat, fwd)
    pc, pc_name, pc_max = library_pairwise_corr(mat)
    ic, icir = summ["ic"], summ["icir"]
    ok = (abs(ic) >= GATE_IC) and (abs(icir) >= GATE_ICIR)
    print("=" * 100, flush=True)
    print(f"{name}: IC={ic:+.4f} ICIR={icir:+.4f} hit={summ['hit']:.3f} n={summ['n_ic_dates']} "
          f"GATE={'PASS' if ok else 'FAIL'} (|IC|>={GATE_IC}, |ICIR|>={GATE_ICIR})", flush=True)
    for k, v in summ["regime"].items():
        print(f"  regime {k}: ic={v['ic']:+.4f} icir={v['icir']:+.4f} n={v['n']}", flush=True)
    print(f"  decay: {dec}", flush=True)
    print(f"  coverage_asset_days={cov_ad:.3f} dates_ge8={cov_d8:.3f} turnover_10d_rank={to:.4f}", flush=True)
    print(f"  max_abs_library_correlation={pc_max:.4f} (vs {pc_name})", flush=True)
    if pc_max > 0.5:
        print(f"  WARNING: |rho|>0.5 vs library -> redundant candidate", flush=True)
    return {"name": name, "ic": ic, "icir": icir, "hit": summ["hit"], "n": summ["n_ic_dates"],
            "regime": summ["regime"], "decay": dec, "coverage_asset_days": cov_ad,
            "coverage_dates_ge8": cov_d8, "turnover_10d_rank": to,
            "max_abs_library_correlation": pc_max, "max_corr_factor": pc_name,
            "gate_pass": ok}
