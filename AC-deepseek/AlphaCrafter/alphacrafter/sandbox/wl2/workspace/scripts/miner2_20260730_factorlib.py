"""Shared factor validation framework for miner_2 (v2).
Loads the 15-asset tradable universe + macro signals, computes rank IC metrics.
Data is restricted to dates <= visible_through (no future leakage).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path("../persistent")
STOCK = DATA / "stock_data"
INDEX = DATA / "index_data"

TRADABLE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
            "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
OBS = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]


def load_series(symbol, kind="stock"):
    path = (STOCK if kind == "stock" else INDEX) / f"{symbol}.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").set_index("date")
    df.index = pd.to_datetime(df.index)
    return df


def load_visible_through():
    d = json.loads((DATA / "date.json").read_text())
    return pd.Timestamp(d["visible_through"])


def load_panel(symbols, kind="stock", end=None):
    """Return DataFrame of close prices, rows <= end (visible date)."""
    end = end or load_visible_through()
    cols = {}
    for s in symbols:
        try:
            df = load_series(s, kind)
            df = df[df.index <= end]
            if len(df) >= 30:
                cols[s] = df["close"].astype(float)
        except Exception as e:
            print(f"  load fail {s}: {e}")
    panel = pd.DataFrame(cols)
    return panel.sort_index()


def factor_metrics(factor_df, forward_ret_df, horizon, min_assets=8, direction=1):
    """Rank IC series between factor values and forward returns (same date)."""
    common = factor_df.join(forward_ret_df, how="inner", lsuffix="_f", rsuffix="_r")
    fcols = [c for c in common.columns if c.endswith("_f")]
    rcols = [c for c in common.columns if c.endswith("_r")]
    common = common.replace([np.inf, -np.inf], np.nan)
    dates, ics = [], []
    for dt, row in common.iterrows():
        f = row[fcols].values
        r = row[rcols].values
        mask = np.isfinite(f) & np.isfinite(r)
        if mask.sum() < min_assets:
            continue
        fr = pd.Series(f[mask]).rank().values
        rr = pd.Series(r[mask]).rank().values
        if np.std(fr) < 1e-12 or np.std(rr) < 1e-12:
            continue
        ic = np.corrcoef(fr, rr)[0, 1]
        if not np.isfinite(ic):
            continue
        dates.append(dt)
        ics.append(ic)
    ics = np.array(ics)
    if len(ics) < 30:
        return None
    mean_ic = float(ics.mean())
    std_ic = float(ics.std(ddof=1)) if len(ics) > 1 else 0.0
    icir = mean_ic / std_ic if std_ic > 1e-12 else 0.0
    hit = float((np.sign(ics) == np.sign(direction)).mean())
    return {
        "horizon": horizon,
        "ic": mean_ic,
        "icir": icir,
        "ic_hit_ratio": hit,
        "n_ic_dates": int(len(ics)),
        "first_date": str(dates[0].date()),
        "last_date": str(dates[-1].date()),
    }


def turnover_10d(factor_df):
    """Mean abs change in cross-sectional rank between rows ~10 days apart."""
    f = factor_df.replace([np.inf, -np.inf], np.nan)
    f = f.dropna(how="all")
    if len(f) < 40:
        return float("nan")
    idx = f.index[::10]
    prev = None
    tots, cnt = 0.0, 0
    for dt in idx:
        row = f.loc[dt].dropna()
        if len(row) < 5:
            continue
        r = row.rank().values
        if prev is not None:
            tots += float(np.abs(r - prev).mean())
            cnt += 1
        prev = r
    return tots / cnt if cnt else float("nan")


def coverage(factor_df):
    f = factor_df.replace([np.inf, -np.inf], np.nan)
    total = float(f.size)
    valid = float(f.notna().sum().sum())
    dates_ge8 = float((f.notna().sum(axis=1) >= 8).mean())
    return {
        "coverage_asset_days": valid / total if total else 0.0,
        "coverage_dates_ge8": dates_ge8,
    }


def decay_profile(factor_df, ret_panel, horizons=(1, 2, 3, 5, 10, 20)):
    out = {}
    for h in horizons:
        fwd = ret_panel.shift(-h)
        m = factor_metrics(factor_df, fwd, h)
        if m:
            out[str(h)] = round(m["ic"], 4)
    return out


def _recompute_library_factor(doc, panel):
    """Recompute known persisted library factors from close prices (and VIX
    where needed) for correlation audit. Returns DataFrame aligned to panel."""
    fid = doc.get("factor_id", "")
    if fid == "mom_10d_skip5":
        return panel.shift(5) / panel.shift(15) - 1.0
    if fid == "mom_120d_skip5":
        return panel.shift(5) / panel.shift(125) - 1.0
    if fid == "vol_of_vol20x60":
        return panel.pct_change().rolling(20).std().rolling(60).std()
    if fid == "vix_beta_cond_60x20":
        vix = load_panel(["VIX"], "index", end=panel.index.max())["VIX"]
        vix_ret = vix.pct_change()
        out = pd.DataFrame(index=panel.index, columns=panel.columns, dtype=float)
        ret = panel.pct_change()
        for sym in panel.columns:
            z = pd.concat([ret[sym].rename("a"), vix_ret.rename("v")], axis=1).dropna()
            beta = z["a"].rolling(60).cov(z["v"]) / z["v"].rolling(60).var()
            vixm = vix / vix.shift(20) - 1.0
            out[sym] = -beta * vixm
        return out
    return None


def max_library_correlation(factor_df, library_dir="factors"):
    lib = Path(library_dir)
    best = 0.0
    if not lib.exists():
        return 0.0
    f = factor_df.replace([np.inf, -np.inf], np.nan)
    for fp in lib.glob("*.json"):
        try:
            doc = json.loads(fp.read_text())
        except Exception:
            continue
        if "factor_ensemble" in fp.name:
            continue
        if doc.get("validation", {}).get("status") != "EFFECTIVE":
            continue
        other = _recompute_library_factor(doc, f)
        if other is None or len(other) < 30:
            continue
        joined = f.join(other, how="inner", lsuffix="_a", rsuffix="_b").dropna()
        if len(joined) < 30:
            continue
        for c in [c for c in joined.columns if c.endswith("_a")]:
            a = joined[c].rank().values
            b = joined[c.replace("_a", "_b")].rank().values
            if np.std(a) < 1e-12 or np.std(b) < 1e-12:
                continue
            rho = abs(float(np.corrcoef(a, b)[0, 1]))
            if np.isfinite(rho):
                best = max(best, rho)
    return best


def full_validate(factor_df, panel, horizon=10, direction=1, min_assets=8,
                  library_dir="factors", label=""):
    """One-stop validation returning the metric dict used for admission."""
    ret_panel = panel.pct_change()
    fwd = ret_panel.shift(-horizon)
    m = factor_metrics(factor_df, fwd, horizon, min_assets, direction)
    if m is None:
        print(f"[{label}] INSUFFICIENT IC observations")
        return None
    m.update(coverage(factor_df))
    m["turnover_10d_rank"] = turnover_10d(factor_df)
    m["decay_ic_by_horizon"] = decay_profile(factor_df, ret_panel)
    m["max_abs_library_correlation"] = max_library_correlation(factor_df, library_dir)
    m["n_assets"] = int(factor_df.notna().sum(axis=1).max())
    print(f"[{label}] h={horizon} IC={m['ic']:.4f} ICIR={m['icir']:.4f} "
          f"hit={m['ic_hit_ratio']:.3f} ndates={m['n_ic_dates']} "
          f"cov={m['coverage_asset_days']:.3f} to={m['turnover_10d_rank']:.2f} "
          f"rho={m['max_abs_library_correlation']:.3f} "
          f"({m['first_date']}..{m['last_date']})")
    return m
