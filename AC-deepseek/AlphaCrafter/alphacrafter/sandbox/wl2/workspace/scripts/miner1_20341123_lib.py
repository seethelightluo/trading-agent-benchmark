"""Shared factor-validation library for miner_1 (2034-11-23 cycle).

Loads the 15 tradable assets + macro observation signals, cuts at the last
completed trading day before the current date (2034-11-22), computes factor
panels, and evaluates cross-sectional IC/ICIR/coverage/turnover/decay plus
max-abs correlation vs the persisted library artifacts.

IMPORTANT: no lookahead - all data used ends <= ASOF (2034-11-22).
"""
import os
import json
import glob
import numpy as np
import pandas as pd

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["VIX", "DXY", "USDCNY", "USDJPY", "EURUSD"]
ASOF = pd.Timestamp("2034-11-22")  # last completed trading day before current date

DATA_DIR = "../persistent/stock_data"
IDX_DIR = "../persistent/index_data"
FACTOR_DIR = "factors"

IC_THRESH = 0.0070
ICIR_THRESH = 0.0840


def load_asset_prices(symbol: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATA_DIR, f"{symbol}.csv"))
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_macro(symbol: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(IDX_DIR, f"{symbol}.csv"))
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def build_price_panel() -> pd.DataFrame:
    """Wide close-price panel indexed by date (aligned to a union calendar)."""
    closes = {}
    for s in WATCH:
        df = load_asset_prices(s)
        closes[s] = df.set_index("date")["close"]
    panel = pd.DataFrame(closes).sort_index()
    return panel[panel.index <= ASOF]


def build_macro_panel() -> pd.DataFrame:
    closes = {}
    for s in MACRO:
        df = load_macro(s)
        closes[s] = df.set_index("date")["close"]
    panel = pd.DataFrame(closes).sort_index()
    return panel[panel.index <= ASOF]


def asset_returns(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.pct_change()


def forward_returns(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Forward return over `horizon` trading days (per-asset own calendar via union index shift)."""
    return panel.shift(-horizon) / panel - 1.0


def daily_cross_sectional_ic(factor: pd.DataFrame, fwd_ret: pd.DataFrame,
                             min_valid=8) -> pd.Series:
    """Spearman rank IC per date between factor values and forward returns."""
    ics = {}
    for dt in factor.index:
        f = factor.loc[dt]
        r = fwd_ret.loc[dt]
        mask = f.notna() & r.notna() & np.isfinite(f) & np.isfinite(r)
        if mask.sum() < min_valid:
            continue
        ics[dt] = f[mask].corr(r[mask], method="spearman")
    s = pd.Series(ics, dtype=float)
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def summarize_ic(ic: pd.Series, horizon: int, label: str = ""):
    if len(ic) == 0:
        return {"horizon": horizon, "n_ic_dates": 0, "ic": np.nan, "icir": np.nan}
    ic = ic.dropna()
    mean_ic = ic.mean()
    std_ic = ic.std(ddof=1)
    icir = mean_ic / std_ic if std_ic > 0 else np.nan
    hit = (ic > 0).mean()
    return {
        "horizon": horizon,
        "n_ic_dates": int(len(ic)),
        "ic": float(mean_ic),
        "icir": float(icir),
        "ic_hit_ratio": float(hit),
        "ic_std": float(std_ic),
    }


def factor_coverage(factor: pd.DataFrame) -> dict:
    valid = factor.notna() & np.isfinite(factor)
    asset_days = valid.sum().sum() / (factor.shape[0] * factor.shape[1])
    dates_ge8 = (valid.sum(axis=1) >= 8).mean()
    return {
        "coverage_asset_days": float(asset_days),
        "coverage_dates_ge8": float(dates_ge8),
    }


def turnover_rank(factor: pd.DataFrame, period: int = 10) -> float:
    """Mean abs change in cross-sectional rank between observations `period` apart."""
    ranks = factor.rank(axis=1)
    r = ranks.diff(period).abs().mean().mean()
    if pd.isna(r):
        return float("nan")
    return float(r / (factor.shape[1] - 1))  # normalized 0..1


def decay_profile(factor: pd.DataFrame, panel: pd.DataFrame,
                  horizons=(1, 2, 3, 5, 10, 20), min_valid=8) -> dict:
    out = {}
    for h in horizons:
        fwd = forward_returns(panel, h)
        ic = daily_cross_sectional_ic(factor, fwd, min_valid=min_valid)
        out[str(h)] = float(ic.mean()) if len(ic) else float("nan")
    return out


def load_library_signal_panels():
    """Load existing factor signal artifacts (embedded JSON values or .npy) as DataFrames."""
    panels = {}
    for p in sorted(glob.glob(os.path.join(FACTOR_DIR, "*.json"))):
        if p.endswith(".bak") or "factor_ensemble" in p:
            continue
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception:
            continue
        sa = d.get("signal_artifact")
        fid = d.get("factor_id")
        if not fid:
            continue
        if isinstance(sa, dict) and sa.get("format") == "daily_panel":
            cols = sa.get("columns")
            dates = sa.get("dates")
            vals = sa.get("values")
            if cols and dates and vals:
                arr = np.array(vals, dtype=float).reshape(len(dates), len(cols))
                panels[fid] = pd.DataFrame(arr, index=pd.to_datetime(dates), columns=cols)
        elif isinstance(sa, str) and sa.endswith(".npy"):
            npy_path = os.path.join(FACTOR_DIR, sa)
            if os.path.exists(npy_path):
                arr = np.load(npy_path, allow_pickle=True)
                if arr.ndim == 2 and arr.shape[1] == len(WATCH):
                    # try to find the date axis from a nearby json with same id
                    panels[fid] = pd.DataFrame(arr, columns=WATCH)
        # fallback: npy file with same stem as factor_id
        npy_path = os.path.join(FACTOR_DIR, f"{fid}.signal.npy")
        if fid not in panels and os.path.exists(npy_path):
            arr = np.load(npy_path, allow_pickle=True)
            if arr.ndim == 2 and arr.shape[1] == len(WATCH):
                panels[fid] = pd.DataFrame(arr, columns=WATCH)
    return panels


def max_abs_library_corr(factor: pd.DataFrame, lib_panels: dict,
                         common_dates: pd.DatetimeIndex) -> float:
    """Max |Pearson rho| between candidate factor cross-sections and library factors
    on dates where both have >= 8 valid values."""
    best = 0.0
    best_id = None
    for fid, lp in lib_panels.items():
        if lp.shape[1] != factor.shape[1]:
            continue
        if isinstance(lp.index, pd.RangeIndex):
            # align by position to factor index if lengths match
            if len(lp) != len(factor):
                continue
            lp2 = lp.copy()
            lp2.index = factor.index
        else:
            lp2 = lp.reindex(common_dates)
            if lp2.shape[0] != len(common_dates):
                continue
        rhos = []
        for dt in common_dates:
            if dt not in factor.index:
                continue
            f = factor.loc[dt]
            g = lp2.loc[dt] if dt in lp2.index else None
            if g is None:
                continue
            m = f.notna() & g.notna() & np.isfinite(f) & np.isfinite(g)
            if m.sum() >= 8:
                r = float(np.corrcoef(f[m], g[m])[0, 1])
                if np.isfinite(r):
                    rhos.append(abs(r))
        if rhos:
            m = float(np.mean(rhos))
            if m > best:
                best = m
                best_id = fid
    return best, best_id


def run_validation(factor_id, factor_name, expression, description, dependencies,
                   parameters, expected_direction, factor_df: pd.DataFrame,
                   panel: pd.DataFrame, tags, regime_notes, lib_panels,
                   horizons=(1, 2, 3, 5, 10, 20), admission_horizon=10,
                   min_valid=8):
    """Full validation pipeline. factor_df: DataFrame indexed by date with WATCH columns."""
    fwd = forward_returns(panel, admission_horizon)
    ic = daily_cross_sectional_ic(factor_df, fwd, min_valid=min_valid)
    summ = summarize_ic(ic, admission_horizon)
    cov = factor_coverage(factor_df)
    turn = turnover_rank(factor_df, period=admission_horizon)
    decay = decay_profile(factor_df, panel, horizons=horizons, min_valid=min_valid)
    common_dates = panel.index.intersection(factor_df.index)
    maxrho, best_id = max_abs_library_corr(factor_df, lib_panels, common_dates)

    metrics = {
        "ic": summ["ic"],
        "icir": summ["icir"],
        "ic_hit_ratio": summ["ic_hit_ratio"],
        "n_ic_dates": summ["n_ic_dates"],
        "ic_std": summ["ic_std"],
        **cov,
        "turnover_10d_rank": turn,
        "decay_ic_by_horizon": decay,
        "max_abs_library_correlation": maxrho,
        "max_corr_library_factor": best_id,
    }
    passed = (abs(metrics["ic"]) >= IC_THRESH) and (abs(metrics["icir"]) >= ICIR_THRESH)
    return {
        "factor_id": factor_id,
        "factor_name": factor_name,
        "expression": expression,
        "description": description,
        "dependencies": dependencies,
        "parameters": parameters,
        "expected_direction": expected_direction,
        "metrics": metrics,
        "passed": passed,
        "regime_notes": regime_notes,
        "tags": tags,
        "asof": str(ASOF.date()),
    }


def persist_factor_json(doc: dict, factor_df: pd.DataFrame, panel: pd.DataFrame):
    """Write factor JSON with embedded daily_panel signal artifact (dates/columns/values)."""
    factor_id = doc["factor_id"]
    path = os.path.join(FACTOR_DIR, f"{factor_id}.json")
    dates = [d.strftime("%Y-%m-%d") for d in factor_df.index]
    values = []
    for dt in factor_df.index:
        row = factor_df.loc[dt]
        values.append([None if (pd.isna(v) or not np.isfinite(v)) else float(v) for v in row])
    artifact = {
        "format": "daily_panel",
        "dates": dates,
        "columns": WATCH,
        "values": values,
    }
    doc["signal_artifact"] = artifact
    # also save npy for robustness
    arr = factor_df.values.astype(float)
    np.save(os.path.join(FACTOR_DIR, f"{factor_id}.signal.npy"), arr)
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    return path
