"""miner_1 2026-11-19: shared validation harness.
Reads 15-asset close panel directly from persistent/stock_data, truncated to the
simulator-visible date (visible_through from persistent/date.json = 2026-11-18).
No future leakage: all research uses data through visible_through only.
"""
import json, math, zlib, base64
import numpy as np
import pandas as pd

WATCHLIST = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
             "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
STOCK_DIR = "../persistent/stock_data/"
DATE_JSON = "../persistent/date.json"

IC_THRESHOLD = 0.0070
ICIR_THRESHOLD = 0.0840
MIN_IC_DATES = 60
MIN_ASSETS_PER_DATE = 8
ADMISSION_HORIZON = 10

_current = None

def visible_through():
    global _current
    if _current is None:
        _current = pd.to_datetime(json.load(open(DATE_JSON))["visible_through"])
    return _current


def load_panel(start="2020-01-01"):
    """Close panel + volume panel, dates old->new, truncated at visible_through."""
    cutoff = visible_through()
    closes, vols = {}, {}
    for s in WATCHLIST:
        df = pd.read_csv(f"{STOCK_DIR}/{s}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= pd.to_datetime(start)) & (df["date"] <= cutoff)]
        df = df.sort_values("date").reset_index(drop=True)
        closes[s] = df.set_index("date")["close"].astype(float)
        if "volume" in df.columns and df["volume"].astype(float).abs().sum() > 0:
            vols[s] = df.set_index("date")["volume"].astype(float)
    panel = pd.DataFrame(closes).sort_index()
    vpanel = pd.DataFrame(vols).sort_index()
    return panel, vpanel


def load_macro_panel(name, start="2020-01-01"):
    cutoff = visible_through()
    df = pd.read_csv(f"../persistent/index_data/{name}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= pd.to_datetime(start)) & (df["date"] <= cutoff)]
    df = df.sort_values("date").reset_index(drop=True)
    return df.set_index("date")["close"].astype(float).sort_index()


def forward_returns(panel, horizon=ADMISSION_HORIZON):
    return panel.shift(-horizon) / panel - 1.0


def spearman_ic_series(factor_df, fwd, min_assets=MIN_ASSETS_PER_DATE):
    """Cross-sectional Spearman IC per date. factor_df & fwd: date x asset."""
    ics, dates = [], []
    common = factor_df.index.intersection(fwd.index)
    fs, fr = factor_df.loc[common], fwd.loc[common]
    for dt in common:
        x = fs.loc[dt].dropna()
        y = fr.loc[dt].reindex(x.index)
        m = x.notna() & y.notna()
        if m.sum() < min_assets:
            continue
        xx, yy = x[m], y[m]
        if xx.nunique() < 3 or yy.nunique() < 3:
            continue
        rho = xx.rank().corr(yy.rank())
        if not np.isnan(rho):
            ics.append(rho)
            dates.append(dt)
    return pd.Series(ics, index=dates)


def ic_metrics(ics):
    if len(ics) < MIN_IC_DATES:
        return {"ic": float("nan"), "icir": float("nan"), "n_ic_dates": len(ics),
                "hit": float("nan"), "tstat": float("nan")}
    ic = float(ics.mean())
    sd = float(ics.std(ddof=1))
    icir = ic / sd if sd > 0 else float("nan")
    tstat = ic / (sd / math.sqrt(len(ics))) if sd > 0 else float("nan")
    hit = float((ics > 0).mean())
    return {"ic": ic, "icir": icir, "n_ic_dates": len(ics), "hit": hit, "tstat": tstat}


def coverage(series, panel):
    sub = series.loc[series.index.intersection(panel.index)]
    valid = int(sub.notna().sum().sum())
    total = panel.shape[0] * panel.shape[1]
    return valid / total if total else float("nan")


def turnover_rank_chg(series, panel):
    """Mean cross-sectional rank change per step (0=stable .. 1=full reshuffle)."""
    sub = series.loc[series.index.intersection(panel.index)]
    ranks = sub.rank(axis=1)
    diff = ranks.diff().abs().mean(axis=1)
    return float(diff.mean()) if len(diff) else float("nan")


def regime_slices(ics):
    """Split IC series into 3 regime windows and summarize."""
    out = {}
    bounds = [("2020-01-01", "2021-12-31"), ("2022-01-01", "2023-12-31"), ("2024-01-01", None)]
    for lo, hi in bounds:
        sub = ics.loc[ics.index >= pd.to_datetime(lo)]
        if hi:
            sub = sub.loc[sub.index <= pd.to_datetime(hi)]
        if len(sub) >= MIN_IC_DATES:
            m = ic_metrics(sub)
            out[f"{lo}..{hi or 'now'}"] = [round(m["ic"], 4), round(m["icir"], 4), len(sub)]
        else:
            out[f"{lo}..{hi or 'now'}"] = [float("nan"), float("nan"), len(sub)]
    return out


def decay_by_horizon(panel, factor_df, horizons=(1, 2, 3, 5, 10, 20)):
    out = {}
    for h in horizons:
        fwd = forward_returns(panel, horizon=h)
        ics = spearman_ic_series(factor_df, fwd)
        out[str(h)] = round(float(ics.mean()), 4) if len(ics) >= MIN_IC_DATES else float("nan")
    return out


def zlib_b64_panel(factor_df):
    csv = factor_df.to_csv()
    comp = zlib.compress(csv.encode("utf-8"))
    return base64.b64encode(comp).decode("ascii")


def max_library_corr(factor_df, panel):
    """Datewise pairwise Spearman rho vs persisted library signals (from factors/ active jsons)."""
    sigs = {}
    import os, glob
    for fn in glob.glob("factors/*.json"):
        if fn.endswith(".bak") or os.path.basename(fn) in ("factor_ensemble.json",):
            continue
        try:
            d = json.load(open(fn))
        except Exception:
            continue
        art = (d.get("validation") or {}).get("signal_artifact") or {}
        raw = art.get("data")
        if not raw:
            continue
        try:
            csv = zlib.decompress(base64.b64decode(raw)).decode("utf-8")
            sigs[d["factor_id"]] = pd.read_csv(pd.io.common.StringIO(csv), index_col=0)
        except Exception:
            continue
    if not sigs:
        return None, {}
    out = {}
    for fid, sig in sigs.items():
        common = factor_df.index.intersection(sig.index)
        if len(common) < 30:
            out[fid] = float("nan")
            continue
        rhos = []
        for dt in common:
            x = factor_df.loc[dt]
            y = sig.loc[dt].reindex(x.index)
            m = x.notna() & y.notna()
            if m.sum() < 8:
                continue
            xx, yy = x[m], y[m]
            if xx.nunique() < 3 or yy.nunique() < 3:
                continue
            rho = xx.rank().corr(yy.rank())
            if not np.isnan(rho):
                rhos.append(rho)
        out[fid] = float(np.mean(rhos)) if rhos else float("nan")
    mx = max((abs(v) for v in out.values() if not np.isnan(v)), default=None)
    return mx, out