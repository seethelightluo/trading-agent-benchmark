"""Shared factor validation utilities for miner_3.

Data is truncated at CURRENT_DATE (no future data). Cross-section = 15 tradable
assets. ICIR = mean(IC)/std(IC) on daily cross-sectional Spearman rank IC.
"""
import json
import math
import numpy as np
import pandas as pd

CURRENT_DATE = pd.Timestamp("2026-07-30")
DATA_DIR = "../persistent/stock_data"
INDEX_DIR = "../persistent/index_data"

TRADABLES = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX",
             "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
OBSERVABLES = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]


def load_close(symbol, data_dir=DATA_DIR, current_date=CURRENT_DATE):
    """Load close series (and OHLCV) for a symbol, truncated at current date."""
    df = pd.read_csv(f"{data_dir}/{symbol}.csv", parse_dates=["date"])
    df = df[df["date"] <= current_date].reset_index(drop=True)
    df = df.set_index("date").sort_index()
    return df


def load_panel(symbols=None):
    """Panel of closes for all tradables, inner-joined on dates."""
    if symbols is None:
        symbols = TRADABLES
    closes = {}
    vols = {}
    for s in symbols:
        df = load_close(s)
        closes[s] = df["close"].astype(float)
        vols[s] = df["volume"].astype(float) if "volume" in df else pd.Series(np.nan, index=df.index)
    px = pd.DataFrame(closes)
    vol = pd.DataFrame(vols)
    px = px.dropna(how="all")
    return px, vol


def forward_returns(px, horizon):
    """Forward return over horizon trading days (close t -> close t+h)."""
    return px.shift(-horizon) / px - 1.0


def rank_ic_series(factor_vals, fwd_ret, min_valid=8):
    """Daily cross-sectional Spearman rank IC series."""
    dates, ics = [], []
    idx = factor_vals.index.intersection(fwd_ret.index)
    for d in idx:
        f = factor_vals.loc[d]
        r = fwd_ret.loc[d]
        pair = pd.concat([f.rename("f"), r.rename("r")], axis=1).dropna()
        if len(pair) >= min_valid:
            ic = pair["f"].corr(pair["r"], method="spearman")
            if ic is not None and math.isfinite(ic):
                dates.append(d)
                ics.append(ic)
    return pd.Series(ics, index=dates)


def summarize_ic(ic_series, factor_name, horizon, n_assets=15):
    """Standard IC summary dict consistent with library format."""
    ic = float(ic_series.mean())
    std = float(ic_series.std())
    icir = ic / std if std and math.isfinite(std) and std > 0 else 0.0
    hit = float((ic_series > 0).mean())
    return {
        "factor": factor_name,
        "horizon": horizon,
        "n_ic_dates": int(len(ic_series)),
        "ic": round(ic, 4),
        "icir": round(icir, 4),
        "ic_hit_ratio": round(hit, 3),
        "pass_gate": abs(ic) >= 0.0070 and abs(icir) >= 0.0840,
    }


def decay_profile(factor_vals, px, horizons=(1, 2, 3, 5, 10, 20), min_valid=8):
    """IC at multiple forward horizons."""
    out = {}
    for h in horizons:
        fr = forward_returns(px, h)
        s = rank_ic_series(factor_vals, fr, min_valid)
        out[str(h)] = round(float(s.mean()), 4) if len(s) else None
    return out


def coverage_stats(factor_vals, px):
    """Fraction of asset-days with valid factor value; fraction of dates with >=8 valid."""
    valid = factor_vals.notna()
    asset_days = float(valid.sum().sum()) / float(factor_vals.shape[0] * factor_vals.shape[1])
    dates_ge8 = float((valid.sum(axis=1) >= 8).mean())
    return {"coverage_asset_days": round(asset_days, 3),
            "coverage_dates_ge8": round(dates_ge8, 3)}


def turnover_rank(factor_vals):
    """Mean abs change of cross-sectional rank (0..1) between consecutive dates."""
    ranks = factor_vals.rank(axis=1, pct=True)
    d = ranks.diff().abs().mean(axis=1)
    return round(float(d.mean()), 3)


def library_signals(px, library_ids=("mom_10d_skip5", "mom_120d_skip5",
                                     "vix_beta_cond_60x20", "vol_of_vol20x60")):
    """Recompute library factor signal matrices on the same panel."""
    ret = px.pct_change()
    vix = load_close("VIX", INDEX_DIR)["close"].astype(float)
    sig = {}
    for fid in library_ids:
        if fid == "mom_10d_skip5":
            sig[fid] = px.shift(5) / px.shift(15) - 1.0
        elif fid == "mom_120d_skip5":
            sig[fid] = px.shift(5) / px.shift(125) - 1.0
        elif fid == "vol_of_vol20x60":
            sig[fid] = ret.rolling(20).std().rolling(60).std()
        elif fid == "vix_beta_cond_60x20":
            vixr = vix.pct_change()
            beta = ret.rolling(60).cov(vixr) / vixr.rolling(60).var()
            sig[fid] = -beta * (vix / vix.shift(20) - 1.0)
        sig[fid] = sig[fid].reindex(px.index)
    return sig


def max_abs_library_correlation(factor_vals, library_sigs):
    """Mean over dates of max abs cross-sectional corr with library signals."""
    obs = []
    for fid, lsig in library_sigs.items():
        cors = []
        idx = factor_vals.index.intersection(lsig.index)
        for d in idx:
            pair = pd.concat([factor_vals.loc[d].rename("f"),
                              lsig.loc[d].rename("l")], axis=1).dropna()
            if len(pair) >= 8:
                c = pair["f"].corr(pair["l"], method="spearman")
                if c is not None and math.isfinite(c):
                    cors.append(c)
        if cors:
            obs.append({"factor": fid, "mean_abs_corr": round(float(np.mean(np.abs(cors))), 3),
                        "mean_corr": round(float(np.mean(cors)), 3)})
    if not obs:
        return None, obs
    return round(max(o["mean_abs_corr"] for o in obs), 3), obs


def ic_metric_path(fid):
    return f"validation.metrics.{fid}"
