"""Miner_2 validation framework (2034-08-03 cycle).
Cross-sectional rank IC / ICIR / decay / turnover / coverage on the 15-asset
tradable universe. Validation window ends at 2034-08-02 (last visible day).
No lookahead: factors use data up to t; forward returns use t..t+h.
"""
import json
import base64
import zlib
import io
import hashlib

import numpy as np
import pandas as pd

ASSETS = ["000300.SH", "000688.SH", "BTC", "CN10Y", "COPPER", "ETH", "HSI",
          "N225", "NDX", "SOX", "SPX", "SX5E", "US10Y", "WTI", "XAU"]
DATA_DIR = "../persistent/stock_data"
INDEX_DIR = "../persistent/index_data"
CURRENT_DATE = pd.Timestamp("2034-08-02")
IC_GATE = 0.0070
ICIR_GATE = 0.0840
MIN_ASSETS_PER_DATE = 8


def load_closes(end_date=CURRENT_DATE):
    closes, vols, opens, highs, lows = {}, {}, {}, {}, {}
    for a in ASSETS:
        df = pd.read_csv(f"{DATA_DIR}/{a}.csv", parse_dates=["date"])
        df = df[df["date"] <= end_date].set_index("date").sort_index()
        closes[a] = df["close"].astype(float)
        vols[a] = df["volume"].astype(float)
        opens[a] = df["open"].astype(float)
        highs[a] = df["high"].astype(float)
        lows[a] = df["low"].astype(float)
    close = pd.DataFrame(closes)
    vol = pd.DataFrame(vols)
    open_ = pd.DataFrame(opens)
    high = pd.DataFrame(highs)
    low = pd.DataFrame(lows)
    return close, vol, open_, high, low


def load_index(name, end_date=CURRENT_DATE):
    df = pd.read_csv(f"{INDEX_DIR}/{name}.csv", parse_dates=["date"])
    df = df[df["date"] <= end_date].set_index("date").sort_index()
    return df["close"].astype(float)


def dense_per_asset(close, vol, open_, high, low):
    dense = {}
    for a in ASSETS:
        idx = close[a].dropna().index
        dense[a] = {
            "close": close[a].reindex(idx),
            "vol": None if vol is None else vol[a].reindex(idx),
            "open": None if open_ is None else open_[a].reindex(idx),
            "high": None if high is None else high[a].reindex(idx),
            "low": None if low is None else low[a].reindex(idx),
        }
    return dense


def factor_panel(fn, close, vol, open_, high, low, macro, **params):
    dense = dense_per_asset(close, vol, open_, high, low)
    out = {}
    for a in ASSETS:
        d = dense[a]
        try:
            s = fn(d["close"], d["vol"], d["open"], d["high"], d["low"], macro, **params)
            out[a] = pd.Series(s.values, index=d["close"].index).reindex(close.index)
        except Exception:
            out[a] = pd.Series(np.nan, index=close.index)
    return pd.DataFrame(out)


def fwd_returns(close, horizon):
    out = {}
    dense = dense_per_asset(close, None, None, None, None)
    for a in ASSETS:
        c = dense[a]["close"]
        fr = (c.shift(-horizon) / c - 1.0).reindex(close.index)
        out[a] = fr
    return pd.DataFrame(out)


def ic_series(factor, fwd_ret, min_assets=MIN_ASSETS_PER_DATE):
    dates, ics = [], []
    for dt in factor.index:
        x = factor.loc[dt]
        y = fwd_ret.loc[dt]
        m = x.notna() & y.notna()
        if m.sum() >= min_assets:
            ics.append(x[m].rank().corr(y[m].rank()))
            dates.append(dt)
    return pd.Series(ics, index=pd.DatetimeIndex(dates))


def turnover_rank(factor, lag=10):
    ranks = factor.rank(axis=1)
    d = ranks.diff(lag).abs().mean(axis=1)
    return float(d.dropna().mean())


def coverage(factor):
    n_total = float(factor.notna().sum().sum())
    denom = factor.shape[0] * factor.shape[1]
    ge8 = float((factor.notna().sum(axis=1) >= MIN_ASSETS_PER_DATE).mean())
    return n_total / denom, ge8


def validate_factor(fn, close, vol, open_, high, low, macro,
                    horizons=(1, 2, 3, 5, 10, 20), admission_horizon=10, **params):
    panel = factor_panel(fn, close, vol, open_, high, low, macro, **params)
    cov_ad, cov_ge8 = coverage(panel)
    decay, ic_series_by_h = {}, {}
    for h in horizons:
        fr = fwd_returns(close, h)
        ic = ic_series(panel, fr)
        ic_series_by_h[h] = ic
        decay[h] = float(ic.mean()) if len(ic) else np.nan
    ic_main = ic_series_by_h[admission_horizon]
    ic = float(ic_main.mean())
    icir = float(ic_main.mean() / ic_main.std()) if len(ic_main) > 2 else np.nan
    hit = float((ic_main > 0).mean()) if np.isfinite(ic) else np.nan
    if ic < 0:
        hit = float((ic_main < 0).mean())
    return {
        "panel": panel,
        "ic": ic,
        "icir": icir,
        "ic_hit_ratio": hit,
        "n_ic_dates": int(len(ic_main)),
        "coverage_asset_days": round(cov_ad, 4),
        "coverage_dates_ge8": round(cov_ge8, 4),
        "turnover_10d_rank": round(turnover_rank(panel), 4),
        "decay_ic_by_horizon": {str(h): round(decay[h], 4) for h in horizons},
    }


def load_library_panels():
    """Decode signal artifacts of existing library factors (backups only)."""
    lib = {}
    for fid in ["mom_10d_skip5", "mom_120d_skip5", "vix_beta_cond_60x20",
                "yield_beta_cond_60x20", "vol_of_vol20x60"]:
        for base in [f"factors/{fid}.json", f"factors/evicted/{fid}.json",
                     f"factors/{fid}.json.bak"]:
            try:
                d = json.load(open(base))
                data = d["validation"]["signal_artifact"]["data"]
                raw = base64.b64decode(data)
                csv_text = zlib.decompress(raw).decode()
                panel = pd.read_csv(io.StringIO(csv_text), index_col=0, parse_dates=True)
                panel.index = pd.DatetimeIndex(panel.index)
                lib[fid] = panel
                break
            except Exception:
                continue
    return lib


def max_library_corr(panel, lib):
    best = 0.0
    for fid, lp in lib.items():
        common = panel.index.intersection(lp.index)
        cols = [c for c in panel.columns if c in lp.columns]
        if len(common) < 60 or len(cols) < 5:
            continue
        a = panel.loc[common, cols].values.ravel()
        b = lp.loc[common, cols].values.ravel()
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 200:
            continue
        rho = float(np.corrcoef(a[m], b[m])[0, 1])
        best = max(best, abs(rho))
    return best


def artifact_b64(panel):
    csv_text = panel.to_csv()
    compressed = zlib.compress(csv_text.encode())
    return base64.b64encode(compressed).decode()


def print_result(name, res):
    print(f"\n=== {name} ===")
    for k in ["ic", "icir", "ic_hit_ratio", "n_ic_dates