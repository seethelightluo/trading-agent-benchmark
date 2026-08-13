"""Shared data-loading + factor validation framework for miner_2 (2033-10-17 cycle).

Data: ../persistent/stock_data/{symbol}.csv (15 tradable) and
../persistent/index_data/{symbol}.csv (5 observation-only).
Visible through 2033-10-14 per ../persistent/date.json (current_date 2033-10-17).
Any rows after that are future data relative to the simulation clock and MUST be excluded.
"""
import json
import base64
import zlib
import hashlib
import numpy as np
import pandas as pd

VISIBLE_THROUGH = "2033-10-14"
TRADABLE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX",
            "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
OBS_ONLY = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]

ADMISSION = {"ic_threshold": 0.0070, "icir_threshold": 0.0840, "horizon": 10}


def load_asset(symbol, obs=False):
    path = f"../persistent/{'index_data' if obs else 'stock_data'}/{symbol}.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[df["date"] <= pd.Timestamp(VISIBLE_THROUGH)].copy()
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_price_panel():
    """Return DataFrame of close prices: index=date, columns=symbols (15 tradable)."""
    panels = {}
    for s in TRADABLE:
        df = load_asset(s)
        panels[s] = df.set_index("date")["close"]
    px = pd.DataFrame(panels)
    px = px.sort_index()
    return px


def load_obs_panel():
    obs = {}
    for s in OBS_ONLY:
        df = load_asset(s, obs=True)
        obs[s] = df.set_index("date")["close"]
    return pd.DataFrame(obs).sort_index()


def rets(px):
    return px.pct_change()


def ic_series(factor_panel, fwd_ret_panel):
    """Per-date Spearman IC between factor values and forward returns.
    Uses at least 8 valid instruments per date. Returns Series of IC per date."""
    ic_vals, dates = [], []
    common = factor_panel.index.intersection(fwd_ret_panel.index)
    for dt in common:
        f = factor_panel.loc[dt]
        r = fwd_ret_panel.loc[dt]
        mask = f.notna() & r.notna()
        if mask.sum() < 8:
            continue
        fv, rv = f[mask], r[mask]
        if fv.nunique() < 3 or rv.nunique() < 3:
            continue
        ic = fv.rank().corr(rv.rank())
        if np.isfinite(ic):
            ic_vals.append(ic)
            dates.append(dt)
    return pd.Series(ic_vals, index=pd.DatetimeIndex(dates), name="ic")


def summarize_ic(ic_s, label=""):
    if len(ic_s) == 0:
        return {"label": label, "n_ic_dates": 0}
    m = ic_s.mean()
    s = ic_s.std(ddof=1)
    icir = m / s if s > 0 else 0.0
    hit = (ic_s > 0).mean()
    return {
        "label": label,
        "n_ic_dates": int(len(ic_s)),
        "ic": float(m),
        "icir": float(icir),
        "ic_std": float(s),
        "ic_hit_ratio": float(hit),
        "first_date": str(ic_s.index[0].date()),
        "last_date": str(ic_s.index[-1].date()),
    }


def decay_analysis(factor_panel, px, horizons=(1, 2, 3, 5, 10, 20)):
    out = {}
    for h in horizons:
        fwd = px.shift(-h) / px - 1.0
        ic = ic_series(factor_panel, fwd)
        out[str(h)] = float(ic.mean()) if len(ic) else np.nan
    return out


def turnover_10d(factor_panel):
    """Mean absolute change of cross-sectional ranks between t and t+10, averaged over dates."""
    r = factor_panel.rank(axis=1)
    d = r.diff(10).abs().mean(axis=1)
    return float(d.mean()) if len(d) else np.nan


def coverage_stats(factor_panel, n_assets=15):
    valid = factor_panel.notna()
    asset_days = float(valid.mean().mean())
    dates_ge8 = float((valid.sum(axis=1) >= 8).mean())
    return {"coverage_asset_days": asset_days, "coverage_dates_ge8": dates_ge8}


def signal_artifact(factor_panel):
    """Encode signal panel as base64:zlib:csv (rows=dates, cols=assets)."""
    csv_str = factor_panel.reset_index().to_csv(index=False)
    comp = zlib.compress(csv_str.encode("utf-8"))
    b64 = base64.b64encode(comp).decode("ascii")
    sha = hashlib.sha256(csv_str.encode("utf-8")).hexdigest()[:16]
    return {
        "format": "base64:zlib:csv",
        "description": f"Factor signal panel: rows = dates, cols = assets. Shape {factor_panel.shape}",
        "columns": list(factor_panel.columns),
        "shape": list(factor_panel.shape),
        "n_valid_values": int(factor_panel.notna().sum().sum()),
        "sha256": sha,
        "data": b64,
    }


def std_roll(x, w):
    return x.rolling(w).std(ddof=0)


def roll_beta(y, x, w=60, min_obs=40):
    """Rolling beta of y (DataFrame) on x (Series). NaN where insufficient obs."""
    out = pd.DataFrame(np.nan, index=y.index, columns=y.columns)
    yv, xv = y.values, x.values
    for i in range(w - 1, len(y)):
        xw = xv[i - w + 1:i + 1]
        yw = yv[i - w + 1:i + 1]
        xm = np.nanmean(xw)
        xd = xw - xm
        denom = np.nansum(xd ** 2)
        if denom <= 0:
            continue
        for j in range(y.shape[1]):
            ym = np.nanmean(yw[:, j])
            num = np.nansum((yw[:, j] - ym) * xd)
            out.iloc[i, j] = num / denom
    return out


if __name__ == "__main__":
    px = load_price_panel()
    print("price panel shape:", px.shape, px.index.min().date(), "->", px.index.max().date())
    print("dates with >=8 valid:", int((px.notna().sum(axis=1) >= 8).sum()))
    obs = load_obs_panel()
    print("obs panel shape:", obs.shape, obs.index.min().date(), "->", obs.index.max().date())
