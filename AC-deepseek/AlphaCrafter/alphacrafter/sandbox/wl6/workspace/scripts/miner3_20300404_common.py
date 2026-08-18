"""Shared data loading & factor evaluation utilities for miner_3."""
import pandas as pd
import numpy as np
import json

TRADABLE = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
MACRO = ["DXY","USDCNY","USDJPY","EURUSD","VIX"]
VISIBLE_THROUGH = "2030-04-03"

def load_asset_panel():
    """Load all tradable + macro close prices aligned on common trading dates."""
    closes = {}
    for sym in TRADABLE + MACRO:
        df = pd.read_csv(f"../persistent/stock_data/{sym}.csv") if sym in TRADABLE else pd.read_csv(f"../persistent/index_data/{sym}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= pd.Timestamp(VISIBLE_THROUGH)].sort_values("date")
        closes[sym] = df.set_index("date")["close"]
    px = pd.DataFrame(closes)
    px = px.dropna(how="all")
    # forward-fill macro series for alignment (observation signals)
    px = px.ffill()
    px = px.dropna()
    rets = px.pct_change()
    return px, rets

def load_library_signal(factor_file):
    """Decode a library factor's signal_artifact (base64:zlib:csv) into DataFrame."""
    import base64, zlib, io
    d = json.load(open(f"factors/{factor_file}"))
    art = d["validation"]["signal_artifact"]
    raw = base64.b64decode(art["data"])
    csv_txt = zlib.decompress(raw).decode()
    df = pd.read_csv(io.StringIO(csv_txt), index_col=0, parse_dates=True)
    df.columns = art["columns"]
    return df

def factor_ic_series(factor_df, fwd_ret, horizon=10, min_valid=8):
    """Daily cross-sectional Spearman IC of factor vs forward return."""
    ics, dates = [], []
    for t in factor_df.index:
        if t not in fwd_ret.index:
            continue
        fv = factor_df.loc[t]
        fr = fwd_ret.loc[t]
        mask = fv.notna() & fr.notna()
        if mask.sum() < min_valid:
            continue
        ic = fv[mask].corr(fr[mask], method="spearman")
        if np.isfinite(ic):
            ics.append(ic)
            dates.append(t)
    return pd.Series(ics, index=dates)

def summarize_ic(ic_series, label=""):
    if len(ic_series) == 0:
        return None
    ic = ic_series.mean()
    icir = ic / ic_series.std(ddof=1) if ic_series.std(ddof=1) > 0 else 0.0
    hit = (ic_series > 0).mean()
    return {"label": label, "n_dates": len(ic_series), "ic": ic, "icir": icir,
            "hit": hit, "std": ic_series.std(ddof=1)}

def evaluate_factor(signal_df, fwd_ret_10, fwd_ret_5=None, fwd_ret_20=None, label=""):
    """Full evaluation: 10d IC/ICIR + decay + coverage."""
    out = {"label": label}
    ic10 = factor_ic_series(signal_df, fwd_ret_10, horizon=10)
    s10 = summarize_ic(ic10, label + "_h10")
    out["h10"] = s10
    if fwd_ret_5 is not None:
        ic5 = factor_ic_series(signal_df, fwd_ret_5, horizon=5)
        out["h5"] = summarize_ic(ic5, label + "_h5")
    if fwd_ret_20 is not None:
        ic20 = factor_ic_series(signal_df, fwd_ret_20, horizon=20)
        out["h20"] = summarize_ic(ic20, label + "_h20")
    # coverage
    valid_assets = signal_df.notna().mean(axis=1)
    out["coverage_asset_days"] = signal_df.notna().mean().mean()
    out["coverage_dates_ge8"] = (valid_assets >= 8).mean()
    # sub-window stability (2-year recent vs full)
    if len(ic10) > 0:
        recent = ic10[ic10.index >= "2028-01-01"]
        if len(recent) >= 20:
            out["recent_h10"] = summarize_ic(recent, label + "_recent")
        last60 = ic10[ic10.index >= "2029-10-01"]
        if len(last60) >= 20:
            out["last6m_h10"] = summarize_ic(last60, label + "_last6m")
    return out

def build_fwd_returns(rets, horizons=(5, 10, 20)):
    """Forward returns over given horizons (t -> t+h)."""
    out = {}
    for h in horizons:
        fr = pd.DataFrame(index=rets.index, columns=rets.columns, dtype=float)
        arr = rets.values
        for i in range(len(rets) - h):
            fr.iloc[i] = np.prod(1 + arr[i+1:i+1+h], axis=0) - 1
        out[h] = fr
    return out
