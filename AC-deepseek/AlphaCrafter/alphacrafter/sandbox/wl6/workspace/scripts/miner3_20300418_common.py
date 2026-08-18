"""Shared data loading & fast factor evaluation utilities for miner_3 (2030-04-18 cycle)."""
import pandas as pd
import numpy as np
import json

TRADABLE = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
MACRO = ["DXY","USDCNY","USDJPY","EURUSD","VIX"]
VISIBLE_THROUGH = "2030-04-17"

def load_asset_panel():
    """Load all tradable + macro close prices aligned on common trading dates."""
    closes = {}
    for sym in TRADABLE + MACRO:
        path = f"../persistent/stock_data/{sym}.csv" if sym in TRADABLE else f"../persistent/index_data/{sym}.csv"
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= pd.Timestamp(VISIBLE_THROUGH)].sort_values("date")
        closes[sym] = df.set_index("date")["close"]
    px = pd.DataFrame(closes)
    px = px.ffill()
    px = px.dropna()
    rets = px.pct_change()
    return px, rets

def build_fwd_returns(rets, horizons=(5, 10, 20)):
    """Forward returns t -> t+h (vectorized via shift)."""
    out = {}
    for h in horizons:
        out[h] = rets.shift(-h).rolling(h).apply(lambda x: np.prod(1 + x) - 1, raw=True)
    return out

def fast_rank_ic_matrix(factor_df, fwd_df, min_valid=8):
    """
    Vectorized daily cross-sectional Spearman IC.
    Ranks per row (per date) then Pearson correlation of ranks across dates.
    Returns Series of daily IC indexed by date.
    """
    common_idx = factor_df.index.intersection(fwd_df.index)
    if len(common_idx) == 0:
        return pd.Series(dtype=float)
    F = factor_df.loc[common_idx]
    Y = fwd_df.loc[common_idx]
    valid = F.notna() & Y.notna()
    # rank across assets per date (np.argsort based), nan-safe
    Fr = pd.DataFrame(np.nan, index=F.index, columns=F.columns)
    Yr = pd.DataFrame(np.nan, index=Y.index, columns=Y.columns)
    for i in range(len(F)):
        frow = F.iloc[i].values
        yrow = Y.iloc[i].values
        m = valid.iloc[i].values
        if m.sum() < min_valid:
            continue
        fr = np.full(len(frow), np.nan)
        yr = np.full(len(yrow), np.nan)
        fr[m] = pd.Series(frow[m]).rank().values
        yr[m] = pd.Series(yrow[m]).rank().values
        Fr.iloc[i] = fr
        Yr.iloc[i] = yr
    # Pearson correlation per row (vectorized over dates)
    Fm = Fr.sub(Fr.mean(axis=1), axis=0)
    Ym = Yr.sub(Yr.mean(axis=1), axis=0)
    num = (Fm * Ym).sum(axis=1)
    den = np.sqrt((Fm ** 2).sum(axis=1) * (Ym ** 2).sum(axis=1))
    ic = num / den
    return ic.replace([np.inf, -np.inf], np.nan).dropna()

def summarize_ic(ic_series, label=""):
    if ic_series is None or len(ic_series) == 0:
        return None
    ic = ic_series.mean()
    std = ic_series.std(ddof=1)
    icir = ic / std if std > 0 else 0.0
    hit = (ic_series > 0).mean()
    return {"label": label, "n_dates": len(ic_series), "ic": ic, "icir": icir,
            "hit": hit, "std": std}

def evaluate_factor(signal_df, fwd_map, label="", min_valid=8):
    """Full evaluation: IC/ICIR at h5/h10/h20 + coverage + subwindows."""
    out = {"label": label}
    ic10 = fast_rank_ic_matrix(signal_df, fwd_map[10], min_valid=min_valid)
    out["h10"] = summarize_ic(ic10, label + "_h10")
    for h in (5, 20):
        if h in fwd_map:
            ic_h = fast_rank_ic_matrix(signal_df, fwd_map[h], min_valid=min_valid)
            out[f"h{h}"] = summarize_ic(ic_h, label + f"_h{h}")
    valid_assets = signal_df.notna().mean(axis=1)
    out["coverage_asset_days"] = float(signal_df.notna().mean().mean())
    out["coverage_dates_ge8"] = float((valid_assets >= 8).mean())
    if len(ic10) > 0:
        recent = ic10[ic10.index >= "2028-01-01"]
        if len(recent) >= 20:
            out["recent_h10"] = summarize_ic(recent, label + "_recent")
        last6m = ic10[ic10.index >= "2029-10-01"]
        if len(last6m) >= 20:
            out["last6m_h10"] = summarize_ic(last6m, label + "_last6m")
    return out

def load_library_signals():
    """Load all effective library factor signal_artifacts into dict {factor_id: DataFrame}."""
    import base64, zlib, io, glob, os
    out = {}
    for fp in sorted(glob.glob("factors/*.json")):
        if fp.endswith(".bak") or "ensemble" in fp or "evicted" in fp or "quarantine" in fp:
            continue
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        fid = d.get("factor_id")
        art = (d.get("validation") or {}).get("signal_artifact")
        if not fid or not art:
            continue
        try:
            raw = base64.b64decode(art["data"])
            csv_txt = zlib.decompress(raw).decode()
            df = pd.read_csv(io.StringIO(csv_txt), index_col=0, parse_dates=True)
            df.columns = art["columns"]
            out[fid] = df
        except Exception:
            continue
    return out

def library_correlation(new_sig, lib_signals, sample_dates=300):
    """
    Pooled Pearson correlation of new factor values vs each library factor over
    overlapping (date, asset) pairs, sampled for speed. Returns (max_abs, detail).
    """
    rows = []
    for fid, lsig in lib_signals.items():
        common = new_sig.index.intersection(lsig.index)
        if len(common) < 60:
            continue
        idx = common
        if len(common) > sample_dates:
            step = len(common) // sample_dates
            idx = common[::step]
        a = new_sig.loc[idx].values.ravel()
        b = lsig.loc[idx].values.ravel()
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 500:
            continue
        rho = float(np.corrcoef(a[m], b[m])[0, 1])
        rows.append((fid, rho))
    if not rows:
        return None, {}
    detail = {fid: round(rho, 4) for fid, rho in rows}
    max_abs = max(abs(rho) for _, rho in rows)
    return max_abs, detail
