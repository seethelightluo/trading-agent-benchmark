"""Shared factor validation framework for miner_2.

Loads the 15-asset tradable panel + 5 macro signals, computes factor values,
cross-sectional rank IC at configurable horizons, ICIR, hit ratio, coverage,
turnover, decay, and max-abs correlation vs the persisted library.
All research is restricted to the VISIBLE window (<= visible_through).
"""
import json
import os
import numpy as np
import pandas as pd

ROOT = "../persistent"
STOCK_DIR = os.path.join(ROOT, "stock_data")
INDEX_DIR = os.path.join(ROOT, "index_data")
LIB_DIR = "factors"

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]


def load_panel(symbols=WATCH, source="stock", visible_through="2026-07-29"):
    """Return dict symbol -> DataFrame (date, close, volume...) restricted to visible window."""
    d = STOCK_DIR if source == "stock" else INDEX_DIR
    out = {}
    for s in symbols:
        fp = os.path.join(d, s + ".csv")
        if not os.path.exists(fp):
            continue
        df = pd.read_csv(fp, parse_dates=["date"])
        df = df[df["date"] <= pd.Timestamp(visible_through)].reset_index(drop=True)
        if "close" in df.columns:
            out[s] = df
    return out


def closes_panel(visible_through="2026-07-29"):
    """Return DataFrame of closes (columns = symbols, index = date) for the tradable universe."""
    frames = load_panel(WATCH, "stock", visible_through)
    closes = {s: df.set_index("date")["close"].astype(float) for s, df in frames.items()}
    return pd.DataFrame(closes).sort_index()


def macro_closes(visible_through="2026-07-29"):
    frames = load_panel(MACRO, "index", visible_through)
    closes = {s: df.set_index("date")["close"].astype(float) for s, df in frames.items()}
    return pd.DataFrame(closes).sort_index()


def ic_series(factor, fwd_ret, min_valid=8):
    """Daily cross-sectional Spearman rank IC between factor and forward return.

    factor: DataFrame (dates x assets) of factor values (aligned dates).
    fwd_ret: DataFrame (dates x assets) of forward h-day returns.
    Returns Series of IC per date (NaN where < min_valid valid pairs).
    """
    dates = factor.index.intersection(fwd_ret.index)
    ics = {}
    for d in dates:
        f = factor.loc[d]
        r = fwd_ret.loc[d]
        pair = pd.concat([f.rename("f"), r.rename("r")], axis=1).dropna()
        if len(pair) < min_valid:
            continue
        if pair["f"].nunique() < 3 or pair["r"].nunique() < 2:
            continue
        ic = pair["f"].corr(pair["r"], method="spearman")
        if np.isfinite(ic):
            ics[d] = ic
    return pd.Series(ics, dtype=float)


def forward_returns(close, h=10):
    """Forward h-day return: fwd_ret_t = close_{t+h}/close_t - 1 (visible-window safe)."""
    return close.shift(-h) / close - 1.0


def summary_metrics(ic_ser, factor, fwd_ret, close, h=10, step=10):
    """Aggregate metrics for admission."""
    ic = ic_ser.dropna()
    n = len(ic)
    if n < 30:
        return None
    ic_mean = float(ic.mean())
    ic_std = float(ic.std(ddof=1)) if n > 1 else float("nan")
    icir = float(ic.mean() / ic.std(ddof=1)) if ic_std and np.isfinite(ic_std) and ic_std > 0 else float("nan")
    hit = float((ic > 0).mean()) if ic_mean >= 0 else float((ic < 0).mean())
    # coverage: asset-days with valid factor value
    valid_mask = factor.notna()
    coverage_asset_days = float(valid_mask.sum().sum() / (factor.shape[0] * factor.shape[1])) if factor.shape[0] else 0.0
    ge8 = factor.dropna(thresh=8)
    coverage_dates_ge8 = float(len(ge8) / len(factor)) if len(factor) else 0.0
    # turnover: mean abs change of normalized ranks over `step`-day gaps
    r = factor.rank(axis=1, pct=True)
    r_step = r.shift(step)
    turn = float((r - r_step).abs().mean().mean()) if r_step.notna().any().any() else float("nan")
    # decay
    decay = {}
    for hh in (1, 2, 3, 5, 10, 20):
        if hh == h:
            decay[str(hh)] = round(ic_mean, 4)
        else:
            fr = forward_returns(close, hh)
            ics = ic_series(factor, fr, min_valid=8)
            decay[str(hh)] = round(float(ics.mean()), 4) if len(ics) else None
    return {
        "ic": round(ic_mean, 4),
        "icir": round(icir, 4) if np.isfinite(icir) else None,
        "ic_hit_ratio": round(hit, 3),
        "n_ic_dates": int(n),
        "coverage_asset_days": round(coverage_asset_days, 3),
        "coverage_dates_ge8": round(coverage_dates_ge8, 3),
        "turnover_10d_rank": round(turn, 3) if np.isfinite(turn) else None,
        "decay_ic_by_horizon": decay,
    }


def library_ic_series_map(close, h=10, min_valid=8):
    """Recompute persisted library factor signals and their IC series (for rho)."""
    import importlib.util
    lib = {}
    for fn in sorted(os.listdir(LIB_DIR)):
        if not fn.endswith(".json") or fn == "factor_ensemble.json":
            continue
        with open(os.path.join(LIB_DIR, fn)) as f:
            meta = json.load(f)
        expr = meta.get("calculation", {}).get("expression", "")
        # only handle expressions we can evaluate via simple known builders
        lib[meta["factor_id"]] = expr
    # Build factor signals per expression
    signals = {}
    ret = close.pct_change()
    for fid, expr in lib.items():
        try:
            if "close.shift(5) / close.shift(15)" in expr:
                sig = close.shift(5) / close.shift(15) - 1.0
            elif "close.shift(5) / close.shift(125)" in expr:
                sig = close.shift(5) / close.shift(125) - 1.0
            elif expr.startswith("std(pct_change,20)"):
                sig = ret.rolling(20).std().rolling(60).std()
            elif "beta(asset_ret, VIX_ret, 60)" in expr:
                vix = macro_closes()["VIX"]
                vix_ret = vix.pct_change()
                beta60 = {}
                for a in close.columns:
                    pair = pd.concat([ret[a].rename("a"), vix_ret.rename("v")], axis=1).dropna()
                    b = pair["a"].rolling(60).cov(pair["v"]) / pair["v"].rolling(60).var()
                    beta60[a] = b
                bdf = pd.DataFrame(beta60)
                sig = -bdf * (vix / vix.shift(20) - 1.0)
            else:
                continue
            sig = sig.reindex(close.index)
            signals[fid] = sig
        except Exception:
            continue
    fr = forward_returns(close, h)
    ics = {}
    for fid, sig in signals.items():
        s = ic_series(sig, fr, min_valid)
        if len(s) > 30:
            ics[fid] = s
    return ics


def max_abs_library_corr(my_ic, lib_ics):
    """Max abs Pearson correlation between candidate IC series and library IC series."""
    if lib_ics is None or len(lib_ics) == 0:
        return 0.0
    best = 0.0
    for fid, s in lib_ics.items():
        pair = pd.concat([my_ic.rename("a"), s.rename("b")], axis=1).dropna()
        if len(pair) < 30:
            continue
        r = pair["a"].corr(pair["b"])
        if np.isfinite(r):
            best = max(best, abs(float(r)))
    return round(best, 4)


def regime_split(ic_ser):
    """IC/ICIR per sub-period to assess robustness across regimes."""
    out = {}
    for name, lo, hi in [("2020-2022", "2020-01-01", "2022-12-31"),
                         ("2023-2024", "2023-01-01", "2024-12-31"),
                         ("2025-2026", "2025-01-01", "2026-07-29")]:
        s = ic_ser[(ic_ser.index >= pd.Timestamp(lo)) & (ic_ser.index <= pd.Timestamp(hi))].dropna()
        if len(s) >= 20:
            std = s.std(ddof=1)
            out[name] = {"ic": round(float(s.mean()), 4),
                         "icir": round(float(s.mean() / std), 4) if std > 0 else None,
                         "n": int(len(s))}
    return out
