"""miner_1 shared validation harness, updated for visible date 2035-12-05.
Cross-asset 15-instrument universe. Per-asset own calendar (no NaN gaps) then
reindexed to union panel. No lookahead: factor at t uses data <= t; forward
return t+1..t+h.
"""
import numpy as np
import pandas as pd
from pathlib import Path

TRADABLES = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
             "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]
VISIBLE_THROUGH = "2035-12-05"

DATA_DIR = Path("../persistent/stock_data")
INDEX_DIR = Path("../persistent/index_data")


def load_asset(symbol: str) -> pd.DataFrame:
    p = (INDEX_DIR if symbol in MACRO else DATA_DIR) / f"{symbol}.csv"
    df = pd.read_csv(p, parse_dates=["date"])
    df = df[df["date"] <= pd.Timestamp(VISIBLE_THROUGH)].copy()
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_panel() -> pd.DataFrame:
    frames = {}
    for a in TRADABLES:
        df = load_asset(a)
        frames[a] = pd.Series(df["close"].astype(float).values,
                              index=pd.to_datetime(df["date"]), name=a)
    return pd.concat(frames, axis=1).sort_index()


def load_ohlcv() -> dict:
    out = {}
    for a in TRADABLES:
        df = load_asset(a)
        out[a] = pd.DataFrame({
            "open": df["open"].astype(float).values,
            "high": df["high"].astype(float).values,
            "low": df["low"].astype(float).values,
            "close": df["close"].astype(float).values,
            "volume": df["volume"].astype(float).values,
        }, index=pd.to_datetime(df["date"]))
    return out


def macro_series(name: str) -> pd.Series:
    df = load_asset(name)
    return pd.Series(df["close"].astype(float).values, index=pd.to_datetime(df["date"]), name=name)


def per_asset(panel: pd.DataFrame, func, *args, **kwargs) -> pd.DataFrame:
    out = {}
    for a in panel.columns:
        s = panel[a].dropna()
        out[a] = func(s, *args, **kwargs).reindex(panel.index)
    return pd.DataFrame(out, index=panel.index)


def fwd_ret_series(s: pd.Series, h: int) -> pd.Series:
    return s.shift(-h) / s - 1.0


def forward_returns(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    return per_asset(panel, fwd_ret_series, horizon)


def compute_ic(factor_panel: pd.DataFrame, ret_panel: pd.DataFrame,
               min_assets: int = 8) -> pd.Series:
    dates = factor_panel.index.intersection(ret_panel.index)
    F = factor_panel.loc[dates]
    R = ret_panel.loc[dates]
    Fr = F.rank(axis=1).values
    Rr = R.rank(axis=1).values
    m = (~np.isnan(Fr)) & (~np.isnan(Rr))
    valid = m.sum(axis=1) >= min_assets
    ics = np.full(len(dates), np.nan)
    idx = np.where(valid)[0]
    for i in idx:
        f = Fr[i, m[i]]
        r = Rr[i, m[i]]
        f = f - f.mean()
        r = r - r.mean()
        denom = np.sqrt((f * f).sum() * (r * r).sum())
        ics[i] = (f * r).sum() / denom if denom > 0 else np.nan
    return pd.Series(ics, index=dates, name="ic")


def panel_rank_corr(a: pd.DataFrame, b: pd.DataFrame, min_assets: int = 8) -> float:
    dates = a.index.intersection(b.index)
    Ar = a.loc[dates].rank(axis=1).values
    Br = b.loc[dates].rank(axis=1).values
    m = (~np.isnan(Ar)) & (~np.isnan(Br))
    valid = m.sum(axis=1) >= min_assets
    cs = []
    idx = np.where(valid)[0]
    for i in idx:
        x = Ar[i, m[i]]
        y = Br[i, m[i]]
        x = x - x.mean()
        y = y - y.mean()
        denom = np.sqrt((x * x).sum() * (y * y).sum())
        if denom > 0:
            cs.append((x * y).sum() / denom)
    return float(np.mean(cs)) if cs else 0.0


def library_correlation(candidate: pd.DataFrame, library: dict, min_assets: int = 8) -> dict:
    out = {}
    for fid, sig in library.items():
        out[fid] = panel_rank_corr(candidate, sig, min_assets)
    max_abs = max((abs(v) for v in out.values()), default=0.0)
    return {"pairwise": out, "max_abs": max_abs}


def turnover_rank(factor_panel: pd.DataFrame, step: int = 10) -> float:
    ranked = factor_panel.rank(axis=1, pct=True)
    vals = []
    for i in range(step, len(ranked), step):
        a, b = ranked.iloc[i - step], ranked.iloc[i]
        m = a.notna() & b.notna()
        if m.sum() >= 8:
            vals.append(float((b[m] - a[m]).abs().mean()))
    return float(np.mean(vals)) if vals else float("nan")


def coverage_stats(factor_panel: pd.DataFrame, n_assets: int = 15, min_assets: int = 8) -> dict:
    total_cells = len(factor_panel) * n_assets
    valid_cells = int(factor_panel.notna().sum().sum())
    ge8 = int((factor_panel.notna().sum(axis=1) >= min_assets).sum())
    return {
        "coverage_asset_days": round(valid_cells / total_cells, 4),
        "coverage_dates_ge8": round(ge8 / len(factor_panel), 4),
        "n_dates_total": int(len(factor_panel)),
        "n_dates_ge8": ge8,
    }


def validate_factor(factor_panel: pd.DataFrame, panel: pd.DataFrame,
                    horizons=(1, 2, 3, 5, 10, 20), admission_horizon: int = 10,
                    library: dict = None, min_assets: int = 8,
                    fwd_cache: dict = None) -> dict:
    ic_by_h = {}
    for h in horizons:
        ret = fwd_cache.get(str(h)) if fwd_cache else None
        if ret is None:
            ret = forward_returns(panel, h)
            if fwd_cache is not None:
                fwd_cache[str(h)] = ret
        ic_by_h[str(h)] = float(compute_ic(factor_panel, ret, min_assets).mean())
    ret_a = fwd_cache.get(str(admission_horizon)) if fwd_cache else None
    if ret_a is None:
        ret_a = forward_returns(panel, admission_horizon)
        if fwd_cache is not None:
            fwd_cache[str(admission_horizon)] = ret_a
    ic_ser = compute_ic(factor_panel, ret_a, min_assets).dropna()
    ic = float(ic_ser.mean())
    icir = float(ic_ser.mean() / ic_ser.std()) if ic_ser.std() > 0 else 0.0
    hit = float((np.sign(ic_ser) == np.sign(ic)).mean()) if ic != 0 else 0.0
    cov = coverage_stats(factor_panel, min_assets=min_assets)
    to = turnover_rank(factor_panel, step=admission_horizon)
    out = {
        "ic": round(ic, 4),
        "icir": round(icir, 4),
        "ic_hit_ratio": round(hit, 3),
        "n_ic_dates": int(len(ic_ser)),
        "coverage_asset_days": cov["coverage_asset_days"],
        "coverage_dates_ge8": cov["coverage_dates_ge8"],
        "n_dates_total": cov["n_dates_total"],
        "n_dates_ge8": cov["n_dates_ge8"],
        "turnover_%d_rank" % admission_horizon: round(to, 3) if to == to else None,
        "decay_ic_by_horizon": {k: round(v, 4) for k, v in ic_by_h.items()},
    }
    if library is not None:
        lc = library_correlation(factor_panel, library, min_assets)
        out["max_abs_library_correlation"] = round(lc["max_abs"], 4)
        out["library_pairwise_corr"] = {k: round(v, 4) for k, v in lc["pairwise"].items()}
    return out


def report(name: str, metrics: dict, gate_ic: float = 0.007, gate_icir: float = 0.084):
    ic = abs(metrics.get("ic", 0.0))
    icir = abs(metrics.get("icir", 0.0))
    passed = (ic >= gate_ic) and (icir >= gate_icir)
    print(f"[{name}] IC={metrics.get('ic')} ICIR={metrics.get('icir')} "
          f"hit={metrics.get('ic_hit_ratio')} n={metrics.get('n_ic_dates')} "
          f"cov_asset={metrics.get('coverage_asset_days')} cov_dates={metrics.get('coverage_dates_ge8')} "
          f"turnover={metrics.get('turnover_10d_rank')} "
          f"maxlibcorr={metrics.get('max_abs_library_correlation')} => {'PASS' if passed else 'FAIL'}")
    return passed
