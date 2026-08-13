"""miner_1 validation harness updated for visible date 2031-11-26 (sim date 2031-11-27).

Data source: ../persistent/stock_data/*.csv and ../persistent/index_data/*.csv
Visible cutoff: date <= 2031-11-26 (previous completed trading day).
No lookahead: factor at t uses data up to t; forward return uses t+1..t+h.
Each asset has its own trading calendar; factor/fwd-return computed per-asset on its OWN
calendar (no NaN gaps), then reindexed to the union panel index for cross-sectional IC.
"""
import numpy as np
import pandas as pd
from pathlib import Path

TRADABLES = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
             "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]
VISIBLE_THROUGH = "2031-11-26"

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
        "turnover_10d_rank": round(to, 3) if to == to else None,
        "decay_ic_by_horizon": {k: round(v, 4) for k, v in ic_by_h.items()},
    }
    if library is not None:
        lc = library_correlation(factor_panel, library, min_assets)
        out["max_abs_library_correlation"] = round(lc["max_abs"], 4)
        out["library_pairwise_corr"] = {k: round(v, 4) for k, v in lc["pairwise"].items()}
    return out


def regime_split_ic(factor_panel: pd.DataFrame, ret_panel: pd.DataFrame,
                    admission_horizon: int = 10, min_assets: int = 8) -> str:
    ic_ser = compute_ic(factor_panel, ret_panel, min_assets).dropna()
    parts = []
    splits = [("2020-2021", "2020-01-01", "2021-12-31"),
              ("2022", "2022-01-01", "2022-12-31"),
              ("2023-2024", "2023-01-01", "2024-12-31"),
              ("2025-2026", "2025-01-01", "2026-12-31"),
              ("2027-2028", "2027-01-01", "2028-12-31"),
              ("2029-2031", "2029-01-01", "2031-12-31")]
    for name, lo, hi in splits:
        sub = ic_ser[(ic_ser.index >= lo) & (ic_ser.index <= hi)]
        if len(sub) >= 20:
            parts.append(f"{name}:ic={sub.mean():+.4f} icir={sub.mean()/sub.std():+.3f} n={len(sub)}")
        else:
            parts.append(f"{name}:n={len(sub)}")
    last = ic_ser.tail(250)
    if len(last) >= 20:
        parts.append(f"last250:ic={last.mean():+.4f} icir={last.mean()/last.std():+.3f} n={len(last)}")
    return " | ".join(parts)


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


def build_active_library(panel: pd.DataFrame) -> dict:
    """Recompute the 5 currently-effective factor signals (per-asset calendars, aligned)."""
    close = panel
    sig = {}
    # max_consec_gain_20: max length of consecutive positive daily returns in 20d window
    def max_consec_gain(s, win=20):
        r = (s.pct_change() > 0).astype(float)
        out = pd.Series(np.nan, index=s.index)
        vals = r.values
        for i in range(len(s)):
            if i < 1:
                continue
            j0 = max(0, i - win + 1)
            cnt = 0
            mx = 0
            for j in range(i, j0 - 1, -1):
                if vals[j] == 1:
                    cnt += 1
                    mx = max(mx, cnt)
                else:
                    cnt = 0
            out.iloc[i] = mx
        return out
    sig["max_consec_gain_20"] = per_asset(close, max_consec_gain, 20)

    # mom_180d_skip5
    sig["mom_180d_skip5"] = per_asset(close, lambda s: s.shift(5) / s.shift(185) - 1.0)

    # range_pos_252: (close - 252d min) / (252d max - 252d min)
    def range_pos(s, win=252):
        mn = s.rolling(win).min()
        mx = s.rolling(win).max()
        return (s - mn) / (mx - mn)
    sig["range_pos_252"] = per_asset(close, range_pos, 252)

    # spx_corr60: rolling 60d correlation of asset returns with SPX returns
    spx = close["SPX"].dropna().pct_change()
    corr_parts = {}
    for a in close.columns:
        s = close[a].dropna()
        ar = s.pct_change()
        df = pd.concat([ar.rename("a"), spx.reindex(ar.index).rename("s")], axis=1).dropna()
        c = df["a"].rolling(60).corr(df["s"])
        corr_parts[a] = c.reindex(panel.index)
    sig["spx_corr60"] = pd.DataFrame(corr_parts, index=panel.index)

    # downbeta_spx_60: rolling beta of asset returns on SPX returns restricted to SPX-down days
    spx_ret = spx
    down_parts = {}
    for a in close.columns:
        s = close[a].dropna()
        ar = s.pct_change()
        df = pd.concat([ar.rename("a"), spx_ret.reindex(ar.index).rename("s")], axis=1).dropna()
        d = df[df["s"] < 0]
        b = d["a"].rolling(60).cov(d["s"]) / d["s"].rolling(60).var()
        b[d["s"].rolling(60).count() < 15] = np.nan
        down_parts[a] = b.reindex(panel.index)
    sig["downbeta_spx_60"] = pd.DataFrame(down_parts, index=panel.index)
    return sig
