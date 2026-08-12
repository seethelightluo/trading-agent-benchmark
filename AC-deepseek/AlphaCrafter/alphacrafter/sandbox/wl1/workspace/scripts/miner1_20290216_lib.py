"""Shared helper for miner_1 factor research (2029-02-16 cycle).

Data scope: trading days <= visible_through (2029-02-15) ONLY. No future data.
Provides: load_prices(), factor IC analysis (daily cross-sectional Spearman IC,
ICIR, hit ratio, coverage, turnover, decay by horizon), library signal correlation.
"""
import json
import numpy as np
import pandas as pd

VISIBLE = "2029-02-15"
ASSETS = ["000300.SH", "000688.SH", "BTC", "CN10Y", "COPPER", "ETH", "HSI",
          "N225", "NDX", "SOX", "SPX", "SX5E", "US10Y", "WTI", "XAU"]
OBS = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]
LIB_FACTORS = ["miner2_20260715_nbody_1d", "miner2_20260715_rev_5d",
               "miner2_20260715_id_rev_1d", "miner2_20260715_nclv_5d",
               "mom_120d_skip5", "vol_of_vol20x60", "miner2_20260715_nclv_3d",
               "miner2_20260715_rev_2d", "miner2_20260715_rev_1d_vs",
               "miner2_20260715_nclv_2d", "miner2_20260715_nclv_1d",
               "miner2_20260715_rev_3d", "miner2_20260715_rev_1d",
               "vix_beta_cond_60x20"]


def trading_days():
    d = json.load(open("../persistent/date.json"))
    return [x for x in d["trading_days"] if x <= VISIBLE]


def load_prices():
    """Return dict asset -> DataFrame indexed by trading day with close/ret/vol."""
    days = trading_days()
    idx = pd.Index(days, name="date")
    out = {}
    for a in ASSETS:
        df = pd.read_csv(f"../persistent/stock_data/{a}.csv")
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df = df[df["date"] <= VISIBLE].set_index("date")
        df = df.reindex(idx).ffill()
        close = df["close"].astype(float)
        ret = close.pct_change()
        vol = df["volume"].astype(float) if "volume" in df else pd.Series(np.nan, index=idx)
        out[a] = pd.DataFrame({"close": close, "ret": ret, "volume": vol,
                               "high": df["high"].astype(float),
                               "low": df["low"].astype(float),
                               "open": df["open"].astype(float)}, index=idx)
    return out


def load_obs():
    """Return dict obs -> Series indexed by trading day."""
    days = trading_days()
    idx = pd.Index(days, name="date")
    out = {}
    for o in OBS:
        df = pd.read_csv(f"../persistent/index_data/{o}.csv")
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df = df[df["date"] <= VISIBLE].set_index("date")
        s = df["close"].astype(float).reindex(idx).ffill()
        out[o] = s
    return out


def factor_panel(fn, frames, obs=None, min_valid=8):
    """Compute factor values per asset on the common calendar.
    fn(asset_df, obs_dict, asset) -> pd.Series indexed by date (NaN where undefined).
    Returns DataFrame date x asset and a mask of dates with >= min_valid values.
    """
    cols = {}
    for a in ASSETS:
        try:
            s = fn(frames[a], obs, a)
        except Exception as e:  # noqa
            print(f"  WARN {a}: {e}")
            s = pd.Series(np.nan, index=frames[a].index)
        s = pd.Series(s, index=frames[a].index).astype(float)
        cols[a] = s
    panel = pd.DataFrame(cols)
    valid = panel.notna().sum(axis=1)
    good_dates = valid[valid >= min_valid].index
    return panel, good_dates


def ic_analysis(panel, good_dates, frames, horizons=(1, 2, 3, 5, 10, 20),
                adm_horizon=10, min_valid=8, direction=1):
    """Daily cross-sectional Spearman IC between factor panel and forward returns."""
    res = {}
    for h in horizons:
        fwd = pd.DataFrame({a: frames[a]["close"].shift(-h) / frames[a]["close"] - 1.0
                            for a in ASSETS})
        ics = []
        for dt in good_dates:
            if dt not in fwd.index:
                continue
            fv = panel.loc[dt]
            fr = fwd.loc[dt]
            m = fv.notna() & fr.notna()
            if m.sum() < min_valid:
                continue
            ic = fv[m].corr(fr[m], method="spearman")
            if np.isfinite(ic):
                ics.append(ic)
        ics = np.array(ics)
        if len(ics) == 0:
            res[h] = {"ic": np.nan, "icir": np.nan, "hit": np.nan, "n": 0}
        else:
            ic_mean = float(np.nanmean(ics))
            ic_std = float(np.nanstd(ics))
            icir = ic_mean / ic_std if ic_std > 0 else 0.0
            hit = float(np.mean(ics * direction > 0))
            res[h] = {"ic": ic_mean, "icir": icir, "hit": hit, "n": int(len(ics))}
    cov_asset_days = float(panel.notna().sum().sum() / (panel.shape[0] * panel.shape[1]))
    n_dates_ge8 = int((panel.notna().sum(axis=1) >= min_valid).sum())
    ranks = panel.rank(axis=1)
    rank_diff = ranks.diff().abs().mean(axis=1)
    turnover_10d = float(rank_diff[good_dates].mean() * 10.0) if len(good_dates) else np.nan
    adm = res[adm_horizon]
    return {
        "admission_horizon": adm_horizon,
        "by_horizon": {str(k): v for k, v in res.items()},
        "adm_ic": adm["ic"], "adm_icir": adm["icir"], "adm_hit": adm["hit"],
        "adm_n_dates": adm["n"],
        "coverage_asset_days": cov_asset_days,
        "n_dates_ge8": int(n_dates_ge8),
        "n_total_dates": int(len(good_dates)),
        "turnover_10d_rank": turnover_10d,
    }


def recent_metrics(panel, good_dates, frames, start="2026-07-16", adm_horizon=10):
    sub = panel[panel.index >= start]
    sub_good = good_dates[good_dates >= start]
    return ic_analysis(sub, sub_good, frames, horizons=(adm_horizon,),
                       adm_horizon=adm_horizon)


def print_metrics(tag, m):
    print(f"\n=== {tag} ===")
    print(f"adm(h{m['admission_horizon']}) IC={m['adm_ic']:.4f} ICIR={m['adm_icir']:.4f} "
          f"hit={m['adm_hit']:.3f} n_dates={m['adm_n_dates']}")
    for h in sorted(m["by_horizon"], key=int):
        v = m["by_horizon"][h]
        print(f"  h={h:>2}: IC={v['ic']:.4f} ICIR={v['icir']:+.4f} hit={v['hit']:.3f} n={v['n']}")
    print(f"  coverage_asset_days={m['coverage_asset_days']:.3f} "
          f"n_dates_ge8={m['n_dates_ge8']}/{m['n_total_dates']} "
          f"turnover_10d_rank={m['turnover_10d_rank']:.3f}")


def library_signals(frames, obs=None):
    """Reconstruct library factor signals (date x asset) from persisted JSON definitions.
    Returns dict factor_id -> panel. Handles known factor families."""
    out = {}

    def _fwd_ret(df, n):
        return df["close"].shift(-n) / df["close"] - 1.0

    def _mom(df, n, skip):
        return df["close"].shift(skip) / df["close"].shift(skip + n) - 1.0

    def _rev(df, n):
        return -(df["close"] / df["close"].shift(n) - 1.0)

    def _nclv(df, n):
        return -(np.log(df["close"]) - np.log(df["close"].shift(n)))

    def _nbody(df):
        return -(df["close"] - df["open"]) / df["close"].shift(1)

    def _idrev(df):
        return (df["high"] / df["close"] - 1.0) * np.sign(df["close"] - df["open"])

    def _rev_vs(df):
        return -(df["close"] / df["close"].shift(2) - 1.0) * (df["close"] > df["close"].shift(2))

    panels = {}
    for a in ASSETS:
        df = frames[a]
        panels[a] = pd.DataFrame({
            "miner2_20260715_rev_1d": _rev(df, 1),
            "miner2_20260715_rev_2d": _rev(df, 2),
            "miner2_20260715_rev_3d": _rev(df, 3),
            "miner2_20260715_rev_5d": _rev(df, 5),
            "miner2_20260715_rev_1d_vs": _rev_vs(df),
            "miner2_20260715_nclv_1d": _nclv(df, 1),
            "miner2_20260715_nclv_2d": _nclv(df, 2),
            "miner2_20260715_nclv_3d": _nclv(df, 3),
            "miner2_20260715_nclv_5d": _nclv(df, 5),
            "miner2_20260715_nbody_1d": _nbody(df),
            "miner2_20260715_id_rev_1d": _idrev(df),
            "mom_120d_skip5": _mom(df, 120, 5),
        }, index=df.index)
    for fid in panels[a].columns:
        out[fid] = pd.DataFrame({a: panels[a][fid] for a in ASSETS})

    # vol_of_vol20x60: std of 20d realized vol vs 60d
    pov = {}
    for a in ASSETS:
        rv20 = frames[a]["ret"].rolling(20).std()
        rv60 = frames[a]["ret"].rolling(60).std()
        pov[a] = rv20 / rv60 - 1.0
    out["vol_of_vol20x60"] = pd.DataFrame(pov)

    # vix_beta_cond_60x20: beta of asset to VIX changes (60d), conditioned on VIX up regime
    if obs is not None and "VIX" in obs:
        vix = obs["VIX"]
        dvi = vix.diff()
        pvb = {}
        for a in ASSETS:
            r = frames[a]["ret"]
            cov = r.rolling(60).cov(dvi)
            var = dvi.rolling(60).var()
            beta = cov / var
            cond = (vix.diff(20) > 0).astype(float)
            pvb[a] = beta * cond
        out["vix_beta_cond_60x20"] = pd.DataFrame(pvb)

    return out


def max_lib_corr(cand_panel, lib_signals, good_dates, min_valid=8):
    """Max absolute cross-sectional signal correlation with library factors."""
    best = 0.0
    best_fid = None
    for fid, lp in lib_signals.items():
        if lp is None:
            continue
        corrs = []
        for dt in good_dates:
            if dt not in lp.index:
                continue
            a = cand_panel.loc[dt]
            b = lp.loc[dt]
            m = a.notna() & b.notna()
            if m.sum() < min_valid:
                continue
            c = a[m].corr(b[m], method="spearman")
            if np.isfinite(c):
                corrs.append(c)
        if corrs:
            mc = float(np.nanmean(np.abs(corrs)))
            if mc > best:
                best = mc
                best_fid = fid
    return best, best_fid
