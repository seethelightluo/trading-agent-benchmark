"""miner_1 cycle 2027-01-08: explore defensive/risk-off factor family.

Context: trader feedback (memory 20270108) - ensemble still negative in current
downtrend; reversal/vol factors dragging, momentum anchor helped but
insufficient. Explore factors that identify DEFENSIVE assets (low downside
beta, safe-haven correlation, low downside-vol participation, trend breadth,
regime-conditional momentum) which should outperform during risk-off phases.

Universe: 15 tradable cross-asset instruments. Macro (VIX, DXY, USDJPY, ...)
are observation-only signals used as conditioning variables.

Gates: abs daily IC >= 0.0070, abs ICIR >= 0.0840 (15-asset cross-section).
Validates on full history 2020-01-02..2027-01-07 plus recent 2025-01-01..2027-01-07.
"""
import numpy as np
import pandas as pd

TRADABLE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
            "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
GATE_IC, GATE_ICIR = 0.0070, 0.0840


def load_panel():
    with open("scripts/panel_cache.pkl", "rb") as f:
        p = pd.read_pickle(f)
    close = p["close"]
    macro = p["macro"]
    return close, macro


def daily_rank_ic(factor, fwd_ret, min_obs=8):
    dates, ics = [], []
    for dt in factor.index:
        f, r = factor.loc[dt], fwd_ret.loc[dt]
        m = f.notna() & r.notna()
        if m.sum() < min_obs:
            continue
        ic = f[m].rank().corr(r[m].rank())
        if np.isfinite(ic):
            dates.append(dt)
            ics.append(ic)
    return pd.Series(ics, index=pd.DatetimeIndex(dates))


def summarize(ic_series):
    ics = ic_series.dropna()
    n = len(ics)
    if n == 0:
        return None
    mean_ic = ics.mean()
    std_ic = ics.std(ddof=1)
    icir = mean_ic / std_ic if std_ic > 0 else 0.0
    hit = float((ics > 0).mean()) if mean_ic > 0 else float((ics < 0).mean())
    return {"ic": round(float(mean_ic), 5), "icir": round(float(icir), 5),
            "ic_hit_ratio": round(hit, 4), "n_ic_dates": int(n)}


def turnover_10d(factor, horizon_days=10):
    rank = factor.rank(axis=1)
    return float(rank.diff(horizon_days).abs().mean().mean())


def build_library_factors(close):
    lib = {
        "rev_1d": -close.pct_change(),
        "rev_2d": -(close / close.shift(2) - 1),
        "rev_3d": -(close / close.shift(3) - 1),
        "mom_10d_skip5": close.shift(5) / close.shift(15) - 1,
        "mom_120d_skip5": close.shift(5) / close.shift(125) - 1,
        "trend_20d": close / close.rolling(20).mean() - 1,
        "trend_60d": close / close.rolling(60).mean() - 1,
        "nclv_1d": -(close / close.shift(2) - 1) * close.pct_change().abs(),
    }
    return lib


def library_corr(factor, lib, min_obs=8):
    out = {}
    for name, lf in lib.items():
        corrs = []
        for dt in factor.index:
            if dt not in lf.index:
                continue
            f, g = factor.loc[dt], lf.loc[dt]
            m = f.notna() & g.notna()
            if m.sum() < min_obs:
                continue
            c = f[m].rank().corr(g[m].rank())
            if np.isfinite(c):
                corrs.append(c)
        out[name] = round(float(np.mean(corrs)), 4) if corrs else None
    return out


def validate(factor_panel, close, label, win_start=None, win_end=None):
    fp, cl = factor_panel, close
    if win_start is not None:
        fp = fp[fp.index >= win_start]
        cl = cl[cl.index >= win_start]
    if win_end is not None:
        fp = fp[fp.index <= win_end]
        cl = cl[cl.index <= win_end]
    results = {}
    for h in (1, 2, 3, 5, 10, 20):
        fwd = cl.shift(-h) / cl - 1.0
        ic_series = daily_rank_ic(fp, fwd)
        results[h] = summarize(ic_series)
    admitted = None
    for h in (1, 2, 3, 5, 10):
        r = results[h]
        if r and abs(r["ic"]) >= GATE_IC and abs(r["icir"]) >= GATE_ICIR:
            if admitted is None or abs(r["icir"]) > abs(results[admitted]["icir"]):
                admitted = h
    cov_assets = float(fp.notna().mean().mean())
    cov_dates_ge8 = float(fp.notna().sum(axis=1).ge(8).mean())
    to10 = turnover_10d(fp)
    lib = build_library_factors(close)
    lc = library_corr(fp, lib)
    max_abs_lc = max((abs(v) for v in lc.values() if v is not None), default=0.0)
    print(f"\n=== {label}  [{fp.index[0].date()} .. {fp.index[-1].date()}] n_dates={len(fp)} n_assets={fp.shape[1]}")
    for h, r in results.items():
        if r:
            print(f"  h={h:2d}  IC={r['ic']:+.5f}  ICIR={r['icir']:+.5f}  hit={r['ic_hit_ratio']:.3f}  n={r['n_ic_dates']}")
    print(f"  >> ADMISSION h={admitted}" + (f" IC={results[admitted]['ic']:+.5f} ICIR={results[admitted]['icir']:+.5f}" if admitted else " (none)"))
    print(f"  coverage_asset_days={cov_assets:.3f} cov_dates_ge8={cov_dates_ge8:.3f} turnover_10d_rank={to10:.3f}")
    print(f"  library_corr={lc}  max_abs={max_abs_lc:.4f}")
    return {"label": label, "admitted": admitted, "results": results, "max_abs_lc": max_abs_lc}


def main():
    close, macro = load_panel()
    ret = close.pct_change()
    print(f"close panel: {close.shape}  [{close.index[0].date()} .. {close.index[-1].date()}]")
    print(f"macro panel: {macro.shape}  [{macro.index[0].date()} .. {macro.index[-1].date()}]")

    # benchmark equity returns (SPX) for beta computations
    spx_ret = ret["SPX"]
    eq_ret = ret[["SPX", "NDX", "SX5E", "N225", "HSI", "000300.SH"]].mean(axis=1)

    vol20 = ret.rolling(20).std()
    vol60 = ret.rolling(60).std()

    # --- downside beta to SPX (only SPX down days) ---
    down_mask = (spx_ret < 0)
    cov_down = ret.where(down_mask, 0.0).rolling(120).cov(spx_ret.where(down_mask, 0.0))
    var_down = spx_ret.where(down_mask, 0.0).rolling(120).var()
    downbeta_spx_120 = cov_down.div(var_down, axis=0)

    # --- downside beta to equal-weight equity basket ---
    down_mask_eq = (eq_ret < 0)
    cov_down_eq = ret.where(down_mask_eq, 0.0).rolling(120).cov(eq_ret.where(down_mask_eq, 0.0))
    var_down_eq = eq_ret.where(down_mask_eq, 0.0).rolling(120).var()
    downbeta_eq_120 = cov_down_eq.div(var_down_eq, axis=0)

    # --- full-sample beta to SPX (60d) ---
    cov60 = ret.rolling(60).cov(spx_ret)
    var60 = spx_ret.rolling(60).var()
    beta_spx_60 = cov60.div(var60, axis=0)

    # --- safe-haven correlation: corr(asset ret, VIX chg) over 60d ---
    vix_chg = macro["VIX"].pct_change()
    vix_chg_aligned = vix_chg.reindex(ret.index).ffill()
    safe_corr_60 = ret.rolling(60).corr(vix_chg_aligned)

    # --- downside volatility ratio: semi-deviation / total vol ---
    neg_ret = ret.clip(upper=0.0)
    semi_vol = np.sqrt((neg_ret ** 2).rolling(60).mean())
    down_vol_ratio = semi_vol / vol60

    # --- trend breadth: fraction of positive days over 60d ---
    pos_cnt = (ret > 0).rolling(60).sum()
    trend_breadth_60 = pos_cnt / 60.0

    # --- momentum z-score: 20d momentum standardized by 120d vol ---
    mom20 = close / close.shift(20) - 1
    mom20_z = mom20 / (vol60 * np.sqrt(20))

    # --- regime-conditional momentum: 60d mom * (VIX below 120d median) ---
    mom60 = close / close.shift(60) - 1
    vix_med = vix_chg_aligned.rolling(120).median()
    vix_low = (macro["VIX"].reindex(ret.index).ffill() <= vix_med).astype(float)
    mom60_vixlow = mom60 * vix_low

    # --- trend-consistency weighted momentum (breadth * mom) ---
    mom60_breadth = mom60 * trend_breadth_60

    # --- yield direction: assets that gain when US10Y falls (rates down = risk-on bonds) ---
    us10y_ret = ret["US10Y"]
    rate_down = (us10y_ret < 0)
    cov_rd = ret.where(rate_down, 0.0).rolling(120).cov(us10y_ret.where(rate_down, 0.0))
    var_rd = us10y_ret.where(rate_down, 0.0).rolling(120).var()
    ratebeta_120 = cov_rd.div(var_rd, axis=0)

    candidates = {
        "downbeta_spx_120": downbeta_spx_120,
        "downbeta_eq_120": downbeta_eq_120,
        "beta_spx_60": beta_spx_60,
        "safe_corr_vix_60": safe_corr_60,
        "down_vol_ratio_60": down_vol_ratio,
        "trend_breadth_60": trend_breadth_60,
        "mom20_z_60vol": mom20_z,
        "mom60_vixlow": mom60_vixlow,
        "mom60_breadth": mom60_breadth,
        "ratebeta_down_120": ratebeta_120,
    }

    print("\n########## FULL WINDOW VALIDATION ##########")
    full = {}
    for name, fac in candidates.items():
        fac = fac.replace([np.inf, -np.inf], np.nan)
        full[name] = validate(fac, close, name)

    print("\n########## RECENT WINDOW 2025-01-01..2027-01-07 ##########")
    recent = {}
    for name, fac in candidates.items():
        fac = fac.replace([np.inf, -np.inf], np.nan)
        recent[name] = validate(fac, close, name + "_recent", win_start="2025-01-01")

    print("\n########## SUMMARY ##########")
    for name in candidates:
        f, r = full.get(name), recent.get(name)
        fa = f["admitted"] if f else None
        ra = r["admitted"] if r else None
        print(f"{name:22s} full_h={fa} recent_h={ra}")


if __name__ == "__main__":
    main()
