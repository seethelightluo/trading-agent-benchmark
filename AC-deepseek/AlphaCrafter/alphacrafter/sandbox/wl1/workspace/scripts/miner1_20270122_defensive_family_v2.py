"""miner_1 cycle 2027-01-22: defensive/risk-off factor family v2 (vectorized).

Context: trader feedback - ensemble still negative in downtrend; reversal/vol
factors dragging, momentum anchor insufficient. Goal: factors identifying
DEFENSIVE assets (low downside beta, safe-haven correlation, low drawdown,
positive risk-adjusted momentum) for long-only 15-asset cross-asset universe.

Gates: abs daily IC >= 0.0070, abs ICIR >= 0.0840 (15-asset cross-section).
Full window 2020-01-02..2027-01-21 and recent 2025-01-01..2027-01-21.
"""
import numpy as np
import pandas as pd

TRADABLE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
            "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
GATE_IC, GATE_ICIR = 0.0070, 0.0840


def load_panel():
    with open("scripts/panel_cache.pkl", "rb") as f:
        p = pd.read_pickle(f)
    return p["close"], p["macro"]


def rank_ic_panel(factor, fwd, min_obs=8):
    """Vectorized daily cross-sectional rank IC between factor and fwd returns."""
    f = factor.rank(axis=1)
    r = fwd.rank(axis=1)
    valid = factor.notna() & fwd.notna()
    n_valid = valid.sum(axis=1)
    # mask rows with < min_obs
    f = f.where(valid)
    r = r.where(valid)
    ic = f.corrwith(r, axis=1)
    ic = ic[n_valid >= min_obs].dropna()
    return ic


def summarize(ic_series):
    ics = ic_series.dropna()
    n = len(ics)
    if n == 0:
        return None
    mean_ic = float(ics.mean())
    std_ic = float(ics.std(ddof=1))
    icir = mean_ic / std_ic if std_ic > 0 else 0.0
    hit = float((ics > 0).mean()) if mean_ic > 0 else float((ics < 0).mean())
    return {"ic": round(mean_ic, 5), "icir": round(icir, 5),
            "ic_hit_ratio": round(hit, 4), "n_ic_dates": int(n)}


def turnover_10d(factor):
    rank = factor.rank(axis=1)
    return float(rank.diff(10).abs().mean().mean())


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


def library_corr(factor, lib, min_obs=8, step=5):
    """Mean daily rank corr vs library factors, sampled every `step` dates."""
    out = {}
    for name, lf in lib.items():
        f = factor.rank(axis=1)
        g = lf.rank(axis=1)
        both = factor.notna() & lf.notna()
        n_valid = both.sum(axis=1)
        f = f.where(both)
        g = g.where(both)
        c = f.corrwith(g, axis=1)
        c = c[(n_valid >= min_obs) & (factor.index % step == 0) if False else (n_valid >= min_obs)]
        c = c.iloc[::step].dropna()
        out[name] = round(float(c.mean()), 4) if len(c) else None
    return out


def validate(factor_panel, close, fwd_cache, label, win_start=None, win_end=None):
    fp = factor_panel
    if win_start is not None:
        fp = fp[fp.index >= win_start]
    if win_end is not None:
        fp = fp[fp.index <= win_end]
    results = {}
    for h in (1, 2, 3, 5, 10, 20):
        ic_series = rank_ic_panel(fp, fwd_cache[h].reindex(fp.index))
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
    print(f"\n=== {label}  [{fp.index[0].date()} .. {fp.index[-1].date()}] n_dates={len(fp)} n_assets={fp.shape[1]}")
    for h, r in results.items():
        if r:
            print(f"  h={h:2d}  IC={r['ic']:+.5f}  ICIR={r['icir']:+.5f}  hit={r['ic_hit_ratio']:.3f}  n={r['n_ic_dates']}")
    print(f"  >> ADMISSION h={admitted}" + (f" IC={results[admitted]['ic']:+.5f} ICIR={results[admitted]['icir']:+.5f}" if admitted else " (none)"))
    print(f"  coverage_asset_days={cov_assets:.3f} cov_dates_ge8={cov_dates_ge8:.3f} turnover_10d_rank={to10:.3f}")
    return {"label": label, "admitted": admitted, "results": results}


def main():
    close, macro = load_panel()
    ret = close.pct_change()
    print(f"close panel: {close.shape}  [{close.index[0].date()} .. {close.index[-1].date()}]")
    print(f"macro panel: {macro.shape}  [{macro.index[0].date()} .. {macro.index[-1].date()}]")

    # forward return cache
    fwd_cache = {h: close.shift(-h) / close - 1.0 for h in (1, 2, 3, 5, 10, 20)}

    spx_ret = ret["SPX"]
    eq_ret = ret[["SPX", "NDX", "SX5E", "N225", "HSI", "000300.SH"]].mean(axis=1)
    vol20 = ret.rolling(20).std()
    vol60 = ret.rolling(60).std()
    vol120 = ret.rolling(120).std()

    # --- downside beta to SPX (only SPX down days) ---
    down_mask = (spx_ret < 0)
    cov_down = ret.where(down_mask, 0.0).rolling(120).cov(spx_ret.where(down_mask, 0.0))
    var_down = spx_ret.where(down_mask, 0.0).rolling(120).var()
    downbeta_spx_120 = cov_down.div(var_down, axis=0)

    # --- downside beta to equity basket ---
    down_mask_eq = (eq_ret < 0)
    cov_down_eq = ret.where(down_mask_eq, 0.0).rolling(120).cov(eq_ret.where(down_mask_eq, 0.0))
    var_down_eq = eq_ret.where(down_mask_eq, 0.0).rolling(120).var()
    downbeta_eq_120 = cov_down_eq.div(var_down_eq, axis=0)

    # --- full beta to SPX 60d ---
    beta_spx_60 = ret.rolling(60).cov(spx_ret).div(spx_ret.rolling(60).var(), axis=0)

    # --- safe-haven correlation: corr(asset ret, VIX chg) 60d ---
    vix_chg = macro["VIX"].pct_change().reindex(ret.index).ffill()
    safe_corr_vix_60 = ret.rolling(60).corr(vix_chg)

    # --- VIX beta 60d ---
    cov_vix = ret.rolling(60).cov(vix_chg)
    var_vix = vix_chg.rolling(60).var()
    vix_beta_60 = cov_vix.div(var_vix, axis=0)

    # --- downside vol ratio ---
    neg_ret = ret.clip(upper=0.0)
    semi_vol = np.sqrt((neg_ret ** 2).rolling(60).mean())
    down_vol_ratio_60 = semi_vol / vol60

    # --- trend breadth 60d ---
    trend_breadth_60 = (ret > 0).rolling(60).sum() / 60.0

    # --- momentum z-score ---
    mom20 = close / close.shift(20) - 1
    mom20_z_60vol = mom20 / (vol60 * np.sqrt(20))

    # --- regime-conditional momentum (VIX low) ---
    mom60 = close / close.shift(60) - 1
    vix_med = macro["VIX"].reindex(ret.index).ffill().rolling(120).median()
    vix_low = (macro["VIX"].reindex(ret.index).ffill() <= vix_med).astype(float)
    mom60_vixlow = mom60 * vix_low

    # --- momentum * breadth ---
    mom60_breadth = mom60 * trend_breadth_60

    # --- beta to US10Y on rate-down days ---
    us10y_ret = ret["US10Y"]
    rate_down = (us10y_ret < 0)
    cov_rd = ret.where(rate_down, 0.0).rolling(120).cov(us10y_ret.where(rate_down, 0.0))
    var_rd = us10y_ret.where(rate_down, 0.0).rolling(120).var()
    ratebeta_down_120 = cov_rd.div(var_rd, axis=0)

    # === NEW v2 candidates ===
    # drawdown from 252d high (less negative = more defensive)
    dd_252 = close / close.rolling(252).max() - 1.0

    # rolling skewness 60d (negative skew = tail risk)
    skew_60 = ret.rolling(60).skew()

    # gold correlation 60d (safe-haven affinity)
    xau_ret = ret["XAU"]
    xau_corr_60 = ret.rolling(60).corr(xau_ret)

    # up/down capture ratio 120d: (avg ret on up days)/(|avg ret on down days|)
    up_ret = ret.where(spx_ret > 0, np.nan).rolling(120).mean()
    dn_ret = ret.where(spx_ret < 0, np.nan).rolling(120).mean()
    capture_ratio_120 = up_ret / dn_ret.abs()

    # risk-adjusted momentum 120d (Sharpe-like)
    mom120 = close / close.shift(120) - 1
    risk_adj_mom_120 = mom120 / (vol120 * np.sqrt(120))

    # vol rank percentile (low vol = defensive)
    vol_rank_252 = vol20.rank(axis=1, pct=True)

    # beta to SPX only in high-VIX regime (VIX above 120d median)
    vix_hi = (macro["VIX"].reindex(ret.index).ffill() > vix_med).astype(float)
    spx_hi = spx_ret * vix_hi
    cov_hi = ret.rolling(120).cov(spx_hi)
    var_hi = spx_hi.rolling(120).var()
    highvix_beta_120 = cov_hi.div(var_hi, axis=0)

    candidates = {
        "downbeta_spx_120": downbeta_spx_120,
        "downbeta_eq_120": downbeta_eq_120,
        "beta_spx_60": beta_spx_60,
        "safe_corr_vix_60": safe_corr_vix_60,
        "vix_beta_60": vix_beta_60,
        "down_vol_ratio_60": down_vol_ratio_60,
        "trend_breadth_60": trend_breadth_60,
        "mom20_z_60vol": mom20_z_60vol,
        "mom60_vixlow": mom60_vixlow,
        "mom60_breadth": mom60_breadth,
        "ratebeta_down_120": ratebeta_down_120,
        "dd_252": dd_252,
        "skew_60": skew_60,
        "xau_corr_60": xau_corr_60,
        "capture_ratio_120": capture_ratio_120,
        "risk_adj_mom_120": risk_adj_mom_120,
        "vol_rank_252": vol_rank_252,
        "highvix_beta_120": highvix_beta_120,
    }

    lib = build_library_factors(close)
    print("\n########## FULL WINDOW VALIDATION ##########")
    full = {}
    for name, fac in candidates.items():
        fac = fac.replace([np.inf, -np.inf], np.nan)
        full[name] = validate(fac, close, fwd_cache, name)
        lc = library_corr(fac, lib)
        mx = max((abs(v) for v in lc.values() if v is not None), default=0.0)
        print(f"    max_abs_library_corr={mx:.4f}  corr={lc}")

    print("\n########## RECENT WINDOW 2025-01-01..2027-01-21 ##########")
    recent = {}
    for name, fac in candidates.items():
        fac = fac.replace([np.inf, -np.inf], np.nan)
        recent[name] = validate(fac, close, fwd_cache, name + "_recent", win_start="2025-01-01")

    print("\n########## SUMMARY ##########")
    for name in candidates:
        f, r = full.get(name), recent.get(name)
        fa = f["admitted"] if f else None
        ra = r["admitted"] if r else None
        print(f"{name:22s} full_h={fa} recent_h={ra}")


if __name__ == "__main__":
    main()
