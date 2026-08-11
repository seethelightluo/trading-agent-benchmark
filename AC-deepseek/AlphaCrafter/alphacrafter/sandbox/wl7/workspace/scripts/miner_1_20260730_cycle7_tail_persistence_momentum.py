"""miner_1: cycle 7 research (fast screen + full eval) - NOVEL factor families on the
15-instrument cross-asset benchmark. Families: (A) tail/asymmetric-risk (downside &
asymmetric beta, kurtosis, leverage effect), (B) path persistence (Hurst R/S,
signed streak, days-since-high), (C) risk-adjusted momentum (rolling Sharpe),
(D) volume/liquidity dynamics (volume regime, amihud trend), (E) pure macro
correlation (VIX corr, US10Y beta).
Admission gates: |IC10| >= 0.007, |ICIR10| >= 0.084. Factor dates <= 2026-07-15.
"""
import sys, time
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from miner_1_metrics import load_panel, load_macro, MIN_ASSETS, FACTOR_LAST, evaluate

WATCH = ["000300.SH", "000688.SH", "SPX", "NDX", "SOX", "HSI", "N225", "SX5E",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]


def per_asset(fn):
    def wrapper(panel, *series_list):
        cols = {}
        for a in panel.columns:
            s = panel[a].dropna()
            args = tuple(sr[a].reindex(s.index) for sr in series_list)
            cols[a] = fn(s, *args)
        return pd.DataFrame(cols, index=panel.index)
    return wrapper


def fast_spearman(a, b):
    ra = rankdata(a); rb = rankdata(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return 0.0 if denom == 0 else float((ra * rb).sum() / denom)


def quick_screen(fvals, closes, horizon=10, min_assets=MIN_ASSETS, factor_last=FACTOR_LAST):
    """Fast h10-only screen: returns (n_dates, ic_mean, icir) or None."""
    fwd = closes.shift(-horizon) / closes - 1.0
    ics = []
    for dt in fvals.index:
        if dt > pd.Timestamp(factor_last):
            continue
        f = fvals.loc[dt]; r = fwd.loc[dt]
        m = f.notna() & r.notna() & np.isfinite(f) & np.isfinite(r)
        n = int(m.sum())
        if n < min_assets:
            continue
        ics.append(fast_spearman(f[m].values.astype(float), r[m].values.astype(float)))
    if len(ics) < 200:
        return None
    arr = np.array(ics)
    return len(arr), float(arr.mean()), float(arr.mean() / arr.std(ddof=1)) if arr.std(ddof=1) > 0 else 0.0


def rolling_beta_mask(x, m, mask, win=60, min_obs=8):
    mask = mask.astype(float)
    n = mask.rolling(win).sum()
    Sx = (x * mask).rolling(win).sum()
    Sy = (m * mask).rolling(win).sum()
    Sxx = (x * x * mask).rolling(win).sum()
    Sxy = (x * m * mask).rolling(win).sum()
    denom = (Sxx - Sx * Sy / n).replace(0, np.nan)
    beta = (Sxy - Sx * Sy / n) / denom
    return beta.where(n >= min_obs)


def hurst_rs(s, win=60):
    lp = np.log(s)
    d = lp.diff()
    mu = d.rolling(win).mean()
    dev = d - mu
    c = dev.rolling(win).sum()
    R = c.rolling(win).max() - c.rolling(win).min()
    S = d.rolling(win).std()
    rs = (R / S.replace(0, np.nan))
    return np.log(rs) / np.log(win)


def signed_streak(s, cap=20):
    r = s.pct_change()
    sign = np.sign(r)
    newrun = sign.ne(sign.shift(1)) | sign.isna()
    grp = newrun.cumsum()
    cnt = r.groupby(grp).cumcount() + 1
    streak = (cnt * sign).clip(-cap, cap)
    return streak


def days_since_high(s, win=60):
    rollmax = s.rolling(win).max()
    newhigh = (s >= rollmax) & rollmax.notna()
    idx = pd.Series(np.arange(len(s)), index=s.index)
    last_true = idx.where(newhigh).ffill()
    days = (idx - last_true).where(rollmax.notna())
    return -np.log(1.0 + days)


def library_signals(closes, vols, vix):
    rets = closes.pct_change()
    mkt_r = closes.mean(axis=1).pct_change()
    out = {}
    out["mom_10d_skip5"] = closes.shift(5) / closes.shift(15) - 1.0
    out["mom_120d_skip5"] = closes.shift(5) / closes.shift(125) - 1.0
    v = rets.rolling(20).std()
    out["vol_of_vol20x60"] = v.rolling(60).std()
    out["max_ret_20d"] = rets.rolling(20).max()
    out["downside_vol_ratio_20"] = -(rets.clip(upper=0).rolling(20).std() / rets.rolling(20).std())
    mom20 = closes.shift(5) / closes.shift(25) - 1.0
    out["rel_mom_20d_skip5"] = mom20.sub(mom20.median(axis=1), axis=0)
    out["beta_ew_60d"] = rets.rolling(60).cov(mkt_r) / mkt_r.rolling(60).var()
    vixr = vix.pct_change()
    out["vix_beta_cond_60x20"] = -(rets.rolling(60).cov(vixr) / vixr.rolling(60).var()) * (vix / vix.shift(20) - 1.0)
    amihud = (rets.abs() / vols.replace(0, np.nan))
    out["amihud_20"] = amihud.rolling(20).mean()
    return out


def library_corr(fvals, closes, libs, n_days=500):
    out = {}
    common = fvals.index.intersection(closes.index)
    for fid, lf in libs.items():
        cs = []
        for dt in common[-n_days:]:
            f = fvals.loc[dt]
            g = lf.loc[dt].reindex(f.index)
            m = f.notna() & g.notna() & np.isfinite(f.astype(float)) & np.isfinite(g.astype(float))
            if int(m.sum()) >= MIN_ASSETS:
                cs.append(fast_spearman(f[m].values.astype(float), g[m].values.astype(float)))
        out[fid] = round(float(np.mean(cs)), 4) if cs else None
    valid = [abs(v) for v in out.values() if v is not None]
    return (round(max(valid), 4) if valid else None), out


if __name__ == "__main__":
    t0 = time.time()
    frames = load_panel()
    closes = pd.DataFrame({s: f["close"].astype(float) for s, f in frames.items()}).sort_index()
    vols = pd.DataFrame({s: f["volume"].astype(float) for s, f in frames.items()}).sort_index()
    macro = {m: load_macro(m)["close"].astype(float) for m in ["VIX", "DXY", "USDJPY", "USDCNY", "EURUSD"]}
    vix = macro["VIX"]
    rets = closes.pct_change()
    mkt_r = closes.mean(axis=1).pct_change()
    print(f"panel {closes.index[0].date()}..{closes.index[-1].date()} assets={closes.shape[1]} rows={len(closes)} "
          f"({time.time()-t0:.1f}s)", flush=True)

    cands = {}

    # ---- Family A: tail / asymmetric risk ----
    asym_cols, dn_cols = {}, {}
    for a in closes.columns:
        s = closes[a].dropna()
        m = mkt_r.reindex(s.index)
        x = s.pct_change()
        bu = rolling_beta_mask(x, m, (m > 0), min_obs=8)
        bd = rolling_beta_mask(x, m, (m < 0), min_obs=8)
        asym_cols[a] = (bu - bd).reindex(closes.index)
        dn_cols[a] = bd.reindex(closes.index)
    cands["asym_beta_60d"] = pd.DataFrame(asym_cols, index=closes.index)
    cands["down_beta_60d"] = pd.DataFrame(dn_cols, index=closes.index)
    cands["kurt_20d"] = rets.rolling(20).kurt()
    cands["leverage_eff_60d"] = per_asset(lambda s: s.pct_change().rolling(60).corr(s.pct_change().abs().shift(1)))(closes)

    # ---- Family B: path persistence ----
    cands["hurst_60d"] = per_asset(hurst_rs)(closes)
    cands["streak_signed_20d"] = per_asset(signed_streak)(closes)
    cands["days_since_high_60d"] = per_asset(days_since_high)(closes)

    # ---- Family C: risk-adjusted momentum ----
    def sharpe(s, win):
        r = s.pct_change()
        return r.rolling(win).mean() / r.rolling(win).std().replace(0, np.nan)
    cands["sharpe_20d"] = per_asset(lambda s: sharpe(s, 20))(closes)
    cands["sharpe_60d"] = per_asset(lambda s: sharpe(s, 60))(closes)

    # ---- Family D: volume / liquidity dynamics ----
    cands["vol_regime_20x250"] = vols.rolling(20).mean() / vols.rolling(250).mean().replace(0, np.nan) - 1.0
    amihud = (rets.abs() / vols.replace(0, np.nan)).rolling(20).mean()
    cands["amihud_trend_20x60"] = amihud / amihud.shift(60) - 1.0

    # ---- Family E: pure macro correlation / sensitivity ----
    vixr = vix.pct_change()
    cands["vix_corr_60d"] = rets.rolling(60).corr(vixr)
    u10 = closes["US10Y"].pct_change()
    cands["us10y_beta_60d"] = rets.rolling(60).cov(u10) / u10.rolling(60).var()

    print(f"factors built ({time.time()-t0:.1f}s)", flush=True)

    # ---- quick screen h10 ----
    print("\n=== Quick screen (h10 only) ===", flush=True)
    passed = {}
    for fid, fv in cands.items():
        q = quick_screen(fv, closes)
        if q is None:
            print(f"{fid:22s} INSUFFICIENT dates", flush=True)
            continue
        n, ic, icir = q
        ok = abs(ic) >= 0.007 and abs(icir) >= 0.084
        print(f"{fid:22s} IC10={ic:+.4f} ICIR10={icir:+.4f} n={n} -> {'PASS' if ok else 'fail'}", flush=True)
        if ok:
            passed[fid] = fv
    print(f"screen done ({time.time()-t0:.1f}s)", flush=True)

    libs = library_signals(closes, vols, vix)
    results = {}
    for fid, fv in passed.items():
        res = evaluate(fv, closes, label=fid)
        if res is None:
            continue
        maxrho, per = library_corr(fv, closes, libs)
        res["max_abs_library_correlation"] = maxrho
        results[fid] = res
        flag = "OK" if (maxrho is not None and maxrho < 0.5) else "HIGH-CORR"
        print(f"\n{fid}: full eval -> IC10={res['ic']:+.4f} ICIR10={res['icir']:+.4f} hit={res['ic_hit_ratio']:.3f} "
              f"turn={res['turnover_10d_rank']:.2f} cov={res['coverage_asset_days']:.2f} "
              f"decay={res['decay_ic_by_horizon']} max_rho={maxrho} {flag}", flush=True)
        if maxrho is not None and maxrho >= 0.5:
            print(f"  -> excluded: library correlation too high", flush=True)

    # per-year robustness for passers
    if results:
        fwd = closes.shift(-10) / closes - 1.0
        print("\n=== Per-year h10 IC (passers) ===", flush=True)
        for fid in results:
            fv = passed[fid]
            yrs = {}
            for dt in fv.index:
                if dt > pd.Timestamp(FACTOR_LAST):
                    continue
                f = fv.loc[dt]; r = fwd.loc[dt]
                m = f.notna() & r.notna() & np.isfinite(f) & np.isfinite(r)
                if m.sum() < MIN_ASSETS:
                    continue
                yrs.setdefault(dt.year, []).append(fast_spearman(f[m].values.astype(float), r[m].values.astype(float)))
            parts = []
            for y in sorted(yrs):
                arr = np.array(yrs[y])
                parts.append(f"{y}: ic={arr.mean():+.4f} icir={arr.mean()/arr.std() if arr.std()>0 else 0:+.3f} n={len(arr)}")
            print(f"{fid:22s} " + " | ".join(parts), flush=True)
    print(f"total time {time.time()-t0:.1f}s", flush=True)
