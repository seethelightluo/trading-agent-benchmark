"""miner_1: validate NEW factor families for the cross-asset benchmark.
Cycle: price-location/range structure, trend consistency, gap/overnight structure,
cross-sectional relative strength, new macro conditional betas, volume trend.
Admission: |IC10| >= 0.007 and |ICIR10| >= 0.084; library |rho| < 0.5.
Factor dates <= 2026-07-15; data visible through 2026-07-29.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from miner_1_metrics import load_panel, load_macro, WATCH, MIN_ASSETS, FACTOR_LAST, MAX_VISIBLE

WATCH = ["000300.SH", "000688.SH", "SPX", "NDX", "SOX", "HSI", "N225", "SX5E",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]


def per_asset(fn):
    """Apply fn on each asset's own trading calendar, reindex to union panel."""
    def wrapper(panel, *macro_series):
        cols = {}
        for a in panel.columns:
            s = panel[a].dropna()
            if macro_series:
                idx = s.index
                for ms in macro_series:
                    idx = idx.intersection(ms.dropna().index)
                s = s.loc[idx]
                args = tuple(ms.reindex(idx) for ms in macro_series)
                cols[a] = fn(s, *args)
            else:
                cols[a] = fn(s)
        return pd.DataFrame(cols, index=panel.index)
    return wrapper


def fwd_returns(closes, h):
    return closes.shift(-h) / closes - 1.0


def ic_series_fast(factor, closes, h):
    fwd = fwd_returns(closes, h)
    dates = factor.index.intersection(fwd.index)
    dates = dates[dates <= pd.Timestamp(FACTOR_LAST)]
    F = factor.reindex(dates).values.astype(float)
    R = fwd.reindex(dates).values.astype(float)
    A = np.argsort(np.argsort(F, axis=1), axis=1).astype(float)
    B = np.argsort(np.argsort(R, axis=1), axis=1).astype(float)
    out, idx = [], []
    for i in range(len(dates)):
        m = np.isfinite(F[i]) & np.isfinite(R[i])
        if int(m.sum()) < MIN_ASSETS:
            continue
        a_, b_ = A[i][m], B[i][m]
        ma, mb = a_.mean(), b_.mean()
        num = float(((a_ - ma) * (b_ - mb)).sum())
        den = float(np.sqrt(((a_ - ma) ** 2).sum() * ((b_ - mb) ** 2).sum()))
        out.append(num / den if den > 0 else 0.0)
        idx.append(dates[i])
    return pd.Series(out, index=idx)


def evaluate(factor, closes, horizons=(1, 2, 3, 5, 10, 20)):
    ics = {h: ic_series_fast(factor, closes, h) for h in horizons}
    ic10 = ics[10]
    if len(ic10) < 200:
        return None
    direction = float(np.sign(ic10.mean())) if np.isfinite(ic10.mean()) and ic10.mean() != 0 else 1.0
    out = {"direction": direction}
    for h in horizons:
        ic = ics[h] * direction
        out[f"ic_h{h}"] = float(ic.mean())
        out[f"icir_h{h}"] = float(ic.mean() / ic.std()) if len(ic) > 2 and ic.std() > 0 else float("nan")
        out[f"hit_h{h}"] = float((ic > 0).mean())
        out[f"n_h{h}"] = len(ic)
    fw = factor.loc[:FACTOR_LAST]
    out["coverage_asset_days"] = float(fw.notna().mean().mean())
    out["coverage_dates_ge8"] = float((fw.notna().sum(axis=1) >= MIN_ASSETS).mean())
    ranks = fw.rank(axis=1)
    out["turnover_10d_rank"] = float((ranks - ranks.shift(10)).abs().mean(axis=1).mean())
    return out


# ---------------- library signal recomputation ----------------
def library_signals(closes, vols, vix, dxy, usdjpy, eurusd):
    rets = closes.pct_change()
    mkt = closes.mean(axis=1)
    mkt_r = mkt.pct_change()
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


def library_corr(fvals, libs, n_days=500):
    """Max abs mean per-date cross-sectional rank corr vs library factors."""
    out = {}
    common = fvals.index.intersection(closes.index)
    for fid, lf in libs.items():
        cs = []
        for dt in common[-n_days:]:
            f = fvals.loc[dt]
            g = lf.reindex(f.index)
            m = f.notna() & g.notna() & np.isfinite(f.astype(float)) & np.isfinite(g.astype(float))
            if int(m.sum()) >= MIN_ASSETS:
                r, _ = spearmanr(f[m].astype(float), g[m].astype(float))
                cs.append(r)
        out[fid] = round(float(np.mean(cs)), 4) if cs else None
    valid = [abs(v) for v in out.values() if v is not None]
    return (round(max(valid), 4) if valid else None), out


# ---------------- main ----------------
if __name__ == "__main__":
    frames = load_panel()
    closes = pd.DataFrame({s: f["close"].astype(float) for s, f in frames.items()}).sort_index()
    opens = pd.DataFrame({s: f["open"].astype(float) for s, f in frames.items()}).sort_index()
    highs = pd.DataFrame({s: f["high"].astype(float) for s, f in frames.items()}).sort_index()
    lows = pd.DataFrame({s: f["low"].astype(float) for s, f in frames.items()}).sort_index()
    vols = pd.DataFrame({s: f["volume"].astype(float) for s, f in frames.items()}).sort_index()
    macro = {m: load_macro(m)["close"].astype(float) for m in ["VIX", "DXY", "USDJPY", "USDCNY", "EURUSD"]}
    vix, dxy, usdjpy, eurusd = macro["VIX"], macro["DXY"], macro["USDJPY"], macro["EURUSD"]
    rets = closes.pct_change()
    print(f"panel {closes.index[0].date()}..{closes.index[-1].date()} assets={closes.shape[1]} rows={len(closes)}")

    cands = {}

    # ---- Family A: price location in N-day range ----
    def loc_range(win):
        def f(s, hi, lo):
            rng = hi.rolling(win).max() - lo.rolling(win).min()
            return (s - lo.rolling(win).min()) / rng.replace(0, np.nan)
        cols = {}
        for a in closes.columns:
            s = closes[a].dropna(); hi = highs[a].dropna(); lo = lows[a].dropna()
            idx = s.index.intersection(hi.index).intersection(lo.index)
            cols[a] = f(s.loc[idx], hi.loc[idx], lo.loc[idx])
        return pd.DataFrame(cols, index=closes.index)
    cands["loc_range_20d"] = loc_range(20)
    cands["loc_range_60d"] = loc_range(60)

    # ---- Family B: trend consistency / quality ----
    cands["winrate_20d"] = per_asset(lambda s: s.pct_change().gt(0).rolling(20).mean() - 0.5)(closes)
    cands["cci_20d"] = per_asset(
        lambda s: (s - s.rolling(20).mean()) / (0.015 * (s - s.rolling(20).mean()).abs().rolling(20).mean()))(closes)
    cands["accel_30x90"] = per_asset(
        lambda s: (s / s.shift(30) - 1.0) - (s / s.shift(90) - 1.0))(closes)

    # ---- Family C: gap / overnight structure ----
    gaps = opens / closes.shift(1) - 1.0
    intra = closes / opens - 1.0
    cands["gap_z_20d"] = per_asset(lambda s: (s / s.shift(1) - 1.0).rolling(20).mean()
                                   / s.pct_change().rolling(20).std())(opens)
    cands["gap_vs_intra_20d"] = gaps.rolling(20).mean() - intra.rolling(20).mean()

    # ---- Family D: cross-sectional relative ----
    mom30 = per_asset(lambda s: s.shift(5) / s.shift(35) - 1.0)(closes)
    cands["z_mom_30d_skip5"] = mom30.sub(mom30.mean(axis=1), axis=0).div(mom30.std(axis=1).replace(0, np.nan), axis=0)
    mom20r = per_asset(lambda s: s.shift(5) / s.shift(25) - 1.0)(closes)
    cands["rank_accel_10d"] = mom20r.rank(axis=1) - mom20r.rank(axis=1).shift(10)

    # ---- Family E: macro conditional beta (new macros) ----
    def cond_beta(macro_series):
        mr = macro_series.pct_change()
        def f(s, m):
            z = pd.concat([s.pct_change().rename("r"), m.pct_change().rename("m")], axis=1).dropna()
            beta = z["r"].rolling(60).cov(z["m"]) / z["m"].rolling(60).var().replace(0, np.nan)
            return beta * (m / m.shift(20) - 1.0)
        return per_asset(f)(closes, macro_series)
    cands["usdjpy_beta_cond_60x20"] = cond_beta(usdjpy)
    cands["eurusd_beta_cond_60x20"] = cond_beta(eurusd)
    cands["dxy_beta_cond_60x20"] = cond_beta(dxy)

    # ---- Family F: volume trend ----
    cands["vol_ratio_20x60"] = (vols.rolling(20).mean() / vols.rolling(60).mean() - 1.0)

    # ---- evaluate ----
    libs = library_signals(closes, vols, vix, dxy, usdjpy, eurusd)
    print("\n=== Metrics (h=10 admission) ===")
    results = {}
    for fid, fv in cands.items():
        res = evaluate(fv, closes)
        if res is None:
            print(f"{fid}: INSUFFICIENT dates")
            continue
        results[fid] = res
        gate = abs(res["ic_h10"]) >= 0.007 and abs(res["icir_h10"]) >= 0.084
        print(f"{fid:26s} IC10={res['ic_h10']:+.4f} ICIR10={res['icir_h10']:+.4f} "
              f"hit={res['hit_h10']:.3f} n={res['n_h10']} dir={res['direction']:+.0f} "
              f"turn={res['turnover_10d_rank']:.2f} cov={res['coverage_asset_days']:.2f} "
              f"decay20={res['ic_h20']:+.4f} -> {'PASS' if gate else 'fail'}")

    print("\n=== Library correlation (max |rho| over last 500d) ===")
    for fid in results:
        maxrho, per = library_corr(cands[fid], libs)
        results[fid]["max_abs_library_correlation"] = maxrho
        flag = "OK" if (maxrho is not None and maxrho < 0.5) else "HIGH-CORR"
        print(f"{fid:26s} max_rho={maxrho if maxrho is not None else float('nan'):.3f} {flag} "
              + " ".join(f"{k}={v:.2f}" for k, v in sorted(per.items()) if v is not None and abs(v) >= 0.3))

    print("\n=== Per-year h10 IC (robustness) ===")
    for fid in results:
        fv = cands[fid]
        ics = ic_series_fast(fv, closes, 10)
        yrs = {}
        for dt, v in ics.items():
            yrs.setdefault(dt.year, []).append(v)
        parts = []
        for y in sorted(yrs):
            arr = np.array(yrs[y])
            parts.append(f"{y}: ic={arr.mean():+.4f} icir={arr.mean()/arr.std() if arr.std()>0 else 0:+.3f} n={len(arr)}")
        print(f"{fid:26s} " + " | ".join(parts))
