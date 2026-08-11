"""miner_1 candidate validation v3: per-asset own-calendar computation.
Universe: 15 cross-asset instruments. Factor window: 2020-01-01..2026-07-15.
Admission: |IC|>=0.007 and |ICIR|>=0.084 @ h=10; pairwise rank corr < 0.5 vs library.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from miner_2_lib import load_panel, load_macro, fwd_returns, rank_ic_series, \
    turnover_10d_rank, MIN_ASSETS, FACTOR_LAST


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



def ic_series_fast(factor, panel, h):
    """Vectorized per-date Spearman IC (numpy), all dates <= FACTOR_LAST."""
    fwd = fwd_returns(panel, h)
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


def evaluate(factor, panel, horizons=(1, 2, 3, 5, 10, 20)):
    ics = {h: ic_series_fast(factor, panel, h) for h in horizons}
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
    out["turnover_10d_rank"] = turnover_10d_rank(fw)
    return out

def pairwise_rho_all(factors, names):
    """Vectorized: for each date (subsampled), compute full Spearman corr matrix
    of factors on assets with complete values; return mean rho per pair."""
    dates = factors[names[0]].index
    dates = dates[dates <= pd.Timestamp(FACTOR_LAST)][::3]
    acc = {n: {m: [] for m in names if m != n} for n in names}
    import numpy as _np
    for dt in dates:
        cols = {}
        ok = None
        for n in names:
            v = factors[n].loc[dt].astype(float)
            cols[n] = v
            m = v.notna() & _np.isfinite(v)
            ok = m if ok is None else (ok & m)
        if int(ok.sum()) < MIN_ASSETS:
            continue
        mat = _np.array([cols[n][ok].values for n in names])
        # Pearson on ranks == Spearman
        mat = _np.argsort(_np.argsort(mat, axis=1), axis=1).astype(float)
        cm = _np.corrcoef(mat)
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                if i < j and _np.isfinite(cm[i, j]):
                    acc[a][b].append(float(cm[i, j]))
    out = {}
    for a in names:
        for b in names:
            if a != b:
                v = acc[a][b]
                out[(a, b)] = float(_np.mean(v)) if v else float("nan")
    return out

def pairwise_rho(a, b):
    cs = []
    common = a.index.intersection(b.index)
    for dt in common:
        if dt > pd.Timestamp(FACTOR_LAST):
            continue
        f, g = a.loc[dt], b.loc[dt]
        if isinstance(f, pd.DataFrame) or isinstance(g, pd.DataFrame):
            continue
        m = f.notna() & g.notna() & np.isfinite(f.astype(float)) & np.isfinite(g.astype(float))
        if int(m.sum()) >= MIN_ASSETS:
            cs.append(spearmanr(f[m].astype(float), g[m].astype(float))[0])
    return float(np.mean(cs)) if cs else float("nan")


def main():
    panel = load_panel()
    macro = load_macro()
    vix = macro["VIX"]
    dxy = macro["DXY"]
    print(f"panel {panel.index[0].date()} .. {panel.index[-1].date()}, assets={panel.shape[1]}")

    cands = {}

    def pct(s): return s.pct_change()
    def rolstd(s, n): return s.pct_change().rolling(n).std()

    # ---- existing / recovery factors ----
    cands["mom_10d_skip5"] = per_asset(lambda s: s.shift(5) / s.shift(15) - 1.0)(panel)
    cands["mom_120d_skip5"] = per_asset(lambda s: s.shift(5) / s.shift(125) - 1.0)(panel)
    cands["vol_of_vol20x60"] = per_asset(lambda s: s.pct_change().rolling(20).std().rolling(60).std())(panel)
    cands["vix_beta_cond_60x20"] = per_asset(
        lambda s, m: -((s.pct_change().rolling(60).cov(m.pct_change())
                        / m.pct_change().rolling(60).var().replace(0, np.nan))
                       * (m / m.shift(20) - 1.0)))(panel, vix)
    mkt = panel.mean(axis=1)
    cands["beta_ew_60d"] = per_asset(
        lambda s: (s.pct_change().rolling(60).cov(mkt.reindex(s.index).pct_change())
                   / mkt.reindex(s.index).pct_change().rolling(60).var().replace(0, np.nan)))(panel)
    mom20 = per_asset(lambda s: s.shift(5) / s.shift(25) - 1.0)(panel)
    cands["rel_mom_20d_skip5"] = mom20.sub(mom20.median(axis=1), axis=0)
    cands["max_ret_20d"] = per_asset(lambda s: s.pct_change().rolling(20).max())(panel)
    cands["downside_vol_ratio_20"] = per_asset(
        lambda s: -(s.pct_change().clip(upper=0).rolling(20).std()
                    / s.pct_change().rolling(20).std().replace(0, np.nan)))(panel)

    # ---- family A: trend quality ----
    def eff_ratio(win):
        return per_asset(lambda s: (s - s.shift(win)).abs()
                         / s.pct_change().abs().rolling(win).sum().replace(0, np.nan))(panel)
    cands["eff_ratio_20d"] = eff_ratio(20)
    cands["eff_ratio_60d"] = eff_ratio(60)
    cands["trend_strength_60x20"] = per_asset(
        lambda s: (s / s.shift(60) - 1.0).abs() / s.pct_change().rolling(20).std())(panel)
    cands["macd_10x40"] = per_asset(
        lambda s: (s.ewm(span=10, adjust=False).mean() - s.ewm(span=40, adjust=False).mean()) / s)(panel)

    # ---- family B: vol structure & asymmetry ----
    cands["vol_term_10x60"] = per_asset(
        lambda s: s.pct_change().rolling(10).std() / s.pct_change().rolling(60).std())(panel)
    cands["updown_ratio_60d"] = per_asset(
        lambda s: (s.pct_change().clip(lower=0).rolling(60).mean()
                   / (-s.pct_change().clip(upper=0)).rolling(60).mean().replace(0, np.nan)))(panel)
    cands["skew_60d"] = per_asset(lambda s: s.pct_change().rolling(60).skew())(panel)
    cands["downside_vol_ratio_60"] = per_asset(
        lambda s: -(s.pct_change().clip(upper=0).rolling(60).std()
                    / s.pct_change().rolling(60).std().replace(0, np.nan)))(panel)

    # ---- family C: macro beta ----
    cands["dxy_beta_60d"] = per_asset(
        lambda s, m: ((s.pct_change().rolling(60).cov(m.pct_change())
                       / m.pct_change().rolling(60).var().replace(0, np.nan))
                      * (m / m.shift(20) - 1.0)))(panel, dxy)
    u10 = panel["US10Y"]
    cands["us10y_beta_60d"] = per_asset(
        lambda s, m: ((s.pct_change().rolling(60).cov(m.pct_change())
                       / m.pct_change().rolling(60).var().replace(0, np.nan))
                      * (m / m.shift(20) - 1.0)))(panel, u10)

    # ---- family D: reversal ----
    cands["rsi_rev_14"] = per_asset(
        lambda s: 50 - (100 - 100 / (1 + s.diff().clip(lower=0).rolling(14).mean()
                                     / (-s.diff().clip(upper=0)).rolling(14).mean().replace(0, np.nan))))(panel)
    cands["zscore_rev_60d"] = per_asset(
        lambda s: -(s - s.rolling(60).mean()) / s.rolling(60).std().replace(0, np.nan))(panel)

    # ---- evaluate ----
    results = {}
    print("\n=== Metrics (h=10 admission) ===")
    for fid, fv in cands.items():
        res = evaluate(fv, panel)
        if res is None:
            print(f"{fid}: INSUFFICIENT dates")
            continue
        results[fid] = res
        gate = abs(res["ic_h10"]) >= 0.007 and abs(res["icir_h10"]) >= 0.084
        print(f"{fid:24s} IC10={res['ic_h10']:+.4f} ICIR10={res['icir_h10']:+.4f} "
              f"hit={res['hit_h10']:.3f} n={res['n_h10']} dir={res['direction']:+.1f} "
              f"turn={res['turnover_10d_rank']:.2f} cov={res['coverage_asset_days']:.2f} "
              f"cov8={res['coverage_dates_ge8']:.2f} "
              f"decay20={res['ic_h20']:+.4f} -> {'PASS' if gate else 'fail'}")

    names = list(results.keys())
    rho_mat = pd.DataFrame(index=names, columns=names, dtype=float)
    pr = pairwise_rho_all(cands, names)
    for a in names:
        for b in names:
            rho_mat.loc[a, b] = 1.0 if a == b else pr.get((a, b), float("nan"))

    print("\n=== Candidate conflicts (|rho|>=0.5 flagged) ===")
    for a in names:
        row = []
        for b in names:
            if b == a:
                continue
            r = rho_mat.loc[a, b]
            if np.isfinite(r) and abs(r) >= 0.5:
                row.append(f"{b}={r:.3f}")
        print(f"{a:24s} max_rho={max((abs(rho_mat.loc[a,b]) for b in names if b!=a), default=0):.3f} "
              + (f"CONFLICTS: {', '.join(row)}" if row else ""))

    return results, cands


if __name__ == "__main__":
    main()
