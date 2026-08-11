"""miner_2 cycle 4 v2 (optimized): trend-quality & time-series persistence family.

Idea: raw price momentum already admitted (mom_10d/120d, rel_mom). This cycle
tests whether *quality-adjusted* trend measures - proximity to 52w high,
fraction of up days, vol-scaled momentum, drawdown depth, and variance-ratio
(Hurst-like) trend persistence - carry incremental cross-sectional signal on
the 15-asset tradable universe at h=10.

Universe: 15 tradable cross-asset instruments. IC = cross-sectional Spearman
rank IC per date (>=8 assets). Validation window 2020-01-01..2026-07-15.
Admission gates: |IC|>=0.007 and |ICIR|>=0.084 @ h=10.
"""
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, "scripts")
from miner_2_lib import (load_panel, load_macro, WATCH, MAX_VISIBLE,
                         FACTOR_LAST, MIN_ASSETS, ADMISSION)

EPS = 1e-12
panel = load_panel()
rets = panel.pct_change()
mac = load_macro()


def dist_high_252():
    """close / rolling_max(close,252) - 1 : 52-week-high proximity (trend strength)."""
    return panel / panel.rolling(252, min_periods=120).max() - 1.0


def up_ratio_60():
    """fraction of up days over 60d (trend consistency)."""
    return (rets > 0).rolling(60, min_periods=30).mean()


def vol_adj_mom_20x60():
    """20d momentum (skip 5) scaled by 60d vol (risk-adjusted trend)."""
    mom = panel.shift(5) / panel.shift(25) - 1.0
    vol = rets.rolling(60, min_periods=30).std()
    return mom / (vol + EPS)


def dd_depth_60():
    """current drawdown depth from 60d max (trend health / reversal risk)."""
    return panel / panel.rolling(60, min_periods=30).max() - 1.0


def hurst_vr_60x10():
    """Variance ratio at lag 10 over rolling 60d window; >1 => persistent trend."""
    vr = {}
    for s in WATCH:
        x = rets[s].dropna()
        var1 = x.rolling(60, min_periods=40).var()
        r10 = x.rolling(10).sum()
        var10 = r10.rolling(60, min_periods=40).var()
        vr[s] = var10 / (10.0 * var1 + EPS)
    return pd.DataFrame(vr, index=panel.index)


candidates = {
    "dist_high_252": dist_high_252(),
    "up_ratio_60": up_ratio_60(),
    "vol_adj_mom_20x60": vol_adj_mom_20x60(),
    "dd_depth_60": dd_depth_60(),
    "hurst_vr_60x10": hurst_vr_60x10(),
}

# ---------------- library signals (all 9 effective factors) ----------------
def library_signals_all():
    libs = {}
    libs["mom_10d_skip5"] = panel.shift(5) / panel.shift(15) - 1.0
    libs["mom_120d_skip5"] = panel.shift(5) / panel.shift(125) - 1.0
    m20 = panel.shift(5) / panel.shift(25) - 1.0
    libs["rel_mom_20d_skip5"] = m20.sub(m20.median(axis=1), axis=0)
    libs["vol_of_vol20x60"] = rets.rolling(20).std().rolling(60).std()
    vix = mac["VIX"]
    vixr = vix.pct_change()
    beta = rets.rolling(60).cov(vixr) / vixr.rolling(60).var()
    libs["vix_beta_cond_60x20"] = -beta * (vix / vix.shift(20) - 1.0)
    am = {}
    for s in WATCH:
        df = pd.read_csv(f"../persistent/stock_data/{s}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= MAX_VISIBLE].set_index("date").sort_index()
        rr = df["close"].pct_change()
        am[s] = (rr.abs() / (df["volume"] + EPS)).rolling(20, min_periods=10).mean()
    libs["amihud_20"] = pd.DataFrame(am, index=panel.index)
    ew = rets.mean(axis=1)
    libs["beta_ew_60d"] = rets.rolling(60).cov(ew) / ew.rolling(60).var()
    neg = rets.clip(upper=0.0)
    ds_vol = (neg ** 2).rolling(20, min_periods=10).mean().apply(np.sqrt)
    tot_vol = rets.rolling(20, min_periods=10).std()
    libs["downside_vol_ratio_20"] = -(ds_vol / (tot_vol + EPS))
    libs["max_ret_20d"] = rets.rolling(20, min_periods=10).max()
    return libs


LIBS = library_signals_all()
print("library signals:", list(LIBS.keys()), flush=True)


def fwd_returns(h):
    out = {}
    for s in WATCH:
        c = panel[s].dropna()
        out[s] = c.shift(-h) / c - 1.0
    return pd.DataFrame(out, index=panel.index)


def rank_ic_fast(f, fwd):
    """Per-date Spearman via rankdata + corrcoef (fast path)."""
    ics = {}
    idx = f.index.intersection(fwd.index)
    for d in idx:
        fv = f.loc[d]
        rv = fwd.loc[d]
        m = fv.notna().values & rv.notna().values
        if int(m.sum()) < MIN_ASSETS:
            continue
        a = rankdata(fv.values[m])
        b = rankdata(rv.values[m])
        cc = np.corrcoef(a, b)
        if np.isfinite(cc[0, 1]):
            ics[d] = cc[0, 1]
    return pd.Series(ics).sort_index()


def library_corr_fast(factor, last_n=700):
    """Per-date Spearman with library factors, vectorized over assets."""
    ft = factor.iloc[-last_n:].T
    per = {}
    for fid, lf in LIBS.items():
        lt = lf.reindex(ft.index).T
        common = ft.columns.intersection(lt.columns)
        if len(common) == 0:
            per[fid] = None
            continue
        f2 = ft[common].rank(axis=0)
        l2 = lt[common].rank(axis=0)
        # pairwise pearson on ranks per date (columns), require >=8 valid pairs
        n_valid = f2.notna().sum()
        keep = n_valid[n_valid >= MIN_ASSETS].index
        if len(keep) == 0:
            per[fid] = None
            continue
        cs = []
        for dt in keep:
            a = f2[dt].values
            b = l2[dt].values
            m = np.isfinite(a) & np.isfinite(b)
            if int(m.sum()) >= MIN_ASSETS:
                cc = np.corrcoef(a[m], b[m])
                if np.isfinite(cc[0, 1]):
                    cs.append(cc[0, 1])
        per[fid] = round(float(np.mean(cs)), 4) if cs else None
    valid = [abs(v) for v in per.values() if v is not None]
    return (round(max(valid), 4) if valid else float("nan")), per


def turnover_10d(factor):
    ranks = factor.rank(axis=1)
    out = []
    for i in range(10, len(ranks)):
        a, b = ranks.iloc[i - 10], ranks.iloc[i]
        both = a.dropna().index.intersection(b.dropna().index)
        if len(both) >= MIN_ASSETS:
            out.append(float((a[both] - b[both]).abs().mean()))
    return float(np.mean(out)) if out else float("nan")


horizons = (1, 2, 3, 5, 10, 20)
fwd = {h: fwd_returns(h) for h in horizons}
all_results = {}

for name, fdf in candidates.items():
    fw = fdf.loc[:FACTOR_LAST]
    res = {"name": name, "factor_rows": int(len(fw)), "n_assets": panel.shape[1]}
    ic_by_h = {h: rank_ic_fast(fw, fwd[h]) for h in horizons}
    ic10 = ic_by_h[10]
    direction = float(np.sign(ic10.mean())) if np.isfinite(ic10.mean()) and ic10.mean() != 0 else 1.0
    for h in horizons:
        ic = ic_by_h[h] * direction
        res[f"ic_h{h}"] = float(ic.mean())
        res[f"icir_h{h}"] = float(ic.mean() / ic.std()) if len(ic) > 2 and ic.std() > 0 else float("nan")
        res[f"hit_h{h}"] = float((ic > 0).mean()) if len(ic) else float("nan")
        res[f"n_dates_h{h}"] = int(len(ic))
    res["direction"] = direction
    valid = fw.notna()
    res["coverage_asset_days"] = float(valid.mean().mean())
    res["coverage_dates_ge8"] = float((valid.sum(axis=1) >= MIN_ASSETS).mean())
    res["turnover_10d_rank"] = turnover_10d(fw)
    res["max_abs_library_correlation"], res["library_corrs"] = library_corr_fast(fw)
    ic10d = ic_by_h[10] * direction
    years = {}
    for y in range(2020, 2027):
        sub = ic10d.loc[str(y)]
        if len(sub) > 20:
            years[y] = {"ic": round(float(sub.mean()), 4),
                        "icir": round(float(sub.mean() / sub.std()), 4),
                        "n": int(len(sub))}
    res["per_year_ic_h10"] = years
    gate_ic = abs(res["ic_h10"]) >= ADMISSION["ic"]
    gate_icir = abs(res["icir_h10"]) >= ADMISSION["icir"]
    res["pass"] = bool(gate_ic and gate_icir)
    all_results[name] = res

    print(f"=== {name} ===  direction={direction:+.3f}  rows={len(fw)}  assets={panel.shape[1]}", flush=True)
    for h in horizons:
        print(f"  h{h:>2}: IC={res[f'ic_h{h}']:+.4f}  ICIR={res[f'icir_h{h}']:+.4f}  "
              f"hit={res[f'hit_h{h}']:.3f}  n={res[f'n_dates_h{h}']}", flush=True)
    print(f"  coverage_ad={res['coverage_asset_days']:.3f}  cov_d8={res['coverage_dates_ge8']:.3f}  "
          f"turnover={res['turnover_10d_rank']:.3f}  max_corr={res['max_abs_library_correlation']:.4f}", flush=True)
    print(f"  per-year h10: {years}", flush=True)
    print(f"  ADMISSION h10: |IC|={abs(res['ic_h10']):.4f} (>=0.007: {gate_ic}), "
          f"|ICIR|={abs(res['icir_h10']):.4f} (>=0.084: {gate_icir}) -> "
          f"{'PASS' if res['pass'] else 'FAIL'}", flush=True)
    print(flush=True)

with open("scripts/miner_2_cycle4_results.json", "w") as f:
    json.dump(all_results, f, indent=1)
print("saved scripts/miner_2_cycle4_results.json", flush=True)
