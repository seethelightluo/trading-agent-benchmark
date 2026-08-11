"""miner_3 2026-09-10 batch E round 2: fix orth_resid reporting bug.

report() previously called to_grid(cand) unconditionally, but orth_resid returns
a (T,n) matrix, so orth candidates became all-NaN ("NO VALID IC DATES"). This
script routes matrices directly into the IC pipeline and re-evaluates the six
orthogonalized candidates against the current library correlation gate.

Gates: |IC|>=0.0070, |ICIR|>=0.0840, max_abs_library_correlation < 0.5.
"""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict
from miner_3_20260813_lib import (ASSETS, GRID, HORIZON, to_grid,
                                  cross_sectional_rank, spearman_ic_matrix,
                                  summarize, decay_curve, fwd_by_horizon_dict,
                                  turnover_10d_rank, library_pairwise_corr,
                                  coverage_stats, safe_div)

GATE_IC = 0.0070
GATE_ICIR = 0.0840


def load_asset(sym, days=2300):
    df = get_stock_daily_data(sym, days=days)
    if df is None:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.set_index("date")
    for c in ["open", "close", "high", "low", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ret"] = df["close"].pct_change()
    return df


series = {s: load_asset(s) for s in ASSETS}
series = {s: df for s, df in series.items() if df is not None and len(df) > 100}
print(f"assets loaded: {len(series)}/15 -> {sorted(series.keys())}")
fwd = fwd_by_horizon_dict(series)
dates = np.array(GRID)
results = {}
spx_ret = series["SPX"]["ret"]


def report_mat(name, mat):
    """Report for an already-materialized (T,n) matrix."""
    rank_mat = cross_sectional_rank(mat)
    ics = spearman_ic_matrix(mat, fwd[10])
    summ = summarize(ics, dates, name, HORIZON)
    if summ is None:
        print(name, "NO VALID IC DATES")
        return None
    cov_ad, cov_d8 = coverage_stats(mat)
    to = turnover_10d_rank(rank_mat)
    dec = decay_curve(mat, fwd)
    corrs, mx_name, mx_abs = library_pairwise_corr(mat)
    top = sorted(corrs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
    ic, icir = summ["ic"], summ["icir"]
    ok = (abs(ic) >= GATE_IC) and (abs(icir) >= GATE_ICIR) and (mx_abs < 0.5)
    print("=" * 100)
    print(f"{name}: IC={ic:+.4f} ICIR={icir:+.4f} hit={summ['hit']:.3f} n={summ['n_ic_dates']} "
          f"cov_ad={cov_ad:.3f} cov_d8={cov_d8:.3f} turn={to:.3f} maxlibcorr={mx_abs:.3f} ({mx_name}) "
          f"GATE={ok}")
    print("   regime:", {k: v for k, v in summ["regime"].items()})
    print("   decay:", dec)
    print("   top conflicts:", top)
    results[name] = {
        "ic": round(ic, 5), "icir": round(icir, 5), "hit": round(summ["hit"], 4),
        "n_ic_dates": summ["n_ic_dates"], "regime": summ["regime"],
        "coverage_asset_days": round(cov_ad, 4), "coverage_dates_ge8": round(cov_d8, 4),
        "turnover_10d_rank": round(to, 4), "decay": dec,
        "max_abs_library_correlation": round(mx_abs, 4),
        "max_lib_corr_name": mx_name, "top_conflicts": top, "pass_gate": bool(ok),
    }
    return summ


def _to_series_dict(fn):
    return {s: fn(df) for s, df in series.items()}


def a_downbeta(df):
    r = df["ret"]
    j = pd.concat([r, spx_ret], axis=1, join="outer")
    j.columns = ["a", "b"]
    dn = j["b"] < 0
    cov = j["a"].where(dn).rolling(60, min_periods=15).cov(j["b"].where(dn))
    var = j["b"].where(dn).rolling(60, min_periods=15).var()
    return pd.Series(safe_div(cov, var), index=j.index).reindex(df.index)


def a_volcluster(df):
    a = df["ret"].abs()
    return a.rolling(60, min_periods=40).corr(a.shift(1))


def a_range252(df):
    lo = df["close"].rolling(252, min_periods=30).min()
    hi = df["close"].rolling(252, min_periods=30).max()
    return pd.Series(safe_div(df["close"] - lo, hi - lo), index=df.index)


def a_spxcorr(df):
    r = df["ret"]
    j = pd.concat([r, spx_ret], axis=1, join="outer")
    j.columns = ["a", "b"]
    return j["a"].rolling(60, min_periods=15).corr(j["b"]).reindex(df.index)


def a_calmness(df):
    sd = df["ret"].rolling(20, min_periods=10).std()
    calm = (df["ret"].abs() < 0.5 * sd).astype(float)
    return calm.rolling(20, min_periods=10).mean()


def a_mom20(df):
    raw = df["close"].shift(5) / df["close"].shift(25) - 1.0
    damp = 1.0 / (1.0 + df["close"].pct_change(60).abs())
    return raw * damp


def a_maxcons_gain(df):
    r = df["ret"].to_numpy()
    n = len(r)
    out = np.full(n, np.nan)
    if n >= 21:
        w = sliding_window_view(r, 21)
        for i in range(w.shape[0]):
            v = w[i]
            best = cur = 0
            for x in v:
                if np.isnan(x):
                    cur = 0
                elif x > 0:
                    cur += 1
                    best = max(best, cur)
                else:
                    cur = 0
            out[i + 20] = best
    return pd.Series(out, index=df.index)


def a_gainloss(df):
    d = df["ret"].clip(lower=0.0)
    u = df["ret"].clip(upper=0.0).abs()
    return pd.Series(safe_div(d.rolling(20, min_periods=10).mean(),
                              u.rolling(20, min_periods=10).mean() + 1e-9), index=df.index)


def a_dshigh(df):
    c = df["close"].to_numpy()
    n = len(c)
    out = np.full(n, np.nan)
    if n >= 60:
        w = sliding_window_view(c, 60)
        for i in range(w.shape[0]):
            v = w[i]
            ok = ~np.isnan(v)
            if ok.sum() < 40:
                continue
            vv = v[ok]
            mx = np.nanmax(vv)
            last = len(vv) - 1 - np.argmax(vv[::-1] == mx)
            out[i + 59] = 59 - last
    return pd.Series(out, index=df.index)


ANCHORS = {
    "downbeta_spx_60": a_downbeta,
    "volcluster_60": a_volcluster,
    "range_pos_252": a_range252,
    "spx_corr60": a_spxcorr,
    "calmness_20": a_calmness,
    "mom20_volproxy60": a_mom20,
    "max_consec_gain_20": a_maxcons_gain,
    "gain_loss_20": a_gainloss,
    "days_since_high_60": a_dshigh,
}
anchor_mats = {k: to_grid(_to_series_dict(fn)) for k, fn in ANCHORS.items()}
print("anchors built:", list(anchor_mats.keys()))


def orth_resid(cand_mat, anchor_names, passes=2, min_obs=8):
    """Per-date OLS residual of cross-sectional rank of candidate on anchor ranks."""
    T, n = cand_mat.shape
    Xs = [cross_sectional_rank(anchor_mats[k]) for k in anchor_names]
    y0 = cross_sectional_rank(cand_mat)
    cur = y0.copy()
    for _ in range(passes):
        out = np.full_like(cand_mat, np.nan)
        for t in range(T):
            cols = [np.ones(n)] + [X[t] for X in Xs]
            Xm = np.column_stack(cols)
            y = cur[t]
            ok = ~(np.isnan(y) | np.isnan(Xm).any(axis=1))
            if ok.sum() < min_obs:
                continue
            try:
                beta, *_ = np.linalg.lstsq(Xm[ok], y[ok], rcond=None)
            except Exception:
                continue
            res = y[ok] - Xm[ok] @ beta
            sd = float(np.std(res))
            if sd > 1e-12:
                res = (res - np.mean(res)) / sd
            out[t, ok] = res
        cur = out
    fin = cross_sectional_rank(cur)
    for t in range(T):
        row = fin[t]
        ok = ~np.isnan(row)
        if ok.sum() < min_obs:
            continue
        z = (row[ok] - np.mean(row[ok])) / np.std(row[ok])
        fin[t, ok] = z
    return fin


def tuw252(df):
    pk = df["close"].rolling(252, min_periods=60).max()
    uw = (df["close"] < pk).astype(float)
    return uw.rolling(252, min_periods=120).mean()


def rbreadth20(df):
    r = df["ret"].to_numpy()
    n = len(r)
    out = np.full(n, np.nan)
    if n >= 20:
        w = sliding_window_view(r, 20)
        for i in range(w.shape[0]):
            v = w[i]
            ok = ~np.isnan(v)
            if ok.sum() < 10:
                continue
            vv = v[ok]
            top5 = np.sort(vv)[-5:]
            tot = np.abs(vv).sum()
            out[i + 19] = top5.sum() / tot if tot > 1e-12 else np.nan
    return pd.Series(out, index=df.index)


m_tuw = to_grid(_to_series_dict(tuw252))
m_rb = to_grid(_to_series_dict(rbreadth20))

report_mat("orth_time_underwater", orth_resid(m_tuw,
        ["range_pos_252", "days_since_high_60", "mom20_volproxy60", "volcluster_60", "downbeta_spx_60"]))
report_mat("orth_ret_breadth", orth_resid(m_rb,
        ["max_consec_gain_20", "gain_loss_20", "mom20_volproxy60", "volcluster_60", "range_pos_252"]))
report_mat("orth_mom20", orth_resid(anchor_mats["mom20_volproxy60"],
        ["downbeta_spx_60", "volcluster_60", "range_pos_252", "spx_corr60", "calmness_20"]))
report_mat("orth_volcluster", orth_resid(anchor_mats["volcluster_60"],
        ["mom20_volproxy60", "downbeta_spx_60", "range_pos_252", "calmness_20", "spx_corr60"]))
report_mat("orth_gain_loss", orth_resid(anchor_mats["gain_loss_20"],
        ["max_consec_gain_20", "mom20_volproxy60", "volcluster_60", "range_pos_252", "downbeta_spx_60"]))
report_mat("orth_downbeta", orth_resid(anchor_mats["downbeta_spx_60"],
        ["mom20_volproxy60", "volcluster_60", "range_pos_252", "spx_corr60", "calmness_20"]))

json.dump(results, open("scripts/miner_3_20260910_batchE2_results.json", "w"), indent=1)
print("SAVED batchE2 results")
