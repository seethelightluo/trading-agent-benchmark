"""miner_3 2026-09-10 exploration batch E: decorrelation-focused batch 2.

Motivation: every natural price-structure candidate in batches A-D maps onto a
handful of dominant cross-sectional dimensions already in the library (trend
position, vol level, beta), so max|rho|>0.5 almost always. Batch E:
 (a) conditional/tail constructs (extreme-tail beta, calmness variants,
     quiet-conditional drift, streak-duration asymmetry, gap skew, aroon,
     asymmetric vol) that may sit in a different niche than the library, and
 (b) explicit rank-residual (orthogonalized) versions of the strongest raw
     signals (time_underwater_252 IC -0.060, ret_breadth_20 IC +0.048,
     mom20_volproxy60, volcluster_60, gain_loss_20, downbeta_spx_60), where the
     cross-sectional rank of the raw signal is regressed (per date, OLS) on the
     contemporaneous ranks of kept library anchors and the standardized residual
     is the factor. This directly targets the admission gate (pairwise Spearman
     rho < 0.5) while the IC/ICIR gates still have to be passed by the residual,
     i.e. only genuinely incremental predictive content survives.

Gates: |IC|>=0.0070, |ICIR|>=0.0840 on daily cross-sectional Spearman vs fwd 10d;
max_abs_library_correlation < 0.5 vs factors/*.signal.npy artifacts.
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
                                  coverage_stats, safe_div, load_macro)

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


def report(name, cand):
    mat = to_grid(cand)
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


# ---------------- kept-library anchor signal matrices (recomputed on own calendars)
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
    # final: z-score the residual ranks
    fin = cross_sectional_rank(cur)
    for t in range(T):
        row = fin[t]
        ok = ~np.isnan(row)
        if ok.sum() < min_obs:
            continue
        z = (row[ok] - np.mean(row[ok])) / np.std(row[ok])
        fin[t, ok] = z
    return fin


# ---------------- natural candidates
# 1. tailbeta_spx_60: beta vs SPX on extreme down days (SPX ret < -1.5*sigma60)
spx_sig = spx_ret.rolling(60, min_periods=30).std()
thr = -1.5 * spx_sig.reindex(GRID)
cand = {}
for s, df in series.items():
    r = df["ret"]
    j = pd.concat([r, spx_ret], axis=1, join="outer")
    j.columns = ["a", "b"]
    tail = j["b"] < thr.reindex(j.index)
    cov = j["a"].where(tail).rolling(120, min_periods=8).cov(j["b"].where(tail))
    var = j["b"].where(tail).rolling(120, min_periods=8).var()
    cand[s] = pd.Series(safe_div(cov, var), index=j.index).reindex(df.index)
report("tailbeta_spx_120", cand)

# 2. calmness_60: calmness family with 60d window
cand = {}
for s, df in series.items():
    sd = df["ret"].rolling(60, min_periods=30).std()
    calm = (df["ret"].abs() < 0.5 * sd).astype(float)
    cand[s] = calm.rolling(60, min_periods=30).mean()
report("calmness_60", cand)

# 3. calm_asym_20: calm-fraction on up days minus calm-fraction on down days
cand = {}
for s, df in series.items():
    sd = df["ret"].rolling(20, min_periods=10).std()
    calm = df["ret"].abs() < 0.5 * sd
    up = calm.where(df["ret"] > 0)
    dn = calm.where(df["ret"] < 0)
    cand[s] = (up.rolling(20, min_periods=8).mean() - dn.rolling(20, min_periods=8).mean())
report("calm_asym_20", cand)

# 4. quiet_drift_20: mean return on calm days (|ret|<0.5*std20), min 6 obs
cand = {}
for s, df in series.items():
    sd = df["ret"].rolling(20, min_periods=10).std()
    calm = df["ret"].abs() < 0.5 * sd
    mu = df["ret"].where(calm).rolling(20, min_periods=6).mean()
    cand[s] = mu * 100.0
report("quiet_drift_20", cand)

# 5. aroon_25: trend-timing oscillator (days since 25d high vs low)
cand = {}
for s, df in series.items():
    c = df["close"].to_numpy()
    n = len(c)
    hi = np.full(n, np.nan)
    lo = np.full(n, np.nan)
    if n >= 25:
        w = sliding_window_view(c, 25)
        for i in range(w.shape[0]):
            v = w[i]
            ok = ~np.isnan(v)
            if ok.sum() < 15:
                continue
            vv = v[ok]
            dhi = len(vv) - 1 - np.argmax(vv[::-1] == np.nanmax(vv))
            dlo = len(vv) - 1 - np.argmin(vv[::-1] == np.nanmin(vv))
            hi[i + 24] = dhi
            lo[i + 24] = dlo
    aroon = (25 - hi) / 25.0 * 100.0 - (25 - lo) / 25.0 * 100.0
    cand[s] = pd.Series(aroon, index=df.index)
report("aroon_25", cand)

# 6. vol_sym_20: (std of up-day rets - std of down-day rets)/std(all)
cand = {}
for s, df in series.items():
    r = df["ret"]
    up = r.where(r > 0).rolling(20, min_periods=8).std()
    dn = r.where(r < 0).rolling(20, min_periods=8).std()
    sd = r.rolling(20, min_periods=10).std()
    cand[s] = pd.Series(safe_div(up - dn, sd), index=df.index)
report("vol_sym_20", cand)

# 7. updown_duration_20: avg up-streak length minus avg down-streak length (20d)
cand = {}
for s, df in series.items():
    r = df["ret"].to_numpy()
    n = len(r)
    up_dur = np.full(n, np.nan)
    dn_dur = np.full(n, np.nan)
    if n >= 20:
        w = sliding_window_view(r, 20)
        for i in range(w.shape[0]):
            v = w[i]
            ok = ~np.isnan(v)
            if ok.sum() < 10:
                continue
            vv = v[ok]
            ups, dns = [], []
            cur_u = cur_d = 0
            for x in vv:
                if x > 0:
                    cur_u += 1
                    if cur_d > 0:
                        dns.append(cur_d)
                        cur_d = 0
                elif x < 0:
                    cur_d += 1
                    if cur_u > 0:
                        ups.append(cur_u)
                        cur_u = 0
            if cur_u:
                ups.append(cur_u)
            if cur_d:
                dns.append(cur_d)
            au = np.mean(ups) if ups else 0.0
            ad = np.mean(dns) if dns else 0.0
            up_dur[i + 19] = au
            dn_dur[i + 19] = ad
    cand[s] = pd.Series(up_dur - dn_dur, index=df.index)
report("updown_duration_20", cand)

# 8. gap_skew_20: skewness of overnight gaps over 20d
cand = {}
for s, df in series.items():
    g = df["open"] / df["close"].shift(1) - 1.0
    cand[s] = g.rolling(20, min_periods=10).skew()
report("gap_skew_20", cand)

# ---------------- orthogonalized (rank-residual) candidates
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

report("orth_time_underwater", orth_resid(m_tuw,
        ["range_pos_252", "days_since_high_60", "mom20_volproxy60", "volcluster_60", "downbeta_spx_60"]))
report("orth_ret_breadth", orth_resid(m_rb,
        ["max_consec_gain_20", "gain_loss_20", "mom20_volproxy60", "volcluster_60", "range_pos_252"]))
report("orth_mom20", orth_resid(anchor_mats["mom20_volproxy60"],
        ["downbeta_spx_60", "volcluster_60", "range_pos_252", "spx_corr60", "calmness_20"]))
report("orth_volcluster", orth_resid(anchor_mats["volcluster_60"],
        ["mom20_volproxy60", "downbeta_spx_60", "range_pos_252", "calmness_20", "spx_corr60"]))
report("orth_gain_loss", orth_resid(anchor_mats["gain_loss_20"],
        ["max_consec_gain_20", "mom20_volproxy60", "volcluster_60", "range_pos_252", "downbeta_spx_60"]))
report("orth_downbeta", orth_resid(anchor_mats["downbeta_spx_60"],
        ["mom20_volproxy60", "volcluster_60", "range_pos_252", "spx_corr60", "calmness_20"]))

json.dump(results, open("scripts/miner_3_20260910_batchE_results.json", "w"), indent=1)
print("SAVED batchE results")
