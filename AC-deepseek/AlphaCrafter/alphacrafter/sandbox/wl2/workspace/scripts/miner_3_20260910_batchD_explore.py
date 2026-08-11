"""miner_3 2026-09-10 exploration batch D: DECORRELATION-TARGETED novel families.

Motivation: batches A/B/C all passed IC/ICIR gates but were evicted by the
pairwise-correlation gate (|rho|<0.5 vs kept library members). This batch
deliberately targets signal families orthogonal to the incumbent library:
 - alpha (pricing-error) vs SPX and vs an equal-weight world index (library has
   betas/corrs, not alphas)
 - beta asymmetry spread (upbeta - downbeta; library has downbeta only)
 - volatility relative to macro fear gauge VIX (idiosyncratic vol)
 - trend consistency / breadth (steadiness, not level)
 - calendar seasonality (weekday / month-start effects; uses no price structure)
 - return concentration, time-underwater, drawdown slope, 60d Sharpe/efficiency

Gates: |IC|>=0.0070, |ICIR|>=0.0840 on daily cross-sectional Spearman vs fwd 10d;
max_abs_library_correlation < 0.5 vs factors/*.signal.npy artifacts.
"""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
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
    df["rng_pct"] = (df["high"] - df["low"]) / df["close"]
    return df


def ols_stats(x, y, w=60, mp=40):
    """Vectorized rolling OLS: returns alpha, beta, tstat(alpha)."""
    x = x.astype(float)
    y = y.astype(float)
    n = x.rolling(w, min_periods=mp).count()
    Sx = x.rolling(w, min_periods=mp).sum()
    Sy = y.rolling(w, min_periods=mp).sum()
    Sxx = (x * x).rolling(w, min_periods=mp).sum()
    Syy = (y * y).rolling(w, min_periods=mp).sum()
    Sxy = (x * y).rolling(w, min_periods=mp).sum()
    denom = (n * Sxx - Sx * Sx)
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = (n * Sxy - Sx * Sy) / denom
        alpha = (Sy - beta * Sx) / n
        SSE = Syy - alpha * Sy - beta * Sxy
        s2 = SSE / (n - 2.0)
        var_b = s2 / denom
        var_a = var_b * Sxx / n
        t_a = alpha / np.sqrt(var_a)
    for s in (beta, alpha, t_a):
        s[np.isinf(s)] = np.nan
    return alpha, beta, t_a


series = {s: load_asset(s) for s in ASSETS}
series = {s: df for s, df in series.items() if df is not None and len(df) > 100}
print(f"assets loaded: {len(series)}/15 -> {sorted(series.keys())}")
fwd = fwd_by_horizon_dict(series)
dates = np.array(GRID)
results = {}

# world ex-self mean return on master grid
R = pd.DataFrame(to_grid({s: df["ret"] for s, df in series.items()}), index=GRID, columns=ASSETS)


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


spx_ret = series["SPX"]["ret"]

# 1. spx_alpha_tstat_60: t-stat of 60d Jensen alpha vs SPX (library has beta/corr, not alpha)
cand = {}
for s, df in series.items():
    a, b, t = ols_stats(spx_ret.reindex(df.index), df["ret"], 60, 40)
    cand[s] = t
report("spx_alpha_tstat_60", cand)

# 2. spx_alpha_60: raw 60d alpha vs SPX
cand = {}
for s, df in series.items():
    a, b, t = ols_stats(spx_ret.reindex(df.index), df["ret"], 60, 40)
    cand[s] = a
report("spx_alpha_60", cand)

# 3. world_alpha_tstat_60: alpha t-stat vs equal-weight world (ex-self) index
cand = {}
for s, df in series.items():
    w = R.drop(columns=[s]).mean(axis=1).reindex(df.index)
    a, b, t = ols_stats(w, df["ret"], 60, 40)
    cand[s] = t
report("world_alpha_tstat_60", cand)

# 4. world_beta_60: beta vs world ex-self index
cand = {}
for s, df in series.items():
    w = R.drop(columns=[s]).mean(axis=1).reindex(df.index)
    a, b, t = ols_stats(w, df["ret"], 60, 40)
    cand[s] = b
report("world_beta_60", cand)

# 5. beta_asym_spx_60: upbeta(60d, SPX>0) - downbeta(60d, SPX<0) spread
spx_s = spx_ret.reindex(GRID)
cand = {}
for s, df in series.items():
    r = df["ret"]
    j = pd.concat([r, spx_s], axis=1, join="outer")
    j.columns = ["a", "b"]
    up = j["b"] > 0
    dn = j["b"] < 0
    cov_up = j["a"].where(up).rolling(60, min_periods=15).cov(j["b"].where(up))
    var_up = j["b"].where(up).rolling(60, min_periods=15).var()
    cov_dn = j["a"].where(dn).rolling(60, min_periods=15).cov(j["b"].where(dn))
    var_dn = j["b"].where(dn).rolling(60, min_periods=15).var()
    bu = pd.Series(safe_div(cov_up, var_up), index=j.index)
    bd = pd.Series(safe_div(cov_dn, var_dn), index=j.index)
    cand[s] = (bu - bd).reindex(df.index)
report("beta_asym_spx_60", cand)

# 6. vol_vix_ratio_20: 20d annualized vol scaled by VIX fear-gauge level
vix = load_macro("VIX")
vix_l = vix.reindex(GRID)
cand = {}
for s, df in series.items():
    v20 = df["ret"].rolling(20, min_periods=10).std() * np.sqrt(252)
    cand[s] = pd.Series(safe_div(v20, vix_l.reindex(df.index)), index=df.index)
report("vol_vix_ratio_20", cand)

# 7. mom_consistency_60: fraction of days in last 60 where 20d skip5 momentum > 0
cand = {}
for s, df in series.items():
    m20 = df["close"].pct_change(20).shift(5)
    frac = m20.gt(0).rolling(60, min_periods=30).mean()
    cand[s] = frac
report("mom_consistency_60", cand)

# 8. monday_effect_60: avg Monday ret minus avg other-weekday ret (60d)
cand = {}
for s, df in series.items():
    dow = pd.to_datetime(df.index).dayofweek
    mon = df["ret"].where(dow == 0)
    oth = df["ret"].where(dow != 0)
    diff = mon.rolling(60, min_periods=8).mean() - oth.rolling(60, min_periods=30).mean()
    cand[s] = diff * 100.0
report("monday_effect_60", cand)

# 9. month_start_60: avg ret of first-3 days of month minus rest, trailing 60d
cand = {}
for s, df in series.items():
    dt = pd.to_datetime(df.index)
    dom = dt.day
    # trading-day-of-month via group rank
    todm = df.groupby(dt.to_period("M")).cumcount()
    is_start = todm < 3
    st = df["ret"].where(is_start)
    ot = df["ret"].where(~is_start)
    diff = st.rolling(60, min_periods=6).mean() - ot.rolling(60, min_periods=30).mean()
    cand[s] = diff * 100.0
report("month_start_60", cand)

# 10. ret_breadth_20: share of |20d gross return| from 5 best days (concentration)
cand = {}
for s, df in series.items():
    from numpy.lib.stride_tricks import sliding_window_view
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
    cand[s] = pd.Series(out, index=df.index)
report("ret_breadth_20", cand)

# 11. time_underwater_252: fraction of days below trailing 252d peak
cand = {}
for s, df in series.items():
    pk = df["close"].rolling(252, min_periods=60).max()
    uw = (df["close"] < pk).astype(float)
    cand[s] = uw.rolling(252, min_periods=120).mean()
report("time_underwater_252", cand)

# 12. drawdown_slope_60: depth per day since 252d peak (speed of decline)
cand = {}
for s, df in series.items():
    pk = df["close"].rolling(252, min_periods=60).max()
    depth = safe_div(df["close"] - pk, pk)
    # days since the rolling max was last reached
    c = df["close"].to_numpy()
    n2 = len(c)
    am = np.full(n2, np.nan)
    if n2 >= 252:
        w = sliding_window_view(c, 252)
        for i in range(w.shape[0]):
            v = w[i]
            ok = ~np.isnan(v)
            if ok.sum() < 60:
                continue
            vv = v[ok]
            am[i + 251] = np.argmax(vv)
    days = 251.0 - am
    cand[s] = pd.Series(safe_div(depth, days + 1.0), index=df.index)
report("drawdown_slope_60", cand)

# 13. roll_sharpe_60: rolling 60d Sharpe
cand = {}
for s, df in series.items():
    m = df["ret"].rolling(60, min_periods=30).mean()
    sd = df["ret"].rolling(60, min_periods=30).std()
    cand[s] = pd.Series(safe_div(m, sd) * np.sqrt(252), index=df.index)
report("roll_sharpe_60", cand)

# 14. eff_ratio_60: |net 60d move| / sum of |daily moves| (path efficiency)
cand = {}
for s, df in series.items():
    net = df["close"].pct_change(60).abs()
    gross = df["ret"].abs().rolling(60, min_periods=30).sum()
    cand[s] = pd.Series(safe_div(net, gross), index=df.index)
report("eff_ratio_60", cand)

json.dump(results, open("scripts/miner_3_20260910_batchD_results.json", "w"), indent=1)
print("SAVED batchD results")
