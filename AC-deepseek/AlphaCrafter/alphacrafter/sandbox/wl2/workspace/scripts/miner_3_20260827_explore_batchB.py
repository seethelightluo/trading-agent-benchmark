"""miner_3 2026-08-27 exploration batch B: risk-adjusted / regime-beta family.

Candidates (per-asset own-calendar unless noted):
 1. sortino_60        : 60d return / 60d downside deviation (reward-to-bad-risk)
 2. ratespread_beta_60: rolling 60d beta of asset rets vs (US10Y - CN10Y) daily change
 3. xau_downbeta_60   : rolling 60d beta of asset rets vs XAU ret on XAU-down days (safe-haven unwind sensitivity)
 4. updown_beta_ratio_60: mean asset ret on SPX-up days / |mean asset ret on SPX-down days| (asymmetry)
 5. dxy_downbeta_60   : rolling 60d beta of asset rets vs DXY ret on DXY-down days (risk-on sensitivity)
Gates: |IC|>=0.007, |ICIR|>=0.084 on daily cross-sectional Spearman vs fwd 10d.
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
fwd = fwd_by_horizon_dict(series)
dates = np.array(GRID)
results = {}


def report(name, mat, extra=""):
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
    ic, icir = summ["ic"], summ["icir"]
    ok = (abs(ic) >= 0.007) and (abs(icir) >= 0.084)
    print("=" * 80)
    print(f"{name}: IC={ic:+.4f} ICIR={icir:+.4f} hit={summ['hit']:.3f} n={summ['n_ic_dates']} "
          f"cov_ad={cov_ad:.3f} cov_d8={cov_d8:.3f} turn={to:.3f} maxlibcorr={mx_abs:.3f} ({mx_name}) PASS={ok}")
    print("   regime:", {k: v for k, v in summ["regime"].items()})
    print("   decay:", dec)
    if ok:
        print("   >>> CANDIDATE PASSES GATE", extra)
    results[name] = {
        "ic": round(ic, 5), "icir": round(icir, 5), "hit": round(summ["hit"], 4),
        "n_ic_dates": summ["n_ic_dates"], "regime": summ["regime"],
        "coverage_asset_days": round(cov_ad, 4), "coverage_dates_ge8": round(cov_d8, 4),
        "turnover_10d_rank": round(to, 4), "decay": dec,
        "max_abs_library_correlation": round(mx_abs, 4),
        "max_lib_corr_name": mx_name, "pass_gate": bool(ok),
    }
    return summ


def rolling_beta(y, x, w=60, minp=40):
    """Rolling beta of y on x (both pd.Series aligned to same index)."""
    out = pd.Series(np.nan, index=y.index)
    yv = y.values.astype(float)
    xv = x.values.astype(float)
    n = len(y)
    for i in range(n):
        a = max(0, i - w + 1)
        yy = yv[a:i + 1]
        xx = xv[a:i + 1]
        ok = ~(np.isnan(yy) | np.isnan(xx))
        if ok.sum() < minp:
            continue
        xc = xx[ok] - xx[ok].mean()
        yc = yy[ok] - yy[ok].mean()
        denom = np.sum(xc * xc)
        if denom < 1e-12:
            continue
        out.iloc[i] = np.sum(xc * yc) / denom
    return out


# conditioning series on master grid
us10y = series["US10Y"]["ret"]
cn10y = series["CN10Y"]["ret"]
spread_ret = (us10y - cn10y)
xau_ret = series["XAU"]["ret"]
spx_ret = series["SPX"]["ret"]
dxy = load_macro("DXY")
if dxy is not None:
    dxy = dxy.reindex(GRID)
    dxy_ret = dxy.pct_change()

# 1. sortino 60
so = {}
for s, df in series.items():
    r = df["ret"]
    dd = r.where(r < 0, np.nan)
    ddstd = dd.rolling(60, min_periods=30).std()
    mom = df["close"] / df["close"].shift(60) - 1.0
    so[s] = pd.Series(safe_div(mom, ddstd), index=df.index)
report("sortino_60", to_grid(so))

# 2. ratespread beta 60
rsb = {}
for s, df in series.items():
    x = spread_ret.reindex(df.index)
    rsb[s] = rolling_beta(df["ret"], x, 60, 40)
report("ratespread_beta_60", to_grid(rsb))

# 3. xau down-beta 60 (condition on XAU daily return < 0)
xdb = {}
for s, df in series.items():
    x = xau_ret.reindex(df.index)
    cond = x < 0
    xc = x.where(cond, np.nan)
    y = df["ret"].where(cond, np.nan)
    xdb[s] = rolling_beta(y, xc, 60, 40)
report("xau_downbeta_60", to_grid(xdb))

# 4. up/down beta ratio vs SPX 60
udr = {}
for s, df in series.items():
    x = spx_ret.reindex(df.index)
    up = x > 0
    dn = x < 0
    mu_up = df["ret"].where(up, np.nan).rolling(60, min_periods=25).mean()
    mu_dn = df["ret"].where(dn, np.nan).rolling(60, min_periods=25).mean()
    udr[s] = pd.Series(safe_div(mu_up, mu_dn.abs()), index=df.index)
report("updown_beta_ratio_60", to_grid(udr))

# 5. dxy down-beta 60 (condition on DXY ret < 0: risk-on days)
if dxy_ret is not None:
    ddb = {}
    for s, df in series.items():
        x = dxy_ret.reindex(df.index)
        cond = x < 0
        xc = x.where(cond, np.nan)
        y = df["ret"].where(cond, np.nan)
        ddb[s] = rolling_beta(y, xc, 60, 40)
    report("dxy_downbeta_60", to_grid(ddb))

json.dump(results, open("scripts/miner_3_20260827_batchB_results.json", "w"), indent=1)
print("SAVED batchB results")
