"""miner_3 2028-03-23 full library re-validation (data visible through 2028-03-22).

Re-validates every persisted library factor on full 2020-01-01..2028-03-22 history
plus regime split and last-250d freshness. Uses rank IC at 10d horizon.
Gates: |IC|>=0.0070, |ICIR|>=0.0840.
Fixed cross-asset alignment: macro/SPX series reindexed to each asset's own calendar.
"""
import sys, json, os
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
from miner_3_20260813_lib import (ASSETS, GRID, HORIZON, to_grid,
                                  summarize, safe_div, load_macro, MIN_ASSETS,
                                  cross_sectional_rank, library_pairwise_corr,
                                  turnover_10d_rank, spearman_ic_matrix,
                                  fwd_by_horizon_dict, decay_curve, coverage_stats)

GATE_IC = 0.0070
GATE_ICIR = 0.0840
OUT = "scripts/miner_3_20280323_revalidate_results.json"

print(f"grid rows: {len(GRID)} (2020-01-01 .. {GRID[-1]})", flush=True)


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
    df["gap"] = df["open"] / df["close"].shift(1) - 1.0
    df["rng_pct"] = (df["high"] - df["low"]) / df["close"]
    df["intraday"] = df["close"] / df["open"] - 1.0
    df["hl_pos"] = (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
    return df


series = {s: load_asset(s) for s in ASSETS}
series = {s: df for s, df in series.items() if df is not None and len(df) > 100}
print(f"assets loaded: {len(series)}/15 -> {sorted(series.keys())}", flush=True)
print(f"data end: {max(df.index[-1] for df in series.values())}", flush=True)

spx_ret = series["SPX"]["ret"] if "SPX" in series else None
macro = {m: load_macro(m) for m in ["DXY", "USDJPY", "VIX", "USDCNY", "EURUSD"]}
macro = {m: s for m, s in macro.items() if s is not None}
macro_ret = {m: s.pct_change() for m, s in macro.items()}
print("macro loaded:", sorted(macro.keys()), flush=True)

fwd_by_h = fwd_by_horizon_dict(series)
fwd10 = fwd_by_h[HORIZON]


def rstd(x, w, minp):
    return x.rolling(w, min_periods=minp).std()


def roll_beta(y, x, w, minp):
    """beta of y on x over window w; x reindexed to y's index first."""
    x = x.reindex(y.index)
    y = y.astype(float); x = x.astype(float)
    xy = (y * x).rolling(w, min_periods=minp).mean()
    xx = (x * x).rolling(w, min_periods=minp).mean()
    mx = x.rolling(w, min_periods=minp).mean()
    my = y.rolling(w, min_periods=minp).mean()
    cov = xy - mx * my
    var = xx - mx * mx
    return safe_div(cov, var)


def rmean(x, w, minp):
    return x.rolling(w, min_periods=minp).mean()


factors = {}

# 1 max_consec_gain_20: streak of positive days over 20d
for s, df in series.items():
    pos = (df["ret"] > 0).astype(float)
    streak = pos * (pos.groupby((pos != pos.shift()).cumsum()).cumsum())
    factors.setdefault("max_consec_gain_20", {})[s] = streak.rolling(20, min_periods=10).max()

# 2 max_consec_loss_20
for s, df in series.items():
    neg = (df["ret"] < 0).astype(float)
    streak = neg * (neg.groupby((neg != neg.shift()).cumsum()).cumsum())
    factors.setdefault("max_consec_loss_20", {})[s] = streak.rolling(20, min_periods=10).max()

# 3 mom20_volproxy60
for s, df in series.items():
    mom20 = df["close"].shift(5) / df["close"].shift(25) - 1.0
    vol60 = rstd(df["ret"], 60, 15)
    factors.setdefault("mom20_volproxy60", {})[s] = safe_div(mom20, vol60)

# 4 spx_corr60
for s, df in series.items():
    if spx_ret is None:
        continue
    if s == "SPX":
        factors.setdefault("spx_corr60", {})[s] = pd.Series(1.0, index=df.index)
    else:
        factors.setdefault("spx_corr60", {})[s] = (
            df["ret"].rolling(60, min_periods=30).corr(spx_ret.reindex(df.index)))

# 5 mom_20d_skip5
for s, df in series.items():
    factors.setdefault("mom_20d_skip5", {})[s] = df["close"].shift(5) / df["close"].shift(25) - 1.0

# 6 gain_loss_20
for s, df in series.items():
    g = df["ret"].clip(lower=0).rolling(20, min_periods=10).mean()
    l = (-df["ret"].clip(upper=0)).rolling(20, min_periods=10).mean()
    factors.setdefault("gain_loss_20", {})[s] = safe_div(g - l, g + l)

# 7 downbeta_spx_60
for s, df in series.items():
    if spx_ret is None:
        continue
    if s == "SPX":
        factors.setdefault("downbeta_spx_60", {})[s] = pd.Series(1.0, index=df.index)
    else:
        sret = spx_ret.reindex(df.index)
        down = (sret < 0)
        r_d = df["ret"].where(down)
        x_d = sret.where(down)
        factors.setdefault("downbeta_spx_60", {})[s] = roll_beta(r_d, x_d, 60, minp=30)

# 8 usdjpy_beta_cond_120x60
for s, df in series.items():
    jpy = macro_ret.get("USDJPY")
    if jpy is None:
        continue
    beta = roll_beta(df["ret"], jpy, 120, minp=60)
    jpy_mom = jpy.reindex(df.index).rolling(60, min_periods=30).mean()
    factors.setdefault("usdjpy_beta_cond_120x60", {})[s] = (beta * jpy_mom).reindex(df.index)

# 9 volcluster_60
for s, df in series.items():
    rv = rstd(df["ret"], 20, 5)
    factors.setdefault("volcluster_60", {})[s] = rv.rolling(60, min_periods=15).mean()

# 10 calmness_20
for s, df in series.items():
    factors.setdefault("calmness_20", {})[s] = -rstd(df["ret"], 20, 10)

# 11 days_since_high_60
for s, df in series.items():
    roll_max = df["close"].rolling(60, min_periods=30).max()
    factors.setdefault("days_since_high_60", {})[s] = df["close"] / roll_max - 1.0

# 12 close_pos_20
for s, df in series.items():
    lo = df["low"].rolling(20, min_periods=10).min()
    hi = df["high"].rolling(20, min_periods=10).max()
    factors.setdefault("close_pos_20", {})[s] = safe_div(df["close"] - lo, hi - lo)

# 13 lagbeta_spx_60
for s, df in series.items():
    if spx_ret is None:
        continue
    if s == "SPX":
        factors.setdefault("lagbeta_spx_60", {})[s] = pd.Series(1.0, index=df.index)
    else:
        factors.setdefault("lagbeta_spx_60", {})[s] = roll_beta(df["ret"], spx_ret.shift(1), 60, minp=30)

# 14 intraday_drift_20
for s, df in series.items():
    factors.setdefault("intraday_drift_20", {})[s] = rmean(df["intraday"], 20, 10)

# 15 dxy_beta_cond_60x20
for s, df in series.items():
    dxy = macro_ret.get("DXY")
    if dxy is None:
        continue
    beta = roll_beta(df["ret"], dxy, 60, minp=30)
    dxy_mom = dxy.reindex(df.index).rolling(20, min_periods=10).mean()
    factors.setdefault("dxy_beta_cond_60x20", {})[s] = (beta * dxy_mom).reindex(df.index)

# 16 vix_beta_cond_60x20
for s, df in series.items():
    vix = macro_ret.get("VIX")
    if vix is None:
        continue
    beta = roll_beta(df["ret"], vix, 60, minp=30)
    vix_mom = vix.reindex(df.index).rolling(20, min_periods=10).mean()
    factors.setdefault("vix_beta_cond_60x20", {})[s] = (beta * vix_mom).reindex(df.index)

# 17 mom_10d_skip5
for s, df in series.items():
    factors.setdefault("mom_10d_skip5", {})[s] = df["close"].shift(5) / df["close"].shift(15) - 1.0

# 18 mom_120d_skip5
for s, df in series.items():
    factors.setdefault("mom_120d_skip5", {})[s] = df["close"].shift(5) / df["close"].shift(125) - 1.0

# 19 mom_180d_skip5
for s, df in series.items():
    factors.setdefault("mom_180d_skip5", {})[s] = df["close"].shift(5) / df["close"].shift(185) - 1.0

# 20 mom30_vol60
for s, df in series.items():
    mom30 = df["close"].shift(5) / df["close"].shift(35) - 1.0
    vol60 = rstd(df["ret"], 60, 15)
    factors.setdefault("mom30_vol60", {})[s] = safe_div(mom30, vol60)

# 21 range_pos_252
for s, df in series.items():
    pos = df["close"] / df["close"].shift(252) - 1.0
    rng = df["close"].rolling(252, min_periods=120).max() - df["close"].rolling(252, min_periods=120).min()
    factors.setdefault("range_pos_252", {})[s] = safe_div(pos, rng / df["close"])

# 22 vol_of_vol20x60
for s, df in series.items():
    rv = rstd(df["ret"], 20, 5)
    factors.setdefault("vol_of_vol20x60", {})[s] = rv.rolling(60, min_periods=15).std()

results = {}
dates = np.array(GRID)
for fid, panel in factors.items():
    mat = to_grid(panel)
    rank_mat = cross_sectional_rank(mat)
    ics = spearman_ic_matrix(rank_mat, fwd10)
    if len(ics) == 0:
        continue
    s = summarize(ics, dates, fid, HORIZON)
    if s is None:
        continue
    rho_dict, rho_name, max_rho = library_pairwise_corr(mat)
    s["max_abs_library_correlation"] = round(max_rho, 4)
    s["max_corr_with"] = rho_name
    s["turnover_10d_rank"] = round(turnover_10d_rank(rank_mat), 4)
    cov, dates_ge8 = coverage_stats(mat)
    s["coverage"] = round(cov, 4)
    s["dates_ge8_frac"] = round(dates_ge8, 4)
    s["decay"] = decay_curve(rank_mat, fwd_by_h)
    s["ok"] = (abs(s["ic"]) >= GATE_IC) and (abs(s["icir"]) >= GATE_ICIR)
    s.pop("idx", None); s.pop("icv", None)
    results[fid] = s
    print(f"{fid:28s} ic={s['ic']:+.4f} icir={s['icir']:+.4f} hit={s['hit']:.3f} "
          f"turn={s['turnover_10d_rank']:.3f} cov={s['coverage']:.3f} maxrho={max_rho:.3f} "
          f"decay={s['decay']} ok={s['ok']}", flush=True)

json.dump(results, open(OUT, "w"), indent=1, default=str)
print("DONE. saved", OUT)
