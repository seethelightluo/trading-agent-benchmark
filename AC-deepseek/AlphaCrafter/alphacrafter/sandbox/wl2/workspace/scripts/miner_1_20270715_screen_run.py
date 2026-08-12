"""miner_1 2027-07-01 exploration: screen ~12 NEW candidate factor ideas.

Data visible through 2027-06-30 (per date.json). Uses per-asset own-calendar
factor computation reindexed on master grid; 10d rank IC gates: |IC|>=0.0070,
|ICIR|>=0.0840. Reports coverage, turnover, decay, max library correlation.
"""
import sys, json, glob, os
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
from miner_3_20260813_lib import (ASSETS, GRID, HORIZON, to_grid, summarize,
                                  safe_div, load_macro, MIN_ASSETS,
                                  cross_sectional_rank, spearman_ic_matrix,
                                  fwd_by_horizon_dict, turnover_10d_rank,
                                  library_pairwise_corr, coverage_stats)

GATE_IC = 0.0070
GATE_ICIR = 0.0840
print(f"grid: {len(GRID)} rows, {GRID[0]} .. {GRID[-1]}", flush=True)


def load_asset(sym, days=2200):
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
    df["overnight"] = df["open"] / df["close"].shift(1) - 1.0
    return df


series = {s: load_asset(s) for s in ASSETS}
series = {s: df for s, df in series.items() if df is not None and len(df) > 100}
print(f"assets loaded: {len(series)}/15 -> {sorted(series.keys())}", flush=True)

macro = {m: load_macro(m) for m in ["DXY", "USDJPY", "VIX", "USDCNY", "EURUSD"]}
macro = {m: s for m, s in macro.items() if s is not None}
macro_ret = {m: s.pct_change() for m, s in macro.items()}
print("macro loaded:", sorted(macro.keys()), flush=True)

eur_ret = macro_ret.get("EURUSD")
wti_ret = series["WTI"]["ret"] if "WTI" in series else None

fwd_ranks = fwd_by_horizon_dict(series)
fwd_rank10 = cross_sectional_rank(fwd_ranks[10])


def roll_skew(s, w, minp):
    return s.rolling(w, min_periods=minp).skew()


def roll_kurt(s, w, minp):
    return s.rolling(w, min_periods=minp).kurt()


def roll_autocorr(s, w, minp):
    return s.rolling(w, min_periods=minp).apply(
        lambda x: pd.Series(x).autocorr(1) if len(x) > 3 and np.std(x) > 0 else np.nan, raw=False)


def roll_beta(a, b, w, minp=30):
    df = pd.concat([a, b], axis=1, join="outer")
    df.columns = ["a", "b"]
    cov = df["a"].rolling(w, min_periods=minp).cov(df["b"])
    var = df["b"].rolling(w, min_periods=minp).var()
    return (cov / var).reindex(a.index)


def avg_corr_60(df, w=60):
    """Mean pairwise corr of this asset's returns vs all other assets (60d).
    Vectorized on the union calendar; windows aligned to df.index."""
    cols = {}
    for s, d in series.items():
        cols[s] = d["ret"]
    m = pd.concat(cols, axis=1)
    m.columns = list(cols.keys())
    self_name = None
    for s, d in series.items():
        if d is df:
            self_name = s
            break
    if self_name is None:
        raise ValueError("df not found in series")
    others = [c for c in m.columns if c != self_name]
    self_ret = m[self_name].reindex(df.index)
    acc = pd.DataFrame(index=df.index)
    for o in others:
        pair = pd.concat([self_ret, m[o].reindex(df.index)], axis=1)
        pair.columns = ["a", "b"]
        acc[o] = pair["a"].rolling(w, min_periods=30).corr(pair["b"])
    out = acc.mean(axis=1, skipna=True)
    out[acc.notna().sum(axis=1) < 5] = np.nan
    return out


CANDIDATES = {}


def add_cand(name, series_dict):
    CANDIDATES[name] = to_grid(series_dict)


# 1. vol term ratio 5x60
vol5 = {s: df["ret"].rolling(5, min_periods=5).std() for s, df in series.items()}
vol60 = {s: df["ret"].rolling(60, min_periods=40).std() for s, df in series.items()}
add_cand("vol_term_5x60", {s: vol5[s] / vol60[s].replace(0, np.nan) for s in series})

# 2. vol trend 60d
add_cand("vol_trend_60", {s: vol60[s] / vol60[s].shift(60) - 1.0 for s in series})

# 3. 60d skewness
add_cand("ret_skew_60", {s: roll_skew(df["ret"], 60, 40) for s, df in series.items()})

# 4. 60d kurtosis
add_cand("kurt_60", {s: roll_kurt(df["ret"], 60, 40) for s, df in series.items()})

# 5. 60d lag-1 autocorrelation
add_cand("autocorr_60", {s: roll_autocorr(df["ret"], 60, 40) for s, df in series.items()})

# 6. 20d gap momentum
add_cand("gap_mom_20", {s: df["gap"].rolling(20, min_periods=15).sum() for s, df in series.items()})

# 7. EURUSD beta 60d
if eur_ret is not None:
    add_cand("eurusd_beta_60", {s: roll_beta(df["ret"], eur_ret, 60, 30) for s, df in series.items()})

# 8. avg correlation 60d
avgc = {}
for s, df in series.items():
    df.attrs["name"] = s
    avgc[s] = avg_corr_60(df, 60)
add_cand("avg_corr_60", avgc)

# 9. range compression 20/60
rng20 = {s: df["rng_pct"].rolling(20, min_periods=15).mean() for s, df in series.items()}
rng60 = {s: df["rng_pct"].rolling(60, min_periods=40).mean() for s, df in series.items()}
add_cand("range_compress_20x60", {s: rng20[s] / rng60[s].replace(0, np.nan) for s in series})

# 10. momentum acceleration ratio 5x60
ret5 = {s: df["close"].pct_change(5) for s, df in series.items()}
ret60 = {s: df["close"].pct_change(60) for s, df in series.items()}
add_cand("mom_ratio_5x60", {s: ret5[s] / (np.abs(ret60[s]) + 0.005) for s in series})

# 11. overnight share 20d
ov20 = {s: df["overnight"].rolling(20, min_periods=15).sum() for s, df in series.items()}
tot20 = {s: (df["overnight"] + df["intraday"]).rolling(20, min_periods=15).sum() for s, df in series.items()}
add_cand("overnight_share_20", {s: ov20[s] / tot20[s].replace(0, np.nan) for s in series})

# 12. WTI beta 60d (energy sensitivity)
if wti_ret is not None:
    add_cand("wti_beta_60", {s: roll_beta(df["ret"], wti_ret, 60, 30) for s, df in series.items()})

print(f"\n=== SCREENING {len(CANDIDATES)} CANDIDATES ===", flush=True)
results = {}
for name, mat in CANDIDATES.items():
    rank_mat = cross_sectional_rank(mat)
    ics = spearman_ic_matrix(rank_mat, fwd_rank10)
    dates = np.array(GRID)
    summ = summarize(ics, dates, name, HORIZON)
    if summ is None:
        print(f"{name}: NO VALID IC DATES", flush=True)
        continue
    ic, icir = summ["ic"], summ["icir"]
    ok = (abs(ic) >= GATE_IC) and (abs(icir) >= GATE_ICIR)
    q = abs(ic) * abs(icir)
    cov_ad, cov_d8 = coverage_stats(mat)
    to = turnover_10d_rank(rank_mat)
    pc, pc_name, pc_max = library_pairwise_corr(mat)
    dec = {}
    for h, fr in fwd_ranks.items():
        icl = spearman_ic_matrix(rank_mat, fr)
        if icl:
            dec[str(h)] = round(float(np.mean([v for _, v in icl])), 4)
    print("=" * 100)
    print(f"{name}: IC={ic:+.4f} ICIR={icir:+.4f} hit={summ['hit']:.3f} n={summ['n_ic_dates']} "
          f"q={q:.5f} cov_ad={cov_ad:.3f} cov_d8={cov_d8:.3f} turn={to:.3f} "
          f"maxlibcorr={pc_max:.3f}({pc_name}) GATE={ok}", flush=True)
    print("   regime:", {k: v for k, v in summ["regime"].items()}, flush=True)
    print("   decay:", dec, flush=True)
    results[name] = {"ic": ic, "icir": icir, "q": q, "ok": ok, "hit": summ["hit"],
                     "n_ic_dates": summ["n_ic_dates"], "regime": summ["regime"], "decay": dec,
                     "coverage_asset_days": round(cov_ad, 3), "coverage_dates_ge8": round(cov_d8, 3),
                     "turnover_10d_rank": round(to, 3),
                     "max_abs_library_correlation": round(pc_max, 4),
                     "max_lib_corr_name": pc_name}

with open("scripts/miner_1_20270715_screen_results.json", "w") as f:
    json.dump(results, f, indent=1, default=str)
print("\nPASSING (|IC|>=0.007 & |ICIR|>=0.084):")
for k, v in sorted(results.items(), key=lambda kv: -abs(kv[1]["q"])):
    if v["ok"]:
        print(f"  {k}: IC={v['ic']:+.4f} ICIR={v['icir']:+.4f} q={v['q']:.5f} maxlibcorr={v['max_abs_library_correlation']:.3f}")
