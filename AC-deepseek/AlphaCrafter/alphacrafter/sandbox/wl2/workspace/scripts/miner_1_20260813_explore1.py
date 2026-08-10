"""miner_1 2026-08-13: explore novel factor family (wick rejection, autocorr, USDCNY-beta, BTC-corr, drawdown)."""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_1_20260813_lib import (ASSETS, GRID, HORIZON, N_GRID, to_grid, load_asset,
                                  load_macro, cross_sectional_rank, spearman_ic_matrix,
                                  summarize, decay_curve, fwd_by_horizon_dict,
                                  turnover_10d_rank, library_pairwise_corr, coverage_stats,
                                  safe_div, roll_mean, roll_std)

series = {s: load_asset(s) for s in ASSETS}
series = {s: df for s, df in series.items() if df is not None and len(df) > 100}
fwd = fwd_by_horizon_dict(series)
dates = np.array(GRID)

# ---- macro: USDCNY ----
usdcny = load_macro("USDCNY").reindex(GRID)
usdcny_ret = usdcny.pct_change()
btc_close = series.get("BTC", None)

cands = {}

# 1. upper_wick_20: rejection sentiment (mean upper shadow share)
uw = {}
for s, df in series.items():
    hi, lo, op, cl = df["high"], df["low"], df["open"], df["close"]
    rng = (hi - lo).replace(0, np.nan)
    upper = safe_div(hi - np.maximum(op, cl), rng)
    uw[s] = pd.Series(upper, index=df.index).rolling(20, min_periods=10).mean()
cands["upper_wick_20"] = uw

# 2. lower_wick_20: support sentiment (mean lower shadow share)
lw = {}
for s, df in series.items():
    hi, lo, op, cl = df["high"], df["low"], df["open"], df["close"]
    rng = (hi - lo).replace(0, np.nan)
    lower = safe_div(np.minimum(op, cl) - lo, rng)
    lw[s] = pd.Series(lower, index=df.index).rolling(20, min_periods=10).mean()
cands["lower_wick_20"] = lw

# 3. wick_asym_20: net rejection (upper - lower)
cands["wick_asym_20"] = {s: uw[s] - lw[s] for s in series}

# 4. autocorr_20 / 5. autocorr_60: 1-lag return autocorrelation
def autocorr_ser(s, w):
    cl = series[s]["close"]
    r = cl.pct_change()
    a = r.rolling(w, min_periods=max(10, w // 2)).corr(r.shift(1))
    return a
cands["autocorr_20"] = {s: autocorr_ser(s, 20) for s in series}
cands["autocorr_60"] = {s: autocorr_ser(s, 60) for s in series}

# 6. usdcny_beta_cond_60x20: rolling beta to USDCNY * 20d CNY momentum
ub = {}
for s, df in series.items():
    r = df["close"].pct_change()
    b = r.rolling(60, min_periods=30).cov(usdcny_ret) / usdcny_ret.rolling(60, min_periods=30).var()
    mom20 = usdcny / usdcny.shift(20) - 1.0
    ub[s] = pd.Series(b, index=df.index) * mom20
cands["usdcny_beta_cond_60x20"] = ub

# 7. btc_corr_60: 60d correlation of asset returns to BTC returns
bc = {}
if btc_close is not None:
    btc_ret = btc_close["close"].pct_change()
    for s, df in series.items():
        c = df["close"].pct_change().rolling(60, min_periods=30).corr(btc_ret)
        bc[s] = c
cands["btc_corr_60"] = bc

# 8. drawdown_60: close/rolling_max(close,60) - 1 (depth below 60d high)
dd = {}
for s, df in series.items():
    mx = df["close"].rolling(60, min_periods=20).max()
    dd[s] = df["close"] / mx - 1.0
cands["drawdown_60"] = dd

print(f"{'cand':22s} {'IC':>8s} {'ICIR':>7s} {'hit':>6s} {'n':>5s} {'covAD':>6s} {'covD8':>6s} {'turn':>6s} {'maxcorr':>8s}  pass")
for name, d in cands.items():
    mat = to_grid(d)
    rank_mat = cross_sectional_rank(mat)
    ics = spearman_ic_matrix(mat, fwd[10])
    summ = summarize(ics, dates, name, HORIZON)
    if summ is None:
        print(name, "NO IC DATES"); continue
    cov_ad, cov_d8 = coverage_stats(mat)
    to_ = turnover_10d_rank(rank_mat)
    corrs, mx_name, mx_abs = library_pairwise_corr(mat)
    ic, icir = summ["ic"], summ["icir"]
    ok = abs(ic) >= 0.007 and abs(icir) >= 0.084
    print(f"{name:22s} {ic:+8.4f} {icir:+7.3f} {summ['hit']:6.3f} {summ['n_ic_dates']:5d} "
          f"{cov_ad:6.3f} {cov_d8:6.3f} {to_:6.3f} {mx_abs:8.3f}  {'PASS' if ok else 'fail'}")
    if abs(ic) >= 0.005:
        print("   regime:", {k: v for k, v in summ["regime"].items()})
        print("   decay:", decay_curve(mat, fwd))
        if mx_name:
            print("   maxcorr with:", mx_name, "| top corrs:", {k: v for k, v in sorted(corrs.items(), key=lambda kv: -abs(kv[1]))[:5]})

# persist exploration summary for later reference
out = {}
for name, d in cands.items():
    mat = to_grid(d)
    ics = spearman_ic_matrix(mat, fwd[10])
    summ = summarize(ics, dates, name, HORIZON)
    if summ is None:
        continue
    cov_ad, cov_d8 = coverage_stats(mat)
    to_ = turnover_10d_rank(cross_sectional_rank(mat))
    corrs, mx_name, mx_abs = library_pairwise_corr(mat)
    out[name] = {"ic": summ["ic"], "icir": summ["icir"], "hit": summ["hit"], "n": summ["n_ic_dates"],
                 "cov_ad": cov_ad, "cov_d8": cov_d8, "turn": to_, "maxcorr": mx_abs, "maxcorr_name": mx_name,
                 "regime": summ["regime"], "decay": decay_curve(mat, fwd)}
json.dump(out, open("scripts/miner_1_20260813_explore1_results.json", "w"), indent=1, default=float)
print("\nsaved scripts/miner_1_20260813_explore1_results.json")
