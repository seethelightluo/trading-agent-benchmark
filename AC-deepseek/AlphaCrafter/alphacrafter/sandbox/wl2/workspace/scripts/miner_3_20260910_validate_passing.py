"""miner_3 2026-09-10 deep validation of batchA/B IC/ICIR-gate passing candidates.

Candidates: updown_vol_ratio_20, downside_freq_20, max_gain_20 (batchA),
            cn10y_corr_60, hl_rank_20 (batchB).
Recompute on data visible through 2026-09-09, save signal matrices (npy) for
persistence, and report pairwise Spearman rho vs every library artifact with
an explicit KEPT/EVICTED/QUARANTINED flag per library member.
"""
import sys, json, glob, os
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict
from miner_3_20260813_lib import (ASSETS, GRID, HORIZON, to_grid,
                                  cross_sectional_rank, spearman_ic_matrix,
                                  summarize, decay_curve, fwd_by_horizon_dict,
                                  turnover_10d_rank, library_pairwise_corr,
                                  coverage_stats, safe_div, MIN_ASSETS)

GATE_IC = 0.0070
GATE_ICIR = 0.0840
CORR_LIMIT = 0.5

# ---- determine kept library (EFFECTIVE status json in main factors/ dir with npy) ----
KEPT = set()
for f in sorted(glob.glob("factors/*.json")):
    base = os.path.basename(f)
    if ".bak" in base or "." in base.split("_")[-1] and base.count(".") > 1:
        pass
    try:
        d = json.load(open(f))
    except Exception:
        continue
    fid = d.get("factor_id", "")
    st = d.get("validation", {}).get("status", "")
    if st == "EFFECTIVE" and os.path.exists(f"factors/{fid}.signal.npy"):
        KEPT.add(fid)
print("KEPT library ids:", sorted(KEPT))


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
    return df


series = {s: load_asset(s) for s in ASSETS}
series = {s: df for s, df in series.items() if df is not None and len(df) > 100}
print(f"assets loaded: {len(series)}/15 -> {sorted(series.keys())}")
fwd = fwd_by_horizon_dict(series)
dates = np.array(GRID)


def roll_mean_s(s, w, minp):
    return s.rolling(w, min_periods=minp).mean()


def roll_std_s(s, w, minp):
    return s.rolling(w, min_periods=minp).std()


def full_report(name, cand, save=True):
    mat = to_grid(cand)
    rank_mat = cross_sectional_rank(mat)
    ics = spearman_ic_matrix(mat, fwd[10])
    summ = summarize(ics, dates, name, HORIZON)
    if summ is None:
        print(name, "NO VALID IC DATES"); return None
    cov_ad, cov_d8 = coverage_stats(mat)
    to = turnover_10d_rank(rank_mat)
    dec = decay_curve(mat, fwd)
    corrs, mx_name, mx_abs = library_pairwise_corr(mat)
    ic, icir = summ["ic"], summ["icir"]
    ok = (abs(ic) >= GATE_IC) and (abs(icir) >= GATE_ICIR)
    q = abs(ic) * abs(icir)
    print("=" * 100)
    print(f"{name}: IC={ic:+.4f} ICIR={icir:+.4f} hit={summ['hit']:.3f} n={summ['n_ic_dates']} "
          f"q={q:.5f} cov_ad={cov_ad:.3f} cov_d8={cov_d8:.3f} turn={to:.3f} GATE={ok}")
    print("   regime:", {k: v for k, v in summ["regime"].items()})
    print("   decay:", dec)
    # conflicts vs kept library only
    conflicts = []
    for fid, rho in sorted(corrs.items(), key=lambda kv: abs(kv[1]), reverse=True):
        flag = "KEPT" if fid in KEPT else "lib"
        if fid in KEPT and abs(rho) >= CORR_LIMIT:
            conflicts.append((fid, rho))
        print(f"     {flag:6s} {fid:28s} rho={rho:+.3f}")
    print(f"   -> max_abs_library_corr={mx_abs:.3f} ({mx_name}) | KEPT conflicts>0.5: {conflicts}")
    if save and ok:
        np.save(f"factors/{name}.signal.npy", mat)
        print(f"   SAVED factors/{name}.signal.npy shape={mat.shape}")
    return {"ic": ic, "icir": icir, "q": q, "ok": ok, "conflicts": conflicts,
            "max_abs_library_correlation": mx_abs, "max_lib_corr_name": mx_name,
            "n_ic_dates": summ["n_ic_dates"], "regime": summ["regime"], "decay": dec}


results = {}

# 1. updown_vol_ratio_20: std of up-day returns / std of down-day returns (20d)
cand = {}
for s, df in series.items():
    r = df["ret"]
    up = r.where(r > 0)
    dn = r.where(r < 0)
    su = up.rolling(20, min_periods=8).std()
    sd_ = dn.rolling(20, min_periods=8).std()
    cand[s] = pd.Series(safe_div(su, sd_), index=df.index)
results["updown_vol_ratio_20"] = full_report("updown_vol_ratio_20", cand)

# 2. downside_freq_20: fraction of negative days over 20d
cand = {}
for s, df in series.items():
    neg = (df["ret"] < 0).astype(float)
    cand[s] = roll_mean_s(neg, 20, 10)
results["downside_freq_20"] = full_report("downside_freq_20", cand)

# 3. max_gain_20: max single-day return over 20d
cand = {s: pd.Series(df["ret"].rolling(20, min_periods=10).max(), index=df.index) for s, df in series.items()}
results["max_gain_20"] = full_report("max_gain_20", cand)

# 4. cn10y_corr_60: 60d rolling corr of asset ret with CN10Y ret
brets = series["CN10Y"]["ret"]
cand = {}
for s, df in series.items():
    joined = pd.concat([df["ret"], brets], axis=1, join="outer")
    joined.columns = ["a", "b"]
    c = joined["a"].rolling(60, min_periods=40).corr(joined["b"])
    cand[s] = c.reindex(df.index)
results["cn10y_corr_60"] = full_report("cn10y_corr_60", cand)

# 5. hl_rank_20: close position in 20d high-low range
cand = {}
for s, df in series.items():
    hi = df["high"].rolling(20, min_periods=10).max()
    lo = df["low"].rolling(20, min_periods=10).min()
    cand[s] = pd.Series(safe_div(df["close"] - lo, hi - lo), index=df.index)
results["hl_rank_20"] = full_report("hl_rank_20", cand)

json.dump(results, open("scripts/miner_3_20260910_validate_passing_results.json", "w"), indent=1, default=str)
print("DONE")
