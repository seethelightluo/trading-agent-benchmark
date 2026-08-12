"""miner_3 2027-02-25 full library re-validation (data visible through 2027-02-24).

Re-validates every persisted library factor on the full 2020-01-01..2027-02-24
history plus regime split and last-250d freshness. Uses rank IC at 10d horizon.
Gates: |IC|>=0.0070, |ICIR|>=0.0840.

Also reports max_abs_library_correlation vs existing factors/*.signal.npy artifacts
(provenance/audit only; the deterministic post-Miner gate recomputes rho from
real signal artifacts).
"""
import sys, json, os
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
from miner_3_20260813_lib import (ASSETS, GRID, HORIZON, to_grid,
                                  summarize, safe_div, load_macro, MIN_ASSETS,
                                  cross_sectional_rank, library_pairwise_corr,
                                  turnover_10d_rank)

GATE_IC = 0.0070
GATE_ICIR = 0.0840
OUT = "scripts/miner_3_20270225_revalidate_results.json"

print(f"grid rows: {len(GRID)} (2020-01-01 .. {GRID[-1]})", flush=True)


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


def cs_rank(mat):
    T, n = mat.shape
    out = np.full_like(mat, np.nan, dtype=float)
    for t in range(T):
        row = mat[t]
        valid = ~np.isnan(row)
        if valid.sum() < MIN_ASSETS:
            continue
        vals = row[valid]
        ranks = pd.Series(vals).rank(pct=True).values
        out[t, valid] = ranks
    return out


def rank_ic(factor_rank, fwd_rank):
    T = factor_rank.shape[0]
    ics = []
    for t in range(T):
        f = factor_rank[t]
        r = fwd_rank[t]
        ok = ~(np.isnan(f) | np.isnan(r))
        if ok.sum() < MIN_ASSETS:
            continue
        v = pd.Series(f[ok]).rank().corr(pd.Series(r[ok]).rank())
        if np.isfinite(v):
            ics.append((t, float(v)))
    return ics


def fwd_rank_matrices(series_dict, horizons=(1, 2, 3, 5, 10, 20)):
    out = {}
    for h in horizons:
        d = {}
        for s, df in series_dict.items():
            close = df["close"]
            d[s] = close.shift(-h) / close - 1.0
        out[h] = cs_rank(to_grid(d))
    return out


def roll_beta(a, b, w, minp=30, cond=None):
    df = pd.concat([a, b], axis=1, join="outer")
    df.columns = ["a", "b"]
    if cond is not None:
        mask = cond.reindex(df.index)
        df = df.where(mask, np.nan)
    cov = df["a"].rolling(w, min_periods=minp).cov(df["b"])
    var = df["b"].rolling(w, min_periods=minp).var()
    return (cov / var).reindex(a.index)


def rmean(s, w, minp):
    return s.rolling(w, min_periods=minp).mean()


def rstd(s, w, minp):
    return s.rolling(w, min_periods=minp).std()


def longest_run(pos_mask):
    arr = pos_mask.astype(float).values
    out = np.full(len(arr), np.nan)
    run = 0
    for i in range(len(arr)):
        run = run + 1 if arr[i] == 1 else 0
        if i >= 19:
            out[i] = run
    return pd.Series(out, index=pos_mask.index).rolling(20, min_periods=10).max()


def days_since_high(df, w=60):
    rollmax = df["close"].rolling(w, min_periods=40).max()
    is_high = (df["close"] >= rollmax).astype(float)
    v = np.full(len(df), np.nan)
    last_high = -1
    for i in range(len(df)):
        if is_high.iloc[i] == 1:
            last_high = i
        if last_high >= 0 and i - (w - 1) >= 0:
            v[i] = i - last_high
    return pd.Series(v, index=df.index)


def report(name, cand, fwd_ranks):
    mat = to_grid(cand)
    rank_mat = cs_rank(mat)
    ics = rank_ic(rank_mat, fwd_ranks[10])
    dates = np.array(GRID)
    summ = summarize(ics, dates, name, HORIZON)
    if summ is None:
        print(name, "NO VALID IC DATES", flush=True)
        return None
    valid = ~np.isnan(mat)
    cov_ad = float(valid.mean())
    cov_d8 = float(np.mean(valid.sum(axis=1) >= MIN_ASSETS))
    to = turnover_10d_rank(rank_mat)
    dec = {}
    for h, fr in fwd_ranks.items():
        icl = rank_ic(rank_mat, fr)
        if icl:
            dec[str(h)] = round(float(np.mean([v for _, v in icl])), 4)
    ic, icir = summ["ic"], summ["icir"]
    ok = (abs(ic) >= GATE_IC) and (abs(icir) >= GATE_ICIR)
    q = abs(ic) * abs(icir)
    pc, pc_name, pc_max = library_pairwise_corr(mat)
    print("=" * 100)
    print(f"{name}: IC={ic:+.4f} ICIR={icir:+.4f} hit={summ['hit']:.3f} n={summ['n_ic_dates']} "
          f"q={q:.5f} cov_ad={cov_ad:.3f} cov_d8={cov_d8:.3f} turn={to:.3f} "
          f"maxlibcorr={pc_max:.3f}({pc_name}) GATE={ok}", flush=True)
    print("   regime:", {k: v for k, v in summ["regime"].items()}, flush=True)
    print("   decay:", dec, flush=True)
    return {"ic": ic, "icir": icir, "q": q, "ok": ok, "hit": summ["hit"],
            "n_ic_dates": summ["n_ic_dates"], "regime": summ["regime"], "decay": dec,
            "coverage_asset_days": cov_ad, "coverage_dates_ge8": cov_d8,
            "turnover_10d_rank": to,
            "max_abs_library_correlation": pc_max,
            "max_corr_factor": pc_name}


results = {}
if os.path.exists(OUT):
    try:
        results = json.load(open(OUT))
        print("resuming:", list(results.keys()), flush=True)
    except Exception:
        results = {}

fwd_ranks = fwd_rank_matrices(series)
print("fwd rank matrices ready", flush=True)


def run(name, cand):
    if name in results and results[name] is not None:
        print(f"skip {name}", flush=True)
        return results[name]
    results[name] = report(name, cand, fwd_ranks)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=1, default=str)
    print(f"  [saved {len(results)}]", flush=True)
    return results[name]


# 1 max_consec_gain_20
cand = {s: longest_run(df["ret"] > 0) for s, df in series.items()}
run("max_consec_gain_20", cand)

# 2 max_consec_loss_20
cand = {s: longest_run(df["ret"] < 0) for s, df in series.items()}
run("max_consec_loss_20", cand)

# 3 mom20_volproxy60
cand = {}
for s, df in series.items():
    mom20 = df["close"].shift(5) / df["close"].shift(25) - 1.0
    mom60p = (df["close"].shift(5) / df["close"].shift(65) - 1.0).abs()
    cand[s] = mom20 / (1.0 + mom60p)
run("mom20_volproxy60", cand)

# 4 spx_corr60
cand = {}
for s, df in series.items():
    if s == "SPX":
        cand[s] = pd.Series(1.0, index=df.index)
    else:
        j = pd.concat([df["ret"], spx_ret], axis=1, join="outer")
        j.columns = ["a", "b"]
        cand[s] = j["a"].rolling(60, min_periods=15).corr(j["b"]).reindex(df.index)
run("spx_corr60", cand)

# 5 mom_20d_skip5
cand = {s: (df["close"].shift(5) / df["close"].shift(25) - 1.0) for s, df in series.items()}
run("mom_20d_skip5", cand)

# 6 gain_loss_20
cand = {}
for s, df in series.items():
    r = df["ret"]
    up = r.clip(lower=0).rolling(20, min_periods=10).mean()
    dn = r.clip(upper=0).rolling(20, min_periods=10).mean().abs()
    cand[s] = up / (dn + 1e-9)
run("gain_loss_20", cand)

# 7 downbeta_spx_60
cand = {}
for s, df in series.items():
    if s == "SPX":
        cand[s] = pd.Series(1.0, index=df.index)
    else:
        cand[s] = roll_beta(df["ret"], spx_ret, 60, minp=30, cond=spx_ret < 0)
run("downbeta_spx_60", cand)

# 8 usdjpy_beta_cond_120x60
cand = {}
usdjpy_r = macro_ret.get("USDJPY")
if usdjpy_r is not None:
    for s, df in series.items():
        beta = roll_beta(df["ret"], usdjpy_r, 120, minp=60)
        mom = usdjpy_r.rolling(60, min_periods=30).mean()
        cand[s] = (beta * mom).reindex(df.index)
run("usdjpy_beta_cond_120x60", cand)

# 9 volcluster_60
cand = {}
for s, df in series.items():
    rv = rstd(df["ret"], 20, 5)
    cand[s] = rv.rolling(60, min_periods=15).std()
run("volcluster_60", cand)

# 10 calmness_20
cand = {}
for s, df in series.items():
    sd = rstd(df["ret"], 20, 10)
    calm = (df["ret"].abs() < 0.5 * sd).astype(float)
    cand[s] = rmean(calm, 20, 10)
run("calmness_20", cand)

# 11 close_pos_20
cand = {s: rmean(df["hl_pos"], 20, 10) for s, df in series.items()}
run("close_pos_20", cand)

# 12 days_since_high_60
cand = {s: days_since_high(df, 60) for s, df in series.items()}
run("days_since_high_60", cand)

# 13 lagbeta_spx_60
cand = {}
for s, df in series.items():
    if s == "SPX":
        cand[s] = pd.Series(1.0, index=df.index)
    else:
        cand[s] = roll_beta(df["ret"], spx_ret.shift(1), 60, minp=30)
run("lagbeta_spx_60", cand)

# 14 intraday_drift_20
cand = {s: rmean(df["intraday"], 20, 10) for s, df in series.items()}
run("intraday_drift_20", cand)

# 15 dxy_beta_cond_60x20
cand = {}
dxy_r = macro_ret.get("DXY")
if dxy_r is not None:
    for s, df in series.items():
        beta = roll_beta(df["ret"], dxy_r, 60, minp=30)
        dxy_mom = dxy_r.rolling(20, min_periods=10).mean()
        cand[s] = (beta * dxy_mom).reindex(df.index)
run("dxy_beta_cond_60x20", cand)

# 16 vix_beta_cond_60x20
cand = {}
vix_r = macro_ret.get("VIX")
if vix_r is not None:
    for s, df in series.items():
        beta = roll_beta(df["ret"], vix_r, 60, minp=30)
        vix_mom = vix_r.rolling(20, min_periods=10).mean()
        cand[s] = (beta * vix_mom).reindex(df.index)
run("vix_beta_cond_60x20", cand)

# 17 mom_10d_skip5
cand = {s: (df["close"].shift(5) / df["close"].shift(15) - 1.0) for s, df in series.items()}
run("mom_10d_skip5", cand)

# 18 mom_120d_skip5
cand = {s: (df["close"].shift(5) / df["close"].shift(125) - 1.0) for s, df in series.items()}
run("mom_120d_skip5", cand)

# 19 mom_180d_skip5
cand = {s: (df["close"].shift(5) / df["close"].shift(185) - 1.0) for s, df in series.items()}
run("mom_180d_skip5", cand)

# 20 mom30_vol60
cand = {}
for s, df in series.items():
    mom30 = df["close"].shift(5) / df["close"].shift(35) - 1.0
    vol60 = rstd(df["ret"], 60, 15)
    cand[s] = safe_div(mom30, vol60)
run("mom30_vol60", cand)

# 21 range_pos_252
cand = {}
for s, df in series.items():
    pos = df["close"] / df["close"].shift(252) - 1.0
    rng = df["close"].rolling(252, min_periods=120).max() - df["close"].rolling(252, min_periods=120).min()
    cand[s] = safe_div(pos, rng / df["close"])
run("range_pos_252", cand)

# 22 vol_of_vol20x60
cand = {}
for s, df in series.items():
    rv = rstd(df["ret"], 20, 5)
    cand[s] = rv.rolling(60, min_periods=15).std()
run("vol_of_vol20x60", cand)

print("\nDONE. saved", OUT)
