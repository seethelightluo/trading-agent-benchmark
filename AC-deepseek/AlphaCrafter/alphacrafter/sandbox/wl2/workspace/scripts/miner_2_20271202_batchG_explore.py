"""miner_2 2027-12-02 Batch-G exploration: regime snapshot + novel candidate factors.

Data visible through 2027-12-01 (current_date 2027-12-02). Reuses miner_3 shared lib
for grid/IC conventions. Gates: |IC|>=0.0070, |ICIR|>=0.0840 at 10d horizon,
rank IC, MIN_ASSETS=8, 15-instrument cross-asset universe.
"""
import sys, json, os
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
from miner_3_20260813_lib import (ASSETS, GRID, HORIZON, to_grid, load_macro, MIN_ASSETS,
                                  cross_sectional_rank, spearman_ic_matrix, summarize,
                                  decay_curve, fwd_by_horizon_dict, turnover_10d_rank,
                                  library_pairwise_corr, safe_div)

GATE_IC = 0.0070
GATE_ICIR = 0.0840
OUT = "scripts/miner_2_20271202_batchG_results.json"

print(f"grid rows: {len(GRID)} (2020-01-01 .. {GRID[-1}])", flush=True) if False else None
print(f"grid rows: {len(GRID)} (2020-01-01 .. {GRID[-1]})", flush=True)


def load_asset(sym, days=2400):
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

# ---------------- regime snapshot ----------------
print("\n=== REGIME SNAPSHOT (through 2027-12-01) ===", flush=True)
for s, df in series.items():
    c = df["close"]
    r5 = c.iloc[-1] / c.iloc[-6] - 1 if len(c) > 6 else np.nan
    r20 = c.iloc[-1] / c.iloc[-21] - 1 if len(c) > 21 else np.nan
    r60 = c.iloc[-1] / c.iloc[-61] - 1 if len(c) > 61 else np.nan
    v20 = df["ret"].iloc[-20:].std() * np.sqrt(252) if len(df) > 20 else np.nan
    print(f"{s:10s} 5d={r5:+.2%} 20d={r20:+.2%} 60d={r60:+.2%} annvol20={v20:.0%}", flush=True)
if "VIX" in macro:
    v = macro["VIX"]
    print(f"VIX last={v.iloc[-1]:.2f} 20d_ago={v.iloc[-21] if len(v)>21 else np.nan:.2f} 60d_ago={v.iloc[-61] if len(v)>61 else np.nan:.2f}", flush=True)
    print(f"VIX 20d chg: {v.iloc[-1]/v.iloc[-21]-1:+.2%}" if len(v) > 21 else "", flush=True)
# volume availability
nvol = {s: int(df["volume"].notna().sum()) for s, df in series.items()}
print("volume non-null:", nvol, flush=True)

# ---------------- candidate factors ----------------
def rmean(s, w, minp):
    return s.rolling(w, min_periods=minp).mean()


def rstd(s, w, minp):
    return s.rolling(w, min_periods=minp).std()


def roll_beta(a, b, w, minp=30, cond=None):
    df = pd.concat([a, b], axis=1, join="outer")
    df.columns = ["a", "b"]
    if cond is not None:
        mask = cond.reindex(df.index)
        df = df.where(mask, np.nan)
    cov = df["a"].rolling(w, min_periods=minp).cov(df["b"])
    var = df["b"].rolling(w, min_periods=minp).var()
    return (cov / var).reindex(a.index)


fwd_ranks_dict = fwd_by_horizon_dict(series)
fwd_ranks = {h: cross_sectional_rank(m) for h, m in fwd_ranks_dict.items()}
dates = np.array(GRID)

cands = {}

# A1 updown_capture_60: mean ret on SPX-up days vs mean |ret| on SPX-down days (60d)
cands["updown_capture_60"] = {}
for s, df in series.items():
    j = pd.concat([df["ret"], spx_ret], axis=1, join="outer")
    j.columns = ["a", "b"]
    up = j["a"].where(j["b"] > 0)
    dn = j["a"].where(j["b"] < 0)
    cup = up.rolling(60, min_periods=15).mean()
    cdn = dn.rolling(60, min_periods=15).mean().abs()
    cands["updown_capture_60"][s] = (cup / (cdn + 1e-9)).reindex(df.index)

# A2 vix_beta_60: 60d beta of asset ret vs dVIX (VIX pct chg). High beta = risk-on sensitivity.
cands["vix_beta_60"] = {}
vix_r = macro_ret.get("VIX")
if vix_r is not None:
    for s, df in series.items():
        cands["vix_beta_60"][s] = roll_beta(df["ret"], vix_r, 60, minp=30).reindex(df.index)

# A3 downside_dev_ratio_20: downside semi-dev / upside semi-dev (20d)
cands["downside_dev_ratio_20"] = {}
for s, df in series.items():
    r = df["ret"]
    dn = r.clip(upper=0).rolling(20, min_periods=10).std()
    up = r.clip(lower=0).rolling(20, min_periods=10).std()
    cands["downside_dev_ratio_20"][s] = dn / (up + 1e-9)

# A4 drawdown_60: close/rolling_max(close,60) - 1 (deep drawdown, contrarian)
cands["drawdown_60"] = {}
for s, df in series.items():
    rm = df["close"].rolling(60, min_periods=40).max()
    cands["drawdown_60"][s] = df["close"] / rm - 1.0

# B1 zscore_20_60: (close - SMA20)/std(close,60)
cands["zscore_20_60"] = {}
for s, df in series.items():
    sma = df["close"].rolling(20, min_periods=10).mean()
    sd = df["close"].rolling(60, min_periods=30).std()
    cands["zscore_20_60"][s] = (df["close"] - sma) / (sd + 1e-9)

# B2 mom60_volproxy20: 60d momentum (skip5) damped by 20d vol proxy
cands["mom60_volproxy20"] = {}
for s, df in series.items():
    mom60 = df["close"].shift(5) / df["close"].shift(65) - 1.0
    vp20 = (df["close"].shift(5) / df["close"].shift(25) - 1.0).abs()
    cands["mom60_volproxy20"][s] = mom60 / (1.0 + vp20)

# B3 eff_ratio_60: |close-close.shift(60)| / sum(|ret|,60) trendiness
cands["eff_ratio_60"] = {}
for s, df in series.items():
    num = (df["close"] - df["close"].shift(60)).abs()
    den = df["ret"].abs().rolling(60, min_periods=30).sum()
    cands["eff_ratio_60"][s] = num / (den + 1e-9)

# B4 reversal_3_20: short-term reversal (3d momentum, expect negative dir)
cands["reversal_3_20"] = {}
for s, df in series.items():
    cands["reversal_3_20"][s] = df["close"] / df["close"].shift(3) - 1.0

# C1 hl_vol_ratio_20: mean range / 20d ret std (range efficiency)
cands["hl_vol_ratio_20"] = {}
for s, df in series.items():
    mrng = df["rng_pct"].rolling(20, min_periods=10).mean()
    sd = rstd(df["ret"], 20, 10)
    cands["hl_vol_ratio_20"][s] = mrng / (sd + 1e-9)

# C2 range_pos_5: close position in 5d range (short-term)
cands["range_pos_5"] = {}
for s, df in series.items():
    hi = df["high"].rolling(5, min_periods=3).max()
    lo = df["low"].rolling(5, min_periods=3).min()
    cands["range_pos_5"][s] = (df["close"] - lo) / (hi - lo).replace(0, np.nan)

# C3 gap_freq_20: fraction of days with |gap|>1% over 20d
cands["gap_freq_20"] = {}
for s, df in series.items():
    big = (df["gap"].abs() > 0.01).astype(float)
    cands["gap_freq_20"][s] = rmean(big, 20, 10)

# C4 skew_20: rolling skewness of 20d returns
cands["skew_20"] = {}
for s, df in series.items():
    cands["skew_20"][s] = df["ret"].rolling(20, min_periods=10).skew()

# D1 usdcny_beta_cond_120x60: beta to USDCNY * USDCNY trend (CN assets)
cands["usdcny_beta_cond_120x60"] = {}
cny_r = macro_ret.get("USDCNY")
if cny_r is not None:
    for s, df in series.items():
        beta = roll_beta(df["ret"], cny_r, 120, minp=60)
        mom = cny_r.rolling(60, min_periods=30).mean()
        cands["usdcny_beta_cond_120x60"][s] = (beta * mom).reindex(df.index)

# D2 dxy_beta_cond_60x20: beta to DXY * DXY trend (all assets; FX sensitivity)
cands["dxy_beta_cond_60x20b"] = {}
dxy_r = macro_ret.get("DXY")
if dxy_r is not None:
    for s, df in series.items():
        beta = roll_beta(df["ret"], dxy_r, 60, minp=30)
        mom = dxy_r.rolling(20, min_periods=10).mean()
        cands["dxy_beta_cond_60x20b"][s] = (beta * mom).reindex(df.index)

# D3 vix_regime_mom20: mom20 * (VIX below its 60d median -> 1 else -1) style gate
cands["vix_regime_mom20"] = {}
if vix_r is not None:
    vix_lvl = macro["VIX"]
    for s, df in series.items():
        mom20 = df["close"].shift(5) / df["close"].shift(25) - 1.0
        gate = (vix_lvl < vix_lvl.rolling(120, min_periods=60).median()).astype(float) * 2 - 1
        cands["vix_regime_mom20"][s] = (mom20 * gate.reindex(df.index)).reindex(df.index)

results = {}
for name, cand in cands.items():
    mat = to_grid(cand)
    rank_mat = cross_sectional_rank(mat)
    ics = spearman_ic_matrix(rank_mat, fwd_ranks[10])
    summ = summarize(ics, dates, name, HORIZON)
    if summ is None:
        print(f"{name}: NO VALID IC DATES", flush=True)
        results[name] = None
        continue
    valid = ~np.isnan(mat)
    cov_ad = float(valid.mean())
    cov_d8 = float(np.mean(valid.sum(axis=1) >= MIN_ASSETS))
    to = turnover_10d_rank(rank_mat)
    dec = decay_curve(rank_mat, fwd_ranks)
    ic, icir = summ["ic"], summ["icir"]
    ok = (abs(ic) >= GATE_IC) and (abs(icir) >= GATE_ICIR)
    q = abs(ic) * abs(icir)
    pc, pc_name, pc_max = library_pairwise_corr(mat)
    print("=" * 100, flush=True)
    print(f"{name}: IC={ic:+.4f} ICIR={icir:+.4f} hit={summ['hit']:.3f} n={summ['n_ic_dates']} "
          f"q={q:.5f} cov_ad={cov_ad:.3f} cov_d8={cov_d8:.3f} turn={to:.3f} "
          f"maxlibcorr={pc_max:.3f}({pc_name}) GATE={ok}", flush=True)
    print("   regime:", {k: v for k, v in summ["regime"].items()}, flush=True)
    print("   decay:", dec, flush=True)
    results[name] = {"ic": ic, "icir": icir, "q": q, "ok": ok, "hit": summ["hit"],
                     "n_ic_dates": summ["n_ic_dates"], "regime": summ["regime"], "decay": dec,
                     "coverage_asset_days": cov_ad, "coverage_dates_ge8": cov_d8,
                     "turnover_10d_rank": to, "max_abs_library_correlation": pc_max,
                     "max_corr_factor": pc_name, "direction": 1 if ic >= 0 else -1}

with open(OUT, "w") as f:
    json.dump(results, f, indent=1, default=str)
print(f"\n[results saved: {OUT}]", flush=True)
