"""miner_2 2034-09-04: (1) revalidate 3 effective library factors for drift;
(2) screen NEW candidate factor ideas (batch 4) on the 15-asset cross-asset universe.
Data visible through the previous completed trading day only (no lookahead).
Admission gates (shared): abs(IC) >= 0.0070 AND abs(ICIR) >= 0.0840 at 10d horizon.
"""
import sys, warnings, json, time
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

sys.path.insert(0, "scripts")
from factor_research_lib import (
    load_panels, close_panel, forward_returns,
    rank_ic_series, summarize_ic, coverage_metrics, turnover_rank, decay_profile,
    full_eval, library_signals, max_library_corr,
)

t_start = time.time()
panels = load_panels(days=4000)
closes = close_panel(panels)
rets = closes.pct_change()
END = closes.index.max().strftime("%Y-%m-%d")
print(f"data through: {END} | n_dates: {len(closes)} | n_assets: {closes.shape[1]}", flush=True)

vix = panels["VIX"]["close"].astype(float) if "VIX" in panels else None
dxy = panels["DXY"]["close"].astype(float) if "DXY" in panels else None
us10y = panels["US10Y"]["close"].astype(float) if "US10Y" in panels else None
cn10y = panels["CN10Y"]["close"].astype(float) if "CN10Y" in panels else None

def eval_factor(name, sig, expected_sign, window=None, library=None):
    s = sig if window is None else sig.loc[window[0]:window[1]]
    c = closes if window is None else closes.loc[window[0]:window[1]]
    m, ics = full_eval(s, c, (1, 2, 3, 5, 10, 20), 8, expected_sign,
                       library=library, admission_horizon=10)
    m["admission_gate"] = {
        "ic_gate_abs": 0.0070, "icir_gate_abs": 0.0840,
        "ic_pass": abs(m["ic"]) >= 0.0070,
        "icir_pass": abs(m["icir"]) >= 0.0840,
    }
    gate = m["admission_gate"]
    ok = gate["ic_pass"] and gate["icir_pass"]
    print(f"=== {name} (dir {expected_sign:+d}) | ic={m['ic']} icir={m['icir']} "
          f"hit={m['ic_hit_ratio']} n={m['n_ic_dates']} cov={m['coverage_dates_ge8']} "
          f"turn={m['turnover_10d_rank']} maxcorr={m.get('max_abs_library_correlation')} "
          f"({m.get('max_corr_factor')}) gate={'PASS' if ok else 'FAIL'}", flush=True)
    return m, ics

# ---------- library reference signals (3 effective factors) ----------
lib_sigs = library_signals(panels, closes, rets, vix)
mom20 = closes / closes.shift(20) - 1.0
mom60 = closes / closes.shift(60) - 1.0
vol20 = rets.rolling(20).std()
lib_sigs["vol_adj_mom_accel_20x60"] = (mom20 - mom60) / vol20
mkt_ret = rets.mean(axis=1)
down = mkt_ret.where(mkt_ret < 0)
beta_down = {}
for a in rets.columns:
    z = pd.concat([rets[a].rename("a"), down.rename("m")], axis=1).dropna()
    beta_down[a] = z["a"].rolling(60).cov(z["m"]) / z["m"].rolling(60).var()
lib_sigs["dn_mkt_beta_60d"] = pd.DataFrame(beta_down, index=rets.index)
cn10y_ret = cn10y.pct_change()
beta_cn = {}
for a in rets.columns:
    z = pd.concat([rets[a].rename("a"), cn10y_ret.rename("r")], axis=1).dropna()
    beta_cn[a] = z["a"].rolling(60).cov(z["r"]) / z["r"].rolling(60).var()
lib_sigs["rate_beta_cn10y_60d"] = pd.DataFrame(beta_cn, index=rets.index)
eff_lib = {k: lib_sigs[k] for k in ["vol_adj_mom_accel_20x60", "dn_mkt_beta_60d", "rate_beta_cn10y_60d"]}
print("effective library reference signals:", list(eff_lib.keys()), flush=True)

print("=" * 70)
print("PART 1: REVALIDATE 3 EFFECTIVE FACTORS (drift check)")
print("=" * 70)
for nm, sg, sd in [("vol_adj_mom_accel_20x60", eff_lib["vol_adj_mom_accel_20x60"], 1),
                   ("dn_mkt_beta_60d", eff_lib["dn_mkt_beta_60d"], 1),
                   ("rate_beta_cn10y_60d", eff_lib["rate_beta_cn10y_60d"], -1)]:
    eval_factor(nm, sg, sd, library=eff_lib)
print("--- RECENT 2Y drift (2032-09-03..END) ---", flush=True)
for nm, sg, sd in [("vol_adj_mom_accel_20x60", eff_lib["vol_adj_mom_accel_20x60"], 1),
                   ("dn_mkt_beta_60d", eff_lib["dn_mkt_beta_60d"], 1),
                   ("rate_beta_cn10y_60d", eff_lib["rate_beta_cn10y_60d"], -1)]:
    eval_factor(nm, sg, sd, ("2032-09-03", END), library=eff_lib)

print("=" * 70)
print("PART 2: NEW CANDIDATE SCREEN (batch 4)")
print("=" * 70)

results = {}

# B1: eff_ratio_60d - Kaufman efficiency ratio |close_t - close_{t-60}| / sum(|ret|,60)
abs_ret = rets.abs()
sig_er = (closes - closes.shift(60)).abs() / abs_ret.rolling(60).sum()
m, _ = eval_factor("eff_ratio_60d", sig_er, 1, library=eff_lib); results["eff_ratio_60d"] = m

# B2: max_gain_60d - max daily return over 60d (lottery/overheat)
sig_mg = rets.rolling(60).max()
m, _ = eval_factor("max_gain_60d", sig_mg, -1, library=eff_lib); results["max_gain_60d"] = m

# B3: min_ret_60d - min daily return over 60d (tail-risk)
sig_mn = rets.rolling(60).min()
m, _ = eval_factor("min_ret_60d", sig_mn, -1, library=eff_lib); results["min_ret_60d"] = m

# B4: inv_vol_20d - inverse 20d volatility (low-vol tilt)
sig_iv = 1.0 / vol20
m, _ = eval_factor("inv_vol_20d", sig_iv, 1, library=eff_lib); results["inv_vol_20d"] = m

# B5: vix_beta_20d - 20d beta of asset returns to VIX changes (short risk sensitivity)
if vix is not None:
    vix_ret = vix.pct_change()
    beta_v20 = {}
    for a in rets.columns:
        z = pd.concat([rets[a].rename("a"), vix_ret.rename("v")], axis=1).dropna()
        beta_v20[a] = z["a"].rolling(20).cov(z["v"]) / z["v"].rolling(20).var()
    sig_vb = pd.DataFrame(beta_v20, index=rets.index)
    m, _ = eval_factor("vix_beta_20d", sig_vb, -1, library=eff_lib); results["vix_beta_20d"] = m
else:
    print("VIX missing; skip vix_beta_20d")

# B6: up_streak_5d - max consecutive up days in last 5 (short sentiment/reversal)
up = (rets > 0).astype(float)
streak = up.copy() * 0
for k in range(5):
    streak = (streak + up.shift(k)) * up
m, _ = eval_factor("up_streak_5d", streak, -1, library=eff_lib); results["up_streak_5d"] = m

# B7: vol_trend_20_60 - 20d vol / 60d vol (vol regime trend; rising vol = worse)
vol60 = rets.rolling(60).std()
sig_vt = vol20 / vol60
m, _ = eval_factor("vol_trend_20_60", sig_vt, -1, library=eff_lib); results["vol_trend_20_60"] = m

# B8: zscore_ma50 - (close - SMA50) / std(close,50) (trend z-score)
sma50 = closes.rolling(50).mean()
std50 = closes.rolling(50).std()
sig_z = (closes - sma50) / std50
m, _ = eval_factor("zscore_ma50", sig_z, 1, library=eff_lib); results["zscore_ma50"] = m

# B9: overnight_ret_5d - mean overnight gap (open/prev_close-1) over 5d
open_panel = pd.concat({a: panels[a]["open"].astype(float) for a in closes.columns if a in panels}, axis=1).sort_index()
sig_ovn = (open_panel / closes.shift(1) - 1.0).rolling(5).mean()
m, _ = eval_factor("overnight_ret_5d", sig_ovn, 1, library=eff_lib); results["overnight_ret_5d"] = m

# B10: mom_90d_skip5 - 90d momentum skipping 5 days (medium-horizon trend)
sig_m90 = closes.shift(5) / closes.shift(95) - 1.0
m, _ = eval_factor("mom_90d_skip5", sig_m90, 1, library=eff_lib); results["mom_90d_skip5"] = m

# B11: cdl_pos_10d - instantaneous candle position over 10d (close-low)/(high-low)
hi10 = closes.rolling(10).max()
lo10 = closes.rolling(10).min()
sig_c10 = (closes - lo10) / (hi10 - lo10).replace(0, np.nan)
m, _ = eval_factor("cdl_pos_10d", sig_c10, 1, library=eff_lib); results["cdl_pos_10d"] = m

# B12: corr_spx_20d - 20d rolling corr of asset returns with SPX (short equity linkage)
spx_ret = rets["SPX"]
sig_cs = rets.rolling(20).corr(spx_ret)
m, _ = eval_factor("corr_spx_20d", sig_cs, -1, library=eff_lib); results["corr_spx_20d"] = m

print("=" * 70)
print("RECENT 2Y WINDOW CHECK for candidates that PASS full-window gate")
print("=" * 70)
cand_defs = {
    "eff_ratio_60d": sig_er, "max_gain_60d": sig_mg, "min_ret_60d": sig_mn,
    "inv_vol_20d": sig_iv, "up_streak_5d": streak, "vol_trend_20_60": sig_vt,
    "zscore_ma50": sig_z, "overnight_ret_5d": sig_ovn, "mom_90d_skip5": sig_m90,
    "cdl_pos_10d": sig_c10, "corr_spx_20d": sig_cs,
}
if vix is not None:
    cand_defs["vix_beta_20d"] = sig_vb
signed = {"max_gain_60d": -1, "min_ret_60d": -1, "up_streak_5d": -1,
          "vol_trend_20_60": -1, "corr_spx_20d": -1, "vix_beta_20d": -1}
for nm, mm in results.items():
    g = mm["admission_gate"]
    if g["ic_pass"] and g["icir_pass"]:
        sd = signed.get(nm, 1)
        m2, _ = eval_factor(nm + "_RECENT2Y", cand_defs[nm], sd, ("2032-09-03", END), library=eff_lib)
        results[nm + "_recent2y"] = m2

print("=" * 70)
print("SUMMARY")
print("=" * 70)
for nm, mm in results.items():
    g = mm["admission_gate"]
    ok = "PASS" if (g["ic_pass"] and g["icir_pass"]) else "fail"
    print(f"{nm:28s} ic={mm['ic']:+.4f} icir={mm['icir']:+.4f} hit={mm['ic_hit_ratio']:.2f} "
          f"n={mm['n_ic_dates']:5d} cov8={mm['coverage_dates_ge8']:.2f} turn={mm['turnover_10d_rank']} "
          f"maxcorr={mm.get('max_abs_library_correlation')} -> {ok}", flush=True)
print("elapsed_s:", round(time.time() - t_start, 1), flush=True)
print("DONE", flush=True)
