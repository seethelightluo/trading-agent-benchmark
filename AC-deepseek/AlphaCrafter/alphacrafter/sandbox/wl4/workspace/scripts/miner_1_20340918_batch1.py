"""miner_1 2034-09-18: (1) revalidate 2 effective library factors for drift;
(2) screen NEW candidate factor ideas (batch 1) on the 15-asset cross-asset universe.
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
usdjpy = panels["USDJPY"]["close"].astype(float) if "USDJPY" in panels else None

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

# ---------- library reference signals (effective + demoted) ----------
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
cn10y = panels["CN10Y"]["close"].astype(float) if "CN10Y" in panels else None
if cn10y is not None:
    cn10y_ret = cn10y.pct_change()
    beta_cn = {}
    for a in rets.columns:
        z = pd.concat([rets[a].rename("a"), cn10y_ret.rename("r")], axis=1).dropna()
        beta_cn[a] = z["a"].rolling(60).cov(z["r"]) / z["r"].rolling(60).var()
    lib_sigs["rate_beta_cn10y_60d"] = pd.DataFrame(beta_cn, index=rets.index)
eff_lib = {k: lib_sigs[k] for k in ["vol_adj_mom_accel_20x60", "dn_mkt_beta_60d", "rate_beta_cn10y_60d"]}
print("effective library reference signals:", list(eff_lib.keys()), flush=True)

print("=" * 70)
print("PART 1: REVALIDATE EFFECTIVE FACTORS (drift check)")
print("=" * 70)
for nm, sg, sd in [("vol_adj_mom_accel_20x60", eff_lib["vol_adj_mom_accel_20x60"], 1),
                   ("dn_mkt_beta_60d", eff_lib["dn_mkt_beta_60d"], 1),
                   ("rate_beta_cn10y_60d", eff_lib["rate_beta_cn10y_60d"], -1)]:
    eval_factor(nm, sg, sd, library=eff_lib)
print("--- RECENT 2Y drift (2032-09-17..END) ---", flush=True)
for nm, sg, sd in [("vol_adj_mom_accel_20x60", eff_lib["vol_adj_mom_accel_20x60"], 1),
                   ("dn_mkt_beta_60d", eff_lib["dn_mkt_beta_60d"], 1),
                   ("rate_beta_cn10y_60d", eff_lib["rate_beta_cn10y_60d"], -1)]:
    eval_factor(nm, sg, sd, ("2032-09-17", END), library=eff_lib)

print("=" * 70)
print("PART 2: NEW CANDIDATE SCREEN (batch 1)")
print("=" * 70)

results = {}

# C1: realized_skew_60d - skewness of daily returns over 60d (lottery/overheat)
sig_sk = rets.rolling(60).skew()
m, _ = eval_factor("realized_skew_60d", sig_sk, -1, library=eff_lib); results["realized_skew_60d"] = m

# C2: dxy_beta_60d - 60d beta of asset returns to DXY returns (USD sensitivity)
if dxy is not None:
    dxy_ret = dxy.pct_change()
    beta_dx = {}
    for a in rets.columns:
        z = pd.concat([rets[a].rename("a"), dxy_ret.rename("d")], axis=1).dropna()
        beta_dx[a] = z["a"].rolling(60).cov(z["d"]) / z["d"].rolling(60).var()
    sig_db = pd.DataFrame(beta_dx, index=rets.index)
    m, _ = eval_factor("dxy_beta_60d", sig_db, -1, library=eff_lib); results["dxy_beta_60d"] = m
else:
    print("DXY missing; skip dxy_beta_60d")

# C3: trend_consistency_60d - fraction of days close>SMA20 over 60d (trend quality)
sig_tc = (closes > closes.rolling(20).mean()).rolling(60).mean()
m, _ = eval_factor("trend_consistency_60d", sig_tc, 1, library=eff_lib); results["trend_consistency_60d"] = m

# C4: reversal_5d - 5d momentum (short-term reversal)
sig_r5 = closes / closes.shift(5) - 1.0
m, _ = eval_factor("reversal_5d", sig_r5, -1, library=eff_lib); results["reversal_5d"] = m

# C5: parkinson_vol_20d - Parkinson range-based volatility over 20d
hi = pd.concat({a: panels[a]["high"].astype(float) for a in closes.columns if a in panels}, axis=1).sort_index()
lo = pd.concat({a: panels[a]["low"].astype(float) for a in closes.columns if a in panels}, axis=1).sort_index()
hl2 = (np.log(hi / lo) ** 2) / (4 * np.log(2))
sig_pv = np.sqrt(hl2.rolling(20).mean())
m, _ = eval_factor("parkinson_vol_20d", sig_pv, -1, library=eff_lib); results["parkinson_vol_20d"] = m

# C6: gain_loss_asym_60d - mean up-day / mean |down-day| over 60d (lottery asymmetry)
pos = rets.where(rets > 0, 0.0)
neg = rets.where(rets < 0, 0.0).abs()
sig_gla = pos.rolling(60).mean() / neg.rolling(60).mean().replace(0, np.nan)
m, _ = eval_factor("gain_loss_asym_60d", sig_gla, -1, library=eff_lib); results["gain_loss_asym_60d"] = m

# C7: corr_btc_20d - 20d rolling correlation with BTC returns (crypto linkage)
sig_cb = rets.rolling(20).corr(rets["BTC"])
m, _ = eval_factor("corr_btc_20d", sig_cb, -1, library=eff_lib); results["corr_btc_20d"] = m

# C8: serial_corr_10d - autocorrelation of daily returns (lag1) over 10d (trend persistence)
def autocorr1(x):
    if len(x) < 4 or np.std(x[:-1]) < 1e-14 or np.std(x[1:]) < 1e-14:
        return np.nan
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])
sig_sc = rets.rolling(10).apply(autocorr1, raw=True)
m, _ = eval_factor("serial_corr_10d", sig_sc, 1, library=eff_lib); results["serial_corr_10d"] = m

# C9: mom_40d_vol20 - 40d momentum scaled by 20d vol (risk-adjusted medium momentum)
sig_mv = (closes / closes.shift(40) - 1.0) / vol20
m, _ = eval_factor("mom_40d_vol20", sig_mv, 1, library=eff_lib); results["mom_40d_vol20"] = m

# C10: usdjpy_beta_60d - 60d beta of asset returns to USDJPY (carry/risk linkage)
if usdjpy is not None:
    jpy_ret = usdjpy.pct_change()
    beta_jy = {}
    for a in rets.columns:
        z = pd.concat([rets[a].rename("a"), jpy_ret.rename("j")], axis=1).dropna()
        beta_jy[a] = z["a"].rolling(60).cov(z["j"]) / z["j"].rolling(60).var()
    sig_jb = pd.DataFrame(beta_jy, index=rets.index)
    m, _ = eval_factor("usdjpy_beta_60d", sig_jb, 1, library=eff_lib); results["usdjpy_beta_60d"] = m
else:
    print("USDJPY missing; skip usdjpy_beta_60d")

# C11: wti_beta_60d - 60d beta of asset returns to WTI (energy linkage)
sig_wb = rets.rolling(60).cov(rets["WTI"]) / rets["WTI"].rolling(60).var()
m, _ = eval_factor("wti_beta_60d", sig_wb, -1, library=eff_lib); results["wti_beta_60d"] = m

# C12: xau_beta_60d - 60d beta of asset returns to XAU (safe-haven linkage)
sig_xb = rets.rolling(60).cov(rets["XAU"]) / rets["XAU"].rolling(60).var()
m, _ = eval_factor("xau_beta_60d", sig_xb, 1, library=eff_lib); results["xau_beta_60d"] = m

print("=" * 70)
print("RECENT 2Y WINDOW CHECK for candidates that PASS full-window gate")
print("=" * 70)
cand_defs = {
    "realized_skew_60d": sig_sk, "trend_consistency_60d": sig_tc, "reversal_5d": sig_r5,
    "parkinson_vol_20d": sig_pv, "gain_loss_asym_60d": sig_gla, "corr_btc_20d": sig_cb,
    "serial_corr_10d": sig_sc, "mom_40d_vol20": sig_mv, "wti_beta_60d": sig_wb,
    "xau_beta_60d": sig_xb,
}
if dxy is not None:
    cand_defs["dxy_beta_60d"] = sig_db
if usdjpy is not None:
    cand_defs["usdjpy_beta_60d"] = sig_jb
signed = {"realized_skew_60d": -1, "reversal_5d": -1, "parkinson_vol_20d": -1,
          "gain_loss_asym_60d": -1, "corr_btc_20d": -1, "wti_beta_60d": -1,
          "dxy_beta_60d": -1}
for nm, mm in results.items():
    g = mm["admission_gate"]
    if g["ic_pass"] and g["icir_pass"]:
        sd = signed.get(nm, 1)
        m2, _ = eval_factor(nm + "_RECENT2Y", cand_defs[nm], sd, ("2032-09-17", END), library=eff_lib)
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
