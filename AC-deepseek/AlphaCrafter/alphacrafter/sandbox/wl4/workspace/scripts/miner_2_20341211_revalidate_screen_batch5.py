"""miner_2 2034-12-11: (1) revalidate 3 effective library factors for drift;
(2) screen NEW candidate factor ideas (batch 5) on the 15-asset cross-asset universe.
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
usdcny = panels["USDCNY"]["close"].astype(float) if "USDCNY" in panels else None
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
print("PART 1: REVALIDATE 3 EFFECTIVE FACTORS (drift check)")
print("=" * 70)
for nm, sg, sd in [("vol_adj_mom_accel_20x60", eff_lib["vol_adj_mom_accel_20x60"], 1),
                   ("dn_mkt_beta_60d", eff_lib["dn_mkt_beta_60d"], 1),
                   ("rate_beta_cn10y_60d", eff_lib["rate_beta_cn10y_60d"], -1)]:
    eval_factor(nm, sg, sd, library=eff_lib)
print("--- RECENT 2Y drift (2032-12-09..END) ---", flush=True)
for nm, sg, sd in [("vol_adj_mom_accel_20x60", eff_lib["vol_adj_mom_accel_20x60"], 1),
                   ("dn_mkt_beta_60d", eff_lib["dn_mkt_beta_60d"], 1),
                   ("rate_beta_cn10y_60d", eff_lib["rate_beta_cn10y_60d"], -1)]:
    eval_factor(nm, sg, sd, ("2032-12-09", END), library=eff_lib)

print("=" * 70)
print("PART 2: NEW CANDIDATE SCREEN (batch 5)")
print("=" * 70)

results = {}

# C1: skew_60d - rolling skewness of 60d returns (left-tail / lottery aversion)
sig_sk = rets.rolling(60, min_periods=40).skew()
m, _ = eval_factor("skew_60d", sig_sk, -1, library=eff_lib); results["skew_60d"] = m

# C2: mom_30d_voladj - 30d momentum / 30d vol (shorter risk-adjusted trend)
vol30 = rets.rolling(30).std()
sig_m30 = (closes / closes.shift(30) - 1.0) / vol30
m, _ = eval_factor("mom_30d_voladj", sig_m30, 1, library=eff_lib); results["mom_30d_voladj"] = m

# C3: rel_strength_20d_z - cross-sectional z-score of 20d return (relative strength vs dispersion)
mom20_2 = closes / closes.shift(20) - 1.0
cs_mean = mom20_2.mean(axis=1)
cs_std = mom20_2.std(axis=1)
sig_rs = (mom20_2.sub(cs_mean, axis=0)).div(cs_std.replace(0, np.nan), axis=0)
m, _ = eval_factor("rel_strength_20d_z", sig_rs, 1, library=eff_lib); results["rel_strength_20d_z"] = m

# C4: dxy_beta_60d - 60d beta of asset returns to DXY (USD strength sensitivity)
if dxy is not None:
    dxy_ret = dxy.pct_change()
    beta_dxy = {}
    for a in rets.columns:
        z = pd.concat([rets[a].rename("a"), dxy_ret.rename("d")], axis=1).dropna()
        beta_dxy[a] = z["a"].rolling(60, min_periods=40).cov(z["d"]) / z["d"].rolling(60, min_periods=40).var()
    sig_db = pd.DataFrame(beta_dxy, index=rets.index)
    m, _ = eval_factor("dxy_beta_60d", sig_db, -1, library=eff_lib); results["dxy_beta_60d"] = m
else:
    print("DXY missing; skip dxy_beta_60d")

# C5: usdjpy_beta_60d - 60d beta to USDJPY (carry proxy sensitivity)
if usdjpy is not None:
    jpy_ret = usdjpy.pct_change()
    beta_jpy = {}
    for a in rets.columns:
        z = pd.concat([rets[a].rename("a"), jpy_ret.rename("j")], axis=1).dropna()
        beta_jpy[a] = z["a"].rolling(60, min_periods=40).cov(z["j"]) / z["j"].rolling(60, min_periods=40).var()
    sig_jb = pd.DataFrame(beta_jpy, index=rets.index)
    m, _ = eval_factor("usdjpy_beta_60d", sig_jb, 1, library=eff_lib); results["usdjpy_beta_60d"] = m
else:
    print("USDJPY missing; skip usdjpy_beta_60d")

# C6: range_ratio_20d - (rolling max high - rolling min low)/close over 20d (range expansion)
hi20 = closes.rolling(20).max()
lo20 = closes.rolling(20).min()
sig_rr = (hi20 - lo20) / closes
m, _ = eval_factor("range_ratio_20d", sig_rr, -1, library=eff_lib); results["range_ratio_20d"] = m

# C7: ret_autocorr_5d - 5-lag autocorrelation of daily returns (trend persistence)
sig_ac = rets.rolling(30, min_periods=20).apply(lambda x: pd.Series(x).autocorr(5) if len(x) > 6 else np.nan, raw=False)
m, _ = eval_factor("ret_autocorr_5d", sig_ac, 1, library=eff_lib); results["ret_autocorr_5d"] = m

# C8: drawdown_20d - close/rolling_max(close,20)-1 (short-term drawdown)
sig_dd = closes / closes.rolling(20).max() - 1.0
m, _ = eval_factor("drawdown_20d", sig_dd, 1, library=eff_lib); results["drawdown_20d"] = m

# C9: bollinger_bw_20 - (2*std20)/sma20 (Bollinger bandwidth)
sma20 = closes.rolling(20).mean()
sig_bw = (2.0 * vol20 * np.sqrt(len(closes)) * 0 + 2.0 * closes.rolling(20).std()) / sma20
m, _ = eval_factor("bollinger_bw_20", sig_bw, -1, library=eff_lib); results["bollinger_bw_20"] = m

# C10: gk_vol_20d - Garman-Klass volatility (open/high/low/close intraday range)
open_panel = pd.concat({a: panels[a]["open"].astype(float) for a in closes.columns if a in panels}, axis=1).sort_index()
hi_panel = pd.concat({a: panels[a]["high"].astype(float) for a in closes.columns if a in panels}, axis=1).sort_index()
lo_panel = pd.concat({a: panels[a]["low"].astype(float) for a in closes.columns if a in panels}, axis=1).sort_index()
gk = 0.5 * (np.log(hi_panel / lo_panel) ** 2) - (2 * np.log(2) - 1) * (np.log(open_panel / closes.shift(1)) ** 2)
sig_gk = np.sqrt(gk.clip(lower=0).rolling(20).mean())
m, _ = eval_factor("gk_vol_20d", sig_gk, -1, library=eff_lib); results["gk_vol_20d"] = m

# C11: cdl_pos_60d - close position within 60d range (longer-horizon trend location)
hi60 = closes.rolling(60).max()
lo60 = closes.rolling(60).min()
sig_c60 = (closes - lo60) / (hi60 - lo60).replace(0, np.nan)
m, _ = eval_factor("cdl_pos_60d", sig_c60, 1, library=eff_lib); results["cdl_pos_60d"] = m

# C12: corr_btc_20d - 20d rolling correlation of asset returns with BTC (crypto linkage)
if "BTC" in rets.columns:
    btc_ret = rets["BTC"]
    sig_cb = rets.rolling(20).corr(btc_ret)
    m, _ = eval_factor("corr_btc_20d", sig_cb, -1, library=eff_lib); results["corr_btc_20d"] = m

# C13: kurt_20d - rolling kurtosis of 20d returns (tail-fatness)
sig_ku = rets.rolling(20, min_periods=15).kurt()
m, _ = eval_factor("kurt_20d", sig_ku, -1, library=eff_lib); results["kurt_20d"] = m

# C14: vol_mom_ratio_10_60 - 10d vol / 60d vol (short-term vol acceleration)
vol10 = rets.rolling(10).std()
vol60 = rets.rolling(60).std()
sig_vr = vol10 / vol60
m, _ = eval_factor("vol_mom_ratio_10_60", sig_vr, -1, library=eff_lib); results["vol_mom_ratio_10_60"] = m

# C15: usdcny_beta_60d - 60d beta to USDCNY (EM/CN sensitivity)
if usdcny is not None:
    cny_ret = usdcny.pct_change()
    beta_cny = {}
    for a in rets.columns:
        z = pd.concat([rets[a].rename("a"), cny_ret.rename("c")], axis=1).dropna()
        beta_cny[a] = z["a"].rolling(60, min_periods=40).cov(z["c"]) / z["c"].rolling(60, min_periods=40).var()
    sig_cny = pd.DataFrame(beta_cny, index=rets.index)
    m, _ = eval_factor("usdcny_beta_60d", sig_cny, -1, library=eff_lib); results["usdcny_beta_60d"] = m
else:
    print("USDCNY missing; skip usdcny_beta_60d")

print("=" * 70)
print("RECENT 2Y WINDOW CHECK for candidates that PASS full-window gate")
print("=" * 70)
cand_defs = {
    "skew_60d": sig_sk, "mom_30d_voladj": sig_m30, "rel_strength_20d_z": sig_rs,
    "range_ratio_20d": sig_rr, "ret_autocorr_5d": sig_ac, "drawdown_20d": sig_dd,
    "bollinger_bw_20": sig_bw, "gk_vol_20d": sig_gk, "cdl_pos_60d": sig_c60,
    "kurt_20d": sig_ku, "vol_mom_ratio_10_60": sig_vr,
}
if dxy is not None:
    cand_defs["dxy_beta_60d"] = sig_db
if usdjpy is not None:
    cand_defs["usdjpy_beta_60d"] = sig_jb
if usdcny is not None:
    cand_defs["usdcny_beta_60d"] = sig_cny
if "BTC" in rets.columns:
    cand_defs["corr_btc_20d"] = sig_cb
signed = {"skew_60d": -1, "range_ratio_20d": -1, "bollinger_bw_20": -1,
          "gk_vol_20d": -1, "kurt_20d": -1, "vol_mom_ratio_10_60": -1,
          "dxy_beta_60d": -1, "usdcny_beta_60d": -1, "corr_btc_20d": -1}
for nm, mm in results.items():
    g = mm["admission_gate"]
    if g["ic_pass"] and g["icir_pass"]:
        sd = signed.get(nm, 1)
        m2, _ = eval_factor(nm + "_RECENT2Y", cand_defs[nm], sd, ("2032-12-09", END), library=eff_lib)
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
