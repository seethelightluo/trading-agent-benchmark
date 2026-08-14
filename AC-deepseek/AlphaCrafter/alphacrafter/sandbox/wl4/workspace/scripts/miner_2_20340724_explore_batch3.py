"""miner_2 2034-07-24: (1) revalidate 3 effective library factors for drift;
(2) screen NEW candidate factor ideas (batch 3) on the 15-asset cross-asset universe.
Data visible through 2034-07-21 (previous completed trading day)."""
import sys, warnings, json
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

sys.path.insert(0, "scripts")
from factor_research_lib import (
    load_panels, close_panel, forward_returns,
    rank_ic_series, summarize_ic, coverage_metrics, turnover_rank, decay_profile,
    full_eval, library_signals, max_library_corr, TRADABLE,
)

panels = load_panels(days=6000)
closes = close_panel(panels)
rets = closes.pct_change()
END = closes.index.max().strftime("%Y-%m-%d")
print("data through:", END, "| n_dates:", len(closes), "| n_assets:", closes.shape[1])

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
    print(f"=== {name} (dir {expected_sign:+d}) | ic={m['ic']} icir={m['icir']} "
          f"hit={m['ic_hit_ratio']} n={m['n_ic_dates']} cov={m['coverage_dates_ge8']} "
          f"turn={m['turnover_10d_rank']} maxcorr={m.get('max_abs_library_correlation')} "
          f"({m.get('max_corr_factor')}) gate={m['admission_gate']['ic_pass'] and m['admission_gate']['icir_pass']}")
    return m, ics

# ---------- effective library reference signals (3 kept factors) ----------
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
print("effective library reference signals:", list(eff_lib.keys()))

print("=" * 70)
print("PART 1: REVALIDATE 3 EFFECTIVE FACTORS (drift check)")
print("=" * 70)
eval_factor("vol_adj_mom_accel_20x60", eff_lib["vol_adj_mom_accel_20x60"], 1, library=eff_lib)
eval_factor("dn_mkt_beta_60d", eff_lib["dn_mkt_beta_60d"], 1, library=eff_lib)
eval_factor("rate_beta_cn10y_60d", eff_lib["rate_beta_cn10y_60d"], -1, library=eff_lib)
print("--- RECENT 1Y (2033-07-21..END) drift ---")
eval_factor("vol_adj_mom_accel_20x60", eff_lib["vol_adj_mom_accel_20x60"], 1, ("2033-07-21", END))
eval_factor("dn_mkt_beta_60d", eff_lib["dn_mkt_beta_60d"], 1, ("2033-07-21", END))
eval_factor("rate_beta_cn10y_60d", eff_lib["rate_beta_cn10y_60d"], -1, ("2033-07-21", END))

print("=" * 70)
print("PART 2: NEW CANDIDATE SCREEN (batch 3)")
print("=" * 70)

# N1: cdl_pos_20d - mean (close-low)/(high-low) over 20d (candle position / buying pressure)
hi = closes.rolling(20).max()
lo = closes.rolling(20).min()
cdl = (closes - lo) / (hi - lo).replace(0, np.nan)
sig_cdl = cdl.rolling(20).mean()
eval_factor("cdl_pos_20d", sig_cdl, 1, library=eff_lib)

# N2: var_ratio_60_20 - 60d variance / (3*20d variance) (trend persistence vs mean reversion)
sig_vr = rets.rolling(60).var() / (3.0 * rets.rolling(20).var())
eval_factor("var_ratio_60_20", sig_vr, 1, library=eff_lib)

# N3: skew_60d - rolling skewness of 60d returns
sig_sk = rets.rolling(60).skew()
eval_factor("skew_60d", sig_sk, -1, library=eff_lib)

# N4: amihud_illiq_20d - mean(|ret|/volume) over 20d (liquidity proxy)
vol_panel = pd.concat({a: panels[a]["volume"].astype(float) for a in closes.columns if a in panels}, axis=1).sort_index()
illiq = (rets.abs() / vol_panel.replace(0, np.nan)).rolling(20).mean()
eval_factor("amihud_illiq_20d", illiq, 1, library=eff_lib)

# N5: global_mom_spill_20d - equal-weight mean of other-asset 20d momentum (global risk-on)
n = len(closes.columns)
mom_all = closes / closes.shift(20) - 1.0
spill = (mom_all.sum(axis=1) - mom_all) / (n - 1)
sig_spill = pd.DataFrame({a: spill for a in closes.columns}, index=closes.index)
eval_factor("global_mom_spill_20d", sig_spill, 1, library=eff_lib)

# N6: us10y_beta_60d - beta of asset returns to US10Y yield changes (rate sensitivity)
us10y_ret = us10y.pct_change()
beta_u10 = {}
for a in rets.columns:
    z = pd.concat([rets[a].rename("a"), us10y_ret.rename("r")], axis=1).dropna()
    beta_u10[a] = z["a"].rolling(60).cov(z["r"]) / z["r"].rolling(60).var()
sig_u10 = pd.DataFrame(beta_u10, index=rets.index)
eval_factor("us10y_beta_60d", sig_u10, -1, library=eff_lib)

# N7: dxy_beta_60d - beta of asset returns to DXY changes (USD sensitivity)
dxy_ret = dxy.pct_change()
beta_dxy = {}
for a in rets.columns:
    z = pd.concat([rets[a].rename("a"), dxy_ret.rename("r")], axis=1).dropna()
    beta_dxy[a] = z["a"].rolling(60).cov(z["r"]) / z["r"].rolling(60).var()
sig_dxy = pd.DataFrame(beta_dxy, index=rets.index)
eval_factor("dxy_beta_60d", sig_dxy, -1, library=eff_lib)

# N8: overnight_ret_20d - mean open/prev_close-1 over 20d (overnight gap persistence)
open_panel = pd.concat({a: panels[a]["open"].astype(float) for a in closes.columns if a in panels}, axis=1).sort_index()
ovn = (open_panel / closes.shift(1) - 1.0).rolling(20).mean()
eval_factor("overnight_ret_20d", ovn, 1, library=eff_lib)

# N9: range_20d - mean((high-low)/close) over 20d (range-based vol proxy)
sig_rng = ((panels_hi := pd.concat({a: panels[a]["high"].astype(float) for a in closes.columns if a in panels}, axis=1).sort_index())
           - lo) / closes
sig_rng = sig_rng.rolling(20).mean()
eval_factor("range_20d", sig_rng, -1, library=eff_lib)

# N10: btc_mom_20d_spill - BTC 20d momentum as cross-sectional signal (crypto leads risk sentiment)
btc_mom = mom_all["BTC"] if "BTC" in mom_all.columns else None
if btc_mom is not None:
    sig_btc = pd.DataFrame({a: btc_mom for a in closes.columns}, index=closes.index)
    eval_factor("btc_mom_20d_spill", sig_btc, 1, library=eff_lib)
else:
    print("BTC not in panel; skip btc_mom_20d_spill")

# N11: recovery_speed_60d - close/rolling_min(close,60)-1 (distance from recent low, recovery speed)
sig_rec = closes / closes.rolling(60).min() - 1.0
eval_factor("recovery_speed_60d", sig_rec, 1, library=eff_lib)

# N12: vol_regime_mom_accel - momentum accel gated by low-vol regime
vol_med = vol20.rolling(250).median()
lowvol = (vol20 <= vol_med).astype(float)
sig_vrm = ((mom20 - mom60) / vol20) * lowvol
eval_factor("vol_regime_mom_accel_20x60", sig_vrm, 1, library=eff_lib)

print("DONE")
