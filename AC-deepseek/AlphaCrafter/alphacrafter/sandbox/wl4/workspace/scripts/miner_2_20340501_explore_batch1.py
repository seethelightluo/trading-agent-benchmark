"""miner_2 2034-05-01: (1) revalidate 3 effective library factors for drift;
(2) screen new candidate factor ideas on the 15-asset cross-asset universe.
Data visible through 2034-04-28 (previous completed trading day)."""
import sys, warnings, json
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

sys.path.insert(0, "scripts")
from factor_research_lib import (
    load_panels, close_panel, forward_returns,
    rank_ic_series, summarize_ic, coverage_metrics, turnover_rank, decay_profile,
    full_eval, library_signals, max_library_corr,
)

panels = load_panels(days=6000)
closes = close_panel(panels)
rets = closes.pct_change()
END = closes.index.max().strftime("%Y-%m-%d")
print("data through:", END, "| n_dates:", len(closes), "| n_assets:", closes.shape[1])

# macro panels (observation-only)
vix = panels["VIX"]["close"].astype(float) if "VIX" in panels else None
dxy = panels["DXY"]["close"].astype(float) if "DXY" in panels else None
dxy_ret = dxy.pct_change() if dxy is not None else None

def eval_factor(name, sig, expected_sign, window=None, library=None):
    s = sig if window is None else sig.loc[window[0]:window[1]]
    c = closes if window is None else closes.loc[window[0]:window[1]]
    m, ics = full_eval(s, c, (1,2,3,5,10,20), 8, expected_sign,
                       library=library, admission_horizon=10)
    m["admission_gate"] = {
        "ic_gate_abs": 0.0070, "icir_gate_abs": 0.0840,
        "ic_pass": abs(m["ic"]) >= 0.0070,
        "icir_pass": abs(m["icir"]) >= 0.0840,
    }
    print(f"=== {name} (expected dir {expected_sign:+d}) ===")
    print(json.dumps(m, indent=1))
    return m, ics

# ---------- library reference signals for correlation ----------
lib_sigs = library_signals(panels, closes, rets, vix)
# add the 3 effective factor panels
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
cn10y = panels["CN10Y"]["close"].astype(float)
cn10y_ret = cn10y.pct_change()
beta_cn = {}
for a in rets.columns:
    z = pd.concat([rets[a].rename("a"), cn10y_ret.rename("r")], axis=1).dropna()
    beta_cn[a] = z["a"].rolling(60).cov(z["r"]) / z["r"].rolling(60).var()
lib_sigs["rate_beta_cn10y_60d"] = pd.DataFrame(beta_cn, index=rets.index)

print("=" * 70)
print("PART 1: REVALIDATE 3 EFFECTIVE FACTORS (drift check)")
print("=" * 70)
eval_factor("vol_adj_mom_accel_20x60", lib_sigs["vol_adj_mom_accel_20x60"], 1)
eval_factor("dn_mkt_beta_60d", lib_sigs["dn_mkt_beta_60d"], 1)
eval_factor("rate_beta_cn10y_60d", lib_sigs["rate_beta_cn10y_60d"], -1)
print("--- RECENT 1Y (2033-05-02..END) drift ---")
eval_factor("vol_adj_mom_accel_20x60", lib_sigs["vol_adj_mom_accel_20x60"], 1, ("2033-05-02", END))
eval_factor("dn_mkt_beta_60d", lib_sigs["dn_mkt_beta_60d"], 1, ("2033-05-02", END))
eval_factor("rate_beta_cn10y_60d", lib_sigs["rate_beta_cn10y_60d"], -1, ("2033-05-02", END))

print("=" * 70)
print("PART 2: NEW CANDIDATE SCREEN")
print("=" * 70)

# C1: dxy_beta_60d - beta of asset returns to DXY returns (USD strength)
if dxy_ret is not None:
    beta_dxy = {}
    for a in rets.columns:
        z = pd.concat([rets[a].rename("a"), dxy_ret.rename("d")], axis=1).dropna()
        beta_dxy[a] = z["a"].rolling(60).cov(z["d"]) / z["d"].rolling(60).var()
    sig_dxy = pd.DataFrame(beta_dxy, index=rets.index)
    eval_factor("dxy_beta_60d", sig_dxy, -1, library=lib_sigs)

# C2: skew_60d - rolling skewness of daily returns (crash-risk / lottery)
sig_skew = rets.rolling(60).skew()
eval_factor("skew_60d", sig_skew, 1, library=lib_sigs)

# C3: beta_asym_60d = downside beta - upside beta (asymmetric tail sensitivity)
up = mkt_ret.where(mkt_ret > 0)
beta_up = {}
beta_dn2 = {}
for a in rets.columns:
    z = pd.concat([rets[a].rename("a"), up.rename("m")], axis=1).dropna()
    beta_up[a] = z["a"].rolling(60).cov(z["m"]) / z["m"].rolling(60).var()
    z2 = pd.concat([rets[a].rename("a"), down.rename("m")], axis=1).dropna()
    beta_dn2[a] = z2["a"].rolling(60).cov(z2["m"]) / z2["m"].rolling(60).var()
sig_asym = pd.DataFrame(beta_dn2, index=rets.index) - pd.DataFrame(beta_up, index=rets.index)
eval_factor("beta_asym_60d", sig_asym, -1, library=lib_sigs)

# C4: autocorr_5d - 5-day lag autocorrelation of daily returns (reversal)
sig_ac = rets.apply(lambda x: x.rolling(30).apply(lambda y: pd.Series(y).autocorr(5), raw=False))
eval_factor("autocorr_5d_30w", sig_ac, -1, library=lib_sigs)

# C5: coskew_120d - coskewness with market return (systematic skew)
sig_cosk = pd.DataFrame(index=rets.index, columns=rets.columns, dtype=float)
mkt_cent = mkt_ret - mkt_ret.rolling(120).mean()
for a in rets.columns:
    ra = rets[a] - rets[a].rolling(120).mean()
    num = (ra * mkt_cent**2).rolling(120).mean()
    den = ra.rolling(120).std() * (mkt_cent**2).rolling(120).mean()
    sig_cosk[a] = num / den
eval_factor("coskew_120d", sig_cosk, 1, library=lib_sigs)

# C6: amihud_20d - illiquidity |ret|/volume, 20d mean
vol_abs = rets.abs()
amihud = {}
for a in closes.columns:
    v = panels[a]["volume"].astype(float) if "volume" in panels[a].columns else pd.Series(np.nan, index=rets.index)
    amihud[a] = (vol_abs[a] / v.replace(0, np.nan)).rolling(20).mean()
sig_amihud = pd.DataFrame(amihud, index=rets.index)
eval_factor("amihud_20d", sig_amihud, 1, library=lib_sigs)

# C7: dn_vol_share_60d - downside volatility share
downside_dev = rets.where(rets < 0, 0.0).rolling(60).std()
total_dev = rets.rolling(60).std()
sig_dnshare = downside_dev / total_dev.replace(0, np.nan)
eval_factor("dn_vol_share_60d", sig_dnshare, -1, library=lib_sigs)

# C8: mom20_voladj - 20d momentum scaled by 60d vol (risk-adjusted momentum, not accel)
sig_mv = (closes / closes.shift(20) - 1.0) / rets.rolling(60).std()
eval_factor("mom20_voladj60", sig_mv, 1, library=lib_sigs)

print("DONE")
