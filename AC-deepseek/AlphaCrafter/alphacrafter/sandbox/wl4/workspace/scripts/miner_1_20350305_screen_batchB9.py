"""miner_1 2035-03-05 - screen NEW candidate factors (batch B9).

New ideas not present in library and not obvious duplicates of evicted/rejected:
skew, idiosyncratic vol, positive-day ratio (win-rate momentum), Amihud illiquidity,
idiosyncratic (residual) momentum, Sortino momentum, distance-from-52w-high,
raw cross-asset market beta, DXY beta, USDJPY beta, yield-trend x US10Y-beta interaction.

Admission gates (shared): abs(IC) >= 0.0070 AND abs(ICIR) >= 0.0840 at 10d horizon.
"""
import sys, time, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from factor_research_lib import (load_panels, close_panel, ret_panel, full_eval,
                                 library_signals, TRADABLE)

t0 = time.time()
panels = load_panels(days=6000)
closes = close_panel(panels)
rets = ret_panel(panels)
print("closes:", closes.shape, closes.index.min().date(), "..", closes.index.max().date(), flush=True)

# ---------- library reference signals ----------
lib_sigs = library_signals(panels, closes, rets,
                           vix=panels.get("VIX", pd.DataFrame()).get("close", None) if "VIX" in panels else None)
mom20 = closes / closes.shift(20) - 1.0
mom60 = closes / closes.shift(60) - 1.0
vol20 = rets.rolling(20).std()
vol60 = rets.rolling(60).std()
lib_sigs["vol_adj_mom_accel_20x60"] = (mom20 - mom60) / vol20.replace(0, np.nan)
mkt_ret = rets.mean(axis=1)
down = mkt_ret.where(mkt_ret < 0)
beta_down = {}
for a in rets.columns:
    z = pd.concat([rets[a].rename("a"), down.rename("m")], axis=1).dropna()
    beta_down[a] = z["a"].rolling(60, min_periods=40).cov(z["m"]) / z["m"].rolling(60, min_periods=40).var()
lib_sigs["dn_mkt_beta_60d"] = pd.DataFrame(beta_down, index=rets.index)
cn10y_ret = rets["CN10Y"] if "CN10Y" in rets else closes["CN10Y"].pct_change()
beta_cn = {}
for a in rets.columns:
    z = pd.concat([rets[a].rename("a"), cn10y_ret.rename("c")], axis=1).dropna()
    beta_cn[a] = z["a"].rolling(60, min_periods=40).cov(z["c"]) / z["c"].rolling(60, min_periods=40).var()
lib_sigs["rate_beta_cn10y_60d"] = pd.DataFrame(beta_cn, index=rets.index)

# ---------- candidate factor construction ----------
sig = {}

# C1: skew_60d - rolling skewness of daily returns (lottery / tail asymmetry)
sig["skew_60d"] = rets.rolling(60, min_periods=40).skew()

# C2: ivol_60d - idiosyncratic vol: std of residuals vs equal-weight cross-asset market
resid = {}
for a in rets.columns:
    z = pd.concat([rets[a].rename("a"), mkt_ret.rename("m")], axis=1).dropna()
    b = z["a"].rolling(60, min_periods=40).cov(z["m"]) / z["m"].rolling(60, min_periods=40).var()
    resid[a] = z["a"] - b * z["m"]
resid_df = pd.DataFrame(resid, index=rets.index)
sig["ivol_60d"] = resid_df.rolling(60, min_periods=40).std()

# C3: pos_day_ratio_60d - fraction of positive daily returns over 60d (win-rate momentum)
sig["pos_day_ratio_60d"] = (rets > 0).rolling(60, min_periods=40).mean()

# C4: amihud_illiq_20d - mean(|ret|/volume) over 20d (illiquidity proxy)
vol_panel = pd.concat({a: panels[a]["volume"].astype(float) for a in closes.columns if a in panels}, axis=1).sort_index()
illiq = (rets.abs() / vol_panel.replace(0, np.nan))
sig["amihud_illiq_20d"] = illiq.rolling(20, min_periods=10).mean() * 1e6

# C5: res_mom_20d - idiosyncratic momentum (asset 20d ret minus market 20d ret) / vol20
mkt20 = mkt_ret.rolling(20).sum()
sig["res_mom_20d"] = (mom20 - mkt20) / vol20.replace(0, np.nan)

# C6: sortino_60d - 60d momentum / downside deviation (std of negative returns)
neg = rets.where(rets < 0, 0.0)
downside_dev = np.sqrt((neg ** 2).rolling(60, min_periods=40).mean()) * np.sqrt(252)
sig["sortino_60d"] = mom60 / downside_dev.replace(0, np.nan)

# C7: dist_252d_high - distance from 52-week high
sig["dist_252d_high"] = closes / closes.rolling(252, min_periods=120).max() - 1.0

# C8: mkt_beta_60d - raw beta to equal-weight cross-asset market
beta_mkt = {}
for a in rets.columns:
    z = pd.concat([rets[a].rename("a"), mkt_ret.rename("m")], axis=1).dropna()
    beta_mkt[a] = z["a"].rolling(60, min_periods=40).cov(z["m"]) / z["m"].rolling(60, min_periods=40).var()
sig["mkt_beta_60d"] = pd.DataFrame(beta_mkt, index=rets.index)

# C9: dxy_beta_60d - beta to DXY changes (dollar sensitivity)
dxy_ret = panels["DXY"]["close"].pct_change() if "DXY" in panels else None
if dxy_ret is not None:
    beta_dxy = {}
    for a in rets.columns:
        z = pd.concat([rets[a].rename("a"), dxy_ret.rename("d")], axis=1).dropna()
        beta_dxy[a] = z["a"].rolling(60, min_periods=40).cov(z["d"]) / z["d"].rolling(60, min_periods=40).var()
    sig["dxy_beta_60d"] = pd.DataFrame(beta_dxy, index=rets.index)

# C10: usdjpy_beta_60d - beta to USDJPY changes (risk-on/off)
jpy_ret = panels["USDJPY"]["close"].pct_change() if "USDJPY" in panels else None
if jpy_ret is not None:
    beta_jpy = {}
    for a in rets.columns:
        z = pd.concat([rets[a].rename("a"), jpy_ret.rename("j")], axis=1).dropna()
        beta_jpy[a] = z["a"].rolling(60, min_periods=40).cov(z["j"]) / z["j"].rolling(60, min_periods=40).var()
    sig["usdjpy_beta_60d"] = pd.DataFrame(beta_jpy, index=rets.index)

# C11: yield_trend_beta - interaction: US10Y-beta * sign(US10Y 20d momentum)
us10y_ret = rets["US10Y"]
beta_u10 = {}
for a in rets.columns:
    z = pd.concat([rets[a].rename("a"), us10y_ret.rename("u")], axis=1).dropna()
    beta_u10[a] = z["a"].rolling(60, min_periods=40).cov(z["u"]) / z["u"].rolling(60, min_periods=40).var()
beta_u10_df = pd.DataFrame(beta_u10, index=rets.index)
yield_mom20 = np.sign(us10y_ret.rolling(20).sum())
sig["yield_trend_beta"] = beta_u10_df * yield_mom20

# ---------- evaluation ----------
def eval_factor(name, s, expected_sign):
    m, ics = full_eval(s, closes, (1, 2, 3, 5, 10, 20), 8, expected_sign,
                       library=lib_sigs, admission_horizon=10)
    m["admission_gate"] = {
        "ic_gate_abs": 0.0070, "icir_gate_abs": 0.0840,
        "ic_pass": abs(m["ic"]) >= 0.0070,
        "icir_pass": abs(m["icir"]) >= 0.0840,
    }
    g = m["admission_gate"]
    ok = g["ic_pass"] and g["icir_pass"]
    print(f"{name:22s} dir={expected_sign:+d} | ic={m['ic']:+.4f} icir={m['icir']:+.4f} "
          f"hit={m['ic_hit_ratio']:.2f} n={m['n_ic_dates']:5d} cov8={m['coverage_dates_ge8']:.2f} "
          f"turn={m['turnover_10d_rank']:.2f} maxcorr={m.get('max_abs_library_correlation')} "
          f"({m.get('max_corr_factor')}) -> {'PASS' if ok else 'fail'}", flush=True)
    return m

results = {}
sign_map = {
    "skew_60d": 1, "ivol_60d": -1, "pos_day_ratio_60d": 1, "amihud_illiq_20d": -1,
    "res_mom_20d": 1, "sortino_60d": 1, "dist_252d_high": 1, "mkt_beta_60d": -1,
    "dxy_beta_60d": 1, "usdjpy_beta_60d": 1, "yield_trend_beta": -1,
}
for nm, s in sig.items():
    results[nm] = eval_factor(nm, s, sign_map[nm])

# reverse-sign probes for sign-ambiguous candidates
for nm in ["skew_60d", "ivol_60d", "amihud_illiq_20d", "dist_252d_high",
           "mkt_beta_60d", "dxy_beta_60d", "usdjpy_beta_60d"]:
    results[nm + "_rev"] = eval_factor(nm + "_rev", sig[nm], -sign_map[nm])

print("=" * 100)
print("SUMMARY")
print("=" * 100)
for nm, mm in results.items():
    g = mm["admission_gate"]
    ok = "PASS" if (g["ic_pass"] and g["icir_pass"]) else "fail"
    print(f"{nm:24s} ic={mm['ic']:+.4f} icir={mm['icir']:+.4f} hit={mm['ic_hit_ratio']:.2f} "
          f"n={mm['n_ic_dates']:5d} cov8={mm['coverage_dates_ge8']:.2f} turn={mm['turnover_10d_rank']:.2f} "
          f"maxcorr={mm.get('max_abs_library_correlation')} -> {ok}", flush=True)
print("elapsed_s:", round(time.time() - t0, 1), flush=True)
print("DONE", flush=True)
