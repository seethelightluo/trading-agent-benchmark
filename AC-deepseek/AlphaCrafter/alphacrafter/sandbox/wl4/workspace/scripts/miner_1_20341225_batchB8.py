"""miner_1 2034-12-25 - batch screen of NEW candidate factors (batch B8).

Explores factor ideas NOT tested in miner_2 batch5 (2034-12-11) and not in the
evicted library: trend-efficiency (Kaufman ER), risk-adjusted momentum (Sharpe),
US10Y rate beta, cross-asset betas (WTI/XAU), tail ratio, longer-horizon momentum
acceleration, Bollinger %B, volume trend, and 1-lag return autocorrelation.

Admission gates (shared): abs(IC) >= 0.0070 AND abs(ICIR) >= 0.0840 at 10d horizon.
"""
import sys, time, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from factor_research_lib import (load_panels, close_panel, ret_panel, full_eval,
                                 library_signals, TRADABLE, MACRO)

t_start = time.time()
panels = load_panels(days=6000)
closes = close_panel(panels)
rets = ret_panel(panels)
print("closes:", closes.shape, closes.index.min().date(), "..", closes.index.max().date(), flush=True)

# ---------- library reference signals (3 effective factors) ----------
lib_sigs = library_signals(panels, closes, rets, vix=panels.get("VIX", pd.DataFrame()).get("close", None) if "VIX" in panels else None)
mom20 = closes / closes.shift(20) - 1.0
mom60 = closes / closes.shift(60) - 1.0
vol20 = rets.rolling(20).std()
vol60 = rets.rolling(60).std()
lib_sigs["vol_adj_mom_accel_20x60"] = (mom20 - mom60) / vol20
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

# C1: kaufman_eff_20d - Kaufman efficiency ratio (trend quality)
path = (closes - closes.shift(20)).abs()
vol_sum = rets.abs().rolling(20).sum()
sig["kaufman_eff_20d"] = path / vol_sum.replace(0, np.nan)

# C2: sharpe_60d - rolling 60d Sharpe ratio (risk-adjusted momentum)
sig["sharpe_60d"] = (mom60) / vol60

# C3: us10y_beta_60d - beta of asset rets to US10Y yield changes
us10y_ret = rets["US10Y"]
beta_u10 = {}
for a in rets.columns:
    z = pd.concat([rets[a].rename("a"), us10y_ret.rename("u")], axis=1).dropna()
    beta_u10[a] = z["a"].rolling(60, min_periods=40).cov(z["u"]) / z["u"].rolling(60, min_periods=40).var()
sig["us10y_beta_60d"] = pd.DataFrame(beta_u10, index=rets.index)

# C4: wti_beta_60d - beta to WTI (energy linkage)
wti_ret = rets["WTI"]
beta_wti = {}
for a in rets.columns:
    z = pd.concat([rets[a].rename("a"), wti_ret.rename("w")], axis=1).dropna()
    beta_wti[a] = z["a"].rolling(60, min_periods=40).cov(z["w"]) / z["w"].rolling(60, min_periods=40).var()
sig["wti_beta_60d"] = pd.DataFrame(beta_wti, index=rets.index)

# C5: xau_beta_60d - beta to XAU (safe-haven linkage)
xau_ret = rets["XAU"]
beta_xau = {}
for a in rets.columns:
    z = pd.concat([rets[a].rename("a"), xau_ret.rename("g")], axis=1).dropna()
    beta_xau[a] = z["a"].rolling(60, min_periods=40).cov(z["g"]) / z["g"].rolling(60, min_periods=40).var()
sig["xau_beta_60d"] = pd.DataFrame(beta_xau, index=rets.index)

# C6: tail_ratio_60d - 95th/5th percentile ratio of 60d returns (fat tail)
tail_hi = rets.rolling(60, min_periods=45).quantile(0.95)
tail_lo = rets.rolling(60, min_periods=45).quantile(0.05)
sig["tail_ratio_60d"] = (tail_hi - tail_lo) / vol60.replace(0, np.nan)

# C7: mom_accel_60_120 - 60d vs 120d momentum acceleration (skip 5), vol-adj
mom120 = closes.shift(5) / closes.shift(125) - 1.0
sig["mom_accel_60_120"] = (mom60 - mom120) / vol60

# C8: bollinger_pctb_20 - Bollinger %B (position within 2-sigma band)
sma20 = closes.rolling(20).mean()
sig["bollinger_pctb_20"] = (closes - sma20) / (2.0 * vol20 * np.sqrt(20)).replace(0, np.nan)

# C9: volume_trend_20d - slope of log(volume) over 20d (volume momentum)
vol_panel = pd.concat({a: panels[a]["volume"].astype(float) for a in closes.columns if a in panels}, axis=1).sort_index()
logv = np.log(vol_panel.replace(0, np.nan))
sig["volume_trend_20d"] = (logv - logv.shift(20)) / 20.0

# C10: ret_autocorr_1d_60 - 1-lag autocorrelation of daily returns over 60d
def autocorr_1d(x, win=60, minp=30):
    out = pd.Series(np.nan, index=x.index)
    r = x.rolling(win, min_periods=minp).apply(lambda w: np.corrcoef(w[:-1], w[1:])[0, 1] if len(w) > 3 and np.std(w[:-1]) > 0 and np.std(w[1:]) > 0 else np.nan, raw=True)
    return r
sig["ret_autocorr_1d_60"] = pd.concat({a: autocorr_1d(rets[a]) for a in rets.columns}, axis=1).sort_index()

def eval_factor(name, s, expected_sign, window=None, library=None):
    s = s if window is None else s.loc[window[0]:window[1]]
    c = closes if window is None else closes.loc[window[0]:window[1]]
    m, ics = full_eval(s, c, (1, 2, 3, 5, 10, 20), 8, expected_sign,
                       library=library, admission_horizon=10)
    m["admission_gate"] = {
        "ic_gate_abs": 0.0070, "icir_gate_abs": 0.0840,
        "ic_pass": abs(m["ic"]) >= 0.0070,
        "icir_pass": abs(m["icir"]) >= 0.0840,
    }
    g = m["admission_gate"]
    ok = g["ic_pass"] and g["icir_pass"]
    print(f"=== {name} (dir {expected_sign:+d}) | ic={m['ic']} icir={m['icir']} "
          f"hit={m['ic_hit_ratio']} n={m['n_ic_dates']} cov8={m['coverage_dates_ge8']} "
          f"turn={m['turnover_10d_rank']} maxcorr={m.get('max_abs_library_correlation')} "
          f"({m.get('max_corr_factor')}) gate={'PASS' if ok else 'FAIL'}", flush=True)
    return m

results = {}
signed = {
    "kaufman_eff_20d": 1, "sharpe_60d": 1, "us10y_beta_60d": -1,
    "wti_beta_60d": -1, "xau_beta_60d": -1, "tail_ratio_60d": -1,
    "mom_accel_60_120": 1, "bollinger_pctb_20": 1, "volume_trend_20d": 1,
    "ret_autocorr_1d_60": 1,
}
for nm, s in sig.items():
    m = eval_factor(nm, s, signed[nm], library=lib_sigs)
    results[nm] = m

# also check reverse-sign for beta factors (signs unknown a priori)
for nm in ["us10y_beta_60d", "wti_beta_60d", "xau_beta_60d", "volume_trend_20d", "ret_autocorr_1d_60", "bollinger_pctb_20"]:
    m = eval_factor(nm + "_rev", sig[nm], -signed[nm], library=lib_sigs)
    results[nm + "_rev"] = m

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
