"""miner_2 exploration cycle 2026-07-30 (v6): NEW factor families not tried in
previous cycles (v1-v5, miner_1/3 rounds). Distinct from existing/quarantined
library: mom*, vol_of_vol, vix_beta_cond, dxy/btc/eth/wti beta, skew_vol_comp,
breadth_cond_mom, trend_r2, eff_ratio, vol_ratio, dist_ma, range_pos, etc.

Candidates (h=10, 15-asset cross-asset universe, min_valid=8):
  A. downside_beta_120 : SPX downside-beta minus upside-beta (asymmetric crash sensitivity)
  B. amihud_illiq_20   : Amihud illiquidity = mean(|ret|/volume, 20d)  [liquidity premium]
  C. vol_price_corr_60 : rolling 60d corr(daily ret, dlog(volume))     [volume confirmation]
  D. trend_tstat_60    : OLS slope t-stat of close over 60d            [trend significance]
  E. kurt_ret_60       : kurtosis of 60d daily returns                 [tail risk]
  F. gk_ratio_20       : Garman-Klass OHLC vol / close-based realized vol [gap/range efficiency]
  G. amihud_illiq_z20  : time-series z-score of amihud_illiq_20        [relative liquidity change]
"""
import sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
from factor_validate import (closes_panel, macro_closes, forward_returns,
                             ic_series, summary_metrics, library_ic_series_map,
                             max_abs_library_corr, regime_split, load_panel, WATCH)

VIS = "2026-07-29"
H = 10
close = closes_panel(VIS)
macro = macro_closes(VIS)
frames = load_panel(visible_through=VIS)
ret = close.pct_change()
high = pd.DataFrame({s: df.set_index("date")["high"].astype(float) for s, df in frames.items()}).sort_index().reindex(close.index)
low = pd.DataFrame({s: df.set_index("date")["low"].astype(float) for s, df in frames.items()}).sort_index().reindex(close.index)
opn = pd.DataFrame({s: df.set_index("date")["open"].astype(float) for s, df in frames.items()}).sort_index().reindex(close.index)
vol = pd.DataFrame({s: df.set_index("date")["volume"].astype(float) for s, df in frames.items()}).sort_index().reindex(close.index)
fr = forward_returns(close, H)
print(f"close panel {close.shape} | dates {close.index.min().date()}..{close.index.max().date()} | assets {len(close.columns)}")

spx_ret = ret["SPX"]


def rolling_beta(asset_ret, mkt_ret, win, mask=None):
    out = {}
    for a in asset_ret.columns:
        pair = pd.concat([asset_ret[a].rename("a"), mkt_ret.rename("m")], axis=1)
        if mask is not None:
            pair = pair[mask]
        pair = pair.dropna()
        b = pair["a"].rolling(win).cov(pair["m"]) / pair["m"].rolling(win).var()
        out[a] = b
    return pd.DataFrame(out).reindex(asset_ret.index)


def eval_factor(name, factor, verbose=True):
    ics = ic_series(factor, fr, min_valid=8)
    if len(ics) < 100:
        print(f"=== {name} === INSUFFICIENT IC dates: {len(ics)} (need >=100)")
        return None
    m = summary_metrics(ics, factor, fr, close, h=H, step=10)
    if m is None:
        print(f"=== {name} === metrics None")
        return None
    reg = regime_split(ics)
    lib_ics = library_ic_series_map(close, h=H)
    m["max_abs_library_correlation"] = max_abs_library_corr(ics, lib_ics)
    m["gate_pass"] = abs(m["ic"]) >= 0.0070 and abs(m["icir"] or 0) >= 0.0840
    if verbose:
        print(f"=== {name} === IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} "
              f"n={m['n_ic_dates']} covA={m['coverage_asset_days']:.3f} covD={m['coverage_dates_ge8']:.3f} "
              f"turn={m['turnover_10d_rank']} maxrho={m['max_abs_library_correlation']} gate={m['gate_pass']}")
        for k, v in reg.items():
            print(f"    regime {k}: IC={v['ic']:+.4f} ICIR={v['icir']:+.4f} n={v['n']}")
    return {"metrics": m, "regime": reg}


results = {}

# A. downside beta 120: beta on SPX-down days minus beta on SPX-up days
spx_down = spx_ret < 0
spx_up = spx_ret >= 0
down_beta = rolling_beta(ret, spx_ret, 120, mask=spx_down)
up_beta = rolling_beta(ret, spx_ret, 120, mask=spx_up)
downside_beta = down_beta - up_beta
results["downside_beta_120"] = eval_factor("downside_beta_120", downside_beta)

# B. Amihud illiquidity 20d
amihud = (ret.abs() / vol.replace(0, np.nan)).rolling(20).mean()
results["amihud_illiq_20"] = eval_factor("amihud_illiq_20", amihud)

# G. Amihud z-score (time-series standardized)
amihud_z = (amihud - amihud.rolling(120).mean()) / amihud.rolling(120).std()
results["amihud_illiq_z20"] = eval_factor("amihud_illiq_z20", amihud_z)

# C. volume-price correlation 60d: corr(ret, dlog vol)
dlogv = np.log(vol.replace(0, np.nan)).diff()
vpc = {}
for a in ret.columns:
    pair = pd.concat([ret[a].rename("r"), dlogv[a].rename("v")], axis=1)
    vpc[a] = pair["r"].rolling(60).corr(pair["v"])
vpc = pd.DataFrame(vpc).reindex(close.index)
results["vol_price_corr_60"] = eval_factor("vol_price_corr_60", vpc)

# D. trend t-stat 60: OLS slope / SE of close over 60d
def trend_tstat(px, win=60):
    out = {}
    x = np.arange(win, dtype=float)
    xm = x - x.mean()
    ssx = (xm ** 2).sum()
    for a in px.columns:
        s = px[a]
        slope = s.rolling(win).apply(lambda y: np.polyfit(x, y, 1)[0], raw=True)
        resid = s.rolling(win).apply(lambda y: np.polyval(np.polyfit(x, y, 1), x) - y, raw=True)
        se = resid.rolling(win).apply(lambda e: np.sqrt((e ** 2).sum() / (win - 2) / ssx), raw=True)
        out[a] = slope / se
    return pd.DataFrame(out).reindex(px.index)

trend_t = trend_tstat(close, 60)
results["trend_tstat_60"] = eval_factor("trend_tstat_60", trend_t)

# E. kurtosis 60d
kurt = ret.rolling(60).kurt()
results["kurt_ret_60"] = eval_factor("kurt_ret_60", kurt)

# F. Garman-Klass vol (OHLC) / realized vol (close) over 20d
log_h = np.log(high / low)
log_c = np.log(close)
log_o = np.log(opn)
gk = 0.5 * (log_h ** 2) - (2 * np.log(2) - 1) * (np.log(close / opn) ** 2)
gk_vol = np.sqrt(gk.rolling(20).mean())
rv_vol = ret.rolling(20).std()
gk_ratio = gk_vol / rv_vol
results["gk_ratio_20"] = eval_factor("gk_ratio_20", gk_ratio)

print("\n=== SUMMARY ===")
for k, v in results.items():
    if v:
        m = v["metrics"]
        print(f"{k}: IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} n={m['n_ic_dates']} "
              f"maxrho={m['max_abs_library_correlation']} gate={m['gate_pass']}")
    else:
        print(f"{k}: FAILED")

with open("scripts/miner2_20260730_explore_v6_results.json", "w") as f:
    json.dump(results, f, indent=1, default=str)
print("saved scripts/miner2_20260730_explore_v6_results.json")
