"""miner_1 2030-03-07: screen candidate cross-asset factors on full history through visible_through.
Validates against the benchmark admission gates (|IC|>=0.007, |ICIR|>=0.084) at H=10.
No future leakage: all data truncated at visible_through."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from miner_1_common import (load_panel, load_macro_panel, forward_returns,
                            spearman_ic_series, ic_metrics, decay_by_horizon,
                            turnover_rank_chg, coverage, regime_slices, MIN_IC_DATES)

px, vol = load_panel(start="2020-01-01")
rets = px.pct_change()
vol20 = rets.rolling(20).std()
vol60 = rets.rolling(60).std()

def bdf(s):
    return s.to_frame("x").join(px).ffill()

def beta_to(asset_ret_series, macro_ret, window=60):
    x = asset_ret_series.to_frame("a").join(macro_ret.to_frame("m"))
    cov = x["a"].rolling(window).cov(x["m"])
    var = x["m"].rolling(window).var()
    return cov / var

macros = {m: load_macro_panel(m, start="2019-06-01") for m in ["VIX", "DXY", "USDCNY", "USDJPY", "EURUSD"]}
mret = {m: macros[m].pct_change() for m in macros}

candidates = {}

# 1) short-term reversal 5d
candidates["short_rev_5"] = -px.pct_change(5)

# 2) distance from 60d high (drawdown depth) - value/rebound tilt
candidates["dd_dist_60"] = px / px.rolling(60).max() - 1.0

# 3) vol-adjusted momentum 20d (sign normalized by vol gives stability)
candidates["vol_adj_mom_20"] = px.pct_change(20) / vol20

# 4) long-run mean reversion 120d scaled by 0
candidates["long_rev_120"] = -(px / px.rolling(120).mean() - 1.0)

# 5) USDJPY beta (risk-on carry/complexity) 60d
for s in px.columns:
    candidates.setdefault("usdjpy_beta_60", pd.DataFrame(index=px.index))
    candidates["usdjpy_beta_60"][s] = beta_to(rets[s], mret["USDJPY"], 60)

# 6) EURUSD beta 60d (EM risk sentiment)
for s in px.columns:
    candidates.setdefault("eurusd_beta_60", pd.DataFrame(index=px.index))
    candidates["eurusd_beta_60"][s] = beta_to(rets[s], mret["EURUSD"], 60)

# 7) reverse volume-price momentum: volume change * price return
vpx = px.pct_change(10)
vvol = vol.pct_change(10)
candidates["vol_px_damp_20"] = vpx * np.sign(vvol)  # price mom damped by volume direction

# 8) tail/flight factor: 3-day max drawdown (risk-off pressure)
candidates["max_drawdown_10"] = px / px.rolling(10).max() - 1.0

fwd = forward_returns(px, horizon=10)
print("="*90)
for name, f in candidates.items():
    fm = f.reindex(px.index)
    ics = spearman_ic_series(fm, fwd)
    met = ic_metrics(ics)
    if np.isnan(met["ic"]):
        print(f"{name:20s} n_ic={met['n_ic_dates']:<5} INSUFFICIENT")
        continue
    cov = coverage(fm, px)
    to10 = turnover_rank_chg(fm.resample('10D').last().reindex(px.index).ffill(), px)
    dec = decay_by_horizon(px, fm)
    reg = regime_slices(ics)
    rec = ic_metrics(ics.loc[ics.index >= pd.Timestamp('2029-01-01')]) if (ics.index >= pd.Timestamp('2029-01-01')).sum() >= MIN_IC_DATES else {}
    flag = "PASS" if (abs(met["ic"]) >= 0.007 and abs(met["icir"]) >= 0.084) else "fail"
    print(f"{name:20s} ic={met['ic']:.4f} icir={met['icir']:.4f} hit={met['hit']:.3f} "
          f"n={met['n_ic_dates']:<5} cov={cov:.3f} to10={to10:.3f} flag={flag}")
    if rec:
        print(f"   {name:18s} recent2029+ ic={rec['ic']:.4f} icir={rec['icir']:.4f} n={rec['n_ic_dates']}")
    print(f"   decay { {k: round(v,4) for k,v in dec.items()} }")
    print(f"   regimes { {k: [round(x,3) if not isinstance(x,str) else x for x in v] for k,v in reg.items()} }")
print("="*90)