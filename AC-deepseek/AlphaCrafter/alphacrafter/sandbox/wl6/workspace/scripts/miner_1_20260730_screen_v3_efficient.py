"""miner_1 efficient broad screen v3 (2026-07-30).

Single data load; NEW factor families distinct from the 4 quarantined library
factors (mom_10d_skip5, mom_120d_skip5, vol_of_vol20x60, vix_beta_cond_60x20):
  - relative cross-sectional strength / rotation
  - cross-asset betas (market, defensive basket, risk-on, US10Y, USD, VIX)
  - range position (stochastic-like)
  - volume-price correlation, high-low intraday position
  - residual momentum, risk-adjusted trend strength
Admission gate: |IC| >= 0.007, |ICIR| >= 0.084 at horizon 10 (15-asset universe).
Regime robustness: subperiods 2020-2022 vs 2023-2026.
"""
import sys, math, json
sys.path.insert(0, "scripts")
import pandas as pd
import numpy as np
from alphacrafter.sim.utils import get_stock_daily_data
from factor_validation_lib import (TRADABLE, MIN_INSTR, load_panel, load_macro,
                                   align_fwd_returns, rank_ic_series, ic_analysis,
                                   library_corr)

VISIBLE = "2026-07-29"
panel = load_panel(max_date=VISIBLE)
ret = panel.pct_change()
print(f"panel: {panel.shape} assets={panel.shape[1]} dates={panel.shape[0]} through {panel.index.max().date()}")

vix = load_macro("VIX", max_date=VISIBLE); vixr = vix.pct_change()
dxy = load_macro("DXY", max_date=VISIBLE); dxy_r = dxy.pct_change()
us10y = ret["US10Y"]; cn10y = ret["CN10Y"]

vol20 = ret.rolling(20).std(); vol60 = ret.rolling(60).std()
basket = ret[["XAU", "US10Y", "CN10Y"]].mean(axis=1)
risk_on = ret[["SPX", "NDX", "SOX"]].mean(axis=1)
market = ret.mean(axis=1)  # equal-weight cross-asset market

def beta_of(a, m, win):
    return a.rolling(win).cov(m) / m.rolling(win).var()

def corr_of(a, m, win):
    return a.rolling(win).corr(m)

C = {}
# --- relative cross-sectional strength (rotation) ---
for lb in (10, 20, 60):
    C[f"rel_strength_{lb}d"] = ret.rolling(lb).sum() - ret.rolling(lb).sum().median(axis=1)
# --- cross-asset betas / sensitivities ---
C["xasset_beta_60d"] = beta_of(ret, market, 60)                 # participation in cross-asset mkt
C["xasset_beta_neg_60d"] = -beta_of(ret, market, 60)
C["def_beta_60d"] = beta_of(ret, basket, 60)                    # defensive hedge demand
C["risk_on_beta_60d"] = beta_of(ret, risk_on, 60)
C["us10y_sens_60d"] = corr_of(ret, us10y, 60)                   # rate sensitivity (bond-like)
C["usd_sens_60d"] = corr_of(ret, dxy_r, 60)                     # USD sensitivity (risk-off)
C["vix_sens_neg_60d"] = -corr_of(ret, vixr, 60)                 # risk-on sensitivity
C["yield_curv_sens_60d"] = corr_of(ret, us10y - cn10y, 60)      # yield-curve slope sensitivity
# --- range position (stochastic) ---
for lb in (10, 20, 60):
    lo = panel.rolling(lb).min(); hi = panel.rolling(lb).max()
    C[f"range_pos_{lb}d"] = (panel - lo) / (hi - lo) - 0.5
C["range_pos_20d_voladj"] = ((panel - panel.rolling(20).min()) / (panel.rolling(20).max() - panel.rolling(20).min()) - 0.5) / (vol20 / vol60)
# --- volume-price / intraday ---
raw_vol = {}
for s in TRADABLE:
    df = get_stock_daily_data(symbol=s, days=4000)
    if df is not None and "volume" in df and len(df) > 30:
        raw_vol[s] = df.set_index(pd.to_datetime(df["date"]))["volume"].astype(float)
vol_panel = pd.DataFrame(raw_vol).sort_index()
vol_panel = vol_panel[vol_panel.index <= pd.Timestamp(VISIBLE)]
vol_chg = vol_panel.pct_change()
C["vol_price_corr_60d"] = corr_of(ret, vol_chg, 60)             # accumulation/distribution
C["hilo_pos_10d"] = ((panel - panel.rolling(10).min()) / (panel.rolling(10).max() - panel.rolling(10).min())).rolling(10).mean() - 0.5
# --- residual momentum / trend strength ---
resid = ret - beta_of(ret, market, 60).mul(market, axis=0)
C["resid_mom_60d"] = resid.rolling(60).sum()
C["trend_strength_60d"] = ret.rolling(60).sum() / (vol60 * math.sqrt(60))
C["sma_ratio_20x60"] = panel.rolling(20).mean() / panel.rolling(60).mean() - 1.0
C["mom20_highvol_cond"] = ret.rolling(20).sum() * (vol20 > vol20.rolling(120).median())
C["neg_skew_60d"] = -ret.rolling(60).skew()

# --- library signals (quarantined former library) for correlation audit ---
lib = {}
lib['mom_10d_skip5'] = panel.shift(5) / panel.shift(15) - 1.0
lib['mom_120d_skip5'] = panel.shift(5) / panel.shift(125) - 1.0
lib['vol_of_vol20x60'] = vol20.rolling(60).std()
lib['vix_beta_cond_60x20'] = -beta_of(ret, vixr, 60) * (vix / vix.shift(20) - 1.0)

print("=" * 118)
results = {}
for name, f in C.items():
    res = ic_analysis(f, panel, horizon=10, label=name)
    results[name] = res
    ic10 = rank_ic_series(f, align_fwd_returns(panel, 10)).dropna()
    sub1 = ic10[(ic10.index >= "2020-01-01") & (ic10.index <= "2022-12-31")]
    sub2 = ic10[ic10.index >= "2023-01-01"]
    s1 = f"{sub1.mean():+.4f}(n={len(sub1)})" if len(sub1) else "na"
    s2 = f"{sub2.mean():+.4f}(n={len(sub2)})" if len(sub2) else "na"
    rho = library_corr(f, lib)
    ok = (abs(res["ic"] or 0) >= 0.007) and (abs(res["icir"] or 0) >= 0.084)
    print(f"{'PASS' if ok else '----'} {name:<26} ic={res['ic']:+.4f} icir={res['icir']:+.4f} "
          f"hit={res['ic_hit_ratio']:.3f} n={res['n_ic_dates']} cov={res['coverage_asset_days']:.2f} "
          f"turn={res['turnover_10d_rank']:.2f} librho={rho:.3f} | 20-22:{s1} 23-26:{s2}")
print("=" * 118)

summary = {k: {kk: vv for kk, vv in v.items() if kk != "decay_ic_by_horizon"} for k, v in results.items()}
json.dump(summary, open("scripts/_screen_v3_results.json", "w"), indent=1)
print("saved scripts/_screen_v3_results.json")
