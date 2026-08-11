"""miner_1 2026-07-20 screen batch A: re-validate prior passers + new candidates.

Gate (benchmark worldline, h=10, min_valid=8): |IC|>=0.007, |ICIR|>=0.084,
max_abs_library_correlation < 0.5 vs the 7 persisted library factors.

Candidates:
  [prior passers re-check]
  1. vol_price_corr_20       : 20d rolling corr(ret, volume)           (passed 2026-08-10)
  2. down_up_vol_ratio_20    : 20d downside vol / upside vol           (passed 2026-07-30)
  3. crypto_beta_btc_60x20   : -beta(asset,BTC,60)*BTCmom20            (passed 2026-07-30)
  [new]
  4. amihud_illiq_20d        : mean(|ret|/volume,20d) negated
  5. efficiency_ratio_20     : |close-close[-20]| / sum(|ret|,20)
  6. gk_vol_inv_20d          : -sqrt(Garman-Klass var, 20d)
  7. trend_r2_60d            : R^2 of 60d linear fit on log close
  8. rsi_14                  : Wilder RSI 14
  9. usdcny_beta_60d         : 60d beta on USDCNY returns
 10. usdjpy_beta_60d         : 60d beta on USDJPY returns
 11. xau_beta_60d            : 60d beta on XAU returns
 12. copper_beta_60d         : 60d beta on COPPER returns
 13. spx_beta_60d            : 60d beta on SPX returns
 14. mom_60d_skip5           : 60d momentum skip5
 15. range_pos_20d           : (close-min20)/(max20-min20)
No lookahead: factor at t uses data <= t; fwd ret = close[t+h]/close[t]-1.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, ret_panel,
                                 forward_returns, rank_ic_series, summarize_ic,
                                 coverage_metrics, turnover_rank, decay_profile,
                                 library_signals, max_library_corr, TRADABLE)

panels = load_panels()
closes = close_panel(panels)
rets = ret_panel(panels)
H_ADM = 10
HORIZONS = (1, 2, 3, 5, 10, 20)
MIN_VALID = 8
print(f"closes {closes.shape} | {closes.index[0].date()}..{closes.index[-1].date()}\n")

# ---- library: 7 persisted factors ----
lib = library_signals(panels, closes, rets)


def rolling_beta(asset_ret, driver_ret, win=60, min_obs=40):
    out = {}
    for a in asset_ret.columns:
        z = pd.concat([asset_ret[a].rename("a"), driver_ret.rename("m")], axis=1).dropna()
        b = (z["a"].rolling(win).cov(z["m"]) / z["m"].rolling(win).var()).where(
            z["m"].rolling(win).count() >= min_obs)
        out[a] = b
    return pd.DataFrame(out, index=asset_ret.index)


def cond_beta(asset_ret, macro_ret, win_beta=60, mom_win=20):
    b = rolling_beta(asset_ret, macro_ret, win_beta)
    mmom = macro_ret.rolling(mom_win).mean()
    return -b * mmom * mom_win


mkt = rets.mean(axis=1)
dn = mkt.where(mkt < 0).fillna(0.0)
lib["dn_mkt_beta_60d"] = rolling_beta(rets, dn, 60)
lib["eurusd_beta_60d"] = rolling_beta(rets, panels["EURUSD"]["close"].astype(float).pct_change(), 60)
lib["rate_beta_cn10y_60d"] = rolling_beta(rets, panels["CN10Y"]["close"].astype(float).pct_change(), 60)
lib = {k: v.reindex(closes.index) for k, v in lib.items()}

V = pd.concat({a: panels[a]["volume"].astype(float) for a in TRADABLE}, axis=1).sort_index()
H = pd.concat({a: panels[a]["high"].astype(float) for a in TRADABLE}, axis=1).sort_index()
L = pd.concat({a: panels[a]["low"].astype(float) for a in TRADABLE}, axis=1).sort_index()

cands = {}

# 1. vol_price_corr_20
vp = {}
for a in closes.columns:
    z = pd.concat([rets[a].rename("r"), V[a].rename("v")], axis=1).dropna()
    vp[a] = z["r"].rolling(20).corr(z["v"])
cands["vol_price_corr_20"] = pd.DataFrame(vp, index=closes.index)

# 2. down_up_vol_ratio_20
down = rets.clip(upper=0)
up = rets.clip(lower=0)
cands["down_up_vol_ratio_20"] = down.rolling(20).std() / up.rolling(20).std()

# 3. crypto_beta_btc_60x20
cands["crypto_beta_btc_60x20"] = cond_beta(rets, closes["BTC"].pct_change(), 60, 20)

# 4. amihud illiq 20d (negated: high value = liquid)
amihud = (rets.abs() / V).rolling(20).mean()
cands["amihud_illiq_20d"] = -amihud

# 5. efficiency ratio 20
cands["efficiency_ratio_20"] = (closes - closes.shift(20)).abs() / rets.abs().rolling(20).sum()

# 6. Garman-Klass inverse vol 20d
gk = ((np.log(H / L) ** 2) / 2 - (2 * np.log(2) - 1) * (rets ** 2)).rolling(20).mean()
cands["gk_vol_inv_20d"] = -np.sqrt(gk.clip(lower=1e-12))

# 7. trend R^2 60d
def trend_r2(s, win=60):
    out = pd.Series(np.nan, index=s.index)
    x = np.arange(win, dtype=float)
    for i in range(win - 1, len(s)):
        y = np.log(s.iloc[i - win + 1:i + 1].values.astype(float))
        if np.isfinite(y).sum() < win // 2:
            continue
        if np.std(y) < 1e-12:
            continue
        r = np.corrcoef(x, y)[0, 1]
        out.iloc[i] = r * r
    return out

cands["trend_r2_60d"] = pd.DataFrame({a: trend_r2(closes[a]) for a in closes.columns})

# 8. RSI 14 (Wilder)
def rsi_wilder(s, win=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / win, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / win, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

cands["rsi_14"] = pd.DataFrame({a: rsi_wilder(closes[a]) for a in closes.columns})

# 9-13. macro/asset betas
cands["usdcny_beta_60d"] = rolling_beta(rets, panels["USDCNY"]["close"].astype(float).pct_change(), 60)
cands["usdjpy_beta_60d"] = rolling_beta(rets, panels["USDJPY"]["close"].astype(float).pct_change(), 60)
cands["xau_beta_60d"] = rolling_beta(rets, closes["XAU"].pct_change(), 60)
cands["copper_beta_60d"] = rolling_beta(rets, closes["COPPER"].pct_change(), 60)
cands["spx_beta_60d"] = rolling_beta(rets, closes["SPX"].pct_change(), 60)

# 14. mom_60d_skip5
cands["mom_60d_skip5"] = closes.shift(5) / closes.shift(65) - 1.0

# 15. range_pos_20d
cands["range_pos_20d"] = (closes - L.rolling(20).min()) / (H.rolling(20).max() - L.rolling(20).min())

# ---- evaluate ----
fwd = forward_returns(closes, H_ADM)
rows = []
for name, panel in cands.items():
    panel = panel.reindex(closes.index)
    ics = rank_ic_series(panel, fwd, MIN_VALID)
    if len(ics) < 200:
        print(f"{name:24s} SKIP (n_ic={len(ics)})")
        continue
    exp_sign = 1
    m = summarize_ic(ics, exp_sign)
    m.update(coverage_metrics(panel, min_valid=MIN_VALID))
    m["turnover_10d_rank"] = turnover_rank(panel, 10)
    m["decay_ic_by_horizon"] = decay_profile(panel, closes, HORIZONS, MIN_VALID, exp_sign)
    corr, key = max_library_corr(panel, lib)
    m["max_abs_library_correlation"] = corr
    m["max_corr_factor"] = key
    gate_ic = abs(m["ic"]) >= 0.007
    gate_icir = abs(m["icir"]) >= 0.084
    gate_corr = corr < 0.5
    ok = gate_ic and gate_icir and gate_corr
    rows.append((name, m, ok))
    print(f"{name:24s} IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} "
          f"n={m['n_ic_dates']:5d} covAD={m['coverage_asset_days']:.3f} covD8={m['coverage_dates_ge8']:.3f} "
          f"to={m['turnover_10d_rank']:.3f} rho={corr:.3f}({key}) "
          f"decay={ {k: round(v,4) for k,v in m['decay_ic_by_horizon'].items()} } "
          f"-> {'PASS' if ok else ''}")

print("\n===== SUMMARY =====")
for name, m, ok in rows:
    print(f"{'PASS' if ok else 'FAIL':4s} {name:24s} IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} "
          f"rho={m['max_abs_library_correlation']:.3f} ({m['max_corr_factor']})")
