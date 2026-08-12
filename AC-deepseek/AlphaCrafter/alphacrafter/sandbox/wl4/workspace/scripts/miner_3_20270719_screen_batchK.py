"""miner_3 batch K screen (2027-07-19) - vectorized rank-IC through last completed day.

A) drift re-validation of 4 active library factors (full + recent250/500)
B) new batch K candidates (fresh ideas, low overlap with batches H-J):
   - cross-asset beta family (worked before): us10y_beta_60d, ndx_beta_60d, spx_beta_60d,
     sx5e_beta_60d, n225_beta_60d, hsi_beta_60d, comm_beta_60d (XAU+COPPER+WTI),
     equity_beta_60d (equity composite), crypto_beta_60d (BTC+ETH)
   - vol regime: vol_ratio_5x20, vol_z_60d, vol_change_5x20
   - corr dynamics: corr_mkt_60d, corr_change_20x60, beta_change_20x60
   - volume-price: obv_slope_20d, vpt_20d, body_ratio_20d, upper_shadow_20d, lower_shadow_20d
   - variants of prior families: win_rate_20d, max_gain_20d, max_loss_20d, skew_20d,
     serial_corr_20d, overnight_share_60d, gap_avg_10d, high_breakout_60d, low_prox_60d

Gate: |IC| >= 0.0070 and |ICIR| >= 0.0840 at h=10 (15-asset cross-asset universe, min_valid=8).
Robustness: full-period + recent250/500; report frozen (HSI/ETH flat since 2026-10-14).
"""
import sys, time, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, forward_returns,
                                 rank_ic_series, summarize_ic, coverage_metrics,
                                 turnover_rank, max_library_corr, TRADABLE)

t0 = time.time()
panels = load_panels(days=3200)
closes = close_panel(panels)
rets = closes.pct_change()
vol_panel = pd.DataFrame({a: panels[a]["volume"].astype(float) for a in closes.columns}).reindex(closes.index)
opens = pd.DataFrame({a: panels[a]["open"].astype(float) for a in closes.columns}).reindex(closes.index)
highs = pd.DataFrame({a: panels[a]["high"].astype(float) for a in closes.columns}).reindex(closes.index)
lows = pd.DataFrame({a: panels[a]["low"].astype(float) for a in closes.columns}).reindex(closes.index)
print(f"panels loaded {time.time()-t0:.1f}s | closes {closes.shape} | {closes.index.min().date()}..{closes.index.max().date()}", flush=True)

LAST = closes.index.max()
print("last completed trading day:", LAST.date(), flush=True)

for a in ["HSI", "ETH"]:
    s = closes[a].dropna()
    last20 = s.tail(20)
    print(f"{a}: n_flat_last20={int((last20.diff()==0).sum())} last_close={last20.iloc[-1]:.4f}", flush=True)

H_ADM = 10
MIN_VALID = 8
GATE_IC, GATE_ICIR = 0.0070, 0.0840
mkt = rets.mean(axis=1)

def rolling_beta(asset_ret, driver_ret, win=60, min_obs=40):
    beta = {}
    for a in asset_ret.columns:
        z = pd.concat([asset_ret[a].rename("a"), driver_ret.rename("m")], axis=1).dropna()
        cov = z["a"].rolling(win).cov(z["m"])
        var = z["m"].rolling(win).var()
        beta[a] = (cov / var).where(z["m"].rolling(win).count() >= min_obs)
    return pd.DataFrame(beta, index=asset_ret.index)

def rolling_corr(asset_ret, driver_ret, win=60, min_obs=40):
    out = {}
    for a in asset_ret.columns:
        z = pd.concat([asset_ret[a].rename("a"), driver_ret.rename("m")], axis=1).dropna()
        c = z["a"].rolling(win).corr(z["m"])
        out[a] = c.where(z["a"].rolling(win).count() >= min_obs)
    return pd.DataFrame(out, index=asset_ret.index)

# ---------- library factor signal artifacts (recompute from definitions) ----------
def lib_vol_price_corr_20():
    return pd.DataFrame({a: rets[a].rolling(20).corr(vol_panel[a]) for a in closes.columns}, index=rets.index)

def lib_dn_mkt_beta_60d():
    dn = mkt.where(mkt < 0)
    return rolling_beta(rets, dn, 60, 40)

def lib_eurusd_beta_60d():
    eur = panels["EURUSD"]["close"].pct_change()
    return rolling_beta(rets, eur, 60, 40)

def lib_rate_beta_cn10y_60d():
    cn = rets["CN10Y"]
    return rolling_beta(rets, cn, 60, 40)

LIBRARY = {
    "vol_price_corr_20": lib_vol_price_corr_20(),
    "dn_mkt_beta_60d": lib_dn_mkt_beta_60d(),
    "eurusd_beta_60d": lib_eurusd_beta_60d(),
    "rate_beta_cn10y_60d": lib_rate_beta_cn10y_60d(),
}

def evaluate(tag, panel, expected_sign=1):
    fwd = forward_returns(closes, H_ADM)
    ics = rank_ic_series(panel, fwd, MIN_VALID)
    m = summarize_ic(ics, expected_sign)
    m.update(coverage_metrics(panel, min_valid=MIN_VALID))
    m["turnover_10d_rank"] = turnover_rank(panel, 10)
    m["decay_ic_by_horizon"] = {}
    for h in (1, 2, 3, 5, 10, 20):
        fh = forward_returns(closes, h)
        ih = rank_ic_series(panel, fh, MIN_VALID)
        if len(ih):
            m["decay_ic_by_horizon"][str(h)] = round(float(ih.mean()), 4)
    corr, key = max_library_corr(panel, LIBRARY)
    m["max_abs_library_correlation"] = corr
    m["max_corr_factor"] = key
    return m, ics

results = {}
# ---------- A) active library drift ----------
print("\n=== A) ACTIVE LIBRARY DRIFT (h=10) ===", flush=True)
for name, panel in LIBRARY.items():
    fwd = forward_returns(closes, H_ADM)
    ics = rank_ic_series(panel, fwd, MIN_VALID)
    m = summarize_ic(ics, expected_sign=1)
    results[f"active_{name}"] = m
    for cut_name, cut in (("recent250", closes.index[-250]), ("recent500", closes.index[-500])):
        sub = ics[ics.index >= cut]
        if len(sub):
            results[f"active_{name}"][f"ic_{cut_name}"] = round(float(sub.mean()), 4)
            results[f"active_{name}"][f"icir_{cut_name}"] = round(float(sub.mean() / sub.std(ddof=1)), 4) if sub.std(ddof=1) > 0 else 0.0
    print(f"{name}: full_ic={m['ic']:.4f} icir={m['icir']:.4f} hit={m['ic_hit_ratio']:.3f} "
          f"n={m['n_ic_dates']} | recent250_ic={results[f'active_{name}'].get('ic_recent250'):.4f} "
          f"recent500_ic={results[f'active_{name}'].get('ic_recent500'):.4f}", flush=True)

# ---------- B) batch K candidates ----------
cands = {}
# cross-asset beta family
cands["K_us10y_beta_60d"] = rolling_beta(rets, rets["US10Y"], 60, 40)
cands["K_ndx_beta_60d"] = rolling_beta(rets, rets["NDX"], 60, 40)
cands["K_spx_beta_60d"] = rolling_beta(rets, rets["SPX"], 60, 40)
cands["K_sx5e_beta_60d"] = rolling_beta(rets, rets["SX5E"], 60, 40)
cands["K_n225_beta_60d"] = rolling_beta(rets, rets["N225"], 60, 40)
cands["K_hsi_beta_60d"] = rolling_beta(rets, rets["HSI"], 60, 40)
comm = rets[["XAU", "COPPER", "WTI"]].mean(axis=1)
cands["K_comm_beta_60d"] = rolling_beta(rets, comm, 60, 40)
equity_set = ["SPX", "NDX", "SX5E", "N225", "000300.SH", "000688.SH", "HSI", "SOX"]
cands["K_equity_beta_60d"] = rolling_beta(rets, rets[equity_set].mean(axis=1), 60, 40)
crypto = rets[["BTC", "ETH"]].mean(axis=1)
cands["K_crypto_beta_60d"] = rolling_beta(rets, crypto, 60, 40)
# vol regime
vol5 = rets.rolling(5).std()
vol20 = rets.rolling(20).std()
cands["K_vol_ratio_5x20"] = vol5 / vol20.replace(0, np.nan)
cands["K_vol_change_5x20"] = vol5 / vol20.replace(0, np.nan) - 1.0
cands["K_vol_z_60d"] = (vol20 - vol20.rolling(60).mean()) / vol20.rolling(60).std().replace(0, np.nan)
# corr dynamics
cands["K_corr_mkt_60d"] = rolling_corr(rets, mkt, 60, 40)
corr20 = rolling_corr(rets, mkt, 20, 15)
cands["K_corr_change_20x60"] = corr20 - rolling_corr(rets, mkt, 60, 40)
b20 = rolling_beta(rets, mkt, 20, 15)
b60 = rolling_beta(rets, mkt, 60, 40)
cands["K_beta_change_20x60"] = b20 - b60
# volume-price
close_shift = closes.shift(1)
obv = ((np.sign(closes.diff()) * vol_panel).fillna(0)).cumsum()
cands["K_obv_slope_20d"] = obv.diff(20) / vol_panel.rolling(20).mean().replace(0, np.nan)
vpt = (rets * vol_panel).fillna(0).cumsum()
cands["K_vpt_20d"] = vpt.diff(20) / vol_panel.rolling(20).mean().replace(0, np.nan)
hl = (highs - lows).replace(0, np.nan)
cands["K_body_ratio_20d"] = ((closes - opens).abs() / hl).rolling(20).mean()
cands["K_upper_shadow_20d"] = ((highs - pd.concat([closes, opens], axis=1).max(axis=1)) / hl).rolling(20).mean()
cands["K_lower_shadow_20d"] = ((pd.concat([closes, opens], axis=1).min(axis=1) - lows) / hl).rolling(20).mean()
# variants of prior families
cands["K_win_rate_20d"] = (rets > 0).rolling(20).mean()
cands["K_max_gain_20d"] = rets.rolling(20).max()
cands["K_max_loss_20d"] = rets.rolling(20).min()
cands["K_skew_20d"] = rets.rolling(20).skew()
cands["K_serial_corr_20d"] = rets.rolling(20).apply(lambda x: pd.Series(x).autocorr() if len(x) > 3 and x.std() > 1e-14 else np.nan, raw=False)
cands["K_overnight_share_60d"] = ((opens - close_shift).abs() / (closes - close_shift).abs().replace(0, np.nan)).rolling(60).mean()
cands["K_gap_avg_10d"] = ((opens / close_shift - 1.0).abs()).rolling(10).mean()
cands["K_high_breakout_60d"] = closes / closes.rolling(60).max() - 1.0
cands["K_low_prox_60d"] = closes / closes.rolling(60).min() - 1.0

print(f"\n=== B) BATCH K SCREEN ({len(cands)} candidates, h=10) ===", flush=True)
for tag, panel in cands.items():
    m, _ = evaluate(tag, panel, expected_sign=1)
    results[tag] = m
    cut_recent = closes.index[-250]
    cut_mid = closes.index[-500]
    fwd = forward_returns(closes, H_ADM)
    ics = rank_ic_series(panel, fwd, MIN_VALID)
    for cut_name, cut in (("recent250", cut_recent), ("recent500", cut_mid)):
        sub = ics[ics.index >= cut]
        if len(sub):
            results[tag][f"ic_{cut_name}"] = round(float(sub.mean()), 4)

df = pd.DataFrame(results).T
df["pass"] = (df["ic"].abs() >= GATE_IC) & (df["icir"].abs() >= GATE_ICIR)
cols = ["ic", "icir", "ic_hit_ratio", "n_ic_dates", "ic_recent250", "ic_recent500",
        "coverage_asset_days", "coverage_dates_ge8", "turnover_10d_rank",
        "max_abs_library_correlation", "max_corr_factor", "pass"]
print("\n=== FULL SCREEN (h=10, min_valid=8) ===")
print(df[cols].to_string(float_format=lambda x: f"{x:.4f}"))
print(f"\nPASSERS ({int(df['pass'].sum())}):", list(df.index[df["pass"]]), flush=True)
print(f"\nelapsed {time.time()-t0:.1f}s", flush=True)
