"""miner_1 batch K screen (2027-06-07) - vectorized rank-IC through last completed day. V2 vectorized.

A) drift re-validation of 4 active library factors (full + recent250/500)
B) new batch K candidates (fresh ideas, low overlap with library + batches D-J):
   - trend quality:    eff_ratio_60d (Kaufman), accel_20d (momentum-of-momentum),
                       hi_lo_pos_20d (stochastic position), vol_adj_mom_60d
   - crash/risk:       skew_20d, dn_up_ratio_60d, semi_vol_60d, profit_factor_60d,
                       dd_from_60d_max
   - microstructure:   autocorr_5_60d, vol_price_lead_20d (lagged vol-price corr),
                       range_ratio_20d, vol_disp_20x60
   - cross-sectional:  rel_strength_20d, mom_quality_blend (z-mom * eff_ratio)

Gate: |IC| >= 0.0070 and |ICIR| >= 0.0840 at h=10 (15-asset cross-asset universe, min_valid=8).
Robustness: full-period + recent250/500; report frozen (HSI/ETH flat since 2026-10-14).
"""
import sys, time
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, forward_returns,
                                 rank_ic_series, summarize_ic, coverage_metrics,
                                 turnover_rank, max_library_corr, TRADABLE)

t0 = time.time()
panels = load_panels(days=3000)
closes = close_panel(panels)
rets = closes.pct_change()
vol_panel = pd.DataFrame({a: panels[a]["volume"].astype(float) for a in closes.columns}).reindex(closes.index)
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

def autocorr_lag_vec(x, lag=5, win=60):
    y = x.shift(lag)
    mx = x.rolling(win).mean()
    my = y.rolling(win).mean()
    cov = (x * y).rolling(win).mean() - mx * my
    vx = x.rolling(win).var()
    vy = y.rolling(win).var()
    return cov / np.sqrt(vx * vy).replace(0, np.nan)

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
for name, sig in LIBRARY.items():
    exp = -1 if name in ("eurusd_beta_60d", "rate_beta_cn10y_60d") else 1
    m, ics = evaluate(f"active_{name}", sig, expected_sign=exp)
    m["expected_sign"] = exp
    results[f"active_{name}"] = m
    cut_recent = closes.index[-250]
    cut_mid = closes.index[-500]
    for cut_name, cut in (("recent250", cut_recent), ("recent500", cut_mid)):
        sub = ics[ics.index >= cut]
        if len(sub):
            results[f"active_{name}"][f"ic_{cut_name}"] = round(float(sub.mean()), 4)
            results[f"active_{name}"][f"icir_{cut_name}"] = round(float(sub.mean() / sub.std(ddof=1)), 4) if sub.std(ddof=1) > 0 else 0.0

# ---------- B) batch K candidates ----------
cands = {}
# trend quality
cands["K_eff_ratio_60d"] = (closes - closes.shift(59)).abs() / rets.abs().rolling(60).sum().replace(0, np.nan)
m20 = closes.shift(5) / closes.shift(25) - 1.0
cands["K_accel_20d"] = m20 - m20.shift(20)
hh20 = highs.rolling(20).max()
ll20 = lows.rolling(20).min()
cands["K_hi_lo_pos_20d"] = (closes - ll20) / (hh20 - ll20).replace(0, np.nan)
vol20 = rets.rolling(20).std()
mom60 = closes.shift(5) / closes.shift(65) - 1.0
cands["K_vol_adj_mom_60d"] = mom60 / vol20.replace(0, np.nan)
# crash / risk
cands["K_skew_20d"] = rets.rolling(20).skew()
up = rets.clip(lower=0)
dn = (-rets.clip(upper=0))
cands["K_dn_up_ratio_60d"] = dn.rolling(60).mean() / up.rolling(60).mean().replace(0, np.nan)
cands["K_semi_vol_60d"] = dn.pow(2).rolling(60).mean().pow(0.5)
cands["K_profit_factor_60d"] = up.rolling(60).sum() / dn.rolling(60).sum().replace(0, np.nan)
cands["K_dd_from_60d_max"] = closes / closes.rolling(60).max() - 1.0
# microstructure
cands["K_autocorr_5_60d"] = pd.DataFrame({a: autocorr_lag_vec(rets[a], 5, 60) for a in closes.columns}, index=rets.index)
cands["K_vol_price_lead_20d"] = pd.DataFrame({a: rets[a].rolling(20).corr(vol_panel[a].shift(1)) for a in closes.columns}, index=rets.index)
cands["K_range_ratio_20d"] = ((highs - lows) / closes.replace(0, np.nan)).rolling(20).mean()
cands["K_vol_disp_20x60"] = vol20 / rets.rolling(60).std().replace(0, np.nan)
# cross-sectional
rets20 = closes / closes.shift(20) - 1.0
cands["K_rel_strength_20d"] = rets20 - rets20.mean(axis=1)
z_mom60 = (mom60 - mom60.rolling(250).mean()) / mom60.rolling(250).std().replace(0, np.nan)
cands["K_mom_quality_blend"] = z_mom60 * cands["K_eff_ratio_60d"]

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
