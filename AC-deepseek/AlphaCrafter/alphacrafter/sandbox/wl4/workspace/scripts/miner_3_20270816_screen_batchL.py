"""miner_3 batch L screen (2027-08-16) - vectorized rank-IC through last completed day (2027-08-13).

A) drift re-validation of 4 active library factors (full + recent250/500/750)
B) new batch L candidates (fresh families, low overlap with batches H-K):
   - trend efficiency/consistency: eff_ratio_20d/60d (Kaufman), trend_r2_20d/60d,
     trend_slope_t_20d (linreg t-stat), stoch_k_20d/60d (range position)
   - drawdown/underwater: dd_depth_120d, underwater_days_120d
   - vol asymmetry / range vol: dnside_vol_ratio_20d/60d, parkinson_vol_20d
   - liquidity/volume: amihud_illiq_20d, volume_z_60d (abnormal volume)
   - macro driver: dxy_beta_60d, dxy_corr_60d
   - reversal/intraday: rev_5d, rev_10d, intraday_mom_20d
   - higher moments: skew_60d, kurt_20d

Gate: |IC| >= 0.0070 and |ICIR| >= 0.0840 at h=10 (15-asset cross-asset universe, min_valid=8).
Robustness: full-period + recent windows; report frozen (HSI/ETH flat since 2026-10-14).
"""
import sys, time, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, forward_returns,
                                 rank_ic_series, summarize_ic, coverage_metrics,
                                 turnover_rank, max_library_corr, TRADABLE)

t0 = time.time()
panels = load_panels(days=3300)
closes = close_panel(panels)
rets = closes.pct_change()
vol_panel = pd.DataFrame({a: panels[a]["volume"].astype(float) for a in closes.columns}).reindex(closes.index)
opens = pd.DataFrame({a: panels[a]["open"].astype(float) for a in closes.columns}).reindex(closes.index)
highs = pd.DataFrame({a: panels[a]["high"].astype(float) for a in closes.columns}).reindex(closes.index)
lows = pd.DataFrame({a: panels[a]["low"].astype(float) for a in closes.columns}).reindex(closes.index)
print(f"panels loaded {time.time()-t0:.1f}s | closes {closes.shape} | {closes.index.min().date()}..{closes.index.max().date()}", flush=True)
LAST = closes.index.max()
print("last completed trading day:", LAST.date(), flush=True)

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

# ---------- library factor signal artifacts ----------
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

# ---------- batch L candidates ----------
cands = {}
close_shift = closes.shift(1)

# trend efficiency / consistency
def kaufman_eff(win):
    tr = (closes - closes.shift(win)).abs()
    path = rets.abs().rolling(win).sum()
    return (tr / path.replace(0, np.nan))
cands["L_eff_ratio_20d"] = kaufman_eff(20)
cands["L_eff_ratio_60d"] = kaufman_eff(60)

def trend_r2(win):
    """R^2 of linear trend fit of log price over window (trend consistency)."""
    x = np.arange(win)
    xm = x - x.mean()
    out = {}
    for a in closes.columns:
        lp = np.log(closes[a])
        def _r2(s):
            if len(s) < win or np.isnan(s).any() or s.std() < 1e-14:
                return np.nan
            y = s.values
            ym = y - y.mean()
            b = (ym @ xm) / (xm @ xm)
            yhat = b * xm
            ss_res = ((y - yhat) ** 2).sum()
            ss_tot = (ym ** 2).sum()
            return 1.0 - ss_res / ss_tot if ss_tot > 1e-14 else np.nan
        out[a] = lp.rolling(win).apply(_r2, raw=True)
    return pd.DataFrame(out, index=closes.index)

cands["L_trend_r2_20d"] = trend_r2(20)
cands["L_trend_r2_60d"] = trend_r2(60)

def trend_slope_t(win):
    """t-stat of linear trend slope of log price over window."""
    x = np.arange(win)
    xm = x - x.mean()
    out = {}
    for a in closes.columns:
        lp = np.log(closes[a])
        def _t(s):
            if len(s) < win or np.isnan(s).any() or s.std() < 1e-14:
                return np.nan
            y = s.values
            ym = y - y.mean()
            ssx = (xm @ xm)
            b = (ym @ xm) / ssx
            yhat = b * xm
            resid = y - yhat
            s2 = (resid @ resid) / (win - 2)
            if s2 <= 1e-20:
                return np.nan
            se = np.sqrt(s2 / ssx)
            return b / se if se > 1e-14 else np.nan
        out[a] = lp.rolling(win).apply(_t, raw=True)
    return pd.DataFrame(out, index=closes.index)

cands["L_trend_slope_t_20d"] = trend_slope_t(20)

# range position (stochastic %K style)
def stoch_k(win):
    rng = (closes - lows.rolling(win).min()) / (highs.rolling(win).max() - lows.rolling(win).min()).replace(0, np.nan)
    return rng - 0.5
cands["L_stoch_k_20d"] = stoch_k(20)
cands["L_stoch_k_60d"] = stoch_k(60)

# drawdown / underwater
cands["L_dd_depth_120d"] = (closes.rolling(120).max() - closes) / closes.rolling(120).max()
def underwater_days(win):
    out = {}
    for a in closes.columns:
        c = closes[a]
        rmax = c.rolling(win, min_periods=1).max()
        # days since last new high within window: count of trailing days where close < running max
        below = (c < rmax).astype(float)
        def _count(s):
            arr = s.values
            n = 0
            for v in arr[::-1]:
                if v == 1:
                    n += 1
                else:
                    break
            return n
        out[a] = below.rolling(win).apply(_count, raw=True)
    return pd.DataFrame(out, index=closes.index)
cands["L_underwater_days_120d"] = underwater_days(120) / 120.0

# vol asymmetry
def dnside_vol_ratio(win):
    neg = rets.where(rets < 0, 0.0)
    pos = rets.where(rets > 0, 0.0)
    dnvol = np.sqrt((neg ** 2).rolling(win).mean())
    upvol = np.sqrt((pos ** 2).rolling(win).mean())
    totvol = rets.rolling(win).std()
    return dnvol / totvol.replace(0, np.nan)
cands["L_dnside_vol_ratio_20d"] = dnside_vol_ratio(20)
cands["L_dnside_vol_ratio_60d"] = dnside_vol_ratio(60)

# range-based (Parkinson) vol
cands["L_parkinson_vol_20d"] = (np.log(highs / lows) ** 2 / (4 * np.log(2))).rolling(20).mean()

# Amihud illiquidity
cands["L_amihud_illiq_20d"] = (rets.abs() / vol_panel.replace(0, np.nan)).rolling(20).mean() * 1e6

# abnormal volume
cands["L_volume_z_60d"] = vol_panel / vol_panel.rolling(60).mean().replace(0, np.nan) - 1.0

# DXY driver
dxy_ret = panels["DXY"]["close"].pct_change()
cands["L_dxy_beta_60d"] = rolling_beta(rets, dxy_ret, 60, 40)
cands["L_dxy_corr_60d"] = rolling_corr(rets, dxy_ret, 60, 40)

# short-term reversal / intraday momentum
cands["L_rev_5d"] = -rets.rolling(5).sum()
cands["L_rev_10d"] = -rets.rolling(10).sum()
cands["L_intraday_mom_20d"] = (closes / opens - 1.0).rolling(20).mean()

# higher moments
cands["L_skew_60d"] = rets.rolling(60).skew()
cands["L_kurt_20d"] = rets.rolling(20).apply(lambda x: pd.Series(x).kurt() if len(x) > 4 and x.std() > 1e-14 else np.nan, raw=False)

print(f"candidates defined {time.time()-t0:.1f}s ({len(cands)})", flush=True)

# ---------- evaluation ----------
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
print("\n=== A) ACTIVE LIBRARY DRIFT (h=10) ===", flush=True)
for name, panel in LIBRARY.items():
    fwd = forward_returns(closes, H_ADM)
    ics = rank_ic_series(panel, fwd, MIN_VALID)
    m = summarize_ic(ics, expected_sign=1)
    results[f"active_{name}"] = m
    for cut_name, cut in (("recent250", closes.index[-250]), ("recent500", closes.index[-500]), ("recent750", closes.index[-750])):
        sub = ics[ics.index >= cut]
        if len(sub):
            results[f"active_{name}"][f"ic_{cut_name}"] = round(float(sub.mean()), 4)
            results[f"active_{name}"][f"icir_{cut_name}"] = round(float(sub.mean() / sub.std(ddof=1)), 4) if sub.std(ddof=1) > 0 else 0.0
    print(f"{name}: full_ic={m['ic']:.4f} icir={m['icir']:.4f} hit={m['ic_hit_ratio']:.3f} n={m['n_ic_dates']} | "
          f"r250_ic={results[f'active_{name}'].get('ic_recent250'):.4f} r500_ic={results[f'active_{name}'].get('ic_recent500'):.4f} "
          f"r750_ic={results[f'active_{name}'].get('ic_recent750'):.4f}", flush=True)

print(f"\n=== B) BATCH L SCREEN ({len(cands)} candidates, h=10) ===", flush=True)
fwd = forward_returns(closes, H_ADM)
for tag, panel in cands.items():
    m, ics = evaluate(tag, panel, expected_sign=1)
    results[tag] = m
    for cut_name, cut in (("recent250", closes.index[-250]), ("recent500", closes.index[-500])):
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
