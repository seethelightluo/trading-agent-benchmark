"""miner_3 batch J screen (2027-04-26) - vectorized rank-IC through last completed day (2027-04-23).

A) drift re-validation of 4 active library factors (full + recent250/500)
B) new batch J candidates (fresh ideas, low overlap with batches D-I):
   - trend/technical: macd_hist_12_26_9, adx_proxy_14d, williams_r_14d, ma50_200_dist
   - risk/vol: gk_vol_20d, vol_percentile_20x250, maxdd_250d, ulcer_250d, dn_share_20d
   - momentum refinements: mom_90d_skip20, mom_180d_skip20, blend_mom_z
   - cross-asset: eth_beta_60d, semi_beta_60d, chn_beta_60d, corr_mkt_20d

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
panels = load_panels(days=3000)
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
    # sub-period checks
    cut_recent = closes.index[-250]
    cut_mid = closes.index[-500]
    for cut_name, cut in (("recent250", cut_recent), ("recent500", cut_mid)):
        sub = ics[ics.index >= cut]
        if len(sub):
            results[f"active_{name}"][f"ic_{cut_name}"] = round(float(sub.mean()), 4)
            results[f"active_{name}"][f"icir_{cut_name}"] = round(float(sub.mean() / sub.std(ddof=1)), 4) if sub.std(ddof=1) > 0 else 0.0

# ---------- B) batch J candidates ----------
cands = {}
# trend/technical
ema12 = closes.ewm(span=12, adjust=False).mean()
ema26 = closes.ewm(span=26, adjust=False).mean()
macd = ema12 - ema26
cands["J_macd_hist_12_26_9"] = macd - macd.ewm(span=9, adjust=False).mean()
up_sum = rets.clip(lower=0).rolling(14).sum()
dn_sum = (-rets.clip(upper=0)).rolling(14).sum()
cands["J_adx_proxy_14d"] = (up_sum - dn_sum).abs() / rets.abs().rolling(14).sum().replace(0, np.nan)
hh14 = highs.rolling(14).max()
ll14 = lows.rolling(14).min()
cands["J_williams_r_14d"] = (hh14 - closes) / (hh14 - ll14).replace(0, np.nan)
ma50 = closes.rolling(50).mean()
ma200 = closes.rolling(200).mean()
cands["J_ma50_200_dist"] = ma50 / ma200 - 1.0
# risk/vol
log_hl = np.log(highs / lows)
log_co = np.log(closes / opens)
cands["J_gk_vol_20d"] = np.sqrt((0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2).rolling(20).mean())
def vol_percentile(win_vol, win=250):
    out = {}
    for a in win_vol.columns:
        x = win_vol[a].values
        n = len(x)
        r = np.full(n, np.nan)
        for i in range(win - 1, n):
            w = x[i - win + 1:i + 1]
            r[i] = float((w < w[-1]).mean())
        out[a] = pd.Series(r, index=win_vol.index)
    return pd.DataFrame(out)
vol20 = rets.rolling(20).std()
cands["J_vol_percentile_20x250"] = vol_percentile(vol20, 250)
dd = closes / closes.rolling(250).max() - 1.0
cands["J_maxdd_250d"] = dd.rolling(250).min()
cands["J_ulcer_250d"] = (dd**2).rolling(250).mean().pow(0.5)
cands["J_dn_share_20d"] = rets.clip(upper=0).rolling(20).sum().abs() / rets.abs().rolling(20).sum().replace(0, np.nan)
# momentum refinements
cands["J_mom_90d_skip20"] = closes.shift(20) / closes.shift(110) - 1.0
cands["J_mom_180d_skip20"] = closes.shift(20) / closes.shift(200) - 1.0
m20 = closes.shift(5) / closes.shift(25) - 1.0
m60 = closes.shift(10) / closes.shift(70) - 1.0
m120 = closes.shift(20) / closes.shift(140) - 1.0
cands["J_blend_mom_z"] = (m20 - m20.rolling(250).mean()) / m20.rolling(250).std().replace(0, np.nan) \
                       + (m60 - m60.rolling(250).mean()) / m60.rolling(250).std().replace(0, np.nan) \
                       + (m120 - m120.rolling(250).mean()) / m120.rolling(250).std().replace(0, np.nan)
# cross-asset
cands["J_eth_beta_60d"] = rolling_beta(rets, rets["ETH"], 60, 40)
cands["J_semi_beta_60d"] = rolling_beta(rets, rets["SOX"], 60, 40)
cands["J_chn_beta_60d"] = rolling_beta(rets, rets["000300.SH"], 60, 40)
cands["J_corr_mkt_20d"] = pd.DataFrame({a: rets[a].rolling(20).corr(mkt) for a in closes.columns}, index=rets.index)

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
