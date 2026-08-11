"""miner_3 batch I screen (2027-02-15) - vectorized rank-IC on data through last completed day (2027-02-12).

A) drift re-validation of 4 active library factors (full + recent250/500)
B) new batch I candidates (fresh ideas, low overlap with batches E/F/G/H):
   - trend/efficiency: eff_ratio_20d, eff_ratio_60d, cmo_14d, bollinger_pos_20d, mom_120d_skip20
   - vol regime: vol_ratio_20x120, down_up_vol_ratio_60d, kurtosis_20d, serial_corr_10d, atr_norm_20d
   - flow/seasonality: gap_avg_20d, overnight_share_20d
   - cross-asset: btc_beta_20d, corr_btc_60d, corr_xau_60d, corr_wti_60d, dn_mom_20d
   - re-confirm batch G/H passers: mom_20d_skip5, mom_20d

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
    ics_full = rank_ic_series(panel, fwd, MIN_VALID)
    m_full = summarize_ic(ics_full, expected_sign)
    cut_recent = closes.index[-250]
    cut_mid = closes.index[-500]
    out = {"tag": tag, "expected_sign": expected_sign,
           "ic_full": m_full["ic"], "icir_full": m_full["icir"],
           "hit_full": m_full["ic_hit_ratio"], "n_full": m_full["n_ic_dates"],
           "ic_recent250": np.nan, "icir_recent250": np.nan, "n_recent250": 0,
           "ic_recent500": np.nan, "icir_recent500": np.nan, "n_recent500": 0}
    for name, cut in (("recent250", cut_recent), ("recent500", cut_mid)):
        sub = ics_full[ics_full.index >= cut]
        if len(sub) >= 20:
            s = summarize_ic(sub, expected_sign)
            out[f"ic_{name}"] = s["ic"]
            out[f"icir_{name}"] = s["icir"]
            out[f"n_{name}"] = s["n_ic_dates"]
    cov = coverage_metrics(panel, min_valid=MIN_VALID)
    out["cov_dates_ge8"] = cov["coverage_dates_ge8"]
    out["turnover_10d"] = turnover_rank(panel, 10)
    corr, key = max_library_corr(panel, LIBRARY)
    out["max_lib_corr"] = corr
    out["max_corr_factor"] = key
    return out

results = {}

# ---------- A) active library drift re-validation ----------
for name, sig in LIBRARY.items():
    exp = -1 if name in ("eurusd_beta_60d", "rate_beta_cn10y_60d") else 1
    results["active_" + name] = evaluate("active_" + name, sig, exp)

# ---------- B) batch I candidates ----------
cands = {}

# trend / efficiency
cands["I_eff_ratio_20d"] = (closes - closes.shift(20)).abs() / rets.abs().rolling(20).sum()
cands["I_eff_ratio_60d"] = (closes - closes.shift(60)).abs() / rets.abs().rolling(60).sum()
up = rets.where(rets > 0, 0.0)
dn = (-rets).where(rets < 0, 0.0)
su, sd = up.rolling(14).sum(), dn.rolling(14).sum()
cands["I_cmo_14d"] = (su - sd) / (su + sd).replace(0, np.nan)
ma20 = closes.rolling(20).mean()
std20 = rets.rolling(20).std()
cands["I_bollinger_pos_20d"] = (closes - ma20) / (2.0 * closes.rolling(20).std())
cands["I_mom_120d_skip20"] = closes.shift(20) / closes.shift(140) - 1.0

# vol regime
cands["I_vol_ratio_20x120"] = std20 / rets.rolling(120).std()
pos_ret = rets.where(rets > 0)
neg_ret = (-rets).where(rets < 0)
cands["I_down_up_vol_ratio_60d"] = neg_ret.rolling(60).std() / pos_ret.rolling(60).std()
cands["I_kurtosis_20d"] = rets.rolling(20).kurt()
cands["I_serial_corr_10d"] = rets.rolling(10).corr(rets.shift(1))
cands["I_atr_norm_20d"] = (highs - lows).rolling(20).mean() / closes

# flow / intraday
gaps = opens / closes.shift(1) - 1.0
cands["I_gap_avg_20d"] = gaps.rolling(20).mean()
cands["I_overnight_share_20d"] = gaps.rolling(20).sum() / rets.abs().rolling(20).sum()

# cross-asset
btc_ret = rets["BTC"]
cands["I_btc_beta_20d"] = rolling_beta(rets, btc_ret, 20, 15)
cands["I_corr_btc_60d"] = pd.DataFrame({a: rets[a].rolling(60).corr(btc_ret) for a in closes.columns}, index=rets.index)
xau_ret = rets["XAU"]
cands["I_corr_xau_60d"] = pd.DataFrame({a: rets[a].rolling(60).corr(xau_ret) for a in closes.columns}, index=rets.index)
wti_ret = rets["WTI"]
cands["I_corr_wti_60d"] = pd.DataFrame({a: rets[a].rolling(60).corr(wti_ret) for a in closes.columns}, index=rets.index)
mkt_dn = (mkt < 0)
dn_rets = rets.where(mkt_dn, 0.0)
cands["I_dn_mom_20d"] = dn_rets.rolling(20).sum()

# re-confirm prior passers
cands["I_mom_20d_skip5"] = closes.shift(5) / closes.shift(25) - 1.0
cands["I_mom_20d"] = closes / closes.shift(20) - 1.0

print(f"evaluating {len(cands)} candidates + 4 active...", flush=True)
for name, sig in cands.items():
    results["I_" + name if not name.startswith("I_") else name] = evaluate(name, sig, 1)

with open("scripts/_miner3_batchI_results.json", "w") as f:
    json.dump(results, f, indent=1, default=str)

# ---------- summary table ----------
rows = []
for k, v in results.items():
    rows.append({"tag": k, "exp": v["expected_sign"], "ic": v["ic_full"], "icir": v["icir_full"],
                 "hit": v["hit_full"], "n": v["n_full"], "ic250": v["ic_recent250"], "icir250": v["icir_recent250"],
                 "ic500": v["ic_recent500"], "cov": v["cov_dates_ge8"], "turn": v["turnover_10d"],
                 "libcorr": v["max_lib_corr"], "corrwith": v["max_corr_factor"]})
df = pd.DataFrame(rows)
df["pass"] = (df["ic"].abs() >= GATE_IC) & (df["icir"].abs() >= GATE_ICIR)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
print(df.to_string(index=False))
print(f"\nPASSERS ({int(df['pass'].sum())}):")
print(df[df["pass"]][["tag", "exp", "ic", "icir", "n", "ic250", "icir250", "cov", "turn", "libcorr", "corrwith"]].to_string(index=False))
print(f"\nrun time {time.time()-t0:.1f}s")
