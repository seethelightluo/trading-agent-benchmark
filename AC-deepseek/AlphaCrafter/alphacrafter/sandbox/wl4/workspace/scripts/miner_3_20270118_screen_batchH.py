"""miner_3 batch H screen (2027-01-18) - vectorized rank-IC on data through last completed day (2027-01-15).

A) drift re-validation of 4 active library factors (full + recent250/500)
B) new batch H candidates (fresh ideas, low overlap with batch G/E/F):
   - risk-adjusted / smoothed trend: mom20_skip5_voladj, mom60_skip20, ma_dist_20, win_rate_60, stoch_pos_20
   - reversal: gap_rev_10d, low_prox_20, high_breakout_20
   - vol regime: vol_ratio_10x60, skew_60d
   - liquidity: amihud_20d, volume_mom_20x60
   - cross-asset/macro betas: dxy_beta_60d, usdjpy_beta_60d, vix_beta_60d, vix_up_beta_60d,
     wti_beta_60d, xau_beta_60d, btc_beta_60d, downup_beta_60d, corr_dxy_60d, corr_vix_60d
   - re-check batch G passers that were NOT persisted: mom_20d_skip5 (IC 0.0595/ICIR 0.1661), mom_20d

Gate: |IC| >= 0.0070 and |ICIR| >= 0.0840 at h=10 (15-asset cross-asset universe, min_valid=8).
Robustness: full-period + recent250/500; report frozen (HSI/ETH flat since 2026-10-14).
"""
import sys, time, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, forward_returns,
                                 rank_ic_series, summarize_ic, coverage_metrics,
                                 turnover_rank, TRADABLE)

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
    m = mkt
    dn = m.where(m < 0)
    beta = {}
    for a in rets.columns:
        z = pd.concat([rets[a].rename("a"), dn.rename("m")], axis=1).dropna()
        cov = z["a"].rolling(60).cov(z["m"])
        var = z["m"].rolling(60).var()
        beta[a] = (cov / var).where(z["m"].rolling(60).count() >= 40)
    return pd.DataFrame(beta, index=rets.index)

def lib_eurusd_beta_60d():
    eur = panels["EURUSD"]["close"].pct_change() if "EURUSD" in panels else None
    return rolling_beta(rets, eur, 60, 40) if eur is not None else None

def lib_rate_beta_cn10y_60d():
    cn = rets["CN10Y"]
    return rolling_beta(rets, cn, 60, 40)

LIBRARY = {}
LIBRARY["vol_price_corr_20"] = lib_vol_price_corr_20()
LIBRARY["dn_mkt_beta_60d"] = lib_dn_mkt_beta_60d()
LIBRARY["eurusd_beta_60d"] = lib_eurusd_beta_60d()
LIBRARY["rate_beta_cn10y_60d"] = lib_rate_beta_cn10y_60d()

def max_lib_corr(cand):
    best, best_key = 0.0, None
    c = cand.stack()
    for name, s in LIBRARY.items():
        if s is None:
            continue
        both = pd.concat([c.rename("cand"), s.stack().rename("lib")], axis=1).dropna()
        if len(both) < 30:
            continue
        r = float(both["cand"].corr(both["lib"]))
        if abs(r) > best:
            best, best_key = abs(r), name
    return round(best, 4), best_key

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
    corr, key = max_lib_corr(panel)
    out["max_lib_corr"] = corr
    out["max_corr_factor"] = key
    return out

results = {}

# ---------- A) active library drift re-validation ----------
for name, sig in LIBRARY.items():
    if sig is None:
        continue
    exp = -1 if name in ("eurusd_beta_60d", "rate_beta_cn10y_60d") else 1
    results["active_" + name] = evaluate("active_" + name, sig, exp)

# ---------- B) batch H candidates ----------
cands = {}

# risk-adjusted / smoothed trend
cands["H_mom20_skip5_voladj"] = (closes.shift(5) / closes.shift(25) - 1.0) / rets.rolling(20).std().replace(0, np.nan)
cands["H_mom60_skip20"] = closes.shift(20) / closes.shift(80) - 1.0
cands["H_ma_dist_20"] = closes / closes.rolling(20).mean() - 1.0
cands["H_ma_dist_60"] = closes / closes.rolling(60).mean() - 1.0
cands["H_win_rate_60"] = (rets > 0).rolling(60).mean()
cands["H_stoch_pos_20"] = (closes - lows.rolling(20).min()) / (highs.rolling(20).max() - lows.rolling(20).min()).replace(0, np.nan)

# reversal / breakout
cands["H_gap_rev_10d"] = -(closes / closes.shift(10) - 1.0)
cands["H_low_prox_20"] = closes / lows.rolling(20).min() - 1.0
cands["H_high_breakout_20"] = closes / highs.rolling(20).max() - 1.0

# vol regime
cands["H_vol_ratio_10x60"] = rets.rolling(10).std() / rets.rolling(60).std() - 1.0
cands["H_skew_60d"] = rets.rolling(60).skew()

# liquidity
cands["H_amihud_20d"] = -1.0 * (rets.abs() / vol_panel.replace(0, np.nan)).rolling(20).mean()
cands["H_volume_mom_20x60"] = vol_panel.rolling(20).mean() / vol_panel.rolling(60).mean() - 1.0

# cross-asset / macro betas
dxy_ret = panels["DXY"]["close"].pct_change()
usdjpy_ret = panels["USDJPY"]["close"].pct_change()
vix_ret = panels["VIX"]["close"].pct_change()
vix_up = vix_ret.where(vix_ret > 0)
wti_ret = rets["WTI"]
xau_ret = rets["XAU"]
btc_ret = rets["BTC"]

cands["H_dxy_beta_60d"] = rolling_beta(rets, dxy_ret, 60, 40)
cands["H_usdjpy_beta_60d"] = rolling_beta(rets, usdjpy_ret, 60, 40)
cands["H_vix_beta_60d"] = rolling_beta(rets, vix_ret, 60, 40)
cands["H_vix_up_beta_60d"] = rolling_beta(rets, vix_up, 60, 40)
cands["H_wti_beta_60d"] = rolling_beta(rets, wti_ret, 60, 40)
cands["H_xau_beta_60d"] = rolling_beta(rets, xau_ret, 60, 40)
cands["H_btc_beta_60d"] = rolling_beta(rets, btc_ret, 60, 40)

# downside/upside beta ratio (downside risk premium)
def downup_beta():
    out = {}
    up = mkt.where(mkt > 0)
    dn = mkt.where(mkt < 0)
    for a in rets.columns:
        za = pd.concat([rets[a].rename("a"), mkt.rename("m")], axis=1).dropna()
        zu = pd.concat([rets[a].rename("a"), up.rename("m")], axis=1).dropna()
        zd = pd.concat([rets[a].rename("a"), dn.rename("m")], axis=1).dropna()
        bu = (zu["a"].rolling(60).cov(zu["m"]) / zu["m"].rolling(60).var()).where(zu["m"].rolling(60).count() >= 40)
        bd = (zd["a"].rolling(60).cov(zd["m"]) / zd["m"].rolling(60).var()).where(zd["m"].rolling(60).count() >= 40)
        out[a] = bd / bu.replace(0, np.nan)
    return pd.DataFrame(out, index=rets.index)
cands["H_downup_beta_60d"] = downup_beta()

# correlation to macro
cands["H_corr_dxy_60d"] = pd.DataFrame({a: rets[a].rolling(60).corr(dxy_ret) for a in closes.columns}, index=rets.index)
cands["H_corr_vix_60d"] = pd.DataFrame({a: rets[a].rolling(60).corr(vix_ret) for a in closes.columns}, index=rets.index)

# re-check batch G passers not persisted
cands["H_mom_20d_skip5"] = closes.shift(5) / closes.shift(25) - 1.0
cands["H_mom_20d"] = closes / closes.shift(20) - 1.0

print(f"evaluating {len(cands)} candidates + 4 active...", flush=True)
for name, sig in cands.items():
    results["H_" + name if not name.startswith("H_") else name] = evaluate(name, sig, 1)

with open("scripts/_miner3_batchH_results.json", "w") as f:
    json.dump(results, f, indent=1, default=str)

# ---------- summary table ----------
rows = []
for k, v in results.items():
    rows.append({"tag": k, "exp": v["expected_sign"], "ic": v["ic_full"], "icir": v["icir_full"],
                 "hit": v["hit_full"], "n": v["n_full"], "ic250": v["ic_recent250"], "icir250": v["ic_recent250"],
                 "ic500": v["ic_recent500"], "cov": v["cov_dates_ge8"], "turn": v["turnover_10d"],
                 "libcorr": v["max_lib_corr"], "corrwith": v["max_corr_factor"]})
df = pd.DataFrame(rows)
df["pass"] = ((df["ic"].abs() >= GATE_IC) & (df["icir"].abs() >= GATE_ICIR))
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
print(df.to_string(index=False))
print(f"\nPASSERS ({((df['pass']).sum())}):")
print(df[df["pass"]][["tag", "exp", "ic", "icir", "n", "ic250", "icir250", "cov", "turn", "libcorr", "corrwith"]].to_string(index=False))
print(f"\nrun time {time.time()-t0:.1f}s")
