"""miner_2 batch I screen (2027-02-15) - vectorized rank-IC on data through last completed day.

A) drift re-validation of 4 active library factors (full + recent250/500)
B) new batch I candidates (fresh families not covered by miner_3 batches A-H):
   - trend quality:  efr_20, efr_60 (Kaufman efficiency ratio), ema_10_30 spread
   - persistence:    autocorr_10, autocorr_20 (lag-1 return autocorrelation)
   - range vol:      park_vol_20 (log-range std), range_vol_20 ((H-L)/C std)
   - tail/vol shape: semi_vol_60 (downside vol share)
   - liquidity:      vol_z_20 (abnormal volume z-score), vol_up_down_20
   - price location: rel_str_20 (rel strength vs cross-sectional mean), dd_60 (drawdown depth), hl_pos_20
   - information:    gap_20 (mean overnight gap magnitude)
   - cross-asset:    mkt_beta_60d (full), up_mkt_beta_60d, ndx_beta_60d, btc_corr_60

Gate: |IC| >= 0.0070 and |ICIR| >= 0.0840 at h=10 (15-asset cross-asset universe, min_valid=8).
Robustness: full-period + recent250/500; report frozen (HSI/ETH flat since 2026-10-14).
"""
import sys, time, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, forward_returns,
                                 rank_ic_series, summarize_ic, coverage_metrics,
                                 turnover_rank, decay_profile, TRADABLE)

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
        out[a] = z["a"].rolling(win).corr(z["m"]).where(z["m"].rolling(win).count() >= min_obs)
    return pd.DataFrame(out, index=asset_ret.index)


# ---------- library factor signal artifacts (recompute from definitions) ----------
def lib_vol_price_corr_20():
    return pd.DataFrame({a: rets[a].rolling(20).corr(vol_panel[a]) for a in closes.columns}, index=rets.index)

def lib_dn_mkt_beta_60d():
    dn = mkt.where(mkt < 0)
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
    return rolling_beta(rets, rets["CN10Y"], 60, 40)

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

# ---------- B) batch I candidates ----------
cands = {}

# trend quality
cands["I_efr_20"] = (closes - closes.shift(20)).abs() / (rets.abs().rolling(20).sum().replace(0, np.nan))
cands["I_efr_60"] = (closes - closes.shift(60)).abs() / (rets.abs().rolling(60).sum().replace(0, np.nan))
cands["I_ema_10_30"] = closes.ewm(span=10, adjust=False).mean() / closes.ewm(span=30, adjust=False).mean() - 1.0

# persistence (lag-1 autocorrelation of returns)
def autocorr_panel(lag=1, win=10):
    out = {}
    for a in rets.columns:
        r = rets[a]
        out[a] = r.rolling(win).corr(r.shift(lag))
    return pd.DataFrame(out, index=rets.index)

cands["I_autocorr_10"] = autocorr_panel(1, 10)
cands["I_autocorr_20"] = autocorr_panel(1, 20)

# range-based volatility
log_range = (highs / lows).apply(np.log)
cands["I_park_vol_20"] = log_range.rolling(20).std()
cands["I_range_vol_20"] = ((highs - lows) / closes.replace(0, np.nan)).rolling(20).std()

# downside vol share
def semi_vol_ratio(win=60):
    out = {}
    for a in rets.columns:
        r = rets[a]
        dn = r.where(r < 0, 0.0)
        out[a] = dn.rolling(win).std() / r.rolling(win).std().replace(0, np.nan)
    return pd.DataFrame(out, index=rets.index)

cands["I_semi_vol_60"] = semi_vol_ratio(60)

# liquidity / participation
cands["I_vol_z_20"] = (vol_panel - vol_panel.rolling(20).mean()) / vol_panel.rolling(20).std().replace(0, np.nan)
upv = vol_panel.where(rets > 0)
dnv = vol_panel.where(rets < 0)
cands["I_vol_up_down_20"] = upv.rolling(20).mean() / dnv.rolling(20).mean().replace(0, np.nan)

# price location / relative strength
cs_mom20 = closes / closes.shift(20) - 1.0
cands["I_rel_str_20"] = cs_mom20 - cs_mom20.mean(axis=1)
cands["I_dd_60"] = closes / closes.rolling(60).max() - 1.0
cands["I_hl_pos_20"] = ((closes - lows) / (highs - lows).replace(0, np.nan)).rolling(20).mean()

# information flow (overnight gap magnitude)
cands["I_gap_20"] = (opens / closes.shift(1) - 1.0).abs().rolling(20).mean()

# cross-asset betas (full / up / NDX / BTC-corr)
cands["I_mkt_beta_60d"] = rolling_beta(rets, mkt, 60, 40)
up = mkt.where(mkt > 0)
cands["I_up_mkt_beta_60d"] = rolling_beta(rets, up, 60, 40)
cands["I_ndx_beta_60d"] = rolling_beta(rets, rets["NDX"], 60, 40)
cands["I_btc_corr_60"] = rolling_corr(rets, rets["BTC"], 60, 40)

print(f"evaluating {len(cands)} candidates + 4 active...", flush=True)
for name, sig in cands.items():
    results[name] = evaluate(name, sig, 1)

with open("scripts/_miner2_batchI_results.json", "w") as f:
    json.dump(results, f, indent=1, default=str)

# ---------- summary table ----------
rows = []
for k, v in results.items():
    rows.append({"tag": k, "ic": v["ic_full"], "icir": v["icir_full"],
                 "hit": v["hit_full"], "n": v["n_full"], "ic250": v["ic_recent250"],
                 "icir250": v["icir_recent250"], "ic500": v["ic_recent500"],
                 "cov": v["cov_dates_ge8"], "turn": v["turnover_10d"],
                 "libcorr": v["max_lib_corr"], "corrwith": v["max_corr_factor"]})
df = pd.DataFrame(rows)
df["pass"] = ((df["ic"].abs() >= GATE_IC) & (df["icir"].abs() >= GATE_ICIR))
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
print(df.to_string(index=False))
print(f"\nPASSERS ({int(df['pass'].sum())}):")
print(df[df["pass"]][["tag", "ic", "icir", "n", "ic250", "icir250", "ic500", "cov", "turn", "libcorr", "corrwith"]].to_string(index=False))
print(f"\nrun time {time.time()-t0:.1f}s")
