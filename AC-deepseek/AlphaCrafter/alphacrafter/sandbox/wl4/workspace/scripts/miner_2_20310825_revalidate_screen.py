"""miner_2 batch Z (2031-08-25) - re-validate effective factors, confirm batch-Y
passers vs CURRENT live library, and screen novel candidates.

Visible data through the previous completed trading day (2031-08-22). Uses the
simulator API via factor_research_lib (no lookahead). 15-instrument universe,
min_valid=8 per IC date. Admission gates: |IC| >= 0.0070 and |ICIR| >= 0.0840
at h=10 (daily rank IC). Reports decay, recent-window drift, and spearman rho vs
the 3 currently effective library factors (conflict threshold 0.5, matching the
worldline pairwise signal-quality contract). No live-account interaction.
"""
import sys, time, warnings, json
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.insert(0, "scripts")
from factor_research_lib import (load_panels, close_panel, forward_returns,
                                 rank_ic_series, summarize_ic, coverage_metrics,
                                 turnover_rank, decay_profile, TRADABLE)

t0 = time.time()
panels = load_panels(days=4000)
closes = close_panel(panels)
rets = closes.pct_change()
mkt_ret = rets.mean(axis=1)
print(f"closes {closes.shape} | {closes.index.min().date()}..{closes.index.max().date()} | {time.time()-t0:.1f}s", flush=True)

def align(series, idx):
    return series.reindex(idx).ffill()

vix = align(panels["VIX"]["close"].astype(float), closes.index)
dxy = align(panels["DXY"]["close"].astype(float), closes.index)
usdjpy = align(panels["USDJPY"]["close"].astype(float), closes.index)
eurusd = align(panels["EURUSD"]["close"].astype(float), closes.index)
usdcny = align(panels["USDCNY"]["close"].astype(float), closes.index)

H = 10
fwd = forward_returns(closes, H)

def rolling_beta(y, x, win=60, min_obs=40):
    out = {}
    for a in y.columns:
        z = pd.concat([y[a].rename("y"), x.rename("x")], axis=1).dropna()
        cov = z["y"].rolling(win).cov(z["x"])
        var = z["x"].rolling(win).var()
        b = (cov / var).where(z["x"].rolling(win).count() >= min_obs)
        out[a] = b
    return pd.DataFrame(out, index=y.index)

# ---------------- 1) RE-VALIDATE current effective factors ----------------
print("\n=== RE-VALIDATION of current effective factors (full + recent) ===", flush=True)
existing = {}
existing["vol_adj_mom_accel_20x60"] = (closes/closes.shift(20)-1 - (closes/closes.shift(60)-1)) / rets.rolling(20).std()
existing["dn_mkt_beta_60d"] = rolling_beta(rets, mkt_ret.clip(upper=0), 60)
existing["rate_beta_cn10y_60d"] = rolling_beta(rets, closes["CN10Y"].pct_change(), 60)

def report(name, sig, expected_sign=1):
    ics = rank_ic_series(sig, fwd)
    s = summarize_ic(ics, expected_sign=expected_sign)
    cov = coverage_metrics(sig)
    to = turnover_rank(sig, 10)
    recent = {}
    for w in (63, 126, 252, 504):
        sub = ics.iloc[-w:]
        if len(sub) > 2:
            mm, ss = sub.mean(), sub.std(ddof=1)
            recent[w] = (mm, mm/ss if ss and ss > 0 else np.nan)
        else:
            recent[w] = (np.nan, np.nan)
    flag = "  <== FULL-PASS" if (abs(s["ic"]) >= 0.0070 and abs(s["icir"]) >= 0.0840) else ""
    print(f"{name:26s} IC={s['ic']:+.4f} ICIR={s['icir']:+.3f} hit={s['ic_hit_ratio']:.2f} n={s['n_ic_dates']} "
          f"| r63=({recent[63][0]:+.3f},{recent[63][1]:+.2f}) r126=({recent[126][0]:+.3f},{recent[126][1]:+.2f}) "
          f"r252=({recent[252][0]:+.3f},{recent[252][1]:+.2f}) r504=({recent[504][0]:+.3f},{recent[504][1]:+.2f}) "
          f"cov={cov['coverage_dates_ge8']:.2f} to={to if to is not None else float('nan'):.2f}{flag}", flush=True)
    return s, ics

for name, sig in existing.items():
    report(name, sig, expected_sign=1 if name != "rate_beta_cn10y_60d" else -1)

# ---------------- 2) CONFIRM batch-Y passers vs live library ----------------
print("\n=== CONFIRM batch-Y passers (IC gate + rho vs CURRENT 3-factor live library) ===", flush=True)
vol20 = rets.rolling(20).std()
vol60 = rets.rolling(60).std()

def downside_ratio(r, win=60):
    out = {}
    for a in r.columns:
        v = r[a]
        sd = v.clip(upper=0).rolling(win).std()
        tot = v.rolling(win).std()
        out[a] = sd / (tot + 1e-12)
    return pd.DataFrame(out, index=r.index)

C = {}
C["downside_ratio_60d"] = downside_ratio(rets, 60)
C["beta_btc_60d"] = rolling_beta(rets, closes["BTC"].pct_change(), 60)
C["drawdown_60d"] = (closes - closes.rolling(60).max()) / closes.rolling(60).max()
C["hl_pos_20d"] = (closes - closes.rolling(20).min()) / (closes.rolling(20).max() - closes.rolling(20).min())
vix_ret = vix.pct_change()
C["vix_beta_x_level_60d"] = -rolling_beta(rets, vix_ret, 60) * (vix / vix.shift(60)).to_frame(0).values

live_lib = dict(existing)  # only the 3 currently effective factors

def spearman_rho_vs_live(cand, library):
    best, best_key = 0.0, None
    for name, lsig in library.items():
        both = pd.concat([cand.stack().rename("c"), lsig.stack().rename("l")], axis=1).dropna()
        if len(both) < 30:
            continue
        rr = float(both["c"].corr(both["l"], method="spearman"))
        if abs(rr) > best:
            best, best_key = abs(rr), name
    return round(best, 4), best_key

for name, sig in C.items():
    s, ics = report(name, sig, expected_sign=1 if name not in ("drawdown_60d", "hl_pos_20d") else -1)
    rho, key = spearman_rho_vs_live(sig, live_lib)
    dec = decay_profile(sig, closes, horizons=(1, 3, 5, 10, 20), min_valid=8,
                        expected_sign=1 if name not in ("drawdown_60d", "hl_pos_20d") else -1)
    passes = abs(s["ic"]) >= 0.0070 and abs(s["icir"]) >= 0.0840
    conflict = "  <== RHO>0.5 CONFLICT RISK" if rho > 0.5 else ""
    print(f"   rho_vs_live={rho:.4f} ({key}) | decay={dec}{conflict}", flush=True)

# ---------------- 3) SCREEN batch Z (novel candidates) ----------------
print("\n=== CANDIDATE SCREEN (batch Z, full history) ===", flush=True)
Z = {}
vol120 = rets.rolling(120).std()

# Z1: 60d momentum skip 5, vol-normalized (medium-term risk-adjusted trend)
Z["mom60_skip5_voladj"] = (closes.shift(5)/closes.shift(65) - 1) / vol60
# Z2: 5d reversal scaled by 20d vol (short-term mean reversion per unit risk)
Z["rev5_voladj"] = -(closes/closes.shift(5) - 1) / vol20
# Z3: vol term-structure slope (vol20 - vol60)/vol60 (regime anticipation)
Z["vol_ts_slope"] = (vol20 - vol60) / (vol60 + 1e-12)
# Z4: 120d Sharpe-style ratio (momentum per unit long-term vol)
Z["sharpe_120d"] = (closes/closes.shift(120) - 1) / (vol120 + 1e-12)
# Z5: beta to US10Y conditional on US10Y rising (rate-shock sensitivity)
us10y_up = (closes["US10Y"].pct_change() > 0).astype(float)
Z["us10y_up_beta_60d"] = rolling_beta(rets, closes["US10Y"].pct_change() * us10y_up, 60)
# Z6: XAU beta in down-market (safe-haven co-movement in stress)
Z["xau_dn_beta_60d"] = rolling_beta(rets, closes["XAU"].pct_change() * (mkt_ret < 0).astype(float), 60)
# Z7: max drawdown over 120d (deep drawdown memory)
Z["drawdown_120d"] = (closes - closes.rolling(120).max()) / closes.rolling(120).max()
# Z8: skewness 60d (longer-window crash asymmetry)
Z["skew_60d"] = rets.rolling(60).skew()
# Z9: 20d range expansion: (high-low)/close 20d mean vs 60d mean
if "high" in panels["SPX"].columns and "low" in panels["SPX"].columns:
    hl = pd.concat({a: (panels[a]["high"] - panels[a]["low"]) / panels[a]["close"]
                    for a in TRADABLE if a in panels}, axis=1).sort_index()
    Z["hl_range_exp_20x60"] = hl.rolling(20).mean() / (hl.rolling(60).mean() + 1e-12)
# Z10: dispersion regime tilt: asset momentum x market dispersion (trend works in dispersion regimes)
disp = rets.std(axis=1)
disp_z = (disp - disp.rolling(60).mean()) / (disp.rolling(60).std() + 1e-12)
Z["mom20_x_disp"] = (closes/closes.shift(20) - 1) * disp_z.to_frame(0).values
# Z11: 3d momentum skip 1 (ultra-short continuation)
Z["mom3_skip1"] = closes.shift(1)/closes.shift(4) - 1
# Z12: USDCNY beta 60d (China FX pressure sensitivity)
Z["beta_usdcny_60d"] = rolling_beta(rets, usdcny.pct_change(), 60)
# Z13: ETH-BTC relative momentum (crypto rotation)
Z["eth_btc_rel_mom20"] = (closes["ETH"]/closes["ETH"].shift(20)-1).to_frame(0).values - (closes["BTC"]/closes["BTC"].shift(20)-1).to_frame(0).values
# Z14: volume-price trend: 20d volume slope (accumulation proxy)
if "volume" in panels["SPX"].columns:
    vol_df = pd.concat({a: panels[a]["volume"].astype(float) for a in TRADABLE if a in panels}, axis=1).sort_index()
    Z["volume_trend_20d"] = vol_df / (vol_df.rolling(60).mean() + 1e-12) - 1.0

results = {}
for name, sig in Z.items():
    s, ics = report(name, sig, expected_sign=1)
    results[name] = (s, ics, sig)

# ---------------- 4) decay + live-library rho for full-pass batch Z ----------------
print("\n=== DECAY + LIVE-LIBRARY RHO for full-pass batch Z candidates ===", flush=True)
passing = {k: v for k, v in results.items() if abs(v[0]["ic"]) >= 0.0070 and abs(v[0]["icir"]) >= 0.0840}
print(f"Full-pass count (batch Z): {len(passing)}", flush=True)
for name, (s, ics, sig) in passing.items():
    dec = decay_profile(sig, closes, horizons=(1, 3, 5, 10, 20), min_valid=8, expected_sign=1)
    rho, key = spearman_rho_vs_live(sig, live_lib)
    print(f"{name:26s} decay={dec} rho_vs_live={rho:.4f} ({key})", flush=True)

print(f"\ndone {time.time()-t0:.1f}s", flush=True)
