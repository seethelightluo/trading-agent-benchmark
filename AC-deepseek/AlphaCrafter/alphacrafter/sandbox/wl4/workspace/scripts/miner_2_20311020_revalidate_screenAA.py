"""miner_2 batch AA (2031-10-20) - re-validate effective factors + screen novel candidates.

Visible data through the previous completed trading day (2031-10-17). Uses the
simulator API via factor_research_lib (no lookahead). 15-instrument universe,
min_valid=8 per IC date. Admission gates: |IC| >= 0.0070 and |ICIR| >= 0.0840
at h=10 (daily rank IC). Reports decay, recent-window drift, and spearman rho vs
the 3 currently effective library factors (conflict threshold 0.5). No live-account interaction.
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

live_out = {}
for name, sig in existing.items():
    exp = 1 if name != "rate_beta_cn10y_60d" else -1
    s, ics = report(name, sig, expected_sign=exp)
    live_out[name] = s

# ---------------- 2) SCREEN batch AA (novel candidates) ----------------
print("\n=== CANDIDATE SCREEN (batch AA, full history) ===", flush=True)
AA = {}
vol20 = rets.rolling(20).std()
vol60 = rets.rolling(60).std()
vol120 = rets.rolling(120).std()

# AA1: 20d momentum skip 5, vol-normalized (short-term trend per unit risk)
AA["mom20_skip5_voladj"] = (closes.shift(5)/closes.shift(25) - 1) / vol20
# AA2: 120d momentum skip 5, vol-normalized (longer trend per unit risk)
AA["mom120_skip5_voladj"] = (closes.shift(5)/closes.shift(125) - 1) / vol120
# AA3: upside half-vol ratio (upside capture vs total vol)
def upside_ratio(r, win=60):
    out = {}
    for a in r.columns:
        v = r[a]
        out[a] = v.clip(lower=0).rolling(win).std() / (v.rolling(win).std() + 1e-12)
    return pd.DataFrame(out, index=r.index)
AA["upside_ratio_60d"] = upside_ratio(rets, 60)
# AA4: beta to DXY (USD strength sensitivity)
AA["beta_dxy_60d"] = rolling_beta(rets, dxy.pct_change(), 60)
# AA5: beta to EURUSD (risk-on FX sensitivity)
AA["beta_eurusd_60d"] = rolling_beta(rets, eurusd.pct_change(), 60)
# AA6: beta to USDJPY (carry / risk sentiment)
AA["beta_usdjpy_60d"] = rolling_beta(rets, usdjpy.pct_change(), 60)
# AA7: 10d return / 60d vol (recent impulse per unit risk)
AA["impulse10_vol60"] = (closes/closes.shift(10) - 1) / vol60
# AA8: 20d return / 60d vol
AA["mom20_vol60"] = (closes/closes.shift(20) - 1) / vol60
# AA9: conditional mom: 20d return only when market up (up-beta momentum)
mkt_up = (mkt_ret > 0).astype(float)
AA["mom20_cond_up"] = (closes/closes.shift(20) - 1) * mkt_up.to_frame(0).values
# AA10: 60d range position: (close - min60)/(max60 - min60) - 0.5 (cycle position)
AA["range_pos_60d"] = (closes - closes.rolling(60).min()) / (closes.rolling(60).max() - closes.rolling(60).min() + 1e-12) - 0.5
# AA11: vol-of-vol 20x60 (volatility regime instability)
AA["vol_of_vol20x60"] = rets.rolling(20).std().rolling(60).std()
# AA12: volume z-score 20d (abnormal volume)
if "volume" in panels["SPX"].columns:
    vol_df = pd.concat({a: panels[a]["volume"].astype(float) for a in TRADABLE if a in panels}, axis=1).sort_index()
    AA["volume_z_20"] = (vol_df - vol_df.rolling(60).mean()) / (vol_df.rolling(60).std() + 1e-12)
# AA13: 3d reversal vol-adj (ultra-short mean reversion)
AA["rev3_voladj"] = -(closes/closes.shift(3) - 1) / vol20
# AA14: XAU beta 60d (safe-haven co-movement)
AA["beta_xau_60d"] = rolling_beta(rets, closes["XAU"].pct_change(), 60)
# AA15: WTI beta 60d (energy sensitivity)
AA["beta_wti_60d"] = rolling_beta(rets, closes["WTI"].pct_change(), 60)
# AA16: COPPER beta 60d (global growth sensitivity)
AA["beta_copper_60d"] = rolling_beta(rets, closes["COPPER"].pct_change(), 60)
# AA17: 120d trend ratio: close / SMA120 (distance from long-term mean)
AA["trend_ratio_120"] = closes / closes.rolling(120).mean() - 1
# AA18: 20d trend ratio: close / SMA20
AA["trend_ratio_20"] = closes / closes.rolling(20).mean() - 1
# AA19: VIX level z-score (fear regime, cross-sectional constant -> low info, but keep)
AA["vix_level_z"] = ((vix - vix.rolling(60).mean()) / (vix.rolling(60).std() + 1e-12)).to_frame(0).values
# AA20: DXY 20d momentum (USD regime)
AA["dxy_mom20"] = (dxy/dxy.shift(20) - 1).to_frame(0).values

DIR = {}
results = {}
for name, sig in AA.items():
    exp = DIR.get(name, 1)
    s, ics = report(name, sig, expected_sign=exp)
    results[name] = (s, ics, sig)

# ---------------- 3) full validation for full-pass candidates ----------------
print("\n=== DECAY + LIVE-LIBRARY RHO for full-pass batch AA candidates ===", flush=True)
passing = {k: v for k, v in results.items() if abs(v[0]["ic"]) >= 0.0070 and abs(v[0]["icir"]) >= 0.0840}
print(f"Full-pass count (batch AA): {len(passing)}", flush=True)

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

for name, (s, ics, sig) in passing.items():
    dec = decay_profile(sig, closes, horizons=(1, 3, 5, 10, 20), min_valid=8, expected_sign=1)
    rho, key = spearman_rho_vs_live(sig, existing)
    print(f"{name:26s} decay={dec} rho_vs_live={rho:.4f} ({key})", flush=True)

# save results
out = {k: {kk: vv for kk, vv in v[0].items()} for k, v in results.items()}
out["_meta"] = {"asof": str(closes.index.max().date()), "n_assets": closes.shape[1],
                "gates": {"abs_ic": 0.0070, "abs_icir": 0.0840, "min_valid": 8, "h": H},
                "live_revalidation": {k: v for k, v in live_out.items()}}
with open("scripts/_miner2_20311020_batchAA_results.json", "w") as fh:
    json.dump(out, fh, indent=1, default=str)
print(f"\nsaved scripts/_miner2_20311020_batchAA_results.json | done {time.time()-t0:.1f}s", flush=True)
