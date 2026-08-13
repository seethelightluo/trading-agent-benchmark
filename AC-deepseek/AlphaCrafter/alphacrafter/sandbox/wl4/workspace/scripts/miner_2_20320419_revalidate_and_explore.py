"""miner_2 (2032-04-19) - re-validate effective factors + explore new candidates.

Visible data through 2032-04-16. Universe: 15 tradable cross-asset instruments.
Gates: |IC|>=0.0070 and |ICIR|>=0.0840 at h=10. Reports decay, coverage,
turnover, max_abs_library_correlation, recent-window drift, yearly splits.
No live-account interaction (miner only).
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
usdcny = align(panels["USDCNY"]["close"].astype(float), closes.index)
eurusd = align(panels["EURUSD"]["close"].astype(float), closes.index)

hi = pd.concat({a: panels[a]["high"].astype(float) for a in TRADABLE if a in panels}, axis=1).sort_index().reindex(closes.index)
lo = pd.concat({a: panels[a]["low"].astype(float) for a in TRADABLE if a in panels}, axis=1).sort_index().reindex(closes.index)
vol_panel = pd.concat({a: panels[a]["volume"].astype(float) for a in TRADABLE if a in panels}, axis=1).sort_index().reindex(closes.index)
op = pd.concat({a: panels[a]["open"].astype(float) for a in TRADABLE if a in panels}, axis=1).sort_index().reindex(closes.index)

H = 10
fwd = forward_returns(closes, H)
vol20 = rets.rolling(20).std()
vol60 = rets.rolling(60).std()


def rolling_beta(y, x, win=60, min_obs=40):
    out = {}
    for a in y.columns:
        z = pd.concat([y[a].rename("y"), x.rename("x")], axis=1).dropna()
        cov = z["y"].rolling(win).cov(z["x"])
        var = z["x"].rolling(win).var()
        b = (cov / var).where(z["x"].rolling(win).count() >= min_obs)
        out[a] = b
    return pd.DataFrame(out, index=y.index)


def rolling_corr_series(a, b, win=20, min_obs=15):
    out = {}
    for col in a.columns:
        z = pd.concat([a[col].rename("a"), b.rename("b")], axis=1).dropna()
        c = z["a"].rolling(win).corr(z["b"])
        out[col] = c.where(z["b"].rolling(win).count() >= min_obs)
    return pd.DataFrame(out, index=a.index)


# ---------- library reference signals for correlation audit ----------
lib = {}
lib["vol_adj_mom_accel_20x60"] = (closes/closes.shift(20)-1 - (closes/closes.shift(60)-1)) / rets.rolling(20).std()
lib["dn_mkt_beta_60d"] = rolling_beta(rets, mkt_ret.clip(upper=0), 60)
lib["rate_beta_cn10y_60d"] = rolling_beta(rets, closes["CN10Y"].pct_change(), 60)
lib["mom_10d_skip5"] = closes.shift(5) / closes.shift(15) - 1.0
lib["mom_120d_skip5"] = closes.shift(5) / closes.shift(125) - 1.0
lib["vol_of_vol20x60"] = rets.rolling(20).std().rolling(60).std()
lib["vol_ratio_20_60"] = vol20 / vol60
lib["volume_z_20"] = (vol_panel - vol_panel.rolling(60).mean()) / (vol_panel.rolling(60).std() + 1e-12)
lib["usdcny_beta_60d"] = rolling_beta(rets, usdcny.pct_change(), 60)
lib["eurusd_beta_60d"] = rolling_beta(rets, eurusd.pct_change(), 60)
lib["vix_beta_cond_60x20"] = rolling_beta(rets, vix.pct_change(), 60) * (vix.pct_change().rolling(20).mean() > 0).astype(float)
lib["mom_vs_median_60d"] = (closes/closes.shift(60)-1) - (closes/closes.shift(60)-1).rolling(60).median()
lib["us10y_cond_beta_60d"] = rolling_beta(rets, closes["US10Y"].pct_change(), 60) * (closes["US10Y"].pct_change().rolling(60).mean() > 0).astype(float)
lib["downside_ratio_60d"] = rets.clip(upper=0).rolling(60).std() / (rets.rolling(60).std() + 1e-12)
lib["max_dd_60d"] = closes / closes.rolling(60).max() - 1.0
lib["kurt_60d"] = rets.rolling(60).kurt()
lib["corr_asset_mkt_20"] = rolling_corr_series(rets, mkt_ret, 20)
lib["hl_pos_20d"] = (hi - lo) / (closes + 1e-12)
lib["rsi_14"] = closes.rolling(14).apply(lambda s: (s.diff().clip(lower=0).sum() / (s.diff().abs().sum() + 1e-12)), raw=False)
lib["xau_beta_60"] = rolling_beta(rets, rets["XAU"], 60)
lib["comm_beta_60"] = rolling_beta(rets, rets[["WTI", "XAU", "COPPER"]].mean(axis=1), 60)
lib["spread_beta_cnus_60"] = rolling_beta(rets, (closes["CN10Y"] - closes["US10Y"]).diff(), 60)
lib["updown_vol_ratio_20"] = rets.clip(lower=0).rolling(20).std() / (rets.clip(upper=0).rolling(20).std() + 1e-12)
lib["trend_tstat_60"] = (closes/closes.shift(60)-1) / (vol60 * np.sqrt(60) + 1e-12)
lib["crypto_beta_60"] = rolling_beta(rets, rets[["BTC", "ETH"]].mean(axis=1), 60)
lib["max_gain_20"] = rets.rolling(20).max()
lib["skew_60d"] = rets.rolling(60).skew()
lib["beta_btc_60d"] = rolling_beta(rets, closes["BTC"].pct_change(), 60)
lib["vix_level_z"] = pd.DataFrame({a: ((vix - vix.rolling(60).mean()) / (vix.rolling(60).std() + 1e-12)).values for a in closes.columns}, index=closes.index)
lib["dxy_mom20"] = pd.DataFrame({a: (dxy/dxy.shift(20) - 1).values for a in closes.columns}, index=closes.index)
print(f"library refs: {len(lib)} | {time.time()-t0:.1f}s", flush=True)


def pairwise_max_abs_corr(cand, libsig):
    common = cand.columns.intersection(libsig.columns)
    if len(common) == 0:
        return np.nan
    rhos = []
    for a in common:
        pair = pd.concat([cand[a].rename("c"), libsig[a].rename("l")], axis=1).dropna()
        if len(pair) > 60 and pair["c"].std() > 1e-14 and pair["l"].std() > 1e-14:
            rhos.append(pair["c"].corr(pair["l"], method="spearman"))
    if not rhos:
        return np.nan
    return float(np.max(np.abs(rhos)))


def full_report(name, sig, expected_sign=1):
    ics = rank_ic_series(sig, fwd)
    s = summarize_ic(ics, expected_sign=expected_sign)
    cov = coverage_metrics(sig)
    to = turnover_rank(sig, 10)
    dec = decay_profile(sig, closes, horizons=[1, 2, 3, 5, 10, 20])
    mc = {k: pairwise_max_abs_corr(sig, v) for k, v in lib.items()}
    mc = {k: v for k, v in mc.items() if v == v}
    mcl = max(mc.values()) if mc else np.nan
    mcl_name = max(mc, key=mc.get) if mc else ""
    # recent drift
    r63 = ics.iloc[-63:].mean() if len(ics) >= 63 else np.nan
    r126 = ics.iloc[-126:].mean() if len(ics) >= 126 else np.nan
    r252 = ics.iloc[-252:].mean() if len(ics) >= 252 else np.nan
    yr = {}
    for y in range(2020, 2033):
        sub = ics[(ics.index.year == y)]
        if len(sub) > 20:
            mm, ss = sub.mean(), sub.std(ddof=1)
            yr[y] = (round(mm, 4), round(mm/ss, 2) if ss and ss > 0 else np.nan)
    flag = "  <== FULL-PASS" if (abs(s["ic"]) >= 0.0070 and abs(s["icir"]) >= 0.0840) else ""
    print(f"\n=== {name} (es={expected_sign:+d}){flag} ===", flush=True)
    print(f"  IC={s['ic']:+.4f} ICIR={s['icir']:+.3f} hit={s['ic_hit_ratio']:.2f} n={s['n_ic_dates']} "
          f"std={s['ic_std']:.3f} | cov8={cov['coverage_dates_ge8']:.2f} cov_asset={cov['coverage_asset_days']:.2f} "
          f"to10={to if to is not None else float('nan'):.2f}", flush=True)
    print(f"  decay(1,2,3,5,10,20)={[round(x,4) if x==x else None for x in dec]} | "
          f"r63={r63:+.4f} r126={r126:+.4f} r252={r252:+.4f}", flush=True)
    print(f"  max_abs_lib_corr={mcl:.3f} (vs {mcl_name})", flush=True)
    print(f"  yearly_ic(mean,icir): {yr}", flush=True)
    return s, ics, sig, mc


# ============ 1) RE-VALIDATE current effective factors ============
print("\n\n########## RE-VALIDATION (current effective) ##########", flush=True)
existing = {
    "vol_adj_mom_accel_20x60": ((closes/closes.shift(20)-1 - (closes/closes.shift(60)-1)) / rets.rolling(20).std(), 1),
    "dn_mkt_beta_60d": (rolling_beta(rets, mkt_ret.clip(upper=0), 60), 1),
    "rate_beta_cn10y_60d": (rolling_beta(rets, closes["CN10Y"].pct_change(), 60), -1),
}
rev = {}
for name, (sig, es) in existing.items():
    s, ics, _, _ = full_report(name, sig, expected_sign=es)
    rev[name] = s

# ============ 2) NEW CANDIDATES ============
print("\n\n########## NEW CANDIDATES ##########", flush=True)
C = {}
# A: return-path efficiency: |net move| / total path length over 20d (trend purity)
C["path_efficiency_20"] = (closes/closes.shift(20)-1).abs() / (rets.abs().rolling(20).sum() + 1e-12)
# B: 1-day return autocorrelation (5d rolling corr of ret with ret.shift(1)); neg => reversal-prone
C["autocorr_5"] = rets.rolling(5).corr(rets.shift(1))
# C: Amihud illiquidity: mean(|ret|/volume) over 20d
C["amihud_illiq_20"] = (rets.abs() / (vol_panel + 1e-12)).rolling(20).mean()
# D: up-day win rate over 20d
C["win_rate_20"] = (rets > 0).rolling(20).mean()
# E: close position in 20d range (0..1)
C["range_pos_20"] = (closes - lo.rolling(20).min()) / (hi.rolling(20).max() - lo.rolling(20).min() + 1e-12)
# F: z-score of close vs 20d mean (mean reversion / overextension)
C["close_z_20"] = (closes - closes.rolling(20).mean()) / (closes.rolling(20).std() + 1e-12)
# G: 60d return dispersion vs market (idiosyncratic trend): asset ret - mkt ret rolling mean
idio = rets.sub(mkt_ret, axis=0)
C["idio_mom_20"] = idio.rolling(20).mean()
# H: drawdown recovery: 1 - close/rolling_60d_max (same as max_dd but positive direction check) -- use depth only
C["dd_depth_60"] = closes.rolling(60).max() / closes - 1.0
# I: conditional downside beta vs VIX (risk-on sensitivity): -beta(asset, VIX, 60)
C["vix_beta_60"] = -rolling_beta(rets, vix.pct_change(), 60)
# J: gold-relative momentum: asset vs XAU 20d spread (safe-haven rotation)
C["xau_rel_mom_20"] = (closes/closes.shift(20)-1).sub((closes["XAU"]/closes["XAU"].shift(20)-1), axis=0)
# K: US10Y momentum spillover: asset correlation with US10Y 20d changes (rate sensitivity)
C["us10y_corr_20"] = rolling_corr_series(rets, closes["US10Y"].pct_change(), 20)
# L: CN10Y momentum spillover
C["cn10y_corr_20"] = rolling_corr_series(rets, closes["CN10Y"].pct_change(), 20)
# M: volume trend: 20d volume vs 60d volume ratio
C["vol_trend_20_60"] = vol_panel.rolling(20).mean() / (vol_panel.rolling(60).mean() + 1e-12)
# N: overnight/gap behavior: open vs prev close gap, 20d mean of abs gap (gap intensity)
C["gap_intensity_20"] = ((op / closes.shift(1) - 1).abs()).rolling(20).mean()
# O: momentum of volatility (5d vol / 60d vol) - fast vol spike
C["vol_spike_5_60"] = rets.rolling(5).std() / (vol60 + 1e-12)
# P: skew-adj momentum: sign-weighted momentum (win rate * magnitude)
C["mom_winrate_20"] = (rets > 0).rolling(20).mean() * (closes/closes.shift(20)-1)

results = {}
for name, sig in C.items():
    es = 1
    s, ics, sig, mc = full_report(name, sig, expected_sign=es)
    results[name] = (s, ics, sig, mc)

passing = {k: v for k, v in results.items() if abs(v[0]["ic"]) >= 0.0070 and abs(v[0]["icir"]) >= 0.0840}
print(f"\n\nFULL-PASS count (new candidates): {len(passing)}", flush=True)
for k in passing:
    print(f"  PASS: {k} IC={passing[k][0]['ic']:+.4f} ICIR={passing[k][0]['icir']:+.3f}", flush=True)

print(f"\ntotal time {time.time()-t0:.1f}s", flush=True)
