"""miner_3 batch AA (2032-02-23) - novel cross-asset factor screen + re-validation.

Visible data through the previous completed trading day (2032-02-20). Uses the
simulator API via factor_research_lib (no lookahead). 15-instrument universe,
min_valid=8 per IC date. Admission gates: |IC| >= 0.0070 and |ICIR| >= 0.0840
at h=10 (daily rank IC). Reports decay, recent-window drift and max-abs library
correlation for passers. No live-account interaction.
"""
import sys, time, warnings
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

# macro signals
def align(series, idx):
    return series.reindex(idx).ffill()

vix = align(panels["VIX"]["close"].astype(float), closes.index)
dxy = align(panels["DXY"]["close"].astype(float), closes.index)
usdjpy = align(panels["USDJPY"]["close"].astype(float), closes.index)
usdcny = align(panels["USDCNY"]["close"].astype(float), closes.index)
eurusd = align(panels["EURUSD"]["close"].astype(float), closes.index)

# high/low/volume panels
hi = pd.concat({a: panels[a]["high"].astype(float) for a in TRADABLE if a in panels}, axis=1).sort_index().reindex(closes.index)
lo = pd.concat({a: panels[a]["low"].astype(float) for a in TRADABLE if a in panels}, axis=1).sort_index().reindex(closes.index)
vol_panel = pd.concat({a: panels[a]["volume"].astype(float) for a in TRADABLE if a in panels}, axis=1).sort_index().reindex(closes.index)

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


def rolling_corr(a, b, win=20, min_obs=15):
    out = {}
    for col in a.columns:
        z = pd.concat([a[col].rename("a"), b.rename("b")], axis=1).dropna()
        c = z["a"].rolling(win).corr(z["b"])
        out[col] = c.where(z["b"].rolling(win).count() >= min_obs)
    return pd.DataFrame(out, index=a.index)


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
    for w in (63, 126, 252):
        sub = ics.iloc[-w:]
        if len(sub) > 2:
            mm, ss = sub.mean(), sub.std(ddof=1)
            recent[w] = (mm, mm/ss if ss and ss > 0 else np.nan)
        else:
            recent[w] = (np.nan, np.nan)
    flag = "  <== FULL-PASS" if (abs(s["ic"]) >= 0.0070 and abs(s["icir"]) >= 0.0840) else ""
    print(f"{name:26s} IC={s['ic']:+.4f} ICIR={s['icir']:+.3f} hit={s['ic_hit_ratio']:.2f} n={s['n_ic_dates']} "
          f"| r63=({recent[63][0]:+.3f},{recent[63][1]:+.2f}) r126=({recent[126][0]:+.3f},{recent[126][1]:+.2f}) "
          f"r252=({recent[252][0]:+.3f},{recent[252][1]:+.2f}) "
          f"cov={cov['coverage_dates_ge8']:.2f} to={to if to is not None else float('nan'):.2f}{flag}", flush=True)
    return s, ics, sig

for name, sig in existing.items():
    report(name, sig, expected_sign=1 if name != "rate_beta_cn10y_60d" else -1)

# ---------------- 2) CANDIDATE SCREEN (batch AA - novel) ----------------
print("\n=== CANDIDATE SCREEN (batch AA, full history) ===", flush=True)
C = {}
vol20 = rets.rolling(20).std()
vol60 = rets.rolling(60).std()

# AA-1 upside/downside vol asymmetry (20d)
C["updown_vol_ratio_20"] = rets.clip(lower=0).rolling(20).std() / (rets.clip(upper=0).rolling(20).std() + 1e-12)
# AA-2 trend strength t-stat (60d momentum per unit vol)
C["trend_tstat_60"] = (closes/closes.shift(60)-1) / (vol60 * np.sqrt(60) + 1e-12)
# AA-3 crypto risk-appetite beta (beta to BTC+ETH basket, 60d)
crypto_basket = rets[["BTC", "ETH"]].mean(axis=1)
C["crypto_beta_60"] = rolling_beta(rets, crypto_basket, 60)
# AA-4 commodity beta (beta to WTI+XAU+COPPER basket, 60d)
comm_basket = rets[["WTI", "XAU", "COPPER"]].mean(axis=1)
C["comm_beta_60"] = rolling_beta(rets, comm_basket, 60)
# AA-5 beta asymmetry: upside mkt beta - downside mkt beta (60d)
C["beta_asym_60"] = rolling_beta(rets, mkt_ret.clip(lower=0), 60) - rolling_beta(rets, mkt_ret.clip(upper=0), 60)
# AA-6 intraday range ratio (20d mean (high-low)/close)
C["range_ratio_20"] = ((hi - lo) / closes).rolling(20).mean()
# AA-7 VIX correlation (20d rolling corr of asset ret with VIX chg)
C["vix_corr_20"] = rolling_corr(rets, vix.pct_change(), 20)
# AA-8 CN10Y-US10Y spread beta (60d) - relative rate regime transmission
spread = closes["CN10Y"] - closes["US10Y"]
C["spread_beta_cnus_60"] = rolling_beta(rets, spread.diff(), 60)
# AA-9 gold/safe-haven beta (60d)
C["xau_beta_60"] = rolling_beta(rets, rets["XAU"], 60)
# AA-10 lottery: max daily return over 20d
C["max_gain_20"] = rets.rolling(20).max()
# AA-11 Kaufman efficiency ratio (20d net move / total path)
C["efficiency_ratio_20"] = (closes - closes.shift(20)).abs() / (rets.abs().rolling(20).sum() + 1e-12)
# AA-12 volume surge ratio (5d avg vol / 20d avg vol)
C["volume_ratio_5_20"] = vol_panel.rolling(5).mean() / (vol_panel.rolling(20).mean() + 1e-12)
# AA-13 overnight gap magnitude (20d mean |open-prev_close|/prev_close)
prev_close = closes.shift(1)
op = pd.concat({a: panels[a]["open"].astype(float) for a in TRADABLE if a in panels}, axis=1).sort_index().reindex(closes.index)
C["gap_ratio_20"] = ((op - prev_close).abs() / prev_close).rolling(20).mean()

print(f"{len(C)} candidates + 3 re-validation factors; time {time.time()-t0:.1f}s", flush=True)
results = {}
for i, (name, sig) in enumerate(C.items()):
    s, ics, _ = report(name, sig, expected_sign=1)
    results[name] = (s, ics, sig)
    print(f"  [{i+1}/{len(C)}] {name} done {time.time()-t0:.1f}s", flush=True)

# ---------------- 3) decay + library correlation for full-pass ----------------
print("\n=== DECAY + LIBRARY CORRELATION for full-pass candidates ===", flush=True)
lib = dict(existing)
lib.update({k: v[2] for k, v in results.items()})
# historical/reference library signals for correlation reference
lib["mom_10d_skip5"] = closes.shift(5) / closes.shift(15) - 1.0
lib["mom_120d_skip5"] = closes.shift(5) / closes.shift(125) - 1.0
lib["vol_of_vol20x60"] = rets.rolling(20).std().rolling(60).std()
lib["vol_ratio_20_60"] = vol20 / vol60
lib["volume_z_20"] = (vol_panel - vol_panel.rolling(60).mean()) / (vol_panel.rolling(60).std() + 1e-12)
lib["rsi_14"] = closes.rolling(14).apply(lambda s: (s.diff().clip(lower=0).sum() / (s.diff().abs().sum() + 1e-12)), raw=False)
lib["usdcny_beta_60d"] = rolling_beta(rets, usdcny.pct_change(), 60)
lib["eurusd_beta_60d"] = rolling_beta(rets, eurusd.pct_change(), 60)
lib["vix_beta_cond_60x20"] = rolling_beta(rets, vix.pct_change(), 60) * (vix.pct_change().rolling(20).mean() > 0).astype(float)
lib["mom_vs_median_60d"] = (closes/closes.shift(60)-1) - (closes/closes.shift(60)-1).rolling(60).median()
lib["us10y_cond_beta_60d"] = rolling_beta(rets, closes["US10Y"].pct_change(), 60) * (closes["US10Y"].pct_change().rolling(60).mean() > 0).astype(float)
lib["downside_ratio_60d"] = rets.clip(upper=0).rolling(60).std() / (rets.rolling(60).std() + 1e-12)
lib["max_dd_60d"] = closes / closes.rolling(60).max() - 1.0
lib["kurt_60d"] = rets.rolling(60).kurt()
lib["vol_price_corr_20"] = rolling_corr(vol_panel, closes, 20)
lib["corr_asset_mkt_20"] = rolling_corr(rets, mkt_ret, 20)
lib["ma_dist_z_60"] = (closes - closes.rolling(60).mean()) / (vol20 * np.sqrt(60) + 1e-12)
lib["trend_r2_60"] = None  # placeholder (slow; skipped here)
lib["rev_5d"] = -(closes.shift(5)/closes - 1.0)
lib["range_break_20"] = closes / hi.rolling(20).max() - 1.0
lib["corr_mkt_20"] = rolling_corr(rets, mkt_ret, 20)
lib["kelly_frac_60"] = rets.rolling(60).mean() / (rets.rolling(60).var() + 1e-12)
lib["dn_rate_beta_60"] = rolling_beta(rets, closes["US10Y"].pct_change().clip(upper=0), 60)
lib["upside_beta_mkt_60"] = rolling_beta(rets, mkt_ret.clip(lower=0), 60)
lib["volume_trend_20"] = vol_panel.rolling(20).mean() / (vol_panel.rolling(60).mean() + 1e-12)
lib["usdjpy_beta_60"] = rolling_beta(rets, usdjpy.pct_change(), 60)
lib["dxy_beta_60"] = rolling_beta(rets, dxy.pct_change(), 60)
lib["vix_beta_60"] = rolling_beta(rets, vix.pct_change(), 60)
lib["semi_dev_20"] = rets.rolling(20).mean() / (rets.clip(upper=0).rolling(20).std() + 1e-12)
lib["breadth_rs_60"] = (closes/closes.shift(60)-1) - mkt_ret.rolling(60).sum()

def max_lib_corr(sig, exclude):
    best, key = 0.0, None
    for lname, lsig in lib.items():
        if lname == exclude or lsig is None:
            continue
        both = pd.concat([sig.stack().rename("c"), lsig.stack().rename("l")], axis=1).dropna()
        if len(both) < 30:
            continue
        rr = float(both["c"].corr(both["l"]))
        if abs(rr) > best:
            best, key = abs(rr), lname
    return round(best, 4), key

passing = {k: v for k, v in results.items() if abs(v[0]["ic"]) >= 0.0070 and abs(v[0]["icir"]) >= 0.0840}
print(f"FULL-PASS count: {len(passing)}", flush=True)
for name, (s, ics, sig) in passing.items():
    dec = decay_profile(sig, closes, horizons=(1, 3, 5, 10, 20))
    corr, key = max_lib_corr(sig, exclude=name)
    print(f"{name:26s} decay={dec} max_abs_lib_corr={corr:.4f} (vs {key})", flush=True)

print(f"\ndone {time.time()-t0:.1f}s", flush=True)
