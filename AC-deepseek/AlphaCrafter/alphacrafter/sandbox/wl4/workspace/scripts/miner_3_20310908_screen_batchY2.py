"""miner_3 batch Y2 (2031-09-08) - vectorized cross-asset factor screen + re-validation.

Visible data through previous completed trading day (2031-09-05). Simulator API via
factor_research_lib (no lookahead). 15-instrument universe, min_valid=8 per IC date.
Admission gates: |IC| >= 0.0070 and |ICIR| >= 0.0840 at h=10 (daily rank IC).
No live-account interaction.
"""
import sys, time, warnings, math
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.insert(0, "scripts")
from factor_research_lib import (load_panels, close_panel, forward_returns,
                                 summarize_ic, coverage_metrics, turnover_rank,
                                 library_signals, TRADABLE)

t0 = time.time()
panels = load_panels(days=3000)
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

H = 10
fwd = forward_returns(closes, H)

def rank_ic_fast(sig, fwd_, min_valid=8):
    """Vectorized daily Spearman rank IC (row-wise rank then corrwith)."""
    fr = sig.rank(axis=1)
    rr = fwd_.reindex(sig.index).rank(axis=1)
    n_valid = (sig.notna() & fwd_.reindex(sig.index).notna()).sum(axis=1)
    ics = fr.corrwith(rr, axis=1)
    ics = ics.where(n_valid >= min_valid)
    return ics.dropna()

def rolling_beta(y, x, win=60, min_obs=40):
    out = {}
    for a in y.columns:
        z = pd.concat([y[a].rename("y"), x.rename("x")], axis=1).dropna()
        cov = z["y"].rolling(win).cov(z["x"])
        var = z["x"].rolling(win).var()
        b = (cov / var).where(z["x"].rolling(win).count() >= min_obs)
        out[a] = b
    return pd.DataFrame(out, index=y.index)

def roll_autocorr(x, win=10):
    y = x.shift(1)
    cov = x.rolling(win).cov(y)
    vx = x.rolling(win).var()
    vy = y.rolling(win).var()
    return cov / np.sqrt(vx * vy)

# ---------------- 1) RE-VALIDATE current effective factors ----------------
print("\n=== RE-VALIDATION of current effective factors (full + recent) ===", flush=True)
existing = {}
existing["vol_adj_mom_accel_20x60"] = (closes/closes.shift(20)-1 - (closes/closes.shift(60)-1)) / rets.rolling(20).std()
existing["dn_mkt_beta_60d"] = rolling_beta(rets, mkt_ret.clip(upper=0), 60)
existing["rate_beta_cn10y_60d"] = rolling_beta(rets, closes["CN10Y"].pct_change(), 60)

def report(name, sig, expected_sign=1):
    ics = rank_ic_fast(sig, fwd)
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

# ---------------- 2) CANDIDATE SCREEN (batch Y + new Z) ----------------
print("\n=== CANDIDATE SCREEN (batch Y + Z, full history) ===", flush=True)
vol20 = rets.rolling(20).std()
vol60 = rets.rolling(60).std()
mom20 = closes / closes.shift(20) - 1.0
mom60 = closes / closes.shift(60) - 1.0

C = {}
# Y1: return skewness 20d
C["skew_20d"] = rets.rolling(20).skew()
# Y2: downside semi-dev / total vol 20d
dn_ret = rets.clip(upper=0)
C["downside_vol_ratio_20"] = dn_ret.rolling(20).std() / (vol20 + 1e-9)
# Y3: drawdown depth from 60d high (negative = deep drawdown)
C["dd_depth_60d"] = closes / closes.rolling(60).max() - 1.0
# Y4: Kaufman efficiency ratio 20d
C["efficiency_ratio_20d"] = (closes - closes.shift(20)).abs() / (rets.abs().rolling(20).sum() + 1e-9)
# Y5: US10Y beta 60d
C["us10y_beta_60d"] = rolling_beta(rets, closes["US10Y"].pct_change(), 60)
# Y6: WTI beta 60d
C["wti_beta_60d"] = rolling_beta(rets, closes["WTI"].pct_change(), 60)
# Y7: BTC beta 60d
C["btc_beta_60d"] = rolling_beta(rets, closes["BTC"].pct_change(), 60)
# Y8: 1d return autocorrelation 10d (vectorized)
C["autocorr_1d_10d"] = roll_autocorr(rets, 10)
# Y9: intraday range 20d (high-low)/close
C["intraday_range_20d"] = pd.concat({a: (panels[a]["high"] - panels[a]["low"]) / panels[a]["close"] for a in TRADABLE if a in panels}, axis=1).reindex(closes.index).rolling(20).mean()
# Y10: drawup 60d from 60d low
C["drawup_60d"] = closes / closes.rolling(60).min() - 1.0
# Y11: worst rolling 5d return within last 20d
C["worst_5d_ret_20d"] = rets.rolling(5).sum().rolling(20).min()
# Y12: 60d up-day fraction * sign(mom60)
up60 = (rets > 0).rolling(60).mean()
C["up_frac_60d"] = (up60 - 0.5) * np.sign(mom60)
# Y13: mean overnight gap 20d
gap = pd.concat({a: panels[a]["open"] / panels[a]["close"].shift(1) - 1.0 for a in TRADABLE if a in panels}, axis=1).reindex(closes.index)
C["gap_ratio_20d"] = gap.rolling(20).mean()
# Y14: asymmetric beta 60d (dn - up)
up_beta = rolling_beta(rets, mkt_ret.clip(lower=0), 60)
dn_beta = rolling_beta(rets, mkt_ret.clip(upper=0), 60)
C["asym_beta_60d"] = dn_beta - up_beta
# Y15: vol expansion 5d vs 20d
C["vol_expansion_5_20"] = rets.rolling(5).std() / (vol20 + 1e-9) - 1.0
# Y16: 5d reversal per unit 20d vol
C["risk_adj_rev_5d"] = -(closes / closes.shift(5) - 1.0) / (vol20 + 1e-9)
# Y17: DXY beta 20d
C["dxy_beta_20d"] = rolling_beta(rets, dxy.pct_change(), 20, 12)
# Y18: intraday direction 20d
intraday = pd.concat({a: panels[a]["close"] / panels[a]["open"] - 1.0 for a in TRADABLE if a in panels}, axis=1).reindex(closes.index)
C["intraday_dir_20d"] = intraday.rolling(20).mean()
# Y19: high-low range vs close-to-close vol ratio 20d
hl = pd.concat({a: panels[a]["high"] / panels[a]["low"] - 1.0 for a in TRADABLE if a in panels}, axis=1).reindex(closes.index)
C["hl_vol_wedge_20d"] = hl.rolling(20).mean() / (vol20 + 1e-9)
# Z1: 60d skewness (longer-window tail asymmetry)
C["skew_60d"] = rets.rolling(60).skew()
# Z2: 20d return scaled by 60d drawdown depth (momentum per drawdown risk)
C["mom20_per_dd60"] = mom20 / (1.0 + (closes / closes.rolling(60).max() - 1.0).abs())
# Z3: max 1d return within 20d (upside capture)
C["max_1d_ret_20d"] = rets.rolling(20).max()
# Z4: vol ratio 10/60 (short-term vol spike vs baseline)
C["vol_ratio_10_60"] = rets.rolling(10).std() / (vol60 + 1e-9) - 1.0
# Z5: XAU beta 60d (safe-haven sensitivity)
C["xau_beta_60d"] = rolling_beta(rets, closes["XAU"].pct_change(), 60)
# Z6: NDX beta 60d (tech/growth sensitivity)
C["ndx_beta_60d"] = rolling_beta(rets, closes["NDX"].pct_change(), 60)
# Z7: 20d return rank persistence - 5d vs 20d momentum alignment
C["mom_align_5_20"] = np.sign(closes / closes.shift(5) - 1.0) * (mom20 / (vol20 + 1e-9))

results = {}
for i, (name, sig) in enumerate(C.items()):
    s, ics = report(name, sig, expected_sign=1)
    results[name] = (s, ics, sig)
    print(f"  [{i+1}/{len(C)}] {name} done {time.time()-t0:.1f}s", flush=True)

# ---------------- 3) decay + library correlation for full-pass ----------------
print("\n=== DECAY + LIBRARY CORRELATION for full-pass candidates ===", flush=True)
lib = dict(existing)
lib.update(library_signals(panels, closes, rets, vix))
lib["vol_price_corr_20"] = rets.rolling(20).corr(mkt_ret)
lib["vol_ratio_20_60"] = vol20 / vol60
lib["us10y_cond_beta_60d"] = rolling_beta(rets, closes["US10Y"].pct_change(), 60) * (closes["US10Y"].pct_change().rolling(60).mean() > 0).astype(float)
lib["mom_vs_median_60d"] = mom60 - mom60.rolling(60).median()
for name, sig in C.items():
    lib.setdefault(name, sig)

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
    dec = {}
    for h in (1, 3, 5, 10, 20):
        ic_h = rank_ic_fast(sig, forward_returns(closes, h))
        dec[h] = round(float(ic_h.mean()), 4) if len(ic_h) else np.nan
    corr, key = max_lib_corr(sig, exclude=name)
    print(f"{name:26s} decay={dec} max_abs_lib_corr={corr:.4f} (vs {key})", flush=True)

print(f"\ndone {time.time()-t0:.1f}s", flush=True)
