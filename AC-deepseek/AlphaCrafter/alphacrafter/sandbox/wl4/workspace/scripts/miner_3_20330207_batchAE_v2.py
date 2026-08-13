"""miner_3 batch AE v2 (2033-02-07) - optimized numpy rank-IC screen + re-validation.

Visible data through previous completed trading day (2033-02-04). 15-instrument
universe, min_valid=8. Admission gates: |IC| >= 0.0070 and |ICIR| >= 0.0840 at h=10.
Uses fast numpy rank-IC to avoid pandas per-date overhead. No live-account interaction.
"""
import sys, time, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.insert(0, "scripts")
from factor_research_lib import (load_panels, close_panel, forward_returns,
                                 coverage_metrics, turnover_rank, TRADABLE)

t0 = time.time()
panels = load_panels(days=4000)
closes = close_panel(panels)
rets = closes.pct_change()
mkt_ret = rets.mean(axis=1)
print(f"closes {closes.shape} | {closes.index.min().date()}..{closes.index.max().date()} | {time.time()-t0:.1f}s", flush=True)

hi = pd.concat({a: panels[a]["high"].astype(float) for a in TRADABLE if a in panels}, axis=1).sort_index().reindex(closes.index)
lo = pd.concat({a: panels[a]["low"].astype(float) for a in TRADABLE if a in panels}, axis=1).sort_index().reindex(closes.index)
vol_panel = pd.concat({a: panels[a]["volume"].astype(float) for a in TRADABLE if a in panels}, axis=1).sort_index().reindex(closes.index)

H = 10
fwd = forward_returns(closes, H)


def fast_rank_ic(sig, fwd_arr, dates, min_valid=8):
    """numpy Spearman rank IC per date; returns (dates_list, ic_array)."""
    out_dates, out_ics = [], []
    S = sig.values.astype(float)
    F = fwd_arr.astype(float)
    for i, dt in enumerate(dates):
        if i >= len(F):
            break
        s = S[i]
        f = F[i]
        m = ~(np.isnan(s) | np.isnan(f))
        if m.sum() < min_valid:
            continue
        sv, fv = s[m], f[m]
        if np.std(sv) < 1e-14 or np.std(fv) < 1e-14:
            continue
        rs = pd.Series(sv).rank().values
        rf = pd.Series(fv).rank().values
        ic = np.corrcoef(rs, rf)[0, 1]
        if not np.isnan(ic):
            out_dates.append(dt)
            out_ics.append(ic)
    return out_dates, np.array(out_ics)


def summarize(ics, expected_sign=1):
    ic = ics.mean()
    icir = ic / ics.std(ddof=1) if ics.std(ddof=1) > 0 else 0.0
    hit = float((np.sign(ics) == expected_sign).mean())
    return {"ic": round(float(ic), 4), "icir": round(float(icir), 4),
            "ic_hit_ratio": round(hit, 3), "n_ic_dates": int(len(ics)),
            "ic_std": round(float(ics.std(ddof=1)), 4)}


def report(name, sig, expected_sign=1):
    dates, ics = fast_rank_ic(sig, fwd.values, closes.index)
    s = summarize(ics, expected_sign)
    cov = coverage_metrics(sig)
    to = turnover_rank(sig, 10)
    recent = {}
    for w in (63, 126, 252, 504):
        sub = ics[-w:]
        if len(sub) > 2:
            mm, ss = sub.mean(), sub.std(ddof=1)
            recent[w] = (mm, mm / ss if ss and ss > 0 else np.nan)
        else:
            recent[w] = (np.nan, np.nan)
    flag = "  <== FULL-PASS" if (abs(s["ic"]) >= 0.0070 and abs(s["icir"]) >= 0.0840) else ""
    print(f"{name:26s} IC={s['ic']:+.4f} ICIR={s['icir']:+.3f} hit={s['ic_hit_ratio']:.2f} n={s['n_ic_dates']} "
          f"| r63=({recent[63][0]:+.3f},{recent[63][1]:+.2f}) r126=({recent[126][0]:+.3f},{recent[126][1]:+.2f}) "
          f"r252=({recent[252][0]:+.3f},{recent[252][1]:+.2f}) "
          f"cov={cov['coverage_dates_ge8']:.2f} to={to if to is not None else float('nan'):.2f}{flag}", flush=True)
    return s, ics


# ---------------- 1) RE-VALIDATE current effective factors ----------------
print("\n=== RE-VALIDATION of current effective factors (full + recent) ===", flush=True)
existing = {}
existing["vol_adj_mom_accel_20x60"] = ((closes / closes.shift(20) - 1) - (closes / closes.shift(60) - 1)) / rets.rolling(20).std()
dn_mask = (mkt_ret < 0).astype(float)
existing["dn_mkt_beta_60d"] = (rets.where(dn_mask > 0)).rolling(60, min_periods=40).cov(mkt_ret.where(dn_mask > 0)) / mkt_ret.where(dn_mask > 0).rolling(60, min_periods=40).var()
beta_cn = {}
for a in rets.columns:
    z = pd.concat([rets[a].rename("y"), closes["CN10Y"].pct_change().rename("x")], axis=1).dropna()
    b = z["y"].rolling(60, min_periods=40).cov(z["x"]) / z["x"].rolling(60, min_periods=40).var()
    beta_cn[a] = b
existing["rate_beta_cn10y_60d"] = pd.DataFrame(beta_cn, index=rets.index)

for name, sig in existing.items():
    report(name, sig, expected_sign=1 if name != "rate_beta_cn10y_60d" else -1)

# ---------------- 2) CANDIDATE SCREEN (batch AE) ----------------
print("\n=== CANDIDATE SCREEN (batch AE, full history) ===", flush=True)
C = {}
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


def rolling_corr(y, x, win=20, min_obs=15):
    out = {}
    for a in y.columns:
        z = pd.concat([y[a].rename("y"), x.rename("x")], axis=1).dropna()
        c = z["y"].rolling(win).corr(z["x"])
        out[a] = c.where(z["x"].rolling(win).count() >= min_obs)
    return pd.DataFrame(out, index=y.index)


# A. idiosyncratic momentum: 60d regression alpha t-stat vs equal-w market
beta60 = rolling_beta(rets, mkt_ret, 60)
alpha_mean = rets.rolling(60).mean() - beta60 * mkt_ret.rolling(60).mean()
resid_var = rets.rolling(60).var() - beta60 ** 2 * mkt_ret.rolling(60).var()
resid_var = resid_var.clip(lower=1e-14)
C["resid_mom_60d"] = alpha_mean * np.sqrt(60) / np.sqrt(resid_var)

# B. trend consistency: fraction of up days over 60d
C["upday_ratio_60d"] = (rets > 0).rolling(60).mean()

# C. 52-week high proximity
C["hi_250_prox"] = closes / closes.rolling(250).max() - 1.0

# D. long-term trend vs 200d SMA
C["sma200_dist"] = closes / closes.rolling(200).mean() - 1.0

# E. idiosyncratic vol (20d residual std from mkt regression)
beta20 = rolling_beta(rets, mkt_ret, 20, min_obs=15)
resid_var20 = rets.rolling(20).var() - beta20 ** 2 * mkt_ret.rolling(20).var()
C["idio_vol_20d"] = np.sqrt(resid_var20.clip(lower=1e-14))

# F. correlation momentum: 60d mkt-corr minus its value 20d ago
corr60 = rolling_corr(rets, mkt_ret, 60, min_obs=40)
C["corr_delta_60"] = corr60 - corr60.shift(20)

# G. volume-confirmed momentum: 20d return * volume trend z (20/60 means)
vol_tr = vol_panel.rolling(20).mean() / vol_panel.rolling(60).mean() - 1.0
mom20 = closes / closes.shift(20) - 1.0
C["vol_conf_mom_20x60"] = mom20 * vol_tr

# H. long-horizon drawdown depth (250d): min(close/rollmax-1) over 250d
roll_max250 = closes.rolling(250).max()
dd = closes / roll_max250 - 1.0
C["max_dd_250d"] = dd.rolling(250).min()

# I. tech-spillover beta (60d)
C["ndx_beta_60d"] = rolling_beta(rets, rets["NDX"], 60)

# J. 20d vol-adj reversal
C["vol_adj_reversal_20d"] = -(closes / closes.shift(20) - 1.0) / vol20

# library signals for correlation
lib = dict(existing)
from factor_research_lib import max_library_corr

print(f"{len(C)} candidates; time {time.time()-t0:.1f}s", flush=True)
results = {}
for i, (name, sig) in enumerate(C.items()):
    s, ics = report(name, sig, expected_sign=1)
    m = dict(s)
    m["turnover_10d_rank"] = turnover_rank(sig, 10)
    m["coverage_asset_days"] = coverage_metrics(sig)["coverage_asset_days"]
    m["coverage_dates_ge8"] = coverage_metrics(sig)["coverage_dates_ge8"]
    corr, key = max_library_corr(sig, lib)
    m["max_abs_library_correlation"] = corr
    m["max_corr_factor"] = key
    for w in (63, 252):
        sub = ics[-w:]
        if len(sub) > 2:
            mm, ss = sub.mean(), sub.std(ddof=1)
            m[f"ic_last{w}d"] = round(float(mm), 4)
            m[f"icir_last{w}d"] = round(float(mm / ss), 3) if ss and ss > 0 else None
        else:
            m[f"ic_last{w}d"] = None
            m[f"icir_last{w}d"] = None
    results[name] = m
    passed = abs(m["ic"]) >= 0.0070 and abs(m["icir"]) >= 0.0840
    print(f"   | libcorr={m['max_abs_library_correlation']:.3f}({m['max_corr_factor']})"
          f" | cov_asset={m['coverage_asset_days']:.3f} cov_d8={m['coverage_dates_ge8']:.3f}"
          f" | 1y_IC={m['ic_last252d']} 1y_ICIR={m['icir_last252d']} 63d_IC={m['ic_last63d']}", flush=True)

print("\n--- summary table ---")
for name, m in results.items():
    passed = abs(m["ic"]) >= 0.0070 and abs(m["icir"]) >= 0.0840
    print(f"{name:24s} IC={m['ic']:+.4f} ICIR={m['icir']:+.3f} {'PASS' if passed else '---'}")
print(f"total time {time.time()-t0:.1f}s", flush=True)
