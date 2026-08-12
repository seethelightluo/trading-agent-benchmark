"""miner_3 batch O screen (2028-05-08) - vectorized rank-IC, no lookahead.

A) drift re-validation of 3 ACTIVE library factors (full + recent 250/500/750)
   active = vol_adj_mom_accel_20x60, dn_mkt_beta_60d, rate_beta_cn10y_60d
B) batch O candidates (fresh families, low overlap with batches A-N):
   - O_lev_effect_60d        : leverage effect corr(ret, |ret|, 60) - risk-off signature
   - O_drawdown_speed_60d    : drawdown depth / time since 60d high (steepness)
   - O_recovery_margin_60d   : close / rolling_min(close,60) - 1 (upside distance)
   - O_mom_consistency_3h    : mean sign agreement of 20/60/120d momentum (trend breadth)
   - O_rv_breakout_10d       : vol20 / rolling_min(vol20,120) - 1 (vol breakout vs own low)
   - O_cs_vol_spread_20d     : asset vol20 vs cross-sectional median (relative vol)
   - O_high_water_age_120d   : days since 120d high (time since last peak)
   - O_up_down_capture_60d   : mean ret on mkt-up days / |mean ret on mkt-down days|
   - O_gap_vol_ratio_10d     : mean |gap| / vol10 (gap persistence)
   - O_overnight_corr_20d    : rolling corr overnight vs intraday returns
   - O_mom60_lowvol_cond     : mom60 gated by low-vol regime (vol20<med250)
   - O_week_ret_skew_12w     : skew of 5d weekly returns over 60d

Gate: |IC| >= 0.0070 and |ICIR| >= 0.0840 at h=10 (15-asset universe, min_valid=8).
"""
import sys, time, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, forward_returns,
                                 coverage_metrics, turnover_rank, TRADABLE)

t0 = time.time()
panels = load_panels(days=3000)
closes = close_panel(panels)
rets = closes.pct_change()
vol20 = rets.rolling(20).std()
vol60 = rets.rolling(60).std()
opens = pd.DataFrame({a: panels[a]["open"].astype(float) for a in closes.columns}).reindex(closes.index)
highs = pd.DataFrame({a: panels[a]["high"].astype(float) for a in closes.columns}).reindex(closes.index)
lows = pd.DataFrame({a: panels[a]["low"].astype(float) for a in closes.columns}).reindex(closes.index)
mkt = rets.mean(axis=1)
print(f"panels loaded {time.time()-t0:.1f}s | closes {closes.shape} | {closes.index.min().date()}..{closes.index.max().date()}", flush=True)
print("last completed trading day:", closes.index.max().date(), flush=True)

H_ADM = 10
MIN_VALID = 8
GATE_IC, GATE_ICIR = 0.0070, 0.0840


# ---------------- vectorized rank-IC (Spearman via cross-sectional ranks) ----------------
def rank_ic_series_fast(factor_panel: pd.DataFrame, fwd: pd.DataFrame, min_valid: int = 8) -> pd.Series:
    rf = factor_panel.rank(axis=1, method="average")
    rr = fwd.rank(axis=1, method="average")
    valid = rf.notna() & rr.notna()
    nv = valid.sum(axis=1)
    rf2 = rf.where(valid)
    rr2 = rr.where(valid)
    mu_f = rf2.sum(axis=1) / nv.replace(0, np.nan)
    mu_r = rr2.sum(axis=1) / nv.replace(0, np.nan)
    cf = rf2.sub(mu_f, axis=0).fillna(0.0)
    cr = rr2.sub(mu_r, axis=0).fillna(0.0)
    ssf = (cf ** 2).sum(axis=1).astype(float)
    ssr = (cr ** 2).sum(axis=1).astype(float)
    cov = (cf * cr).sum(axis=1).astype(float)
    den = np.sqrt((ssf * ssr).replace([0.0, np.inf, -np.inf], np.nan).to_numpy(dtype=float))
    icv = (cov.to_numpy(dtype=float) / den)
    ic = pd.Series(icv, index=cov.index)
    ok = (nv >= min_valid) & (ssf > 1e-14) & (ssr > 1e-14) & ic.notna()
    return ic[ok].rename("ic")


def summarize_fast(ic_series: pd.Series):
    ic = float(ic_series.mean())
    sd = float(ic_series.std(ddof=1)) if len(ic_series) > 1 else 0.0
    icir = ic / sd if sd > 0 else 0.0
    return {"ic": ic, "icir": icir, "ic_hit_ratio": float((ic_series > 0).mean()),
            "n_ic_dates": int(len(ic_series))}


def max_library_corr(candidate: pd.DataFrame, library: dict):
    best, best_key = 0.0, None
    for name, lib_sig in library.items():
        both = pd.concat([candidate.stack().rename("cand"), lib_sig.stack().rename("lib")], axis=1).dropna()
        if len(both) < 30:
            continue
        r = float(both["cand"].corr(both["lib"]))
        if abs(r) > best:
            best, best_key = abs(r), name
    return round(best, 4), best_key


def evaluate(tag, panel, library=None):
    fwd = forward_returns(closes, H_ADM)
    ics = rank_ic_series_fast(panel, fwd, MIN_VALID)
    m = summarize_fast(ics)
    m.update(coverage_metrics(panel, min_valid=MIN_VALID))
    m["turnover_10d_rank"] = turnover_rank(panel, 10)
    if library is not None:
        corr, key = max_library_corr(panel, library)
        m["max_abs_library_correlation"] = corr
        m["max_corr_factor"] = key
    return m, ics


# ---------------- library reference signals (3 ACTIVE factors) ----------------
m20 = closes / closes.shift(20) - 1.0
m60 = closes / closes.shift(60) - 1.0
m120 = closes / closes.shift(120) - 1.0
LIBRARY = {}
LIBRARY["vol_adj_mom_accel_20x60"] = (m20 - m60) / vol20.replace(0, np.nan)
dn_mkt = pd.DataFrame(np.broadcast_to(np.minimum(mkt.values[:, None], 0.0), rets.shape), index=rets.index, columns=rets.columns)
beta60 = {}
for a in rets.columns:
    z = pd.concat([rets[a].rename("a"), dn_mkt[a].rename("m")], axis=1).dropna()
    b = z["a"].rolling(60).cov(z["m"]) / z["m"].rolling(60).var()
    beta60[a] = b
LIBRARY["dn_mkt_beta_60d"] = pd.DataFrame(beta60, index=rets.index)
cn10y_ret = closes["CN10Y"].pct_change()
rate_beta = {}
for a in rets.columns:
    z = pd.concat([rets[a].rename("a"), cn10y_ret.rename("c")], axis=1).dropna()
    b = z["a"].rolling(60).cov(z["c"]) / z["c"].rolling(60).var()
    rate_beta[a] = b
LIBRARY["rate_beta_cn10y_60d"] = pd.DataFrame(rate_beta, index=rets.index)
print("library signals built", flush=True)

results = {}

print("\n=== A) ACTIVE LIBRARY DRIFT (h=10) ===", flush=True)
for name, panel in LIBRARY.items():
    fwd = forward_returns(closes, H_ADM)
    ics = rank_ic_series_fast(panel, fwd, MIN_VALID)
    m = summarize_fast(ics)
    results[f"active_{name}"] = m
    line = f"{name}: full_ic={m['ic']:.4f} icir={m['icir']:.4f} hit={m['ic_hit_ratio']:.3f} n={m['n_ic_dates']}"
    for cut_name, cut in (("r250", closes.index[-250]), ("r500", closes.index[-500]), ("r750", closes.index[-750])):
        sub = ics[ics.index >= cut]
        if len(sub):
            icc = float(sub.mean())
            icirr = float(sub.mean() / sub.std(ddof=1)) if sub.std(ddof=1) > 0 else 0.0
            results[f"active_{name}"][f"ic_{cut_name}"] = round(icc, 4)
            results[f"active_{name}"][f"icir_{cut_name}"] = round(icirr, 4)
            line += f" | {cut_name}: ic={icc:.4f} icir={icirr:.4f}"
    print(line, flush=True)

# ---------------- batch O candidates ----------------
print(f"\n=== B) BATCH O SCREEN ({12} candidates, h=10) ===", flush=True)
cands = {}

# 1. leverage effect: corr(ret, |ret|, 60)
cands["O_lev_effect_60d"] = pd.DataFrame(
    {a: rets[a].rolling(60).corr(rets[a].abs()) for a in closes.columns}, index=rets.index)

# 2. drawdown speed: depth / time since 60d high
def days_since_max(s, win):
    return s.rolling(win).apply(lambda x: float(np.argmax(x)), raw=True).map(lambda p: win - 1 - p)

dd60 = closes / closes.rolling(60).max() - 1.0
dsm60 = pd.DataFrame({a: days_since_max(closes[a], 60) for a in closes.columns}, index=closes.index)
cands["O_drawdown_speed_60d"] = dd60 / (dsm60 + 1.0)

# 3. recovery margin: close / rolling_min(close,60) - 1
cands["O_recovery_margin_60d"] = closes / closes.rolling(60).min() - 1.0

# 4. momentum consistency across 20/60/120d
cands["O_mom_consistency_3h"] = (np.sign(m20) + np.sign(m60) + np.sign(m120)) / 3.0

# 5. vol breakout vs own 120d low
cands["O_rv_breakout_10d"] = vol20 / vol20.rolling(120).min().replace(0, np.nan) - 1.0

# 6. cross-sectional vol spread
med_v20 = vol20.median(axis=1)
std_v20 = vol20.std(axis=1)
cands["O_cs_vol_spread_20d"] = vol20.sub(med_v20, axis=0).div(std_v20.replace(0, np.nan), axis=0)

# 7. days since 120d high
cands["O_high_water_age_120d"] = pd.DataFrame(
    {a: days_since_max(closes[a], 120) for a in closes.columns}, index=closes.index)

# 8. up/down capture: mean ret on mkt-up days / |mean ret on mkt-down days| (60d)
up_mask = (mkt > 0).astype(float)
dn_mask = (mkt < 0).astype(float)
mean_up = (rets.mul(up_mask, axis=0)).rolling(60).sum() / up_mask.rolling(60).sum().replace(0, np.nan)
mean_dn = (rets.mul(dn_mask, axis=0)).rolling(60).sum() / dn_mask.rolling(60).sum().replace(0, np.nan)
cands["O_up_down_capture_60d"] = (mean_up / mean_dn.abs().replace(0, np.nan)).clip(-10, 10)

# 9. gap / vol ratio
gap = opens / closes.shift(1) - 1.0
cands["O_gap_vol_ratio_10d"] = gap.abs().rolling(10).mean() / rets.rolling(10).std().replace(0, np.nan)

# 10. overnight vs intraday corr (20d)
intraday = closes / opens - 1.0
overnight = opens / closes.shift(1) - 1.0
cands["O_overnight_corr_20d"] = pd.DataFrame(
    {a: overnight[a].rolling(20).corr(intraday[a]) for a in closes.columns}, index=rets.index)

# 11. mom60 gated by low-vol regime (vol20 < rolling median vol20 over 250)
volmed250 = vol20.rolling(250).median()
cands["O_mom60_lowvol_cond"] = m60 * (vol20 <= volmed250).astype(float)

# 12. skew of weekly (5d) returns over 12 weeks
wret = rets.rolling(5).sum()
cands["O_week_ret_skew_12w"] = wret.rolling(60).skew()

for tag, panel in cands.items():
    m, ics = evaluate(tag, panel, library=LIBRARY)
    results[tag] = m
    for cut_name, cut in (("r250", closes.index[-250]), ("r500", closes.index[-500]), ("r750", closes.index[-750])):
        sub = ics[ics.index >= cut]
        if len(sub):
            results[tag][f"ic_{cut_name}"] = round(float(sub.mean()), 4)
            results[tag][f"icir_{cut_name}"] = round(float(sub.mean() / sub.std(ddof=1)), 4) if sub.std(ddof=1) > 0 else 0.0
    print(f"done {tag} {time.time()-t0:.1f}s", flush=True)

df = pd.DataFrame(results).T
df["pass"] = (df["ic"].abs() >= GATE_IC) & (df["icir"].abs() >= GATE_ICIR)
cols = ["ic", "icir", "ic_hit_ratio", "n_ic_dates", "ic_r250", "ic_r500", "ic_r750",
        "coverage_asset_days", "coverage_dates_ge8", "turnover_10d_rank",
        "max_abs_library_correlation", "max_corr_factor", "pass"]
print("\n=== FULL SCREEN (h=10, min_valid=8) ===")
print(df[cols].to_string(float_format=lambda x: f"{x:.4f}"))
print(f"\nPASSERS ({int(df['pass'].sum())}):", list(df.index[df["pass"]]), flush=True)

with open("scripts/_miner3_batchO_results.json", "w") as f:
    json.dump(results, f, indent=1, default=str)
print(f"\nelapsed {time.time()-t0:.1f}s", flush=True)
