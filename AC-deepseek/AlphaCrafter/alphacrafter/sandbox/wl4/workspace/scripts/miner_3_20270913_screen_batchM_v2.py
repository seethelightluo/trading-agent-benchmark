"""miner_3 batch M v2 screen (2027-09-13) - vectorized rank-IC, no lookahead.

A) drift re-validation of 4 active library factors (full + recent 250/500/750)
B) batch M candidates (fresh families, low overlap with batches A-L):
   - trend acceleration: mom_accel_20x60, mom_120d_skip30, mom_90d_skip10
   - weekly win-rate proxy (12w), MFI_14d, TRIX_15d
   - signed gap/overnight return: overnight_ret_20d, gap_mom_20d
   - vol-scaled range position: high_prox_atr_20d, low_prox_atr_20d
   - risk asymmetry: max_gain_loss_ratio_60d
   - idiosyncratic risk: idio_vol_60d, r2_mkt_60d, pair_corr_mean_60d
   - relative (safe-haven) momentum: rel_mom_vs_xau_60d, rel_mom_vs_us10y_60d
   - window variants of active: dn_beta_20d, vol_price_corr_60d
   - range vol of range: vol_of_range_20d; price-volume level corr 60d

Gate: |IC| >= 0.0070 and |ICIR| >= 0.0840 at h=10 (15-asset universe, min_valid=8).
"""
import sys, time, json, math
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, forward_returns,
                                 coverage_metrics, turnover_rank, max_library_corr,
                                 TRADABLE)

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

# ---------------- vectorized rank-IC (Spearman via cross-sectional ranks) ----------------
def rank_ic_series_fast(factor_panel: pd.DataFrame, fwd: pd.DataFrame, min_valid: int = 8) -> pd.Series:
    """Per-date Spearman rank IC, vectorized. NaN pairs dropped per date."""
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
    ssf = (cf ** 2).sum(axis=1)
    ssr = (cr ** 2).sum(axis=1)
    cov = (cf * cr).sum(axis=1)
    ic = cov / np.sqrt(ssf * ssr).replace(0, np.nan)
    ok = (nv >= min_valid) & (ssf > 1e-14) & (ssr > 1e-14) & ic.notna()
    return ic[ok].rename("ic")


def summarize_fast(ic_series: pd.Series):
    ic = float(ic_series.mean())
    sd = float(ic_series.std(ddof=1)) if len(ic_series) > 1 else 0.0
    icir = ic / sd if sd > 0 else 0.0
    return {"ic": ic, "icir": icir, "ic_hit_ratio": float((ic_series > 0).mean()),
            "n_ic_dates": int(len(ic_series))}


def evaluate(tag, panel):
    fwd = forward_returns(closes, H_ADM)
    ics = rank_ic_series_fast(panel, fwd, MIN_VALID)
    m = summarize_fast(ics)
    m.update(coverage_metrics(panel, min_valid=MIN_VALID))
    m["turnover_10d_rank"] = turnover_rank(panel, 10)
    m["decay_ic_by_horizon"] = {}
    for h in (1, 2, 3, 5, 10, 20):
        fh = forward_returns(closes, h)
        ih = rank_ic_series_fast(panel, fh, MIN_VALID)
        if len(ih):
            m["decay_ic_by_horizon"][str(h)] = round(float(ih.mean()), 4)
    corr, key = max_library_corr(panel, LIBRARY)
    m["max_abs_library_correlation"] = corr
    m["max_corr_factor"] = key
    return m, ics


def rolling_beta(asset_ret, driver_ret, win=60, min_obs=40):
    beta = {}
    for a in asset_ret.columns:
        z = pd.concat([asset_ret[a].rename("a"), driver_ret.rename("m")], axis=1).dropna()
        cov = z["a"].rolling(win).cov(z["m"])
        var = z["m"].rolling(win).var()
        beta[a] = (cov / var).where(z["m"].rolling(win).count() >= min_obs)
    return pd.DataFrame(beta, index=asset_ret.index)


def rolling_corr_fast(a, b, win=60, min_obs=40):
    n = a.rolling(win).count()
    cov = (a * b).rolling(win).mean() - a.rolling(win).mean() * b.rolling(win).mean()
    den = a.rolling(win).std() * b.rolling(win).std()
    out = (cov / den.replace(0, np.nan)).where(n >= min_obs)
    return out


# ---------- library factor signal artifacts (same definitions as library) ----------
def lib_vol_price_corr_20():
    out = {}
    for a in closes.columns:
        out[a] = rolling_corr_fast(rets[a], vol_panel[a], 20, 15)
    return pd.DataFrame(out, index=rets.index)


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
print(f"library signals computed {time.time()-t0:.1f}s", flush=True)

# ---------- batch M candidates ----------
cands = {}
close_shift = closes.shift(1)
hl = highs - lows
hl_mean20 = hl.rolling(20).mean().replace(0, np.nan)

# trend acceleration / momentum variants
m20 = closes / closes.shift(20) - 1.0
m60 = closes / closes.shift(60) - 1.0
cands["M_mom_accel_20x60"] = m20 - m60
cands["M_mom_120d_skip30"] = closes.shift(30) / closes.shift(150) - 1.0
cands["M_mom_90d_skip10"] = closes.shift(10) / closes.shift(100) - 1.0

# weekly win-rate proxy (fraction of positive trailing 5d windows over ~60d)
cands["M_week_win_rate_12w"] = (rets.rolling(5).sum() > 0).astype(float).rolling(60).mean()

# Money Flow Index 14d
tp = (highs + lows + closes) / 3.0
tp_chg = tp.diff()
mf = tp * vol_panel
mf_pos = mf.where(tp_chg > 0, 0.0)
mf_neg = mf.where(tp_chg < 0, 0.0)
mfi = 100.0 - 100.0 / (1.0 + mf_pos.rolling(14).sum() / mf_neg.rolling(14).sum().replace(0, np.nan))
cands["M_mfi_14d"] = (mfi - 50.0) / 50.0

# TRIX 15d (triple EMA momentum)
e1 = closes.ewm(span=15, adjust=False).mean()
e2 = e1.ewm(span=15, adjust=False).mean()
e3 = e2.ewm(span=15, adjust=False).mean()
cands["M_trix_15d"] = e3.pct_change(15)

# signed overnight/gap returns
overnight = opens / close_shift - 1.0
cands["M_overnight_ret_20d"] = overnight.rolling(20).mean()
cands["M_gap_mom_20d"] = overnight.rolling(20).sum()

# vol-scaled range position
cands["M_high_prox_atr_20d"] = (closes.rolling(20).max() - closes) / hl_mean20
cands["M_low_prox_atr_20d"] = (closes - closes.rolling(20).min()) / hl_mean20

# risk asymmetry: max 5d gain vs |min 5d loss| over 60d
g5 = rets.rolling(5).sum()
maxg = g5.rolling(60).max()
minl = g5.rolling(60).min()
cands["M_max_gain_loss_ratio_60d"] = (maxg / (-minl).replace(0, np.nan)).clip(upper=10)

# idiosyncratic vol and market R^2 (60d)
var_a = rets.rolling(60).var()
var_m = mkt.rolling(60).var().replace(0, np.nan)
cov_am = rets.rolling(60).cov(mkt)
idio_var = (var_a - cov_am ** 2 / var_m).clip(lower=0)
cands["M_idio_vol_60d"] = np.sqrt(idio_var)
corr_am2 = (cov_am ** 2 / (var_a * var_m)).clip(0, 1)
cands["M_r2_mkt_60d"] = corr_am2

# mean pairwise correlation with other 14 assets (60d)
pair_sum = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
pair_cnt = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
for i, a in enumerate(rets.columns):
    for b in rets.columns[i + 1:]:
        c = rolling_corr_fast(rets[a], rets[b], 60, 40)
        pair_sum[a] = pair_sum[a].add(c, fill_value=0)
        pair_sum[b] = pair_sum[b].add(c, fill_value=0)
        pair_cnt[a] = pair_cnt[a].add(c.notna().astype(float), fill_value=0)
        pair_cnt[b] = pair_cnt[b].add(c.notna().astype(float), fill_value=0)
cands["M_pair_corr_mean_60d"] = pair_sum / pair_cnt.replace(0, np.nan)

# relative momentum vs safe havens
xau = closes["XAU"]
cands["M_rel_mom_vs_xau_60d"] = (closes / closes.shift(60)) / (xau / xau.shift(60)) - 1.0
us10y = closes["US10Y"]
cands["M_rel_mom_vs_us10y_60d"] = (closes / closes.shift(60)) / (us10y / us10y.shift(60)) - 1.0

# window variants of active factors
dn = mkt.where(mkt < 0)
cands["M_dn_beta_20d"] = rolling_beta(rets, dn, 20, 15)
vp60 = {}
for a in closes.columns:
    vp60[a] = rolling_corr_fast(rets[a], vol_panel[a], 60, 40)
cands["M_vol_price_corr_60d"] = pd.DataFrame(vp60, index=rets.index)

# range vol of range + price-volume level corr
cands["M_vol_of_range_20d"] = (hl / closes).rolling(20).std()
pv60 = {}
for a in closes.columns:
    pv60[a] = rolling_corr_fast(np.log(closes[a]), np.log(vol_panel[a].replace(0, np.nan)), 60, 40)
cands["M_price_vol_level_corr_60d"] = pd.DataFrame(pv60, index=rets.index)

print(f"candidates defined {time.time()-t0:.1f}s ({len(cands)})", flush=True)

# ---------- evaluation ----------
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

print(f"\n=== B) BATCH M SCREEN ({len(cands)} candidates, h=10) ===", flush=True)
for tag, panel in cands.items():
    m, ics = evaluate(tag, panel)
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
print(f"\nelapsed {time.time()-t0:.1f}s", flush=True)
