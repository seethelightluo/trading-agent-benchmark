"""
miner_1 | 2026-07-30 cycle | batch-2 cross-asset factor exploration

Motivation: broaden the factor library (currently mom_10d_skip5, mom_120d_skip5,
vol_of_vol20x60, vix_beta_cond_60x20) with macro-beta, distributional-risk and
trend-quality families.

Methodology fix vs batch-1: factor values and forward returns are computed
per-asset on each asset's OWN calendar (shift contract), then aligned for daily
cross-sectional Spearman rank IC. The batch-1 union-index approach produced only
3 valid IC dates because weekend/holiday NaNs broke rolling windows.

Universe: 15 tradable cross-asset instruments; dates with >=8 valid values count
as IC observations. Validation window 2020-01-01..2026-07-29 (data visible through
the last completed trading day; no lookahead).
Admission gates (15-asset cross-asset universe): abs(IC)>=0.0070 and
abs(ICIR)>=0.0840 at h=10.
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from miner_1_20260730_helpers import WATCH, MACRO, MIN_DATE, MAX_DATE, load_close, max_library_correlation

# ---------------- data plumbing ----------------
def C(a):
    df = load_close(a)
    if df is None:
        return None
    s = df["close"].astype(float)
    return s[s.index >= MIN_DATE]

def fwd(h):
    out = {}
    for a in WATCH:
        s = C(a)
        if s is not None:
            out[a] = s.shift(-h) / s - 1.0
    return pd.DataFrame(out).sort_index()

macro = {}
for m in MACRO:
    df = load_close(m)
    if df is not None:
        s = df["close"].astype(float)
        macro[m] = s[s.index >= MIN_DATE]
mret = {m: macro[m].pct_change() for m in macro}

FWD = {h: fwd(h) for h in (1, 2, 3, 5, 10, 20)}

# ---------------- helpers ----------------
def beta_to(r_a, r_b, win):
    """Rolling beta of r_a (asset returns) to r_b (benchmark returns, reindexed to r_a calendar)."""
    df = pd.concat([r_a.rename("x"), r_b.reindex(r_a.index).rename("y")], axis=1).dropna()
    if len(df) < win + 5:
        return pd.Series(np.nan, index=r_a.index)
    return (df["x"].rolling(win).cov(df["y"]) / df["y"].rolling(win).var()).reindex(r_a.index)

def build(fn):
    out = {}
    for a in WATCH:
        s = C(a)
        if s is None:
            continue
        r = s.pct_change()
        out[a] = fn(s, r)
    return pd.DataFrame(out).sort_index()

def rank_ic(panel, fwd_frame, min_valid=8):
    idx = panel.index.intersection(fwd_frame.index)
    ics, dates, ns = [], [], []
    for t in idx:
        f = panel.loc[t]
        rr = fwd_frame.loc[t]
        m = f.notna() & rr.notna()
        n = int(m.sum())
        if n < min_valid:
            continue
        ic = f[m].rank().corr(rr[m].rank())
        if np.isfinite(ic):
            ics.append(ic); dates.append(t); ns.append(n)
    return pd.Series(ics, index=pd.DatetimeIndex(dates)), ns

def turn10(panel):
    r = panel.rank(axis=1)
    return float(r.diff(10).abs().mean().mean())

def cvr(panel):
    cov_ad = float(panel.notna().sum().sum() / (panel.shape[0] * max(panel.shape[1], 1)))
    cov_ge8 = float((panel.notna().sum(axis=1) >= 8).mean())
    return cov_ad, cov_ge8

def full_report(name, panel):
    ic, ns = rank_ic(panel, FWD[10])
    if len(ic) < 30:
        print(f"{name:<26} insufficient (n_ic={len(ic)})")
        return None
    mean_ic = float(ic.mean())
    std_ic = float(ic.std(ddof=1)) if len(ic) > 1 else 0.0
    icir = mean_ic / std_ic if std_ic > 0 else 0.0
    hit = float((np.sign(ic) == np.sign(mean_ic)).mean())
    cov_ad, cov_ge8 = cvr(panel)
    to = turn10(panel)
    decay = {str(h): round(float(rank_ic(panel, FWD[h])[0].mean()), 4) for h in (1, 2, 3, 5, 10, 20)}
    max_r, rho = max_library_correlation(panel)
    passed = abs(mean_ic) >= 0.0070 and abs(icir) >= 0.0840
    print(f"{name:<26}{mean_ic:>8.4f}{icir:>8.4f}{hit:>7.3f}{len(ic):>8d}{int(np.mean(ns)):>6d}"
          f"{cov_ad:>7.3f}{cov_ge8:>7.3f}{to:>7.3f}{max_r:>7.3f}  {decay['1']}/{decay['3']}/{decay['5']}/{decay['10']}/{decay['20']}  {'PASS' if passed else '-'}")
    return {"name": name, "ic": mean_ic, "icir": icir, "hit": hit, "n_ic_dates": len(ic),
            "mean_n_valid": int(np.mean(ns)), "coverage_asset_days": cov_ad, "coverage_dates_ge8": cov_ge8,
            "turnover_10d_rank": to, "decay": decay, "max_abs_library_correlation": max_r,
            "passed": passed}

# ---------------- calibration (methodology check vs persisted records) ----------------
cal = {}
cal["CAL_mom_10d_skip5"] = build(lambda s, r: s.shift(5) / s.shift(15) - 1.0)
cal["CAL_mom_120d_skip5"] = build(lambda s, r: s.shift(5) / s.shift(125) - 1.0)

# ---------------- candidate factor families ----------------
fams = {}
fams["dxy_beta_60d"] = build(lambda s, r: beta_to(r, mret["DXY"], 60))
fams["jpy_beta_60d"] = build(lambda s, r: beta_to(r, mret["USDJPY"], 60))
fams["vix_beta_plain_60d"] = build(lambda s, r: beta_to(r, mret["VIX"], 60))
fams["rate_beta_60d"] = build(lambda s, r: beta_to(r, C("US10Y").pct_change(), 60))
fams["crypto_beta_60d"] = build(lambda s, r: beta_to(r, C("BTC").pct_change(), 60))
fams["oil_beta_60d"] = build(lambda s, r: beta_to(r, C("WTI").pct_change(), 60))
fams["skew_20d"] = build(lambda s, r: r.rolling(20).skew())
fams["vol_ratio_5_60"] = build(lambda s, r: r.rolling(5).std() / r.rolling(60).std())
fams["downside_vol_60d"] = build(lambda s, r: r.clip(upper=0).rolling(60).apply(lambda v: np.sqrt(np.mean(v * v)), raw=True))
fams["trend_quality_60d"] = build(lambda s, r: (r > 0).rolling(60).mean())
fams["range_pos_20d"] = build(lambda s, r: (s - s.rolling(20).min()) / (s.rolling(20).max() - s.rolling(20).min()))
fams["autocorr_10d"] = build(lambda s, r: r.rolling(10).apply(lambda v: pd.Series(v).autocorr() if len(v) > 3 else np.nan, raw=True))
fams["max_drawdown_60d"] = build(lambda s, r: (1 + r).rolling(60).apply(
    lambda v: np.prod(v) / np.maximum.accumulate(np.cumprod(np.r_[1, v]))[-1] - 1, raw=True))
fams["mom60_vol_adj"] = build(lambda s, r: r.rolling(60).sum() / r.rolling(60).std())
fams["dxy_corr_change_20_60"] = build(lambda s, r: r.rolling(20).corr(mret["DXY"].reindex(r.index))
                                      - r.rolling(60).corr(mret["DXY"].reindex(r.index)))

# ---------------- run ----------------
print(f"window: {MIN_DATE} .. {MAX_DATE}  assets: {len(WATCH)}  macro loaded: {list(macro)}")
print("factor                        IC    ICIR    hit n_dates  nv  cov_ad cov_ge8  turn  rho_lib  decay1/3/5/10/20  gate")
results = {}
for name, panel in {**cal, **fams}.items():
    r = full_report(name, panel)
    if r:
        results[name] = r

Path("scripts/batch2_results.json").write_text(json.dumps(results, indent=1, default=str))
print("\nsaved scripts/batch2_results.json")