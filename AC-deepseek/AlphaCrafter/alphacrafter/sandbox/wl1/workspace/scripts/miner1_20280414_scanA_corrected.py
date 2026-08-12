"""miner_1 2028-04-14: quantify union-calendar rolling artifact and re-validate
scanA candidates using per-asset calendars (canonical computation: factor computed
on each asset's own series, then reindexed to the union panel for cross-sectional IC)."""
import pickle
import numpy as np
import pandas as pd

panel = pickle.load(open("scripts/panel_cache.pkl", "rb"))
C, V, M = panel["close"], panel["vol"], panel["macro"]
ret = C.pct_change()

ONLINE_START = pd.Timestamp("2026-07-16")

# ---------------- helpers ----------------
def per_asset_apply(fn_series, name="factor"):
    """Apply a series->series transform to each asset on its own calendar, then reindex."""
    out = {}
    for s in C.columns:
        sser = C[s].dropna()
        out[s] = fn_series(sser).reindex(C.index)
    return pd.DataFrame(out)

def fwd_ret(days=1):
    return C.shift(-days) / C - 1.0

def ic_metrics(fac, fwd, min_names=8, sub=None):
    dates = fac.index.intersection(fwd.index)
    if sub is not None:
        dates = dates[(dates >= sub[0]) & (dates <= sub[1])]
    ics, obs = [], []
    for dt in dates:
        f = fac.loc[dt].dropna()
        r = fwd.loc[dt].reindex(f.index).dropna()
        common = f.index.intersection(r.index)
        if len(common) < min_names:
            continue
        x = f[common].astype(float).rank()
        y = r[common].astype(float).rank()
        if x.std() == 0 or y.std() == 0:
            continue
        ic = np.corrcoef(x, y)[0, 1]
        if np.isfinite(ic):
            ics.append(ic); obs.append(len(common))
    ics = np.array(ics)
    if len(ics) == 0:
        return dict(n=0, ic=np.nan, icir=np.nan, hit=np.nan, std=np.nan)
    return dict(n=int(len(ics)), ic=float(ics.mean()),
                icir=float(ics.mean() / ics.std()) if ics.std() > 0 else np.nan,
                hit=float((ics > 0).mean()), std=float(ics.std()))

def coverage(fac):
    n_valid = int(fac.notna().sum().sum())
    n_total = int(C.notna().sum().sum())
    return n_valid / n_total if n_total else np.nan

def per_asset_valid_counts(fac):
    return {s: int(fac[s].notna().sum()) for s in C.columns}

# ---------------- artifact quantification: dd_dist_60 ----------------
print("=" * 100)
print("ARTIFACT CHECK: dd_dist_60 (1 - close/rolling_max(close,60))")
print("=" * 100)
naive = 1.0 - C / C.rolling(60).max()
corrected = per_asset_apply(lambda s: 1.0 - s / s.rolling(60).max(), "dd_dist_60")
print("naive valid per asset:", per_asset_valid_counts(naive))
print("corr  valid per asset:", per_asset_valid_counts(corrected))
print(f"naive coverage: {coverage(naive):.4f} | corrected coverage: {coverage(corrected):.4f}")

# ---------------- candidate definitions (corrected, per-asset calendar) ----------------
def f_vscaled_mom(sser):
    mom = sser.shift(5) / sser.shift(125) - 1.0
    vol = sser.pct_change().rolling(60).std()
    return mom / vol

def f_mom_ma20_conf(sser):
    mom = sser.shift(5) / sser.shift(125) - 1.0
    ma20 = sser.rolling(20).mean()
    return mom * (sser > ma20)

def f_ma20_dist(sser):
    ma20 = sser.rolling(20).mean()
    return sser / ma20 - 1.0

def f_dd_dist_60(sser):
    return 1.0 - sser / sser.rolling(60).max()

def f_neg_skew_60(sser):
    return -sser.pct_change().rolling(60).skew()

def f_vol_ratio_60_20(sser):
    r = sser.pct_change()
    return r.rolling(60).std() / r.rolling(20).std()

def f_macd_hist(sser):
    e12 = sser.ewm(span=12, adjust=False).mean()
    e26 = sser.ewm(span=26, adjust=False).mean()
    return (e12 - e26) / sser

def f_ma60_slope(sser):
    ma60 = sser.rolling(60).mean()
    return ma60 / ma60.shift(20) - 1.0

def f_dd_recovery_120(sser):
    """drawdown distance on 120d window (slower mean-reversion signal)."""
    return 1.0 - sser / sser.rolling(120).max()

CANDIDATES = {
    "vscaled_mom_120x60": f_vscaled_mom,
    "mom_ma20_conf": f_mom_ma20_conf,
    "ma20_dist": f_ma20_dist,
    "dd_dist_60": f_dd_dist_60,
    "neg_skew_60": f_neg_skew_60,
    "vol_ratio_60_20": f_vol_ratio_60_20,
    "macd_hist": f_macd_hist,
    "ma60_slope_20": f_ma60_slope,
    "dd_recovery_120": f_dd_recovery_120,
}

fwd1 = fwd_ret(1)
fwd5 = fwd_ret(5)
fwd10 = fwd_ret(10)

print("=" * 100)
print("CORRECTED SCAN (per-asset calendar), full 2020-01-01..2028-04-13 + online 2026-07-16..")
print("=" * 100)
print(f"{'factor':<20} {'cov':>5} {'IC1':>8} {'ICIR1':>7} {'hit1':>5} {'n1':>5} {'IC5':>8} {'IC10':>8} | {'IC1on':>8} {'ICIR1on':>7} {'n1on':>5} {'IC5on':>8} {'IC10on':>8}")
print("-" * 130)
for name, fn in CANDIDATES.items():
    fac = per_asset_apply(fn, name)
    cov = coverage(fac)
    m1 = ic_metrics(fac, fwd1)
    m5 = ic_metrics(fac, fwd5)
    m10 = ic_metrics(fac, fwd10)
    m1on = ic_metrics(fac, fwd1, sub=(ONLINE_START, C.index.max()))
    m5on = ic_metrics(fac, fwd5, sub=(ONLINE_START, C.index.max()))
    m10on = ic_metrics(fac, fwd10, sub=(ONLINE_START, C.index.max()))
    print(f"{name:<20} {cov:>5.3f} {m1['ic']:>+8.4f} {m1['icir']:>+7.3f} {m1['hit']:>5.3f} {m1['n']:>5d} "
          f"{m5['ic']:>+8.4f} {m10['ic']:>+8.4f} | {m1on['ic']:>+8.4f} {m1on['icir']:>+7.3f} {m1on['n']:>5d} {m5on['ic']:>+8.4f} {m10on['ic']:>+8.4f}")

# decay for the top two drawdown-distance variants
print("\nDECAY (corrected dd_dist_60 and dd_recovery_120):")
for name in ["dd_dist_60", "dd_recovery_120"]:
    fac = per_asset_apply(CANDIDATES[name], name)
    dec = {}
    for h in [1, 2, 3, 5, 10, 20]:
        mh = ic_metrics(fac, fwd_ret(h))
        dec[h] = round(mh["ic"], 4)
    print(f"  {name}: {dec}")
