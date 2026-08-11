"""miner_1 2026-07-30 cycle6 -- reverse-engineer the post-Miner gate correlation.

Goal: reproduce the eviction rhos (eff_ratio_20 0.554, down_vol_ratio_20x60 0.602,
ret_kurt_30 0.523 vs yield_beta_cond_60x20) and confirm hl_pos_150 passes.
Tests multiple correlation definitions to identify the gate's method.
"""
import json, base64, zlib, io
import numpy as np
import pandas as pd

def load_panel(fid, base="factors"):
    d = json.load(open(f"{base}/{fid}.json"))
    raw = base64.b64decode(d["validation"]["signal_artifact"]["data"])
    p = pd.read_csv(io.StringIO(zlib.decompress(raw).decode()), index_col=0, parse_dates=True)
    p.index = pd.DatetimeIndex(p.index)
    return p

def pooled_spearman(a, b, min_pairs=1):
    common = a.index.intersection(b.index)
    cols = [c for c in a.columns if c in b.columns]
    x = a.loc[common, cols].values.ravel()
    y = b.loc[common, cols].values.ravel()
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < min_pairs:
        return np.nan, int(m.sum())
    return float(pd.Series(x[m]).rank().corr(pd.Series(y[m]).rank())), int(m.sum())

def daily_mean_spearman(a, b, min_days=5):
    common = a.index.intersection(b.index)
    cols = [c for c in a.columns if c in b.columns]
    rhos = []
    for dt in common:
        x = a.loc[dt, cols]; y = b.loc[dt, cols]
        m = x.notna() & y.notna()
        if m.sum() >= 5:
            v = x[m].rank().corr(y[m].rank())
            if np.isfinite(v):
                rhos.append(v)
    rhos = np.array(rhos)
    if len(rhos) < min_days:
        return np.nan, len(rhos)
    return float(rhos.mean()), len(rhos)

def pooled_pearson(a, b, min_pairs=1):
    common = a.index.intersection(b.index)
    cols = [c for c in a.columns if c in b.columns]
    x = a.loc[common, cols].values.ravel()
    y = b.loc[common, cols].values.ravel()
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < min_pairs:
        return np.nan, int(m.sum())
    return float(np.corrcoef(x[m], y[m])[0, 1]), int(m.sum())

yb = load_panel("yield_beta_cond_60x20")
print("yield_beta panel:", yb.shape, "valid:", int(yb.notna().sum().sum()),
      "dates with >=5 valid:", int((yb.notna().sum(axis=1) >= 5).sum()))
print("yield_beta index range:", yb.index.min(), "->", yb.index.max())

for cand in ["eff_ratio_20", "down_vol_ratio_20x60", "ret_kurt_30", "hl_pos_150", "hl_pos_180", "mom_10d_skip5", "vix_beta_cond_60x20"]:
    try:
        p = load_panel(cand, base="factors/evicted" if cand in ("eff_ratio_20","down_vol_ratio_20x60","ret_kurt_30") else "factors")
    except Exception as e:
        print(cand, "load fail", e); continue
    sp1, n1 = pooled_spearman(p, yb)
    sp30, n30 = pooled_spearman(p, yb, min_pairs=30)
    dm, nd = daily_mean_spearman(p, yb)
    pp, np_ = pooled_pearson(p, yb)
    print(f"\n{cand:22s}")
    print(f"  pooled_spear(no min) = {sp1:+.4f} (n={n1})   pooled_spear(min30) = {sp30:+.4f} (n={n30})")
    print(f"  daily_mean_spearman  = {dm:+.4f} (n_days={nd})   pooled_pearson = {pp:+.4f} (n={np_})")

# also cross-check vs other active factors (full method)
print("\n--- vs other active factors (pooled spearman, no min) ---")
for cand in ["hl_pos_150", "hl_pos_180"]:
    p = load_panel(cand)
    for libf in ["mom_10d_skip5", "vix_beta_cond_60x20", "yield_beta_cond_60x20"]:
        lp = load_panel(libf)
        r, n = pooled_spearman(p, lp)
        print(f"  {cand} vs {libf}: {r:+.4f} (n={n})")
