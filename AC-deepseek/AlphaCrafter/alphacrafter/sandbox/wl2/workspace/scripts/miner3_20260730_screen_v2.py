"""miner_3 screening v2 2026-07-30: NEW candidate factor families on the 15-asset
cross-asset universe. Per-asset own-calendar computation, union reindex.
Admission gates: |IC| >= 0.0070 and |ICIR| >= 0.0840 at 10d horizon.
Visible through 2026-07-29 (no lookahead).
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner3_lib import build_panel, forward_returns, spearman_ic, mean_rank_turnover, VISIBLE

prices = build_panel()
panel = pd.DataFrame(prices)
ret = panel.pct_change()
print(f"panel: {panel.shape}, dates {panel.index.min().date()}..{panel.index.max().date()} (visible {VISIBLE})")
print(f"assets: {len(panel.columns)}")


def per_asset(func):
    """Apply func to each asset's own-calendar series, reindex to union panel."""
    out = {}
    for a in panel.columns:
        s = panel[a].dropna()
        out[a] = func(s).reindex(panel.index)
    return pd.DataFrame(out, index=panel.index)


# ---------------- candidate factor families (each per-asset, own calendar) ----------------
cands = {}

# F1 variance ratio 20x1: var of rolling 20d summed returns / (20 * var of 1d returns)
# >1 => trending (momentum), <1 => mean-reverting
def f_var_ratio2(s, w=20):
    r = s.pct_change()
    rsum = r.rolling(w).sum()       # w-day overlapping return sums
    vs = rsum.rolling(w).var()
    v1 = r.rolling(w).var()
    return vs / (w * v1)
cands["var_ratio_20x1"] = per_asset(f_var_ratio2)

# F2 price z-score 60: (close - SMA60) / std(close,60)  -> mean reversion candidate
def f_zscore(s, w=60):
    return (s - s.rolling(w).mean()) / s.rolling(w).std()
cands["price_zscore_60"] = per_asset(lambda s: f_zscore(s, 60))

# F3 12-1 momentum: 252d return skipping last 21d
def f_mom(s, w=252, skip=21):
    return s.shift(skip) / s.shift(skip + w) - 1.0
cands["mom_252_skip21"] = per_asset(lambda s: f_mom(s, 252, 21))

# F4 trend efficiency 60 (Kaufman): |close - close.shift(60)| / sum(|ret|,60)
def f_eff(s, w=60):
    r = s.pct_change().abs()
    return (s - s.shift(w)).abs() / r.rolling(w).sum()
cands["trend_eff_60"] = per_asset(lambda s: f_eff(s, 60))

# F5 up-day ratio 20: participation / breadth
def f_upday(s, w=20):
    r = s.pct_change()
    return (r > 0).rolling(w).mean()
cands["upday_ratio_20"] = per_asset(lambda s: f_upday(s, 20))

# F6 relative momentum 60: 60d ret minus cross-sectional median 60d ret
mom60 = per_asset(lambda s: s / s.shift(60) - 1.0)
cands["rel_mom_60"] = mom60.sub(mom60.median(axis=1), axis=0)

# F7 pullback 5d: (close - rolling_max(close,5)) / rolling_max(close,5)
def f_pullback(s, w=5):
    return s / s.rolling(w).max() - 1.0
cands["pullback_5d"] = per_asset(lambda s: f_pullback(s, 5))

# F8 vol term structure 10x120: 10d vol / 120d vol - 1
def f_volts(s, w1=10, w2=120):
    r = s.pct_change()
    return r.rolling(w1).std() / r.rolling(w2).std() - 1.0
cands["vol_ts_10x120"] = per_asset(lambda s: f_volts(s, 10, 120))

# F9 corr to BTC 20d: rolling correlation of asset returns with BTC returns
def f_corr_btc(s, w=20):
    r = s.pct_change()
    b = panel["BTC"].dropna().pct_change()
    z = pd.concat([r.rename("a"), b.rename("b")], axis=1).dropna()
    c = z["a"].rolling(w).corr(z["b"])
    return c.reindex(r.index)
cands["corr_btc_20"] = per_asset(lambda s: f_corr_btc(s, 20))

# F10 vol-momentum signed 20x60: sign(mom20)*vol60 (trend confidence, direction-scaled vol)
def f_vol_mom(s, w1=20, w2=60):
    r = s.pct_change()
    mom = s / s.shift(w1) - 1.0
    return np.sign(mom) * r.rolling(w2).std()
cands["vol_mom_signed_20x60"] = per_asset(lambda s: f_vol_mom(s, 20, 60))

# ---------------- validation ----------------
HORIZON = 10
fwd = forward_returns(prices, HORIZON)
results = {}
for fid, fdf in cands.items():
    fdf = fdf.replace([np.inf, -np.inf], np.nan)
    ic_series = spearman_ic(fdf, fwd)
    if len(ic_series) < 100:
        print(f"{fid:22s} SKIP n_ic={len(ic_series)}")
        continue
    ic = float(ic_series.mean())
    icir = float(ic_series.mean() / ic_series.std()) if ic_series.std() > 0 else 0.0
    hit = float((ic_series > 0).mean()) if ic >= 0 else float((ic_series < 0).mean())
    dec = {}
    for h in (1, 2, 3, 5, 10, 20):
        s = spearman_ic(fdf, forward_returns(prices, h))
        dec[str(h)] = round(float(s.mean()), 4)
    cov_ad = float(fdf.notna().sum().sum()) / float(fdf.size)
    n_ge8 = sum(1 for d in fdf.index if fdf.loc[d].notna().sum() >= 8)
    cov_d8 = n_ge8 / len(fdf)
    to = mean_rank_turnover(fdf)
    reg = {}
    for b0, b1 in [("2020-01-01", "2021-12-31"), ("2022-01-01", "2022-12-31"),
                   ("2023-01-01", "2024-12-31"), ("2025-01-01", "2026-12-31")]:
        sub = ic_series[(ic_series.index >= b0) & (ic_series.index <= b1)]
        if len(sub) >= 30:
            reg[f"{b0[:4]}-{b1[:4]}"] = (round(float(sub.mean()), 4),
                                          round(float(sub.mean() / sub.std()), 4) if sub.std() > 0 else 0.0)
    results[fid] = {"ic": ic, "icir": icir, "hit": hit, "n": len(ic_series),
                    "cov_ad": cov_ad, "cov_d8": cov_d8, "turn": to, "decay": dec, "regime": reg}
    print(f"==== {fid} ====")
    print(f"  ic={ic:+.4f}  icir={icir:+.4f}  hit={hit:.3f}  n={len(ic_series)}")
    print(f"  cov_asset_days={cov_ad:.3f}  cov_dates_ge8={cov_d8:.3f}  turnover10d={to:.3f}")
    print(f"  decay={dec}")
    print(f"  regime_ic/icir={reg}")
    print(f"  PASS={'YES' if abs(ic)>=0.007 and abs(icir)>=0.084 else 'no'}")

print("\n==== summary sorted by |ic|*|icir| ====")
for fid, r in sorted(results.items(), key=lambda kv: abs(kv[1]["ic"]) * abs(kv[1]["icir"]), reverse=True):
    print(f"{fid:22s} ic={r['ic']:+.4f} icir={r['icir']:+.4f} pass={abs(r['ic'])>=0.007 and abs(r['icir'])>=0.084}")
