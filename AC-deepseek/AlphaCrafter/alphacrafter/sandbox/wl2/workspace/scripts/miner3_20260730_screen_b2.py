"""miner_3 screening batch 2 2026-07-30: additional distinct candidate families
(upside/downside capture, short-term reversal, breakout, vol percentile, market beta,
VIX-conditional trend). Same harness as batch 1. Gates: |IC|>=0.007, |ICIR|>=0.084 @10d."""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner3_lib import build_panel, forward_returns, spearman_ic, mean_rank_turnover, VISIBLE

prices = build_panel()
panel = pd.DataFrame(prices)
ret = panel.pct_change()


def per_asset(func):
    out = {}
    for a in panel.columns:
        s = panel[a].dropna()
        out[a] = func(s).reindex(panel.index)
    return pd.DataFrame(out, index=panel.index)


vix = pd.read_csv("../persistent/index_data/VIX.csv", parse_dates=["date"])
vix = vix[vix["date"] <= pd.Timestamp(VISIBLE)].set_index("date")["close"].astype(float)

cands = {}

# G1 upside/downside capture 60d: mean(pos rets) / |mean(neg rets)|  (asymmetry)
def f_updown(s, w=60):
    r = s.pct_change()
    pos = r[r > 0].rolling(w).mean()
    neg = r[r < 0].rolling(w).mean()
    return pos / neg.abs()
cands["updown_capture_60"] = per_asset(lambda s: f_updown(s, 60))

# G2 max daily return 20d (lottery/attention)
def f_maxret(s, w=20):
    return s.pct_change().rolling(w).max()
cands["max_ret_20"] = per_asset(lambda s: f_maxret(s, 20))

# G3 1d return (short-term reversal)
cands["ret_1d"] = per_asset(lambda s: s.pct_change())

# G4 10d breakout: close / rolling_max(close,10) - 1
def f_breakout(s, w=10):
    return s / s.rolling(w).max() - 1.0
cands["breakout_10"] = per_asset(lambda s: f_breakout(s, 10))

# G5 vol percentile 120: rank of vol20 within trailing 120d vol distribution (0..1)
def f_volpct(s, w1=20, w2=120):
    r = s.pct_change()
    v = r.rolling(w1).std()
    return v.rolling(w2).apply(lambda x: (x.iloc[-1] >= x).mean(), raw=False)
cands["vol_pctile_20x120"] = per_asset(lambda s: f_volpct(s, 20, 120))

# G6 market beta 20d vs equal-weight cross-sectional return
market = ret.mean(axis=1)
def f_mktbeta(s, w=20):
    r = s.pct_change()
    z = pd.concat([r.rename("a"), market.rename("m")], axis=1).dropna()
    cov = z["a"].rolling(w).cov(z["m"])
    var = z["m"].rolling(w).var()
    return (cov / var).reindex(r.index)
cands["mkt_beta_20"] = per_asset(lambda s: f_mktbeta(s, 20))

# G7 VIX-conditional z-score: price_zscore_60 only when VIX < its 60d median, else 0
def f_vixcond_z(s, w=60):
    z = (s - s.rolling(w).mean()) / s.rolling(w).std()
    v = vix.reindex(s.index)
    calm = (v < v.rolling(60).median())
    return z.where(calm, np.nan)
cands["zscore60_vix_calm"] = per_asset(lambda s: f_vixcond_z(s, 60))

# G8 VIX-conditional momentum: mom20 * (VIX percentile rank - 0.5)  (trend in calm, reverse in stress)
def f_vixmom(s, w=20):
    mom = s / s.shift(w) - 1.0
    v = vix.reindex(s.index)
    pct = v.rolling(252).apply(lambda x: (x.iloc[-1] >= x).mean(), raw=False)
    return mom * (pct - 0.5)
cands["mom20_vix_cond"] = per_asset(lambda s: f_vixmom(s, 20))

# G9 60d drawdown-weighted reversal: dd_from_high_20 (negative), low = deep pullback -> contrarian
def f_dd20(s, w=20):
    return s / s.rolling(w).max() - 1.0
cands["dd_20"] = per_asset(lambda s: f_dd20(s, 20))

# G10 3d gap ratio: open-to-close / close-to-open (intraday vs overnight strength)
def f_gap(s, w=20):
    # need open series -> use panel-level data via per-asset csv
    return None

HORIZON = 10
fwd = forward_returns(prices, HORIZON)
results = {}
for fid, fdf in cands.items():
    if fdf is None:
        continue
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
