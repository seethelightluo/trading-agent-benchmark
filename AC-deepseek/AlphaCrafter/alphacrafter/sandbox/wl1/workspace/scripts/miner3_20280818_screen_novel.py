"""miner_3 2028-08-18: screen novel factor candidates on panel through 2028-08-17 (VECTORIZED).
Admission gate: abs IC >= 0.0070, abs ICIR >= 0.0840.
"""
import numpy as np
import pandas as pd
import pickle, json

panel = pickle.load(open("scripts/panel_cache.pkl", "rb"))
C = panel["close"]; O = panel["open"]; H = panel["high"]; L = panel["low"]
V = panel["vol"]; M = panel["macro"]
R_ = C.pct_change()

gate_ic, gate_icir = 0.0070, 0.0840
factors = {}

ma10 = C.rolling(10).mean()
ma20 = C.rolling(20).mean()
ma50 = C.rolling(50).mean()

# --- trend-conditioned momentum ---
for nd, skip, maw in [(40, 5, 20), (60, 5, 20), (60, 5, 50), (120, 5, 20), (120, 5, 50), (90, 5, 20)]:
    mom = C.shift(skip) / C.shift(skip + nd) - 1.0
    ma = {"20": ma20, "50": ma50}[str(maw)]
    cond = (C > ma).astype(float)
    factors[f"mom_{nd}d_skip{skip}_cond_ma{maw}"] = mom * cond

# --- vol-scaled momentum ---
for nd, skip, vw in [(60, 5, 20), (60, 5, 60), (120, 5, 60), (120, 5, 90), (40, 5, 20)]:
    mom = C.shift(skip) / C.shift(skip + nd) - 1.0
    vol = R_.rolling(vw).std()
    factors[f"mom_{nd}d_skip{skip}_voladj{vw}"] = mom / vol

# --- drawdown / trend strength ---
factors["dd_60d"] = 1.0 - C / C.rolling(60).max()
factors["dd_120d"] = 1.0 - C / C.rolling(120).max()
factors["hi_prox_252"] = C / C.rolling(252).max()
factors["er_20d"] = (C - C.shift(20)).abs() / (C.diff().abs().rolling(20).sum())
factors["er_60d"] = (C - C.shift(60)).abs() / (C.diff().abs().rolling(60).sum())
factors["ma_slope_20"] = (ma20 / ma20.shift(20) - 1.0)
factors["ma_slope_50"] = (ma50 / ma50.shift(50) - 1.0)

# --- conditional reversal ---
rev2 = -(C.shift(2) / C - 1.0)
vol20 = R_.rolling(20).std()
vol60 = R_.rolling(60).std()
hi_vol = (vol20 > vol20.rolling(120).quantile(0.7)).astype(float)
factors["rev_2d_cond_hiVol"] = rev2 * hi_vol
factors["rev_2d_cond_maUp"] = rev2 * (C > ma20).astype(float)
factors["rev_2d_voladj"] = rev2 * vol20

# --- vol regime ---
factors["vol_ratio_5_60"] = R_.rolling(5).std() / R_.rolling(60).std()
factors["vol_ratio_10_60"] = R_.rolling(10).std() / R_.rolling(60).std()
factors["volz_20"] = (vol20 - vol20.rolling(120).mean()) / vol20.rolling(120).std()

# --- skew / kurtosis ---
factors["skew_20d"] = R_.rolling(20).skew()
factors["skew_60d"] = R_.rolling(60).skew()
factors["kurt_60d"] = R_.rolling(60).kurt()

# --- cross-sectional relative strength ---
mom60 = C.shift(5) / C.shift(65) - 1.0
mom120 = C.shift(5) / C.shift(125) - 1.0
factors["rel_rank_mom60"] = mom60.rank(axis=1) / mom60.count(axis=1)
factors["rel_rank_mom120"] = mom120.rank(axis=1) / mom120.count(axis=1)
factors["rel_rank_invvol60"] = (1.0 / vol60).rank(axis=1) / (1.0 / vol60).count(axis=1)

# --- macro-beta conditionals ---
vix = M["VIX"]; dxy = M["DXY"]
vix_ret = vix.pct_change(); dxy_ret = dxy.pct_change()
for name, mret in [("vix", vix_ret), ("dxy", dxy_ret)]:
    for w in (60, 120):
        cov = R_.rolling(w).cov(mret)
        var = mret.rolling(w).var()
        beta = cov / var
        for condname, cond in [("up", (mret > 0).astype(float)),
                               ("aboveMA", (mret > mret.rolling(20).mean()).astype(float))]:
            factors[f"{name}_beta_{w}d_cond_{condname}"] = beta * cond

# --- volume-price ---
factors["vol_price_ratio_20"] = C / V.rolling(20).mean()
factors["vol_trend_20"] = V.rolling(5).mean() / V.rolling(60).mean() - 1.0

# --- range position with trend ---
for nd in (5, 10, 20):
    hi = C.rolling(nd).max(); lo = C.rolling(nd).min()
    nclv = (C - lo) / (hi - lo) - 0.5
    factors[f"nclv_{nd}d_cond_ma20"] = nclv * (C > ma20).astype(float)
    factors[f"nclv_{nd}d_cond_ma50"] = nclv * (C > ma50).astype(float)


def ic_series_vec(f, h):
    """Vectorized per-date Spearman IC between factor and h-day forward return."""
    fwd = C.shift(-h) / C - 1.0
    f_rank = f.rank(axis=1)
    r_rank = fwd.rank(axis=1)
    f_dm = f_rank.sub(f_rank.mean(axis=1), axis=0)
    r_dm = r_rank.sub(r_rank.mean(axis=1), axis=0)
    num = (f_dm * r_dm).sum(axis=1)
    den = np.sqrt((f_dm ** 2).sum(axis=1) * (r_dm ** 2).sum(axis=1))
    ic = num / den
    mask = f.notna().sum(axis=1) >= 8
    ic = ic.where(mask & np.isfinite(ic))
    return ic.dropna()


print(f"{'factor':32s} {'h':>2s} {'IC':>8s} {'ICIR':>8s} {'hit':>5s} {'n':>5s} {'IC1y':>8s} {'ICIR1y':>8s}  gate")
out = {}
for name, f in factors.items():
    best = None
    for h in (1, 2, 5, 10):
        s = ic_series_vec(f, h)
        if len(s) == 0:
            continue
        ic = s.mean(); icir = s.mean() / s.std() * np.sqrt(len(s))
        hit = (s > 0).mean()
        rec = s[s.index >= s.index.max() - pd.Timedelta(days=365)]
        ic1y = rec.mean() if len(rec) else np.nan
        icir1y = rec.mean() / rec.std() * np.sqrt(len(rec)) if len(rec) > 2 and rec.std() > 0 else np.nan
        if best is None or abs(icir) > abs(best[2]):
            best = (h, ic, icir, hit, len(s), ic1y, icir1y)
    if best is None:
        continue
    h, ic, icir, hit, n, ic1y, icir1y = best
    flag = "PASS" if (abs(ic) >= gate_ic and abs(icir) >= gate_icir) else ""
    print(f"{name:32s} {h:>2d} {ic:>8.4f} {icir:>8.3f} {hit:>5.2f} {n:>5d} {ic1y:>8.4f} {icir1y:>8.3f}  {flag}")
    out[name] = {"best_h": h, "ic": ic, "icir": icir, "hit": hit, "n": n,
                 "ic_1y": ic1y, "icir_1y": icir1y, "gate": flag}

json.dump(out, open("scripts/miner3_20280818_screen_results.json", "w"), indent=1, default=str)
print("\nsaved -> scripts/miner3_20280818_screen_results.json")
