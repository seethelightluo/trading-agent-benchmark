"""Miner2 broad screen #3: NEW untested factor families on the 15-asset cross-section.

Window: 2020-01 .. 2026-07-15 (warm-up, from panel_cache.pkl, 2388 dates).
Gates (15-name universe): |IC1| >= 0.0070 and |ICIR1| >= 0.0840, >=8 names/date.

Families (all per-asset panels, dates x 15 symbols):
  A. Macro-sensitivity betas: rolling beta of asset returns to DXY/USDJPY/VIX/EURUSD/USDCNY changes
  B. Basket (equal-weight 15-asset) correlation: unconditional, downside-only, upside-only, and the
     downside-minus-upside asymmetry ("flight-to-quality" tilt)
  C. Tail-shape: rolling skewness/kurtosis of daily returns
  D. Downside-vol ratio: semideviation / total vol
  E. Range/body position: mean((close-low)/(high-low)), mean(|close-open|/(high-low))
  F. Volume-based: vol z-score, vol trend, amihud illiquidity, price-volume correlation
  G. Drawdown / distance from high
  H. Up-day fraction (breadth), sign streak
  I. Vol term-structure ratios: vol5/vol20, vol20/vol60
  J. Overnight vs intraday component means
  K. Calibration baselines: mom_10d_skip5, vol_20d, skew_60d

For every candidate that passes the gate, also report gate-style pairwise |spearman|
vs the effective library (miner2 mom factor + seed factors) to anticipate the
correlation gate (threshold 0.5).
"""
import sys, time, json, os, pickle, base64, gzip, zlib
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import miner2_fast as F

SYMBOLS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
           "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
t0 = time.time()
cache = pickle.load(open("scripts/panel_cache.pkl", "rb"))
CP, OP = cache["close"], cache["open"]
HP, LP, V = cache["high"], cache["low"], cache["vol"]
RET, MAC = cache["ret"], cache["macro"]
idx = CP.index
print(f"loaded cache: {len(idx)} dates {idx.min().date()}..{idx.max().date()}, macro={MAC.shape} cols={list(MAC.columns)}")

fwd = {h: CP.shift(-h) / CP - 1.0 for h in (1, 2, 3, 5, 10, 20)}
N_CELLS = len(idx) * len(SYMBOLS)
LOG = np.log(CP / CP.shift(1))

cands = {}

# ---------- A. macro betas ----------
MACR = MAC.pct_change().reindex(idx)
for col in MACR.columns:
    mr = MACR[col]
    var_m = mr.rolling(60).var()
    for win in (20, 60):
        beta = RET.rolling(win).cov(mr) / (var_m if win == 60 else mr.rolling(win).var())
        cands[f"beta_{col.lower()}_{win}d"] = beta

# ---------- B. basket correlations & asymmetry ----------
BASK = RET.mean(axis=1)
cands["basket_corr_60d"] = RET.rolling(60).corr(BASK)
cands["basket_corr_20d"] = RET.rolling(20).corr(BASK)

def cond_corr(win, side):
    """rolling corr of each asset with basket on days basket<0 (side='dn') or >0 (side='up')."""
    cols = {}
    for s in SYMBOLS:
        r = RET[s].values
        out = np.full(len(r), np.nan)
        for i in range(win, len(r)):
            seg_r = r[i - win:i]
            seg_b = BASK.values[i - win:i]
            if side == "dn":
                m = seg_b < 0
            else:
                m = seg_b > 0
            if m.sum() >= 5:
                a, b = seg_r[m], seg_b[m]
                if a.std() > 0 and b.std() > 0:
                    out[i] = np.corrcoef(a, b)[0, 1]
        cols[s] = out
    return pd.DataFrame(cols, index=idx)

for win in (60,):
    up = cond_corr(win, "up")
    dn = cond_corr(win, "dn")
    cands[f"dn_corr_{win}d"] = dn
    cands[f"up_corr_{win}d"] = up
    cands[f"dn_up_corr_gap_{win}d"] = dn - up

# ---------- C. tail shape ----------
for nd in (20, 60, 120):
    cands[f"skew_{nd}d"] = RET.rolling(nd).skew()
    cands[f"kurt_{nd}d"] = RET.rolling(nd).kurt()
    cands[f"nskew_{nd}d"] = -RET.rolling(nd).skew()

# ---------- D. downside vol ratio ----------
def downside_ratio(nd):
    dn = RET.clip(upper=0.0)
    dn_sd = (dn ** 2).rolling(nd).mean().apply(np.sqrt)
    tot_sd = RET.rolling(nd).std()
    return dn_sd / (tot_sd + 1e-12)

cands["downside_ratio_20d"] = downside_ratio(20)
cands["downside_ratio_60d"] = downside_ratio(60)

# ---------- E. range / body position ----------
rg = (HP - LP).replace(0, np.nan)
pos = (CP - LP) / rg
cands["range_pos_5d"] = pos.rolling(5).mean()
cands["range_pos_20d"] = pos.rolling(20).mean()
cands["body_ratio_5d"] = ((CP - OP).abs() / rg).rolling(5).mean()
cands["body_ratio_20d"] = ((CP - OP).abs() / rg).rolling(20).mean()
cands["nbody_ratio_5d"] = -((CP - OP).abs() / rg).rolling(5).mean()

# ---------- F. volume-based ----------
VOL = V.astype(float)
vol20 = VOL.rolling(20).mean()
cands["vol_z_20d"] = (VOL - vol20) / (VOL.rolling(60).std() + 1e-9)
cands["vol_trend_20_60"] = vol20 / (VOL.rolling(60).mean() + 1e-9)
cands["amihud_20d"] = (RET.abs() / (VOL + 1e-9)).rolling(20).mean()
cands["amihud_60d"] = (RET.abs() / (VOL + 1e-9)).rolling(60).mean()
cands["pv_corr_20d"] = RET.rolling(20).corr(VOL.pct_change())
cands["nvol_z_20d"] = -cands["vol_z_20d"]

# ---------- G. drawdown / distance from high ----------
cands["max_dd_60d"] = CP.rolling(60).max() / CP - 1.0
cands["max_dd_252d"] = CP.rolling(252).max() / CP - 1.0
cands["ndist_252_high"] = -(CP / CP.rolling(252).max() - 1.0)
cands["dist_252_high"] = CP / CP.rolling(252).max() - 1.0

# ---------- H. breadth & streak ----------
up = (RET > 0).astype(float)
cands["up_frac_20d"] = up.rolling(20).mean()
cands["up_frac_60d"] = up.rolling(60).mean()

def sign_streak():
    cols = {}
    for s in SYMBOLS:
        r = RET[s].values
        out = np.full(len(r), np.nan)
        prev = 0.0
        for i in range(len(r)):
            if not np.isfinite(r[i]):
                prev = 0.0
                continue
            sg = 1.0 if r[i] > 0 else (-1.0 if r[i] < 0 else 0.0)
            if sg == 0:
                prev = 0.0
                out[i] = 0.0
            elif np.sign(prev) == sg or prev == 0.0:
                prev = prev + sg if np.sign(prev) == sg else sg
                out[i] = prev
            else:
                prev = sg
                out[i] = sg
        cols[s] = out
    return pd.DataFrame(cols, index=idx)

cands["sign_streak"] = sign_streak()

# ---------- I. vol term-structure ratios ----------
v5 = RET.rolling(5).std()
v20 = RET.rolling(20).std()
v60 = RET.rolling(60).std()
cands["vol_ratio_5_20"] = v5 / (v20 + 1e-12)
cands["vol_ratio_20_60"] = v20 / (v60 + 1e-12)
cands["nvol_ratio_5_20"] = -(v5 / (v20 + 1e-12))

# ---------- J. overnight / intraday components ----------
gap = OP / CP.shift(1) - 1.0
intra = CP / OP - 1.0
cands["gap_5d"] = gap.rolling(5).mean()
cands["intra_5d"] = intra.rolling(5).mean()
cands["gap_20d"] = gap.rolling(20).mean()
cands["intra_20d"] = intra.rolling(20).mean()

# ---------- K. calibration baselines ----------
cands["mom_10d_skip5"] = CP.shift(5) / CP.shift(15) - 1.0
cands["vol_20d"] = RET.rolling(20).std() * np.sqrt(252)
cands["skew_60d_base"] = RET.rolling(60).skew()

print(f"built {len(cands)} candidates ({time.time()-t0:.1f}s)")

res = []
for name, panel in cands.items():
    panel = panel.reindex(idx)
    try:
        cov = float(panel.notna().sum().sum()) / N_CELLS
        to = F.turnover10(panel)
        ic1 = F.fast_ic(panel, fwd[1])
        passed = (abs(ic1["ic"]) >= 0.0070) and (abs(ic1["icir"]) >= 0.0840)
        if passed or name in ("mom_10d_skip5", "vol_20d"):
            ic5 = F.fast_ic(panel, fwd[5])
            ic10 = F.fast_ic(panel, fwd[10])
            print(f"{name:26s} cov={cov:.3f} to={to:.3f} | IC1={ic1['ic']:+.4f} ICIR1={ic1['icir']:+.3f} "
                  f"hit1={ic1['hit']:.2f} n1={ic1['n_dates']} | IC5={ic5['ic']:+.4f} | IC10={ic10['ic']:+.4f} | {'PASS' if passed else 'fail'}")
        res.append({"name": name, "cov": cov, "to": to, "ic1": ic1, "passed": passed})
    except Exception as e:
        print(f"{name}: ERROR {e}")

npass = sum(r["passed"] for r in res)
print(f"\nscreen done in {time.time()-t0:.1f}s | {npass} passed gate / {len(cands)} candidates")
for r in res:
    if r["passed"]:
        print("PASSED:", r["name"], f"IC1={r['ic1']['ic']:+.4f} ICIR1={r['ic1']['icir']:+.3f}")
