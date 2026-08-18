"""miner_2 cycle-22 exploration: screen candidate cross-asset factors.

Data: observable through 2026-10-07 (visible_through for sim date 2026-10-08).
Universe: 15 tradable instruments. IC/ICIR gates: |IC|>=0.0070, |ICIR|>=0.0840.
Admission horizon 10 (also report 5/20 for decay). Cross-section needs >=8 valid.
"""
import pandas as pd
import numpy as np
from itertools import combinations

SIM_DATE = "2026-10-08"
VISIBLE = "2026-10-07"
UNIVERSE = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU",
            "COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
OBS_ONLY = ["DXY","USDCNY","USDJPY","EURUSD","VIX"]

def load(name, root="../persistent/stock_data/"):
    df = pd.read_csv(f"{root}{name}.csv", parse_dates=["date"])
    df = df[df["date"] <= pd.Timestamp(VISIBLE)].reset_index(drop=True)
    return df

prices = {}
for s in UNIVERSE:
    prices[s] = load(s).set_index("date")["close"]
px = pd.DataFrame(prices).sort_index()
ret = px.pct_change()
print("price panel:", px.shape, px.index.min().date(), "->", px.index.max().date())

ob = {}
for s in OBS_ONLY:
    d = pd.read_csv(f"../persistent/index_data/{s}.csv", parse_dates=["date"])
    d = d[d["date"] <= pd.Timestamp(VISIBLE)].set_index("date")["close"]
    ob[s] = d
ob = pd.DataFrame(ob).sort_index()
print("obs panel:", ob.shape, ob.index.min().date(), "->", ob.index.max().date())

# ---------------- candidate factor definitions ----------------
fac = {}
# f1 ret-vol correlation 40d (risk-on/off structure)
r5 = ret.rolling(5).sum()
v20 = ret.rolling(20).std()
fac["ret_vol_corr_40"] = r5.rolling(40).corr(v20)

# f2 drawdown level vs 120d high (oversold/overbought)
fac["drawdown_120"] = px / px.rolling(120).max() - 1.0

# f3 gap ratio 20d: mean |overnight gap| / mean |full move|
o = pd.DataFrame({s: load(s).set_index("date")["open"] for s in UNIVERSE}).sort_index()
gap = (o - px.shift(1)).abs()
full = (px - px.shift(1)).abs()
fac["gap_ratio_20"] = gap.rolling(20).mean() / full.rolling(20).mean().replace(0, np.nan)

# f4 short-horizon realized skew 5d vs level (crash proxy)
fac["skew_5_60"] = ret.rolling(5).skew() / ret.rolling(60).std().replace(0, np.nan)

# f5 volume trend 20/60 (flow proxy; yields have no volume -> NaN, fine)
vol = pd.DataFrame({s: load(s).set_index("date")["volume"] for s in UNIVERSE}).sort_index()
fac["vol_trend_20x60"] = vol.rolling(20).mean() / vol.rolling(60).mean().replace(0, np.nan)

# f6 dual momentum: 10d minus 60d momentum (trend vs reversal interaction)
fac["dual_mom_10x60"] = ret.rolling(10).sum() - ret.rolling(60).sum()

# f7 overnight vs intraday return ratio (auction behavior)
overnight = o - px.shift(1)
open2close = px - o
fac["overnight_ratio_20"] = overnight.rolling(30).sum() / (overnight.abs().rolling(30).sum().replace(0, np.nan) + open2close.abs().rolling(30).sum().replace(0, np.nan))

for k, v in fac.items():
    print(k, "valid day-assets:", int(v.notna().sum().sum()))

# ---------------- IC engine ----------------
def ic_series(factor, fwd_h, min_n=8):
    out = {}
    for dt in factor.index:
        fv = factor.loc[dt]
        fr = px[px.index > dt].iloc[:fwd_h]
        if len(fr) < fwd_h:
            break
        fwd = px.loc[fr.index[-1]] / px.loc[dt] - 1.0
        mask = fv.notna() & fwd.notna() & np.isfinite(fv) & np.isfinite(fwd)
        if mask.sum() < min_n:
            continue
        ic = fv[mask].corr(fwd[mask], method="spearman")
        if np.isfinite(ic):
            out[dt] = ic
    s = pd.Series(out)
    return s

def summarize(name, h):
    s = ic_series(fac[name], h)
    if len(s) == 0:
        return None
    ic = s.mean()
    icir = s.mean() / s.std() if s.std() > 0 else 0.0
    hit = (np.sign(s) == np.sign(ic)).mean()
    return {"horizon": h, "n_dates": len(s), "ic": ic, "icir": icir,
            "hit": hit, "ic_std": s.std()}

print("\n=== screen results ===")
for name in fac:
    for h in (5, 10, 20):
        r = summarize(name, h)
        if r:
            print(f"{name:22s} h={h:3d} n={r['n_dates']:5d} IC={r['ic']:+.4f} ICIR={r['icir']:+.4f} hit={r['hit']:.3f}")
    print("-" * 90)