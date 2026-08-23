"""miner_3 (2026-08-13): Cycle-2 fresh sweep - genuinely orthogonal macro/liquidity centers.

Goal: find factors passing IC/ICIR gate (absIC>=0.0070 & absICIR>=0.0840 at h=10 on
15-asset universe) with LOW library correlation (<0.5). Prior sweeps found most
price/vol momentum variants correlate >0.5. Test fresh structural centers:
  - ret_vs_cny_20  : asset 20d momentum X sign of USDCNY 20d change (CNY regime tilt)
  - range_efficiency : (close-open)/range capture of daily efficiency vs intraday range
  - drawdown_120   : 1 - close/rolling_max(close,120)
  - wti_beta_60    : rolling 60d beta vs WTI (commodity/energy center; beta_VIX was on VIX)
  - ret_liquidity : Amihud-style illiquidity but normalized by own price scale (turnover-like)
"""
from __future__ import annotations
import sys, pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner3_20260730_harness import ASSETS, evaluate, load_closes, load_macro

def main():
    closes = load_closes()
    macro = load_macro()
    print("assets:", len(closes), "macro:", len(macro))

    ohlc = {}
    for a in ASSETS:
        f = pathlib.Path(f"../persistent/stock_data/{a}.csv")
        df = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
        df = df[df["date"] <= "2026-07-29"]
        ohlc[a] = df.set_index("date")

    ret = {a: closes[a].pct_change() for a in closes}
    cand = {}

    # 1) CNY-regime-flipped 20d momentum: asset mom * sign(USDCNY 20d change shifted 5)
    cny = macro["USDCNY"]
    cnyr = cny.pct_change(20).shift(5)
    cys = np.where(cnyr.notna(), np.where(cnyr > 0, -1.0, 1.0), np.nan)
    cys = pd.Series(cys, index=cnyr.index)
    for a in closes:
        mom = closes[a] / closes[a].shift(20) - 1.0
        cand.setdefault("ret_cny_flip20", {})[a] = mom * cys

    # 2) daily efficiency: |close-open| / (high-low), averaged 20d
    for a in closes:
        o_ = ohlc[a]["open"]; h_ = ohlc[a]["high"]; l_ = ohlc[a]["low"]; c_ = ohlc[a]["close"]
        eff = (c_ - o_).abs() / (h_ - l_).replace(0, np.nan)
        cand.setdefault("daily_efficiency_20", {})[a] = eff.rolling(20).mean()

    # 3) drawdown from 120d high (already have days_since_high_60; this is depth not time)
    for a in closes:
        dd = 1.0 - closes[a] / closes[a].rolling(120).max()
        cand.setdefault("dd_depth_120", {})[a] = dd

    # 4) WTI beta (energy center) - verify numerically distinct from beta_VIX
    wti = closes["WTI"]
    wti_ret = wti.pct_change()
    def rolling_beta(a, m, w=60, minp=12):
        df = pd.concat([a.rename("a"), m.rename("m")], axis=1)
        out = []
        for i in range(len(df)):
            if i < w - 1:
                out.append(np.nan); continue
            sub = df.iloc[i-w+1:i+1]
            mm = sub["m"].to_numpy(); aa = sub["a"].to_numpy()
            fm = np.isfinite(mm) & np.isfinite(aa)
            if fm.sum() < minp or np.nanstd(mm[fm]) == 0:
                out.append(np.nan); continue
            out.append(np.cov(aa[fm], mm[fm])[0, 1] / np.var(mm[fm]))
        return pd.Series(out, index=df.index)
    for a in closes:
        cand.setdefault("wti_beta_60", {})[a] = rolling_beta(ret[a], wti_ret, 60)

    # 5) intermarket momentum linkage: asset 10d momentum vs COMMODITY basket mean mom
    #    commodity-beta signed (candidate distinct from per-asset momentum)
    w = closes["WTI"].pct_change() + closes["XAU"].pct_change() + closes["COPPER"].pct_change()
    w = w / 3.0
    bmom = w.rolling(20).sum().shift(5)
    for a in closes:
        mom = closes[a] / closes[a].shift(10) - 1.0
        cand.setdefault("comm_beta_mom10", {})[a] = mom * np.sign(bmom)

    # 6) turnover-liquidity: volume z-score (volume is per-asset; distinct center)
    for a in closes:
        v = ohlc[a]["volume"].astype(float).replace(0, np.nan)
        vz = (v - v.rolling(20).mean()) / v.rolling(20).std().replace(0, np.nan)
        cand.setdefault("vol_z20", {})[a] = vz

    for name, vals in cand.items():
        try:
            evaluate(closes, vals, name, horizon=10)
        except Exception as e:
            print(name, "ERROR:", repr(e))
        print()

if __name__ == "__main__":
    main()