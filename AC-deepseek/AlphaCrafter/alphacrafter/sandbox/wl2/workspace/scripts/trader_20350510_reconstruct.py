"""Reconstruct the 04-27 decision view: factor z-composite as of data through 04-25.

Mirrors strategy.py _live_factors / build_target using only closes visible at
the decision date (slice <= 2035-04-25). Reports z, composite pref, and the
cap/floor mapping that produced the executed target.
"""
import json
import math
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
DECISION_VISIBLE = "2035-04-25"

CAP, FLOOR = 0.14, 0.012

ens = json.load(open("factors/factor_ensemble.json"))
sel = {it["factor_id"]: it["weight"] for it in ens["selected_factors"]}
print("ensemble factors:", {k: round(v, 4) for k, v in sel.items()})

closes = {}
for a in ASSETS:
    df = get_stock_daily_data(a, days=300)
    if df is None or "close" not in df or len(df) < 130:
        continue
    s = df.set_index(pd.to_datetime(df["date"]))["close"].astype(float)
    s = s[s.index <= DECISION_VISIBLE]
    closes[a] = s
print("closes through", DECISION_VISIBLE, "for", len(closes), "assets")

spx = closes["SPX"]
spx_ret = spx.pct_change()

def longest_run(x):
    m, cur = 0.0, 0
    for v in x:
        if v == 1:
            cur += 1
            m = max(m, cur)
        else:
            cur = 0
    return m

per = {}
for a, c in closes.items():
    ret = c.pct_change()
    f = {}
    flat = bool(len(ret) >= 15 and float(ret.tail(15).std()) < 1e-12)
    f["_flat"] = flat
    if not flat:
        pos = (ret > 0).astype(int)
        f["max_consec_gain_20"] = pos.rolling(21, min_periods=10).apply(longest_run, raw=True).iloc[-1]
        f["mom_180d_skip5"] = (c.shift(5) / c.shift(185) - 1.0).iloc[-1]
        rmin = c.rolling(252, min_periods=30).min()
        rmax = c.rolling(252, min_periods=30).max()
        f["range_pos_252"] = ((c - rmin) / (rmax - rmin).replace(0, np.nan)).iloc[-1]
        f["spx_corr60"] = ret.rolling(60, min_periods=15).corr(spx_ret).iloc[-1]
        m2 = pd.concat([ret, spx_ret], axis=1, join="inner").dropna()
        m2.columns = ["a", "s"]
        def downbeta(x):
            sub = m2.loc[x.index]
            sub = sub[sub["s"] < 0]
            if len(sub) < 15 or sub["s"].var() < 1e-12:
                return np.nan
            return float(sub["a"].cov(sub["s"]) / sub["s"].var())
        f["downbeta_spx_60"] = m2["a"].rolling(60, min_periods=20).apply(downbeta, raw=False).iloc[-1]
    per[a] = f

def rank_z(vals):
    n = len(vals)
    r = pd.Series(vals).rank(method="average").to_numpy()
    z = (r - 0.5) / max(n, 1)
    z = np.clip((z - z.mean()) / (z.std() + 1e-12), -3, 3)
    return z

raw = {fid: [] for fid in sel}
for a in ASSETS:
    f = per.get(a, {})
    for fid in sel:
        raw[fid].append(float("nan") if f.get("_flat") else f.get(fid, float("nan")))

zs = {fid: rank_z([v if v == v else 0.5 for v in raw[fid]]) for fid in sel}
comp = np.zeros(len(ASSETS))
for fid, w in sel.items():
    comp += w * zs[fid]

# r20 for gates
r20 = {}
for a, c in closes.items():
    r20[a] = (c.iloc[-1] / c.iloc[-21] - 1.0) if len(c) > 21 else float("nan")

print("\n%-10s %8s %8s %8s %8s %8s %8s %8s" % ("asset", "r20%", "streak", "mom180", "range252", "corr60", "dnsbeta", "zcomp"))
order = np.argsort(-comp)
for i in order:
    a = ASSETS[i]
    f = per.get(a, {})
    def g(k):
        v = f.get(k, float("nan"))
        return float("nan") if v is None else v
    print("%-10s %8.2f %8.2f %8.3f %8.3f %8.3f %8.3f %8.4f" % (
        a, r20.get(a, float("nan")) * 100, g("max_consec_gain_20"), g("mom_180d_skip5"),
        g("range_pos_252"), g("spx_corr60"), g("downbeta_spx_60"), comp[i]))

# VIX for v9 gate
try:
    vix = get_stock_daily_data("VIX", days=130)
    if vix is not None:
        vix = vix.set_index(pd.to_datetime(vix["date"]))["close"]
        vix = vix[vix.index <= DECISION_VISIBLE]
        print("\nVIX at decision:", round(float(vix.iloc[-1]), 2))
except Exception as e:
    print("VIX err", e)
