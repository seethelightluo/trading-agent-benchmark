"""Trader validation v2: recompute 9 ensemble factors LIVE and compare vs stored artifacts.

Alignment: artifact rows map onto the MASTER trading calendar (date.json trading_days),
row i <-> trading_days[i], artifact has 2398 rows = sessions 2020-01-01..2026-07-29.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

BASE = Path(__file__).parent.parent
DATE_PATH = BASE.parent / "persistent" / "date.json"
FACTORS = BASE / "factors"

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]

date_state = json.loads(DATE_PATH.read_text())
trading_days = date_state.get("trading_days", [])
visible = date_state.get("visible_through", "2026-09-23")
master = [d for d in trading_days if d <= visible]
print(f"master sessions {master[0]} .. {master[-1]} -> {len(master)} rows")
art_n = 2398
print(f"artifact covers sessions {master[0]} .. {master[art_n-1]} -> expect last=2026-07-29")

# ---------------- load data ----------------
closes = {}
for a in ASSETS:
    df = get_stock_daily_data(a, days=300)
    closes[a] = df.set_index(pd.to_datetime(df["date"]))["close"].astype(float)
usdjpy = get_index_daily_data("USDJPY", days=300)
usdjpy = usdjpy.set_index(pd.to_datetime(usdjpy["date"]))["close"].astype(float)

# ---------------- factor implementations ----------------
def compute_factors(c):
    ret = c.pct_change()
    out = {}
    out["mom_20d_skip5"] = c.shift(5) / c.shift(25) - 1.0
    raw20 = c.shift(5) / c.shift(25) - 1.0
    mom60 = c / c.shift(60) - 1.0
    out["mom20_volproxy60_raw"] = raw20
    out["mom20_volproxy60"] = raw20 / (1.0 + mom60.abs())
    std20 = ret.rolling(20, min_periods=10).std()
    out["calmness_20"] = (ret.abs() < 0.5 * std20).rolling(20, min_periods=10).mean()
    aret = ret.abs()
    out["volcluster_60"] = aret.rolling(60, min_periods=40).corr(aret.shift(1))
    pos = (ret > 0).astype(int)
    def longest_run(x):
        m = 0; cur = 0
        for v in x:
            if v == 1:
                cur += 1; m = max(m, cur)
            else:
                cur = 0
        return m
    out["max_consec_gain_20"] = pos.rolling(21, min_periods=10).apply(longest_run, raw=True)
    g = ret.clip(lower=0).rolling(20, min_periods=10).sum()
    l = ret.clip(upper=0).abs().rolling(20, min_periods=10).sum()
    out["gain_loss_20"] = g / l.replace(0, np.nan)
    out["gain_loss_20_gl"] = g - l
    spx_ret = closes["SPX"].pct_change()
    out["spx_corr60"] = ret.rolling(60, min_periods=15).corr(spx_ret)
    m2 = pd.concat([ret, spx_ret], axis=1, join="inner").dropna()
    m2.columns = ["a", "s"]
    def downbeta(x):
        sub = m2.loc[x.index]
        sub = sub[sub["s"] < 0]
        if len(sub) < 15:
            return np.nan
        if sub["s"].var() < 1e-12:
            return np.nan
        return float(sub["a"].cov(sub["s"]) / sub["s"].var())
    out["downbeta_spx_60"] = m2["a"].rolling(60, min_periods=20).apply(downbeta, raw=False)
    m3 = pd.concat([ret, usdjpy.pct_change()], axis=1, join="inner").dropna()
    m3.columns = ["a", "u"]
    def jpybeta(x):
        sub = m3.loc[x.index]
        if len(sub) < 60 or sub["u"].var() < 1e-12:
            return np.nan
        return float(sub["a"].cov(sub["u"]) / sub["u"].var())
    b = m3["a"].rolling(120, min_periods=60).apply(jpybeta, raw=False)
    mom60j = usdjpy / usdjpy.shift(60) - 1.0
    out["usdjpy_beta_cond_120x60"] = b * mom60j
    return out

master_idx = pd.DatetimeIndex(pd.to_datetime(master))
my_panels = {}
for a in ASSETS:
    facs = compute_factors(closes[a])
    for fid, s in facs.items():
        if fid not in my_panels:
            my_panels[fid] = pd.DataFrame(index=master_idx, columns=ASSETS, dtype=float)
        my_panels[fid][a] = s.reindex(master_idx).values

# ---------------- load artifacts (aligned to master sessions) ----------------
def load_artifact(fid):
    p = FACTORS / f"{fid}.signal.npy"
    if p.exists():
        arr = np.load(p, allow_pickle=True)
        return arr[:art_n]
    j = json.loads((FACTORS / f"{fid}.json").read_text())
    art = j.get("signal_artifact", {})
    dates = pd.to_datetime(art.get("dates", []))
    vals = np.array(art.get("values", []), dtype=float)
    # embedded artifact should already be session-aligned; reindex onto master
    df = pd.DataFrame(vals, index=dates, columns=ASSETS)
    return df.reindex(master_idx).values[:art_n]

def spearman(a, b):
    ma = ~(np.isnan(a) | np.isnan(b))
    if ma.sum() < 5:
        return np.nan
    ra = pd.Series(a[ma]).rank().values
    rb = pd.Series(b[ma]).rank().values
    ra = ra - ra.mean(); rb = rb - rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else np.nan

print()
print(f"{'factor':28s} {'med_rho':>8s} {'n':>4s} {'last_rho':>8s}  live rows nan?")
base_map = {"gain_loss_20_gl": "gain_loss_20", "mom20_volproxy60_raw": "mom20_volproxy60"}
for fid in (["max_consec_gain_20", "mom20_volproxy60", "mom20_volproxy60_raw", "spx_corr60",
             "mom_20d_skip5", "gain_loss_20", "gain_loss_20_gl", "downbeta_spx_60",
             "usdjpy_beta_cond_120x60", "volcluster_60", "calmness_20"]):
    base = base_map.get(fid, fid)
    art = load_artifact(base)
    mine = my_panels[fid].values
    n = min(art.shape[0], mine.shape[0], 40)
    rhos = []
    for k in range(1, n + 1):
        a_row = art[-k, :]
        m_row = mine[-k, :]
        if np.isnan(a_row).sum() > 5:
            continue
        rho = spearman(a_row, m_row)
        if rho == rho:
            rhos.append(rho)
    med = float(np.median(rhos)) if rhos else np.nan
    last_rho = spearman(art[-1, :], mine[-1, :])
    live = mine[-1, :]
    print(f"{fid:28s} {med:+8.3f} {len(rhos):4d} {last_rho:+8.3f}  nan={int(np.isnan(live).sum())}/15")

print()
print("=== live (2026-09-23) ensemble factor values ===")
for fid in ["max_consec_gain_20", "mom20_volproxy60", "spx_corr60", "mom_20d_skip5",
            "gain_loss_20", "downbeta_spx_60", "usdjpy_beta_cond_120x60",
            "volcluster_60", "calmness_20"]:
    row = my_panels[fid].iloc[-1].values
    print(f"{fid:28s}", " ".join(f"{v:+.3f}" if v == v else "  nan" for v in row))
