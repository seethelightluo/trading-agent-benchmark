"""Trader dry-run 2030-01-15: replicate strategy factor branch, verify non-fallback target."""
import json
from pathlib import Path
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data
from math import isfinite

MIN_ROWS = 61
CAP = 0.15
DEF = ("XAU", "US10Y", "CN10Y")

def _stock(sym, days=200):
    try:
        return get_stock_daily_data(sym, days=days)
    except Exception:
        return None

def _index(sym, days=200):
    try:
        return get_index_daily_data(sym, days=days)
    except Exception:
        return None

def _series(df, name=None):
    if df is None or "close" not in df or len(df) < MIN_ROWS:
        return None
    s = pd.Series(df["close"].astype(float), index=pd.to_datetime(df["date"]))
    return s.rename(name) if name else s

def ranks(values, assets):
    valid = sorted((float(v), a) for a, v in values.items() if v is not None and isfinite(float(v)))
    out = {a: 0.5 for a in assets}
    n = len(valid)
    if n == 0:
        return out
    for i, (_, a) in enumerate(valid):
        out[a] = i / (n - 1) if n > 1 else 0.5
    return out

def capped_normalize(w, pref):
    w = {a: max(0.0, float(x)) for a, x in w.items()}
    for _ in range(60):
        excess = sum(max(0.0, x - CAP) for x in w.values())
        w = {a: min(CAP, x) for a, x in w.items()}
        room = [a for a, x in w.items() if x < CAP - 1e-12]
        if excess < 1e-12 or not room:
            break
        den = sum(max(0.0, pref.get(a, 0.0)) for a in room)
        for a in room:
            w[a] += excess * (max(0.0, pref.get(a, 0.0)) / den if den else 1.0 / len(room))
    total = sum(w.values())
    return {a: x / total for a, x in w.items()} if total > 0 else {a: 1.0 / len(w) for a in w}

def def_alloc(budget, series_map, assets, scale=0.10):
    alloc = {a: 0.0 for a in assets}
    alloc["XAU"] = 0.70 * budget
    alloc["US10Y"] = 0.15 * budget
    return alloc

acc = get_account_dict()
assets = list(acc.get("watch_list", []))
frames = {a: _stock(a) for a in assets}
series = {a: _series(f) for a, f in frames.items()}
usable = {a: s.pct_change().rename(a) for a, s in series.items() if s is not None}
print("usable assets:", len(usable), sorted(usable.keys()))
R = pd.concat(usable, axis=1, join="inner").dropna().tail(150)
print("R rows:", len(R))

cp = (1.0 + R).cumprod()
mkt = R.mean(axis=1)

mom = cp.shift(5) / cp.shift(25) - 1.0
rel_mom = mom.sub(mom.median(axis=1), axis=0)
mvar = mkt.rolling(60).var()
beta_ew = R.rolling(60).cov(mkt).div(mvar, axis=0)
neg = R.clip(upper=0.0)
semi = (neg ** 2).rolling(20).mean() ** 0.5
tot = R.rolling(20).std()
dvr = -(semi / tot)
mx = R.rolling(20).max()
dxy_cond = None
dfx = _index("DXY")
if dfx is not None and len(dfx) >= MIN_ROWS:
    dc = pd.Series(dfx["close"].astype(float), index=pd.to_datetime(dfx["date"]))
    dxy_ret = dc.pct_change().reindex(R.index)
    dxy_20 = (dc / dc.shift(20) - 1.0).reindex(R.index)
    if dxy_ret.notna().sum() >= 40 and dxy_20.notna().sum() >= 40:
        dvar = dxy_ret.rolling(60).var()
        bfx = R.rolling(60).cov(dxy_ret).div(dvar, axis=0)
        dxy_cond = -bfx * dxy_20
corr_parts = []
for a in R.columns:
    others = [R[a].rolling(60).corr(R[b]) for b in R.columns if b != a]
    corr_parts.append(pd.concat(others, axis=1).mean(axis=1).rename(a))
corr_ew = pd.concat(corr_parts, axis=1)
kurt = R.shift(5).rolling(20).kurt()

factor_values = {
    "rel_mom_20d_skip5": rel_mom,
    "beta_ew_60d": beta_ew,
    "downside_vol_ratio_20": dvr,
    "max_ret_20d": mx,
    "dxy_beta_cond_60x20": dxy_cond,
    "corr_ew_60": corr_ew,
    "kurt_20d_skip5": kurt,
}

ens = json.loads((Path("factor_ensemble.json")).read_text())
sel = [(str(it["factor_id"]), float(it["weight"]), int(it.get("direction", 1)))
       for it in ens.get("selected_factors", []) if isinstance(it, dict) and it.get("factor_id")]
print("ensemble factors:", [f[0] for f in sel])

score = {a: 0.0 for a in assets}
active = []
for fid, w, d in sel:
    fr = factor_values.get(fid)
    if fr is None or len(fr) == 0:
        print(f"  MISSING factor series: {fid}")
        continue
    last = fr.iloc[-1]
    if last.isna().all():
        print(f"  ALL-NaN factor series: {fid}")
        continue
    rk = ranks(last.to_dict(), assets)
    for a in assets:
        score[a] += w * (d * rk[a])
    active.append(fid)
print("active factors:", active)

mkt_20 = float(mkt.tail(20).mean())
vix_df = _index("VIX")
vix = float(vix_df["close"].iloc[-1]) if vix_df is not None and len(vix_df) else 25.0
risk_off = mkt_20 < -0.0005 or vix > 30.0
risk_on = mkt_20 > 0.001 and vix < 14.0
def_floor = 0.12 if risk_off else (0.09 if risk_on else 0.10)
print(f"mkt_20={mkt_20:.6f} vix={vix:.2f} risk_off={risk_off} def_floor={def_floor}")

score_rk = ranks(score, assets)
def_w = def_alloc(def_floor * len(DEF), series, list(DEF))
pref = {}
for a in assets:
    pref[a] = def_w.get(a, def_floor) if a in DEF else 0.05 + 0.95 * score_rk[a]
weights = capped_normalize(dict(pref), pref)
weights[assets[-1]] += 1.0 - sum(weights.values())

print("\nTarget weights (must sum to 1.0):", round(sum(weights.values()), 8))
for a in sorted(weights, key=lambda x: -weights[x]):
    print(f"  {a}: {weights[a]:.4f}  score_rank={score_rk[a]:.3f}")
print("fallback?", len(active) == 0 or max(weights.values()) - min(weights.values()) < 1e-9)
