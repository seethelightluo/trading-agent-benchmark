"""miner_3 2035-11-07 fast candidate screen (stride=2 sampling for IC).
VIX 24.3 (60d +87.9% from 13), DXY 82.2 flat, SPX +4.2% 60d. Elevated/risk-off spike regime.
Admission gate (15-asset cs): |IC|>=0.0070 and |ICIR|>=0.0840.
"""
import sys, os, math, json
sys.path.insert(0, "scripts")
from factor_validation_lib import TRADABLE, align_fwd_returns
import pandas as pd, numpy as np

VIS = "2035-11-06"
closes = {}
for sym in TRADABLE:
    df = pd.read_csv(f"../persistent/stock_data/{sym}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= pd.Timestamp(VIS)].sort_values("date")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    closes[sym] = df.set_index("date")["close"]
px = pd.DataFrame(closes).sort_index().ffill()
px = px[px.index >= "2021-01-01"]
ret = px.pct_change()
print("panel:", px.shape, px.index.min().date(), "->", px.index.max().date(), flush=True)

def spearman_ic_series(f, fwd, stride=2):
    ics = {}
    idx = f.index
    for i in range(0, len(idx), stride):
        d = idx[i]
        x = f.loc[d].dropna()
        y = fwd.loc[d].reindex(x.index).dropna()
        c = x.index.intersection(y.index)
        if len(c) < 8:
            continue
        xv, yv = x[c].values, y[c].values
        if np.std(xv) == 0 or np.std(yv) == 0:
            continue
        rho = pd.Series(xv).rank().corr(pd.Series(yv).rank())
        if np.isfinite(rho):
            ics[d] = rho
    return pd.Series(ics, dtype=float)

def evalc(f, label, hor=10):
    ic = spearman_ic_series(f, align_fwd_returns(px, hor))
    if len(ic) == 0:
        print(f"[{label}] NO IC DATES", flush=True)
        return None
    icm = float(ic.mean())
    icstd = float(ic.std(ddof=1)) if len(ic) > 1 else np.nan
    icir = icm / icstd if icstd and math.isfinite(icstd) and icstd > 0 else 0.0
    hit = float((ic > 0).mean())
    r3 = ic[ic.index >= "2032-11-01"]
    ricm = float(r3.mean()) if len(r3) else np.nan
    ricir = ricm / r3.std(ddof=1) if len(r3) > 2 and r3.std(ddof=1) > 0 else np.nan
    cov = float(f.notna().mean().mean())
    dec = {}
    for h in (1, 5, 10, 20):
        ih = spearman_ic_series(f, align_fwd_returns(px, h))
        dec[h] = round(float(ih.mean()), 4) if len(ih) else None
    gate = (abs(icm) >= 0.0070) and (abs(icir) >= 0.0840)
    print(f"[{label}] IC10={icm:+.4f} ICIR10={icir:+.4f} hit={hit:.3f} n={len(ic)} "
          f"ric3y={ricm:+.4f} ricir3y={ricir:+.4f} cov={cov:.3f} decay={dec} "
          f"GATE={'PASS' if gate else 'fail'}", flush=True)
    return dict(ic=icm, icir=icir, hit=hit, ric=ricm, ricir=ricir, cov=cov, dec=dec)

vix = pd.read_csv("../persistent/index_data/VIX.csv", parse_dates=["date"])
vix = vix[vix["date"] <= pd.Timestamp(VIS)].set_index("date")["close"].reindex(px.index).ffill()
vix_chg20 = vix / vix.shift(20) - 1
vix_chg60 = vix / vix.shift(60) - 1
dxy = pd.read_csv("../persistent/index_data/DXY.csv", parse_dates=["date"])
dxy = dxy[dxy["date"] <= pd.Timestamp(VIS)].set_index("date")["close"].reindex(px.index).ffill()
dxy_chg20 = dxy / dxy.shift(20) - 1

mom10 = px / px.shift(10) - 1
mom20 = px / px.shift(20) - 1
mom60 = px / px.shift(60) - 1
vol20 = ret.rolling(20).std()
vol60 = ret.rolling(60).std()

candidates = {}
asc20 = np.clip(vix_chg20, -0.5, 0.5)   # high = VIX rising 20d
asc60 = np.clip(vix_chg60, -0.5, 0.5)
candidates["vixrise_cond_mom20"] = mom20.mul(asc20, axis=0)
candidates["vixrise_cond_mom10"] = mom10.mul(asc20, axis=0)
candidates["vixsurge_cond_mom60"] = mom60.mul(asc60, axis=0)
dxydec20 = -np.clip(dxy_chg20, -0.3, 0.3)
candidates["mom20_dxydecl"] = mom20.mul(dxydec20, axis=0)
candidates["vol_drop_20x60"] = -vol20.diff(20) / vol20
candidates["vol_ratio_20_60_neg"] = -vol20 / vol60.shift(1)
candidates["mom10_div_vol20"] = mom10 / vol20.shift(1)
candidates["mom20_div_vol60"] = mom20 / vol60.shift(1)
# novel: 60d return normalized by 60d vol (risk-adjusted momentum, slower)
candidates["mom60_div_vol60"] = mom60 / vol60.shift(1)
# novel: VIX level gate (elevated >= 20) x mom20 - regime-carry
gate_elev = pd.Series(np.where(vix >= 20, 1.0, 0.4), index=vix.index)
candidates["elevvix_gate_mom20"] = mom20.mul(gate_elev, axis=0)
# novel: VIX 60d surge gate x mom60 (risk-off winners)
gate_surge = pd.Series(np.clip(vix_chg60, 0, 0.8), index=vix.index)
candidates["vixsurge_gate_mom60"] = mom60.mul(gate_surge, axis=0)

out = {}
for name, f in candidates.items():
    o = evalc(f, name)
    if o is not None:
        out[name] = o
json.dump(out, open("scripts/miner3_20351107_screen_fast_results.json", "w"), indent=1)
print("saved results", flush=True)