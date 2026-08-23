"""miner3 screen 2035-07-04 v2. Fix Series alignment; add more low-VIX risk-on candidates."""
import sys, os, math
sys.path.insert(0, 'scripts')
from factor_validation_lib import rank_ic_series, align_fwd_returns, load_macro, TRADABLE
import pandas as pd, numpy as np

VIS = "2035-07-03"
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
print("panel shape:", px.shape, "assets:", px.shape[1],
      "date:", px.index.min().date(), "->", px.index.max().date())


def build(f):
    return f.reindex(index=px.index, columns=px.columns)


def evalc(f, label):
    f = build(f)
    ic = rank_ic_series(f, align_fwd_returns(px, 10))
    if len(ic) == 0:
        print(f"[{label}] NO IC DATES"); return None
    icm = float(ic.mean()); icstd = float(ic.std(ddof=1)) if len(ic) > 1 else np.nan
    icir = icm / icstd if icstd and math.isfinite(icstd) and icstd > 0 else np.nan
    hit = float((ic > 0).mean())
    recent = ic[ic.index >= "2033-01-01"]
    ricm = float(recent.mean()) if len(recent) else np.nan
    ricir = ricm / recent.std(ddof=1) if len(recent) > 2 and recent.std(ddof=1) > 0 else np.nan
    cov_ad = float(f.notna().mean().mean())
    cov_d8 = float((f.notna().sum(axis=1) >= 8).mean())
    turn = float(f.rank(axis=1, pct=True).diff().abs().mean(axis=1).mean() * 10) if len(f) > 2 else np.nan
    dec = {}
    for h in (1,2,3,5,10,20):
        ic_h = rank_ic_series(f, align_fwd_returns(px,h))
        dec[h] = round(float(ic_h.mean()),4) if len(ic_h) else None
    gate = (abs(icm) >= 0.0070) and (abs(icir) >= 0.0840)
    flag = "PASS" if gate else "fail"
    print(f"[{label}] n_ic={len(ic)} IC={icm:+.4f} ICIR={icir:+.4f} hit={hit:.3f} "
          f"recentIC={ricm:+.4f} ricir={ricir:+.4f} covAD={cov_ad:.3f} covD8={cov_d8:.3f} "
          f"turn={turn:.3f} decay={dec} GATE={flag}")
    return dict(ic=icm, icir=icir, hit=hit, ricm=ricm, ricir=ricir, cov=cov_ad, turn=turn, dec=dec)

vix = load_macro("VIX", VIS).reindex(px.index).ffill()
vix_lvl = vix
vix_chg20 = (vix / vix.shift(20) - 1)
vix_chg60 = (vix / vix.shift(60) - 1)
dxy = load_macro("DXY", VIS).reindex(px.index).ffill()
dxy_chg60 = (dxy / dxy.shift(60) - 1)

mom20 = px / px.shift(20) - 1
mom40 = px / px.shift(40) - 1
mom60 = px / px.shift(60) - 1
vol20 = ret.rolling(20).std()
vol60 = ret.rolling(60).std()

candidates = {}
dec20 = -np.clip(vix_chg20, -0.5, 0.5)
dec60 = -np.clip(vix_chg60, -0.5, 0.5)
dxydec = -np.clip(dxy_chg60, -0.3, 0.3)

candidates["vixdecl_cond_mom20"] = mom20.mul(dec20, axis=0)
candidates["vixdecl_cond_mom40"] = mom40.mul(dec60, axis=0)
candidates["vixdecl_cond_mom60"] = mom60.mul(dec60, axis=0)
candidates["dxydecl_cond_mom20"] = mom20.mul(dxydec, axis=0)
gate_low = pd.Series(np.where(vix_lvl < 20, 1.0, 0.4), index=vix.index)
candidates["lowvix_gate_mom20"] = mom20.mul(gate_low, axis=0)
candidates["mom40_div_vol20"] = mom40 / vol20.shift(1)
candidates["mom60_div_vol60"] = mom60 / vol60.shift(1)

for name, f in candidates.items():
    evalc(f, name)