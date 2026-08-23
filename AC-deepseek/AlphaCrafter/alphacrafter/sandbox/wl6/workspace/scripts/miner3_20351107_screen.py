"""miner_3 screen 2035-11-07. VIX risen 60d +87.9% to 24.3 (recent spike from very low 13
in Sep). Ensemble frozen 2034-11-08 (8 factors, defensive-heavy directed at highvol regime).
Screen fresh high-VIX-rising (elevated/risk-off) candidates at horizon 10.
Admission gate: |IC|>=0.0070 and |ICIR|>=0.0840 on 15-asset cross-section.
"""
import sys, os, math, json, base64, zlib
sys.path.insert(0, 'scripts')
from factor_validation_lib import rank_ic_series, align_fwd_returns, load_macro, TRADABLE
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
print("panel shape:", px.shape, "date:", px.index.min().date(), "->", px.index.max().date(), flush=True)

def build(f):
    return f.reindex(index=px.index, columns=px.columns)

def evalc(f, label, hor=10):
    f = build(f)
    ic = rank_ic_series(f, align_fwd_returns(px, hor))
    if len(ic) == 0:
        print(f"[{label}] NO IC DATES", flush=True); return None
    icm = float(ic.mean()); icstd = float(ic.std(ddof=1)) if len(ic) > 1 else np.nan
    icir = icm / icstd if icstd and math.isfinite(icstd) and icstd > 0 else 0.0
    hit = float((ic > 0).mean())
    recent = ic[ic.index >= "2033-01-01"]
    ricm = float(recent.mean()) if len(recent) else np.nan
    ricir = ricm/recent.std(ddof=1) if len(recent) > 2 and recent.std(ddof=1) > 0 else np.nan
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
          f"recentIC={ricm:+.4f} ricIR={ricar:+.4f} covAD={cov_ad:.3f} covD8={cov_d8:.3f} "
          f"turn={turn:.3f} decay={dec} GATE={flag}", flush=True)
    return dict(ic=icm, icir=icir, hit=hit, ric=ricm, ricir=ricar,
                cov=cov_ad, covd8=cov_d8, turn=turn, dec=dec)

vix = load_macro("VIX", VIS).reindex(px.index).ffill()
vix_chg20 = (vix / vix.shift(20) - 1)
vix_chg60 = (vix / vix.shift(60) - 1)
vix_chg120 = (vix / vix.shift(120) - 1)
dxy = load_macro("DXY", VIS).reindex(px.index).ffill()
dxy_chg20 = (dxy / dxy.shift(20) - 1)

mom10 = px / px.shift(10) - 1
mom20 = px / px.shift(20) - 1
mom60 = px / px.shift(60) - 1
vol20 = ret.rolling(20).std()
vol60 = ret.rolling(60).std()

# risk-on carry when VIX rising but asset is resilient (low asset-vs-own-vol beta):
rv = vol20 / (ret.rolling(20).std())
## asset vol normalized signal (idiosyncratic strength while market vol spikes)
candidates = {}

# A. momentum gated by VIX uptrend (regime-conditional momentum)
dec20 = -np.clip(vix_chg20, -0.5, 0.5)   # high positive = VIX falling
asc20 = np.clip(vix_chg20, -0.5, 0.5)    # high positive = VIX rising
candidates["vixrise_cond_mom20"] = mom20.mul(asc20, axis=0)
candidates["vixrise_cond_mom10"] = mom10.mul(asc20, axis=0)

# B. risk-on carry: positive momentum gated for non-defensive in low-vol regime
vol_norm = vol20 / vol60.shift(1)
candidates["mom20_lowvolnorm"] = mom20.mul(-1.0*dxy_chg20.reindex(px.index), axis=0)

# C. DXY falling => risk-on: momentum gated by DXY decline
dxydec20 = -np.clip(dxy_chg20, -0.3, 0.3)
candidates["mom20_dxydecl"] = mom20.mul(dxydec20, axis=0)

# D. mean-reversion after volatility compression: low recent vol / high prior vol
candidates["vol_drop_20x60"] = -vol20.diff(20)/vol20  # negative = vol falling
candidates["vol_ratio_20_60_neg"] = -vol20/vol60.shift(1)

# E. risk-normalized momentum (mom / recent vol)
candidates["mom10_div_vol20"] = mom10 / vol20.shift(1)
candidates["mom20_div_vol60"] = mom20 / vol60.shift(1)

for name, f in candidates.items():
    evalc(f, name)