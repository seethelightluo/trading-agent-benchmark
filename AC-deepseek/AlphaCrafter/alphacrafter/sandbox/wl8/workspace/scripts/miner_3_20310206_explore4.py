"""miner_3 2031-02-06 explore batch 4: composite/conditioned constructs to stabilize ICIR.
Focus: momentum/volatility conditioned on macro regime states and risk-adjusted cross-asset
relative momentum. Single-idea-batch screening (each candidate is its own construct)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from miner_3_20261203_common import (WATCH, load_prices, load_macro, load_visible_through,
                                     cross_sectional_ic, ic_stats, regime_split)

ASOF = load_visible_through()
px = load_prices(ASOF)
mac = load_macro(ASOF)
INDEX = px.index
print(f"ASOF={ASOF} rows={len(INDEX)} assets={len(WATCH)}")

def vseries(s): return s.dropna()
def retk(s, k):
    v = vseries(s); return (v / v.shift(k) - 1.0).reindex(INDEX)
def forward(s, h):
    v = vseries(s); return (v.shift(-h)/v - 1.0).reindex(INDEX)

H = 10
fwd = pd.DataFrame({s: forward(px[s], H) for s in WATCH}).sort_index()
def build(df): return df.sort_index().replace([np.inf,-np.inf],np.nan).astype(float)

def rv(s, win=20):
    v = vseries(s); return v.pct_change().rolling(win).std().reindex(INDEX)

def assess(name, factor_df, show_regime=True):
    icd = cross_sectional_ic(factor_df, fwd)
    st = ic_stats(icd)
    cov = (factor_df.notna() & fwd.notna()).mean().mean()
    ic365 = ic_stats(icd[icd.index >= icd.index[-1]-pd.Timedelta(days=365)]) if len(icd) else icd
    ic60 = ic_stats(icd.tail(60))
    line = (f"{name:30s} FULL IC={st['ic']:+.4f} ICIR={st['icir']:+.4f} hit={st['hit']:.3f} "
            f"n={st['n_dates']:5d} cov={cov:.3f} | "
            f"365d {ic365['ic']:+.4f}/{ic365['icir']:+.4f} last60 {ic60['ic']:+.4f}/{ic60['icir']:+.4f}")
    print(line)
    if show_regime:
        for lab, seg in regime_split(icd).items():
            print(f"    {lab}: [{seg[0]:+.4f},{seg[1]:+.4f},n={seg[2]}]")
    gate = abs(st['ic'])>=0.0070 and abs(st['icir'])>=0.0840
    print(f"    GATE: {'PASS' if gate else 'FAIL'}")
    return st

mom10 = build(pd.DataFrame({s: retk(px[s],10) for s in WATCH}))
mom20 = build(pd.DataFrame({s: retk(px[s],20) for s in WATCH}))
mom60 = build(pd.DataFrame({s: retk(px[s],60) for s in WATCH}))
vol20 = build(pd.DataFrame({s: rv(px[s],20) for s in WATCH}))
vol60 = build(pd.DataFrame({s: rv(px[s],60) for s in WATCH}))

cands = {}

# 1) Risk-adjusted 20d momentum (Sharpe-like): mom20 / vol20
cands['sharp_mom20'] = build(mom20 / (vol20*vol20*252).pow(0.5).replace(0,np.nan))

# 2) Risk-adjusted momentum conditioned on regime: mom20/vol20 * sign of VIX 20d change
d_vix20 = mac['VIX'].pct_change(20).reindex(INDEX)
regime_vix = build(pd.DataFrame({s: np.sign(d_vix20.fillna(0)) for s in WATCH}))
cands['sharp_mom20_x_vixsign'] = build((mom20/vol20*252).replace(np.inf,np.nan) * regime_vix)

# 3) Cross-sectional relative 20d momentum minus vol-weighted median (risk parity tilt)
volmed = vol20.median(axis=1)
cands['rel_mom20_volmed'] = build(mom20 - vol60.median(axis=1).to_frame().values)

# 4) Trend-agreement: sign-agreement product of mom20 and mom60 with magnitude mom20
cands['trend_agreement_20x60'] = build(mom20 * np.sign(mom60))

# 5) Momentum consistency across 3 windows (10/20/60 mean) - smoother accel
cands['mom_consist_3w'] = build((mom10+mom20+mom60)/3.0)

# 6) Vol-conditioned trend continuity: consecutive same-sign daily moves weighted by magnitude (streak*mom)
dl = pd.DataFrame({s: px[s].dropna().pct_change().reindex(INDEX) for s in WATCH})
streak = pd.DataFrame(index=INDEX, columns=WATCH, dtype=float)
for c in WATCH:
    arr = dl[c].fillna(0).values
    out = np.zeros(len(arr)); cnt=0
    for i in range(len(arr)):
        if arr[i]>0 and cnt>=0: cnt+=1 if cnt>=0 else 1
        elif arr[i]<0 and cnt<=0: cnt-=1 if cnt<=0 else 1
        else: cnt = (1 if arr[i]>0 else (-1 if arr[i]<0 else 0))
        out[i]=cnt
    streak[c]=out
cands['streak_mom20'] = build(np.sign(streak)*mom20.abs())

for name, fd in cands.items():
    assess(name, fd)
print("\nDONE")