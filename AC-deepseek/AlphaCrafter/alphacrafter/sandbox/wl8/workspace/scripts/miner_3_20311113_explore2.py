"""miner_3 2031-11-13 cycle part 2: gradient-correction. Avoid constant-broadcast
candidates; evaluate genuine per-asset cross-sectional signals.
"""
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
def rv(s, win):
    v = vseries(s); return v.pct_change().rolling(win).std().reindex(INDEX)

H = 10
fwd = pd.DataFrame({s: forward(px[s], H) for s in WATCH}).sort_index()
def build(df): return df.sort_index().replace([np.inf,-np.inf],np.nan).astype(float)

def assess(name, factor_df):
    icd = cross_sectional_ic(factor_df, fwd)
    if len(icd)==0:
        print(f"{name:28s} NO DATES"); return None
    st = ic_stats(icd)
    cov = (factor_df.notna() & fwd.notna()).mean().mean()
    line = (f"{name:28s} FULL IC={st['ic']:+.4f} ICIR={st['icir']:+.4f} hit={st['hit']:.3f} "
            f"n={st['n_dates']:5d} avg={st.get('avg_n',np.nan):4.1f} cov={cov:.3f}")
    rmask = icd.index >= icd.index[-1]-pd.Timedelta(days=365)
    if rmask.any():
        ic365 = ic_stats(icd[rmask]); line += f" | 365d {ic365['ic']:+.4f}/{ic365['icir']:+.4f}"
    g = abs(st['ic'])>=0.0070 and abs(st['icir'])>=0.0840
    line += f" | {'PASS' if g else 'FAIL'}"
    print(line)
    for lab, seg in regime_split(icd).items():
        print(f"    {lab}: [{seg[0]:+.4f},{seg[1]:+.4f},n={seg[2]}]")
    return st, icd

print("===== NEW GENUINE PER-ASSET CANDIDATES =====")
# N2. price vs 20d max (drawdown proximity; low drawdown = momentum quality)
def drawdown(s, w):
    v=vseries(s); roll=v.rolling(w).max(); return (v/roll-1).reindex(INDEX)
f_dd = build(pd.DataFrame({s: drawdown(px[s],20) for s in WATCH}))
assess('rev_20d_drawdown', f_dd)

# N3. price vs 80d MA distance (trend strength)
def dist_ma(s, w):
    v=vseries(s); return (v/v.rolling(w).mean()-1).reindex(INDEX)
f_dma = build(pd.DataFrame({s: dist_ma(px[s],80) for s in WATCH}))
assess('trend_80d_ma_dist', f_dma)

# N4. vol z-score (relative low-vol = quality/defensive)
f_vz = build(pd.DataFrame({s: -((rv(px[s],20)-rv(px[s],60).rolling(120).mean())/rv(px[s],60).rolling(120).std()) for s in WATCH}))
assess('vol_z_20x60(neg)', f_vz)

# N5. downside capture vs upside capture 60d
def down_up(s, w=60):
    v=vseries(s); r=v.pct_change()
    return (r.clip(upper=0).rolling(w).sum()/r.clip(lower=0).rolling(w).sum()).reindex(INDEX)
f_duc = build(pd.DataFrame({s: down_up(px[s]) for s in WATCH}))
assess('down_up_capture_60', f_duc)

# N6. 60d Amihud price-impact (illiquidity premium across assets)
def amihud(s, w=60):
    v=vseries(s); r=v.pct_change().abs()
    return (r.rolling(w).mean()).reindex(INDEX)  # proxy with only price data
f_am = build(pd.DataFrame({s: amihud(px[s]) for s in WATCH}))
assess('realized_abs_ret_60(neg)', -f_am)

print("\nDONE")