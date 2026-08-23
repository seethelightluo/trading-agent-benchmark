"""miner_3 2031-05-15 cycle: re-validate active effective factors + screen new candidate
constructs through visible_through (2031-05-14). Continuous re-validation cycle.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from miner_3_20261203_common import (WATCH, load_prices, load_macro, load_visible_through,
                                     cross_sectional_ic, ic_stats, regime_split, spearman_panel_rho)

ASOF = load_visible_through()
px = load_prices(ASOF)
mac = load_macro(ASOF)
INDEX = px.index
print(f"ASOF={ASOF} rows={len(INDEX)} assets={len(WATCH)} px_last={px.index[-1].date()}")

def vseries(s): return s.dropna()
def retk(s, k):
    v = vseries(s); return (v / v.shift(k) - 1.0).reindex(INDEX)
def forward(s, h):
    v = vseries(s); return (v.shift(-h)/v - 1.0).reindex(INDEX)
def rv(s, win):
    v = vseries(s); return v.pct_change().rolling(win).std().reindex(INDEX)
def mv(s, win):
    return vseries(s).rolling(win).mean().reindex(INDEX)

H = 10
fwd = pd.DataFrame({s: forward(px[s], H) for s in WATCH}).sort_index()
def build(df): return df.sort_index().replace([np.inf,-np.inf],np.nan).astype(float)

def assess(name, factor_df, show_regime=True):
    icd = cross_sectional_ic(factor_df, fwd)
    st = ic_stats(icd)
    cov = (factor_df.notna() & fwd.notna()).mean().mean()
    if len(icd):
        ic365 = ic_stats(icd[icd.index >= icd.index[-1]-pd.Timedelta(days=365)])
        ic180 = ic_stats(icd[icd.index >= icd.index[-1]-pd.Timedelta(days=180)])
        ic60 = ic_stats(icd.tail(60))
    else:
        ic365 = ic180 = ic60 = {}
    gate = abs(st['ic'])>=0.0070 and abs(st['icir'])>=0.0840
    line = (f"{name:28s} FULL IC={st['ic']:+.4f} ICIR={st['icir']:+.4f} hit={st['hit']:.3f} "
            f"n={st['n_dates']:5d} avg={st.get('avg_n',np.nan):4.1f} cov={cov:.3f} | "
            f"365d {ic365.get('ic',np.nan):+.4f}/{ic365.get('icir',np.nan):+.4f} "
            f"180d {ic180.get('ic',np.nan):+.4f}/{ic180.get('icir',np.nan):+.4f} "
            f"60d {ic60.get('ic',np.nan):+.4f}/{ic60.get('icir',np.nan):+.4f} | {'PASS' if gate else 'FAIL'}")
    print(line)
    if show_regime:
        for lab, seg in regime_split(icd).items():
            print(f"    {lab}: [{seg[0]:+.4f},{seg[1]:+.4f},n={seg[2]}]")
    return st, icd

print("\n===== ACTIVE FACTOR RE-VALIDATION =====")
res = {}
f_flip = build(pd.DataFrame({s: retk(px[s],20)*np.sign(retk(px[s],10)) for s in WATCH}))
res['flip_mom_20x10'] = assess('flip_mom_20x10', f_flip)
f_momd = build(pd.DataFrame({s: retk(px[s],20)-retk(px[s],60) for s in WATCH}))
res['mom_diff_20_60'] = assess('mom_diff_20_60', f_momd)
m10_ = build(pd.DataFrame({s: retk(px[s],5) for s in WATCH}))
f_mom10 = build(pd.DataFrame({s: (vseries(px[s])/vseries(px[s]).shift(5)-1.0)*np.sign(vseries(px[s])/vseries(px[s]).shift(5)-1.0) for s in WATCH}))
res['mom_10d_skip5'] = assess('mom_10d_skip5(proxy)', m10_)
vix = mac['VIX']
f_vixb = build(pd.DataFrame({s: (retk(px[s],1).rolling(60).cov(vix.pct_change())/vix.pct_change().rolling(60).var()) for s in WATCH}))
res['vix_beta_cond_60x20'] = assess('vix_beta_cond_60x20', f_vixb)
usd = mac['USDCNY']
f_usd = build(pd.DataFrame({s: (retk(px[s],1).rolling(60).cov(usd.pct_change())/usd.pct_change().rolling(60).var()) for s in WATCH}))
res['usdcny_beta_60'] = assess('usdcny_beta_60', f_usd)

print("\n===== 2024+ regime gate check =====")
for name, fd in [('flip_mom_20x10',f_flip),('mom_diff_20_60',f_momd),('mom_10d_skip5(proxy)',m10_),('vix_beta_cond_60x20',f_vixb)]:
    icd = cross_sectional_ic(fd, fwd)
    segs = regime_split(icd)
    if '2024+ crypto/commodity' in segs:
        s = segs['2024+ crypto/commodity']
        print(f"  {name} 2024+: IC={s[0]:+.4f} ICIR={s[1]:+.4f} n={s[2]}")

print("\n===== LIBRARY CORRELATION =====")
print("flip vs momd:", round(spearman_panel_rho(f_flip, f_momd),4))
print("flip vs mom10:", round(spearman_panel_rho(f_flip, m10_),4))
print("momd vs mom10:", round(spearman_panel_rho(f_momd, m10_),4))
print("flip vs vixb:", round(spearman_panel_rho(f_flip, f_vixb),4))
print("momd vs vixb:", round(spearman_panel_rho(f_momd, f_vixb),4))
print("momd vs usd:", round(spearman_panel_rho(f_momd, f_usd),4))
print("flip vs usd:", round(spearman_panel_rho(f_flip, f_usd),4))

print("\nDONE")