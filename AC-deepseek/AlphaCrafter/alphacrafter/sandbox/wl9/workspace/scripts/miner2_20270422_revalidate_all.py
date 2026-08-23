#!/usr/bin/env python
"""Re-validate all 17 existing factors as of 2027-04-22. Cross-asset 15-instrument pool. Gates: |IC|>=0.0070, |ICIR|>=0.0840"""
import numpy as np, pandas as pd, json, warnings, sys
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
warnings.filterwarnings('ignore')

DT="2027-04-22"
WL=["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
ML=["DXY","USDCNY","USDJPY","EURUSD","VIX"]
print(f"=== REVALIDATION: {DT} ===")
data={}
for s in WL:
    df=get_stock_daily_data(s,750)
    if df is None or len(df)<200: df=get_index_daily_data(s,750)
    if df is not None and len(df)>=200: data[s]=df
for s in ML:
    df=get_index_daily_data(s,750)
    if df is not None and len(df)>=200: data[s]=df
print(f"Loaded {len(data)} symbols")
min_dt=max(d.index[0] for d in data.values() if len(d)>0)
aligned={s:d.loc[d.index>=min_dt] for s,d in data.items()}
common=aligned[WL[0]].index
for s in WL[1:]: common=common.intersection(aligned.get(s,pd.DataFrame()).index)
print(f"Common dates: {len(common)} from {common[0]} to {common[-1]}")
C=np.column_stack([aligned[s]['close'].reindex(common).values for s in WL])
R=np.column_stack([aligned[s]['pct_change'].reindex(common).fillna(0).values for s in WL])
Lo=np.column_stack([aligned[s]['low'].reindex(common).values for s in WL])
Hi=np.column_stack([aligned[s]['high'].reindex(common).values for s in WL])
dxy=aligned['DXY']['close'].values if 'DXY' in aligned else None
vix=aligned['VIX']['close'].values if 'VIX' in aligned else None
cny=aligned['USDCNY']['close'].values if 'USDCNY' in aligned else None
T,N=R.shape; print(f"Panel: {T}d x {N}a")
fwd1=np.full_like(R,np.nan)
for i in range(T-1): fwd1[i]=R[i+1]

def ic_metrics(fac, fwd):
    """Compute IC and ICIR across valid cross-sections."""
    ics=[]
    for i in range(T):
        f=fac[i]; r=fwd[i]
        m=~np.isnan(f)&~np.isnan(r)
        if np.sum(m)>=8:
            ics.append(np.corrcoef(f[m],r[m])[0,1])
    ics=np.array(ics)
    ic=np.nanmean(ics) if len(ics)>0 else 0
    icir=ic/np.maximum(np.nanstd(ics),1e-10) if len(ics)>1 else 0
    hit=np.mean(np.sign(ics)>0) if len(ics)>0 else 0
    cov = np.nanmean(~np.isnan(fac))
    return {'ic':float(ic),'icir':float(icir),'hit':float(hit),'n_dates':int(len(ics)),'coverage':float(cov)}

# compute all factors
print("\nComputing factors...")

# 1. mom_120d_skip5
print("1. mom_120d_skip5")
m120_valid = ic_metrics(mom120, fwd1)

# 2. mom_10d_skip5
print("2. mom_10d_skip5")
m10_valid = ic_metrics(mom10, fwd1)

# 3. vol_z_20d
print("3. vol_z_20d")
vz_valid = ic_metrics(volz, fwd1)

# 4. kaufman_eff_20d
print("4. kaufman_eff_20d")
kf_valid = ic_metrics(kauf, fwd1)

# 5. bb_width_20d
print("5. bb_width_20d")
bbw_valid = ic_metrics(bbw, fwd1)

# 6. beta_VIX_60
print("6. beta_VIX_60")
bv_valid = ic_metrics(bv, fwd1)

# 7. cny_beta_60
print("7. cny_beta_60")
cb_valid = ic_metrics(cb, fwd1)

# 8. ac1_120d (autocorrelation 120d)
print("8. ac1_120d")
ac1_valid = ic_metrics(ac1, fwd1)

# 9. skew_20d
print("9. skew_20d")
sk_valid = ic_metrics(sk, fwd1)

# 10. rng_pos_20d
print("10. rng_pos_20d")
rp_valid = ic_metrics(rp, fwd1)

# 11. kurt_20d
print("11. kurt_20d")
kt_valid = ic_metrics(kt, fwd1)

# 12. streak_len_14
print("12. streak_len_14")
st_valid = ic_metrics(st, fwd1)

# 13. days_since_high_60
print("13. days_since_high_60")
dsh_valid = ic_metrics(dsh, fwd1)

# 14. dxy_corr_change_20_60
print("14. dxy_corr_change_20_60")
dc_valid = ic_metrics(dc, fwd1)

# 15. mom_10_vixreg
print("15. mom_10_vixreg")
mv_valid = ic_metrics(mv, fwd1)

# 16. vix_beta_cond_60x20
print("16. vix_beta_cond_60x20")
vc_valid = ic_metrics(vc, fwd1)

# 17. vol_of_vol
print("17. vol_of_vol")
vov_valid = ic_metrics(vov, fwd1)

# Print results
results = {
    'mom_120d_skip5': m120_valid,
    'mom_10d_skip5': m10_valid,
    'vol_z_20d': vz_valid,
    'kaufman_eff_20d': kf_valid,
    'bb_width_20d': bbw_valid,
    'beta_VIX_60': bv_valid,
    'cny_beta_60': cb_valid,
    'ac1_120d': ac1_valid,
    'skew_20d': sk_valid,
    'rng_pos_20d': rp_valid,
    'kurt_20d': kt_valid,
    'streak_len_14': st_valid,
    'days_since_high_60': dsh_valid,
    'dxy_corr_change_20_60': dc_valid,
    'mom_10_vixreg': mv_valid,
    'vix_beta_cond_60x20': vc_valid,
    'vol_of_vol': vov_valid,
}

print("\n\n=== REVALIDATION RESULTS ===")
print(f"{'Factor':<25} {'IC':>8} {'ICIR':>8} {'Hit%':>6} {'NDates':>8} {'Cov%':>6} {'PASS':>6}")
print("="*75)
gate_ic=0.007; gate_icir=0.084
all_pass=True
for fid, m in results.items():
    ic=abs(m['ic']); icir=abs(m['icir'])
    pass_=ic>=gate_ic and icir>=gate_icir
    if not pass_: all_pass=False
    print(f"{fid:<25} {m['ic']:>8.4f} {m['icir']:>8.4f} {m['hit']:>6.3f} {m['n_dates']:>8d} {m['coverage']:>6.3f} {'YES' if pass_ else 'NO':>6}")

print(f"\nAll pass: {all_pass}")