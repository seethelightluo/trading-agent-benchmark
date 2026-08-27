import os
# append remaining factor revalidation code to the full validation script
append_code = '''
# === 5. cny_beta_60 ===
print("\\n=== 5. cny_beta_60 ===",flush=True)
beta_cny=rets.rolling(60).cov(cny_ret)/cny_ret.rolling(60).var()
report('cny_beta_60',beta_cny,start=FULL)
report('cny_beta_60',beta_cny,start=RECENT)
report('cny_beta_60',beta_cny,start=RECENT2)

# === 6. vol_z_20d ===
print("\\n=== 6. vol_z_20d ===",flush=True)
vol20=rets.rolling(20).std()
vol60=rets.rolling(60).std()
vol_z=(vol20-vol60.rolling(20).mean())/(vol60.rolling(20).std().replace(0,np.nan))
report('vol_z_20d',vol_z,start=FULL)
report('vol_z_20d',vol_z,start=RECENT)
report('vol_z_20d',vol_z,start=RECENT2)

# === 7. ac1_120d ===
print("\\n=== 7. ac1_120d ===",flush=True)
ac1=rets.rolling(120).apply(lambda x: x.autocorr() if len(x)>5 else 0, raw=False)
report('ac1_120d(neg=use)',-ac1,start=FULL)
report('ac1_120d(neg=use)',-ac1,start=RECENT)
report('ac1_120d(neg=use)',-ac1,start=RECENT2)

# === 8. mom_10d_skip5 ===
print("\\n=== 8. mom_10d_skip5 ===",flush=True)
report('mom_10d_skip5',close/close.shift(15)-1,start=FULL)
report('mom_10d_skip5',close/close.shift(15)-1,start=RECENT)
report('mom_10d_skip5',close/close.shift(15)-1,start=RECENT2)

# === 9. dxy_corr_change_20_60 ===
print("\\n=== 9. dxy_corr_change_20_60 ===",flush=True)
corr20=rets.rolling(20).corr(dxy_ret)
corr60=rets.rolling(60).corr(dxy_ret)
dxy_corr_chg=corr20-corr60
report('dxy_corr_change_20_60',dxy_corr_chg,start=FULL)
report('dxy_corr_change_20_60',dxy_corr_chg,start=RECENT)
report('dxy_corr_change_20_60',dxy_corr_chg,start=RECENT2)

# === 10. skew_20d ===
print("\\n=== 10. skew_20d ===",flush=True)
def roll_skew(s):
    return s.rolling(20).skew()
skew=rets.rolling(20).skew()
report('skew_20d',skew,start=FULL)
report('skew_20d',skew,start=RECENT)
report('skew_20d',skew,start=RECENT2)

# === 11. kurt_20d ===
print("\\n=== 11. kurt_20d ===",flush=True)
kurt=rets.rolling(20).kurt()
report('kurt_20d',kurt,start=FULL)
report('kurt_20d',kurt,start=RECENT)
report('kurt_20d',kurt,start=RECENT2)

# === 12. days_since_high_60 ===
print("\\n=== 12. days_since_high_60 ===",flush=True)
roll_high=close.rolling(60).max()
dsh=(close.shift(1)>close.rolling(60).max().shift(1)) 
days_since_high = dsh.cumsum() - dsh.cumsum().where(dsh).ffill()
report('days_since_high(neg=use)',-days_since_high,start=FULL)
report('days_since_high(neg=use)',-days_since_high,start=RECENT)

# === 13. streak_len_14 ===
print("\\n=== 13. streak_len_14 ===",flush=True)
sign=(rets>0).astype(int)
streak=sign*(sign.groupby((sign!=sign.shift()).cumsum()).cumcount()+1)
report('streak_len_14',streak,start=FULL)
report('streak_len_14',streak,start=RECENT)

# === 14. vix_roc_20d ===
print("\\n=== 14. vix_roc_20d ===",flush=True)
vix_roc=vix.pct_change(20)
# cross-asset: for each asset use same macro value
vix_roc_asset=vix_roc.to_frame().reindex(close.columns,axis=1).ffill()
report('vix_roc_20d(neg=use)',-vix_roc_asset,start=FULL)
report('vix_roc_20d(neg=use)',-vix_roc_asset,start=RECENT)

# === 15. rng_pos_20d (range position) ===
print("\\n=== 15. rng_pos_20d ===",flush=True)
hh=high.rolling(20).max()
ll=low.rolling(20).min()
rng_pos=(close-ll)/(hh-ll).replace(0,np.nan)
report('rng_pos_20d(pro)',rng_pos,start=FULL)
report('rng_pos_20d(pro)',rng_pos,start=RECENT)
report('rng_pos_20d(contra)',rng_pos,start=RECENT,flip=True)

# === 16. mom_10_vixreg (vix-regressed momentum) ===
print("\\n=== 16. mom_10_vixreg ===",flush=True)
mom10=close/close.shift(15)-1
vix_mom10=vix_ret.rolling(10).mean()
# regression residual: mom10 - beta*vix_mom10 (approx by demeaning)
c=rets.rolling(60).cov(vix_ret)/vix_ret.rolling(60).var()
resid=mom10 - c*vix_mom10.reindex(close.index).ffill().fillna(0)
report('mom_10_vixreg',resid,start=FULL)
report('mom_10_vixreg',resid,start=RECENT)
'''
with open('scripts/miner_1_20340720_validate_full.py','a') as fh:
    fh.write(append_code)
print("appended")