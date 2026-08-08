from pathlib import Path
s=Path('scripts/miner_2_20300530_downside_close_location_participation_recovery_residual_20.py').read_text()
s=s.replace('visible through 2030-05-29','visible through 2030-06-12').replace("E=pd.Timestamp('2030-05-29')","E=pd.Timestamp('2030-06-12')")
s=s.replace('downside_close_location_participation_recovery_residual_20','negative_overnight_gap_recovery_residual_20')
a=s.index('# Candidate:')
b=s.index('# Reconstructions of currently admitted',a)
new='''# Candidate: normalized recovery following negative overnight gaps. A gap down is
# treated as an information/shock event; subsequent close-to-close performance is
# averaged with severity weights and residualized against common risk and trend.
O=pd.DataFrame({a:rd(a,'open') for a in A})
gap=O/P.shift(1)-1
gapsev=(-gap.shift(1)/(v.shift(1)+1e-12)).clip(0,4)
raw=R.mul(gapsev).rolling(20,min_periods=10).sum().div(gapsev.rolling(20,min_periods=10).sum().replace(0,np.nan),axis=0)/(v+1e-12)
F=res(raw,v,peer,dba,P/P.shift(20)-1)
'''
s=s[:a]+new+s[b:]
# Add exact predecessor to novelty library, after library dict is completed and before next comment.
needle='# Additional admitted signals as of the validation date.'
old='''# Prior admitted factor, included in full novelty screen.
V0=pd.DataFrame({a:rd(a,'volume') for a in A})
part0=(V0/V0.rolling(20,min_periods=15).mean()).clip(0.5,2.0).fillna(1.0)
loc0=pd.DataFrame({a:(P[a]-rd(a,'low'))/(rd(a,'high')-rd(a,'low')).replace(0,np.nan) for a in A}).clip(0,1).fillna(0.5)
pl0=(-R.shift(1)/(v.shift(1)+1e-12)).clip(0,4)
q0=(0.5+loc0.shift(1)).mul(part0.shift(1)); w0=pl0*q0
raw0=R.mul(w0).rolling(20,min_periods=10).sum().div(w0.rolling(20,min_periods=10).sum().replace(0,np.nan),axis=0)/(v+1e-12)
L['downside_close_location_participation_recovery_residual_20']=res(raw0,v,peer,dba,P/P.shift(20)-1)
'''
s=s.replace(needle,old+needle)
Path('scripts/miner_2_20300613_negative_overnight_gap_recovery_residual_20.py').write_text(s)
print('written')
