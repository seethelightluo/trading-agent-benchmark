"""Candidate: trend-orthogonal residual directional close-location asymmetry acceleration.
Tests whether the balance of closing strength following idiosyncratic gains vs losses
is changing, after removing contemporaneous cross-sectional risk-adjusted trend."""
p='scripts/miner_3_20340511_continuous_close_location_pressure_library_audit.py'
s=open(p,encoding='utf-8').read()
s=s.replace("END=pd.Timestamp('2034-05-10')", "END=pd.Timestamp('2035-02-28')")
old="f=clv.rolling(10,min_periods=8).mean()-clv.rolling(60,min_periods=45).mean()"
new="""# Closing-location separation of idiosyncratic up and down sessions, fast versus slow.
up20=clv.where(res>=0).rolling(20,min_periods=8).mean()
dn20=clv.where(res<0).rolling(20,min_periods=8).mean()
up60=clv.where(res>=0).rolling(60,min_periods=22).mean()
dn60=clv.where(res<0).rolling(60,min_periods=22).mean()
raw=(up20-dn20)-(up60-dn60)
# Remove the cross-sectional component attributable to established 20d risk-adjusted trend.
trend=(p/p.shift(20)-1)/v
nf=pd.DataFrame(index=p.index,columns=A,dtype=float)
for t in p.index:
 q=pd.concat([raw.loc[t],trend.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(q)>=8 and q.iloc[:,1].nunique()>1:
  b0=np.polyfit(q.iloc[:,1],q.iloc[:,0],1); nf.loc[t]=raw.loc[t]-(b0[1]+b0[0]*trend.loc[t])
f=nf
print('CANDIDATE trend_orthogonal_residual_directional_close_location_asymmetry_acceleration_20_60obs')"""
assert old in s
s=s.replace(old,new)
exec(compile(s,'miner_1_trend_orthogonal_residual_directional_close_location_asymmetry_acceleration','exec'))
"""
