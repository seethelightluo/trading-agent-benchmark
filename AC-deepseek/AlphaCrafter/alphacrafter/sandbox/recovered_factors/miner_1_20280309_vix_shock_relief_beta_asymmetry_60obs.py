"""Single idea: asymmetric VIX-shock beta, validated through 2028-03-08."""
import numpy as np,pandas as pd,warnings
warnings.filterwarnings('ignore')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2028-03-08')
def ld(a,c='close',idx=False): return pd.read_csv(('../persistent/index_data/' if idx else '../persistent/stock_data/')+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:END,c].astype(float)
p={a:ld(a) for a in A};r=pd.DataFrame({a:p[a].pct_change() for a in A});m=r.median(axis=1);v={a:ld(a,'volume').replace(0,np.nan) for a in A}
def beta(x,y,n,mi,mask=None):
 if mask is not None:x=x.where(mask);y=y.where(mask)
 return x.rolling(n,min_periods=mi).cov(y)/y.rolling(n,min_periods=mi).var()
def csres(x,z):
 o=x*np.nan
 for t in x.index:
  q=pd.concat([x.loc[t],z.loc[t]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,1].var()>0:o.loc[t,q.index]=q.iloc[:,0]-np.polyval(np.polyfit(q.iloc[:,1],q.iloc[:,0],1),q.iloc[:,1])
 return o
def cmean(x,mask,w,mi): return x.where(mask).rolling(w,min_periods=mi).mean()
# Candidate: difference between asset sensitivity to VIX jumps and VIX relief over 60 observations.
vix=ld('VIX',idx=True).pct_change(); f=pd.DataFrame({a:beta(r[a],vix,60,12,vix>0)-beta(r[a],vix,60,12,vix<0) for a in A})
s20=r.rolling(20,min_periods=15).std();s60=r.rolling(60,min_periods=45).std();fast=pd.DataFrame({a:(p[a]/p[a].shift(20)-1)/s20[a] for a in A});slow=pd.DataFrame({a:(p[a]/p[a].shift(60)-1)/s60[a] for a in A})
lib={'risk_adjusted_trend_20d':fast,'relative_volume_participation_20d':pd.DataFrame({a:np.log(v[a]/v[a].rolling(20,min_periods=15).mean()) for a in A}),'realized_volatility_20obs':s20,'volnorm_reversal_5obs':pd.DataFrame({a:-(p[a]/p[a].shift(5)-1)/r[a].rolling(5,min_periods=4).std() for a in A}),'risk_adjusted_trend_acceleration_20_60d':fast-slow,'return_persistence_autocorr_20obs':r.rolling(20,min_periods=15).corr(r.shift()),'return_directional_efficiency_20obs':r.rolling(20,min_periods=15).sum().abs()/r.abs().rolling(20,min_periods=15).sum(),'return_sign_balance_20obs':(r>0).rolling(20,min_periods=15).mean()-.5,'volatility_clustering_autocorr_20obs':r.abs().rolling(20,min_periods=15).corr(r.abs().shift()),'relative_liquidity_stress_20_60obs':pd.DataFrame({a:np.log(v[a].rolling(20,min_periods=15).mean()/v[a].rolling(60,min_periods=45).mean()) for a in A}),'downside_volume_participation_asymmetry_60obs':pd.DataFrame({a:np.log(cmean(v[a],r[a]<0,60,15)/cmean(v[a],r[a]>=0,60,15)) for a in A})}
b=pd.DataFrame({a:beta(r[a],m,60,45) for a in A});dn=m<0;up=~dn
ca=pd.DataFrame({a:r[a].where(dn).rolling(60,min_periods=10).corr(m.where(dn))-r[a].where(up).rolling(60,min_periods=10).corr(m.where(up)) for a in A});lib['correlation_asymmetry_60obs']=ca
res=pd.DataFrame({a:r[a]-(r[a].rolling(60,min_periods=45).mean()+b[a]*(m-m.rolling(60,min_periods=45).mean())) for a in A});lib['residual_downside_semivol_share_60obs']=res.clip(upper=0).pow(2).rolling(60,min_periods=45).mean().pow(.5)/res.pow(2).rolling(60,min_periods=45).mean().pow(.5);lib['vol_orthogonal_median_beta_60obs']=csres(b,s20)
D=r.std(axis=1);lib['dispersion_sensitivity_20obs']=pd.DataFrame({a:[r[a].loc[:t].tail(20).corr(D.loc[:t].tail(20),method='spearman') if pd.concat([r[a].loc[:t].tail(20),D.loc[:t].tail(20)],axis=1).dropna().shape[0]>=15 else np.nan for t in r.index] for a in A},index=r.index)
lib['excess_downside_beta_ca_orthogonal_60obs']=csres(pd.DataFrame({a:beta(r[a],m,60,10,dn)-b[a] for a in A}),ca);lib['adaptive_vix_relief_beta_change_25_60obs']=pd.DataFrame({a:beta(r[a],vix,25,8,vix<0)-beta(r[a],vix,60,18,vix<0) for a in A})
short=pd.DataFrame({a:np.log(cmean(v[a],r[a]<0,20,5)/cmean(v[a],r[a]>=0,20,5)) for a in A});long=pd.DataFrame({a:np.log(cmean(v[a],r[a]<0,60,12)/cmean(v[a],r[a]>=0,60,12)) for a in A});lib['inverted_downside_volume_participation_acceleration_20_60obs']=long-short
g=pd.DataFrame({a:ld(a,'open')/p[a].shift()-1 for a in A});body=pd.DataFrame({a:p[a]/ld(a,'open')-1 for a in A});interaction=-g*body;lib['overnight_daytime_reversal_concordance_20obs']=interaction.rolling(20,min_periods=15).mean()/interaction.rolling(20,min_periods=15).std()
print('CANDIDATE vix_shock_relief_beta_asymmetry_60obs visible_through',END.date(),'assets=15')
best=None
for h in [1,5,10,20]:
 y=pd.DataFrame({a:p[a].shift(-h)/p[a]-1 for a in A});z=[];cv=[]
 for t in f.index:
  q=pd.concat([f.loc[t],y.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8:z.append((t,q.iloc[:,0].corr(q.iloc[:,1],method='spearman')));cv.append(len(q)/15)
 x=pd.Series(dict(z));ic=x.mean();ir=ic/x.std(ddof=1);print(f'H={h} dates={len(x)} IC={ic:.6f} ICIR={ir:.6f} hit={(x>0).mean():.4f} coverage={np.mean(cv):.4f} mean_instruments={15*np.mean(cv):.2f}')
 if best is None or abs(ic*ir)>abs(best[1].mean()*(best[1].mean()/best[1].std(ddof=1))):best=(h,x)
h,x=best;print('BEST_HORIZON',h)
for n,lo,hi in [('2020','2020','2021'),('2021-22','2021','2023'),('2023-24','2023','2025'),('2025-current','2025','2030')]:
 z=x[(x.index>=lo)&(x.index<hi)];print(f'REGIME {n} dates={len(z)} IC={z.mean():.6f} ICIR={z.mean()/z.std(ddof=1):.6f} hit={(z>0).mean():.4f}')
rk=f.rank(axis=1,pct=True);to=[]
for i in range(1,len(rk)):
 q=pd.concat([rk.iloc[i-1],rk.iloc[i]],axis=1).dropna()
 if len(q)>=8:to.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print(f'turnover={np.mean(to):.6f}; signal_cells={f.notna().sum().sum()}/{f.size}={f.notna().mean().mean():.4f}')
mx=-1
for n,o in lib.items():
 q=pd.concat([f.stack(),o.stack()],axis=1).replace([np.inf,-np.inf],np.nan).dropna();rho=q.iloc[:,0].corr(q.iloc[:,1],method='spearman');print(f'LIB {n} rho={rho:.6f} cells={len(q)}')
 if abs(rho)>mx:mx=abs(rho);who=n;cells=len(q)
print(f'max_abs_library_correlation={mx:.6f}; closest={who}; evidence_cells={cells}; library_count={len(lib)}')
