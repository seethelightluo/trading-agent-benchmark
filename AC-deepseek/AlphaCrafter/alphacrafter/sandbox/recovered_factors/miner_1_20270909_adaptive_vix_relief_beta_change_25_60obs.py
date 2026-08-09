"""Validation: adaptive VIX-relief beta change (25 vs 60); vectorized full-library gate."""
import numpy as np,pandas as pd,json,glob,warnings
warnings.filterwarnings('ignore')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-09-08')
def ld(a,c='close',idx=False):
 return pd.read_csv(('../persistent/index_data/' if idx else '../persistent/stock_data/')+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,c].astype(float)
p={a:ld(a) for a in A}; r=pd.DataFrame({a:p[a].pct_change() for a in A}); m=r.median(axis=1); vix=ld('VIX',idx=True).pct_change(); vol=r.rolling(20,min_periods=15).std()
def beta(x,y,n,minn,mask=None):
 if mask is not None: x=x.where(mask);y=y.where(mask)
 return x.rolling(n,min_periods=minn).cov(y)/y.rolling(n,min_periods=minn).var()
# Candidate: short-minus-long sensitivity only on VIX declines.
f=pd.DataFrame({a:beta(r[a],vix,25,8,vix<0)-beta(r[a],vix,60,18,vix<0) for a in A})
fast=pd.DataFrame({a:(p[a]/p[a].shift(20)-1)/vol[a] for a in A}); slow=pd.DataFrame({a:(p[a]/p[a].shift(60)-1)/r[a].rolling(60,min_periods=45).std() for a in A})
lib={'risk_adjusted_trend_20d':fast,'relative_volume_participation_20d':pd.DataFrame({a:np.log(ld(a,'volume').replace(0,np.nan)/ld(a,'volume').replace(0,np.nan).rolling(20,min_periods=15).mean()) for a in A}),'realized_volatility_20obs':vol,'volnorm_reversal_5obs':pd.DataFrame({a:-(p[a]/p[a].shift(5)-1)/r[a].rolling(5,min_periods=4).std() for a in A}),'risk_adjusted_trend_acceleration_20_60d':fast-slow,'return_persistence_autocorr_20obs':r.rolling(20,min_periods=15).corr(r.shift()),'return_directional_efficiency_20obs':r.rolling(20,min_periods=15).sum().abs()/r.abs().rolling(20,min_periods=15).sum(),'return_sign_balance_20obs':(r>0).rolling(20,min_periods=15).mean()-0.5,'volatility_clustering_autocorr_20obs':r.abs().rolling(20,min_periods=15).corr(r.abs().shift()),'relative_liquidity_stress_20_60obs':pd.DataFrame({a:np.log(ld(a,'volume').replace(0,np.nan).rolling(20,min_periods=15).mean()/ld(a,'volume').replace(0,np.nan).rolling(60,min_periods=45).mean()) for a in A})}
b=pd.DataFrame({a:beta(r[a],m,60,60) for a in A}); dn=m<0;up=m>=0
ca=pd.DataFrame({a:r[a].where(dn).rolling(60,min_periods=10).corr(m.where(dn))-r[a].where(up).rolling(60,min_periods=10).corr(m.where(up)) for a in A})
# residual downside semivol using rolling beta/intercept approximation
res=pd.DataFrame({a:r[a]-(r[a].rolling(60,min_periods=60).mean()+b[a]*(m-m.rolling(60,min_periods=60).mean())) for a in A})
ds=(res.clip(upper=0).pow(2).rolling(60,min_periods=60).mean().pow(.5)/res.pow(2).rolling(60,min_periods=60).mean().pow(.5))
lib['correlation_asymmetry_60obs']=ca;lib['residual_downside_semivol_share_60obs']=ds
def csres(x,z):
 out=x*np.nan
 for t in x.index:
  q=pd.concat([x.loc[t],z.loc[t]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,1].var()>0: out.loc[t,q.index]=q.iloc[:,0]-np.polyval(np.polyfit(q.iloc[:,1],q.iloc[:,0],1),q.iloc[:,1])
 return out
lib['vol_orthogonal_median_beta_60obs']=csres(b,vol)
D=r.std(axis=1);lib['dispersion_sensitivity_20obs']=pd.DataFrame({a:r[a].rolling(20,min_periods=15).corr(D,method='spearman') for a in A}) if False else pd.DataFrame({a:[r[a].loc[:t].tail(20).corr(D.loc[:t].tail(20),method='spearman') if pd.concat([r[a].loc[:t].tail(20),D.loc[:t].tail(20)],axis=1).dropna().shape[0]>=15 else np.nan for t in r.index] for a in A},index=r.index)
ex=pd.DataFrame({a:beta(r[a],m,60,10,dn)-b[a] for a in A});lib['excess_downside_beta_ca_orthogonal_60obs']=csres(ex,ca)
print('CANDIDATE adaptive_vix_relief_beta_change_25_60obs visible_through',END.date(),'assets=15')
best=None
for h in [1,5,10,20]:
 y=pd.DataFrame({a:p[a].shift(-h)/p[a]-1 for a in A}); vals=[]; cov=[]
 for t in f.index:
  q=pd.concat([f.loc[t],y.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8: vals.append((t,q.iloc[:,0].corr(q.iloc[:,1],method='spearman')));cov.append(len(q)/15)
 x=pd.Series(dict(vals));ic=x.mean();ir=ic/x.std(ddof=1);print(f'H={h} dates={len(x)} IC={ic:.6f} ICIR={ir:.6f} hit={(x>0).mean():.4f} coverage={np.mean(cov):.4f} mean_instruments={15*np.mean(cov):.2f}')
 if best is None or abs(ic*ir)>abs(best[1].mean()*(best[1].mean()/best[1].std(ddof=1))):best=(h,x)
h,x=best;print('BEST_HORIZON',h)
for n,lo,hi in [('2020','2020','2021'),('2021-22','2021','2023'),('2023-24','2023','2025'),('2025-current','2025','2030')]:
 z=x[(x.index>=lo)&(x.index<hi)];print(f'REGIME {n} dates={len(z)} IC={z.mean():.6f} ICIR={z.mean()/z.std(ddof=1):.6f} hit={(z>0).mean():.4f}')
rk=f.rank(axis=1,pct=True);turn=[]
for i in range(1,len(rk)):
 q=pd.concat([rk.iloc[i-1],rk.iloc[i]],axis=1).dropna()
 if len(q)>=8:turn.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print(f'turnover={np.mean(turn):.6f}; signal_cells={f.notna().sum().sum()}/{f.size}={f.notna().mean().mean():.4f}')
mx=-1
for n,o in lib.items():
 q=pd.concat([f.stack(),o.stack()],axis=1).replace([np.inf,-np.inf],np.nan).dropna();rho=q.iloc[:,0].corr(q.iloc[:,1],method='spearman');print(f'LIB {n} rho={rho:.6f} cells={len(q)}')
 if abs(rho)>mx:mx=abs(rho);who=n;cells=len(q)
print(f'max_abs_library_correlation={mx:.6f}; closest={who}; evidence_cells={cells}; library_count={len(lib)}')
