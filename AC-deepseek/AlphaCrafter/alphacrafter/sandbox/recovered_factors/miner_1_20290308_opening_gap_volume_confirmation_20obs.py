"""Single idea: 20-observation opening-gap volume confirmation.
The signal is the rolling Spearman association between signed overnight gaps and
same-day abnormal log volume. It measures whether overnight repricing attracts
confirming participation, a distinct event-ordering mechanism."""
import numpy as np,pandas as pd,warnings
warnings.filterwarnings('ignore')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2029-03-07')
def ld(a,c='close',idx=False):
 p=('../persistent/index_data/' if idx else '../persistent/stock_data/')+a+'.csv'
 return pd.read_csv(p,parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:END,c].astype(float)
p={a:ld(a) for a in A}; r=pd.DataFrame({a:p[a].pct_change() for a in A}); med=r.median(axis=1)
v={a:ld(a,'volume').replace(0,np.nan) for a in A}; g=pd.DataFrame({a:ld(a,'open')/p[a].shift()-1 for a in A})
vs=pd.DataFrame({a:np.log(v[a]/v[a].rolling(20,min_periods=15).mean()) for a in A})
# Native observation rolling Spearman avoids calendar synchronicity artifacts.
def rollcorr(x,y,n=20,mi=15): return x.rolling(n,min_periods=mi).corr(y,method='spearman')
f=pd.DataFrame({a:rollcorr(g[a],vs[a]) for a in A})
def beta(x,y,n,mi,mask=None):
 if mask is not None:x=x.where(mask);y=y.where(mask)
 return x.rolling(n,min_periods=mi).cov(y)/y.rolling(n,min_periods=mi).var()
def csres(x,z):
 o=x*np.nan
 for t in x.index:
  q=pd.concat([x.loc[t],z.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8 and q.iloc[:,1].var()>0:o.loc[t,q.index]=q.iloc[:,0]-np.polyval(np.polyfit(q.iloc[:,1],q.iloc[:,0],1),q.iloc[:,1])
 return o
def cmean(x,mask,w,mi): return x.where(mask).rolling(w,min_periods=mi).mean()
s20=r.rolling(20,min_periods=15).std();s60=r.rolling(60,min_periods=45).std()
fast=pd.DataFrame({a:(p[a]/p[a].shift(20)-1)/s20[a] for a in A});slow=pd.DataFrame({a:(p[a]/p[a].shift(60)-1)/s60[a] for a in A})
lib={'risk_adjusted_trend_20d':fast,'relative_volume_participation_20d':vs,'realized_volatility_20obs':s20,'volnorm_reversal_5obs':pd.DataFrame({a:-(p[a]/p[a].shift(5)-1)/r[a].rolling(5,min_periods=4).std() for a in A}),'risk_adjusted_trend_acceleration_20_60d':fast-slow,'return_persistence_autocorr_20obs':pd.DataFrame({a:rollcorr(r[a],r[a].shift()) for a in A}),'return_directional_efficiency_20obs':r.rolling(20,min_periods=15).sum().abs()/r.abs().rolling(20,min_periods=15).sum(),'return_sign_balance_20obs':(r>0).rolling(20,min_periods=15).mean()-.5,'volatility_clustering_autocorr_20obs':pd.DataFrame({a:rollcorr(r[a].abs(),r[a].abs().shift()) for a in A}),'relative_liquidity_stress_20_60obs':pd.DataFrame({a:np.log((r[a].abs()/v[a]).rolling(20,min_periods=15).mean()/(r[a].abs()/v[a]).rolling(60,min_periods=45).mean()) for a in A})}
b=pd.DataFrame({a:beta(r[a],med,60,45) for a in A}); dn=med<0;up=~dn
ca=pd.DataFrame({a:rollcorr(r[a].where(dn),med.where(dn),60,10)-rollcorr(r[a].where(up),med.where(up),60,10) for a in A});lib['correlation_asymmetry_60obs']=ca
res=pd.DataFrame({a:r[a]-(r[a].rolling(60,min_periods=45).mean()+b[a]*(med-med.rolling(60,min_periods=45).mean())) for a in A});lib['residual_downside_semivol_share_60obs']=res.clip(upper=0).pow(2).rolling(60,min_periods=45).mean().pow(.5)/res.pow(2).rolling(60,min_periods=45).mean().pow(.5);lib['vol_orthogonal_median_beta_60obs']=csres(b,s20)
D=r.std(axis=1);lib['dispersion_sensitivity_20obs']=pd.DataFrame({a:rollcorr(r[a],D) for a in A});lib['excess_downside_beta_ca_orthogonal_60obs']=csres(pd.DataFrame({a:beta(r[a],med,60,10,dn)-b[a] for a in A}),ca)
vix=ld('VIX',idx=True).pct_change();lib['adaptive_vix_relief_beta_change_25_60obs']=pd.DataFrame({a:beta(r[a],vix,25,8,vix<0)-beta(r[a],vix,60,18,vix<0) for a in A});lib['vix_shock_relief_beta_asymmetry_60obs']=pd.DataFrame({a:beta(r[a],vix,60,12,vix>0)-beta(r[a],vix,60,12,vix<0) for a in A})
dxy=ld('DXY',idx=True).pct_change(); mt=med.rolling(20,min_periods=15).mean();lib['dxy_median_trend_regime_beta_spread_60obs']=pd.DataFrame({a:beta(r[a],dxy,60,12,mt>0)-beta(r[a],dxy,60,12,mt<=0) for a in A});rv=s20/s20.rolling(60,min_periods=45).median();lib['dxy_relative_vol_regime_beta_spread_60obs']=pd.DataFrame({a:beta(r[a],dxy,60,12,rv[a]>1)-beta(r[a],dxy,60,12,rv[a]<=1) for a in A})
dva=pd.DataFrame({a:np.log(cmean(v[a],r[a]<0,60,15)/cmean(v[a],r[a]>=0,60,15)) for a in A});lib['downside_volume_participation_asymmetry_60obs']=dva;lib['inverted_downside_volume_participation_acceleration_20_60obs']=pd.DataFrame({a:dva[a]-np.log(cmean(v[a],r[a]<0,20,8)/cmean(v[a],r[a]>=0,20,8)) for a in A})
rg=pd.DataFrame({a:(ld(a,'high')-ld(a,'low'))/p[a] for a in A}); state=np.sign(np.log(s20.median(axis=1)/s20.median(axis=1).rolling(60,min_periods=45).median()));lib['inverted_dispersion_regime_range_state_20_60obs']=pd.DataFrame({a:-np.log(rg[a].rolling(20,min_periods=15).mean()/rg[a].rolling(60,min_periods=45).mean())*state for a in A})
print('CANDIDATE opening_gap_volume_confirmation_20obs visible_through',END.date(),'assets=15')
best=None
for h in [1,5,10,20]:
 y=pd.DataFrame({a:p[a].shift(-h)/p[a]-1 for a in A}); z=[];cv=[]
 for t in f.index:
  q=pd.concat([f.loc[t],y.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8:z.append((t,q.iloc[:,0].corr(q.iloc[:,1],method='spearman')));cv.append(len(q)/15)
 x=pd.Series(dict(z));ic=x.mean();ir=ic/x.std(ddof=1);print(f'H={h} dates={len(x)} IC={ic:.6f} ICIR={ir:.6f} hit={(x>0).mean():.4f} coverage={np.mean(cv):.4f} mean_instruments={15*np.mean(cv):.2f}')
 if best is None or abs(ic*ir)>abs(best[1].mean()*(best[1].mean()/best[1].std(ddof=1))):best=(h,x)
h,x=best;print('BEST_HORIZON',h)
for n,lo,hi in [('2020-21','2020','2022'),('2022-23','2022','2024'),('2024-25','2024','2026'),('2026-current','2026','2030')]:
 z=x[(x.index>=lo)&(x.index<hi)];print(f'REGIME {n} dates={len(z)} IC={z.mean():.6f} ICIR={z.mean()/z.std(ddof=1):.6f} hit={(z>0).mean():.4f}')
rk=f.rank(axis=1,pct=True);to=[]
for i in range(1,len(rk)):
 q=pd.concat([rk.iloc[i-1],rk.iloc[i]],axis=1).dropna()
 if len(q)>=8:to.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print(f'turnover={np.mean(to):.6f}; signal_cells={f.notna().sum().sum()}/{f.size}={f.notna().mean().mean():.4f}')
mx=-1; missing=[]
for n,o in lib.items():
 q=pd.concat([f.stack(),o.stack()],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(q)==0:missing.append(n);continue
 rho=q.iloc[:,0].corr(q.iloc[:,1],method='spearman');print(f'LIB {n} rho={rho:.6f} cells={len(q)}')
 if abs(rho)>mx:mx=abs(rho);who=n;cells=len(q)
print(f'max_abs_library_correlation={mx:.6f}; closest={who}; evidence_cells={cells}; library_count={len(lib)}; missing={missing}')
"""
