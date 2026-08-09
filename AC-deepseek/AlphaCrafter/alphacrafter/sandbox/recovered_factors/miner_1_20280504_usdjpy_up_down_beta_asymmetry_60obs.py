"""One factor: USDJPY up/down beta asymmetry over 60 visible observations."""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END='2028-05-03'
def load(a,macro=False):
 return pd.read_csv(('../persistent/index_data/' if macro else '../persistent/stock_data/')+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END]
d={a:load(a) for a in A};c=pd.DataFrame({a:x.close for a,x in d.items()});v=pd.DataFrame({a:x.volume.replace(0,np.nan) for a,x in d.items()});o=pd.DataFrame({a:x.open for a,x in d.items()});r=c.pct_change();med=r.median(1)
def cond_beta(x,mask,w=60,minn=12):
 x=x.where(mask); n=mask.astype(float).rolling(w,min_periods=minn).sum(); sx=x.rolling(w,min_periods=minn).sum(); den=(x*x).rolling(w,min_periods=minn).sum()-sx*sx/n
 return pd.DataFrame({a:((x*r[a].where(mask)).rolling(w,min_periods=minn).sum()-sx*r[a].where(mask).rolling(w,min_periods=minn).sum()/n)/den for a in A})
jpy=load('USDJPY',True).close.pct_change().reindex(r.index);f=cond_beta(jpy,jpy>0)-cond_beta(jpy,jpy<0)
# admitted library signals, reconstructed point-in-time
vol20=r.rolling(20,min_periods=15).std();ret20=r.rolling(20,min_periods=15).sum();ret60=r.rolling(60,min_periods=45).sum();vol60=r.rolling(60,min_periods=45).std();rv=np.log(v/v.rolling(20,min_periods=15).mean())
L={'risk_adjusted_trend_20d':ret20/vol20,'relative_volume_participation_20d':rv,'realized_volatility_20obs':vol20,'volnorm_reversal_5obs':-r.rolling(5,min_periods=4).sum()/r.rolling(5,min_periods=4).std(),'risk_adjusted_trend_acceleration_20_60d':ret20/vol20-ret60/vol60,'return_persistence_autocorr_20obs':r.rolling(20,min_periods=15).corr(r.shift(1)),'return_directional_efficiency_20obs':ret20.abs()/r.abs().rolling(20,min_periods=15).sum(),'relative_liquidity_stress_20_60obs':np.log((r.abs()/v).rolling(20,min_periods=15).mean()/(r.abs()/v).rolling(60,min_periods=45).mean()),'return_sign_balance_20obs':(r>0).rolling(20,min_periods=15).mean()-.5,'volatility_clustering_autocorr_20obs':r.abs().rolling(20,min_periods=15).corr(r.abs().shift(1)),'dispersion_sensitivity_20obs':r.rolling(20,min_periods=15).corr(r.std(1))}
ca=pd.DataFrame(index=r.index,columns=A,dtype=float);xdb=ca.copy()
for t in r.index:
 z=r.loc[:t].tail(60);dn=med.loc[z.index]<0
 for a in A:
  q=pd.concat([z[a],med.loc[z.index]],axis=1);x=q[dn].dropna();y=q[~dn].dropna()
  if len(x)>=8 and len(y)>=8:ca.loc[t,a]=x.iloc[:,0].corr(x.iloc[:,1])-y.iloc[:,0].corr(y.iloc[:,1])
  if len(x)>=8 and x.iloc[:,1].var()!=0:xdb.loc[t,a]=x.iloc[:,0].cov(x.iloc[:,1])/x.iloc[:,1].var()
L['correlation_asymmetry_60obs']=ca
beta=r.rolling(60,min_periods=45).cov(med).div(med.rolling(60,min_periods=45).var(),axis=0);resid=r-beta.mul(med,axis=0)
L['residual_downside_semivol_share_60obs']=np.sqrt(resid.where(resid<0).pow(2).rolling(60,min_periods=45).mean())/np.sqrt(resid.pow(2).rolling(60,min_periods=45).mean())
ort=pd.DataFrame(index=r.index,columns=A,dtype=float);ex=ort.copy()
for t in r.index:
 q=pd.concat([beta.loc[t],vol20.loc[t]],axis=1).dropna()
 if len(q)>=3:ort.loc[t,q.index]=q.iloc[:,0]-np.polyval(np.polyfit(q.iloc[:,1],q.iloc[:,0],1),q.iloc[:,1])
 q=pd.concat([xdb.loc[t],ca.loc[t]],axis=1).dropna()
 if len(q)>=3:ex.loc[t,q.index]=q.iloc[:,0]-np.polyval(np.polyfit(q.iloc[:,1],q.iloc[:,0],1),q.iloc[:,1])
L['vol_orthogonal_median_beta_60obs']=ort;L['excess_downside_beta_ca_orthogonal_60obs']=ex
vx=load('VIX',True).close.pct_change().reindex(r.index);L['vix_shock_relief_beta_asymmetry_60obs']=cond_beta(vx,vx>0)-cond_beta(vx,vx<0);L['adaptive_vix_relief_beta_change_25_60obs']=cond_beta(vx,vx<0,25)-cond_beta(vx,vx<0,60)
dv20=np.log(v.where(r<0).rolling(20,min_periods=10).mean()/v.where(r>=0).rolling(20,min_periods=10).mean());dv60=np.log(v.where(r<0).rolling(60,min_periods=30).mean()/v.where(r>=0).rolling(60,min_periods=30).mean());L['downside_volume_participation_asymmetry_60obs']=np.log(v.where(r<0).rolling(60,min_periods=30).mean()/v.where(r>=0).rolling(60,min_periods=30).mean());L['inverted_downside_volume_participation_acceleration_20_60obs']=-(dv20-dv60);L['overnight_daytime_reversal_concordance_20obs']=(-((o/c.shift(1)-1)*(c/o-1))).rolling(20,min_periods=15).mean()
dxy=load('DXY',True).close.pct_change().reindex(r.index);state=ret20.median(1)>0;L['dxy_median_trend_regime_beta_spread_60obs']=cond_beta(dxy,state)-cond_beta(dxy,~state)
eval=f.index[f.notna().sum(1)>=8];out={}
def getic(y):
 z=[];n=[]
 for t in eval:
  q=pd.concat([f.loc[t],y.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8:z.append((t,spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic));n.append(len(q))
 return pd.Series(dict(z)),np.mean(n)
print('FACTOR usdjpy_up_down_beta_asymmetry_60obs visible_through',END,'assets',len(A),'signal_cells',f.notna().sum().sum(),'/',f.size)
for h in [1,5,10,20]:
 s,n=getic(c.shift(-h)/c-1);out[h]=s;print('H',h,'dates',len(s),'IC %.6f ICIR %.6f hit %.4f mean_instruments %.2f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean(),n))
best=max(out,key=lambda h:abs(out[h].mean()*out[h].mean()/out[h].std(ddof=1)));s=out[best];print('SELECTED',best)
for lo,hi,nm in [('2020','2021','2020'),('2021','2023','2021-22'),('2023','2025','2023-24'),('2025','2030','2025-current')]:
 q=s[(s.index>=lo)&(s.index<hi)];print('REGIME',nm,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
rk=f.rank(1);turn=[]
for i in range(1,len(rk)):
 q=pd.concat([rk.iloc[i],rk.iloc[i-1]],axis=1).dropna()
 if len(q)>=8:turn.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
print('TURNOVER',round(np.mean(turn),6),'coverage',round(f.notna().mean().mean(),4))
mx=(-1,None,0);e=0
for nm,g in L.items():
 z=[]
 for t in eval:
  q=pd.concat([f.loc[t],g.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8:z.append(abs(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic))
 k=max(z) if z else np.nan;print('LIB',nm,'max_abs_rho',k,'dates',len(z));e+=len(z)
 if np.isfinite(k) and k>mx[0]:mx=(k,nm,len(z))
print('MAX_ABS_LIBRARY_CORRELATION %.6f closest %s dates %d evidence_cells %d'%(*mx,e))
